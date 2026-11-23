/*
 * Copyright (c) PyPTO Contributors.
 * SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
 */
#pragma once
#include "a1_tensor.h"

template <unsigned Inner, unsigned Width, unsigned NTile>
__aicore__ inline void PrefetchProjectionStage(__gm__ int8_t* input,
                                              __gm__ int8_t* weight,
                                              unsigned row, unsigned col,
                                              unsigned k, unsigned slot) {
  using Input = pto::Tile<pto::TileType::Mat, int8_t, 16, 32,
                          pto::BLayout::RowMajor, 16, 32,
                          pto::SLayout::RowMajor, 512>;
  using Weight = pto::Tile<pto::TileType::Mat, int8_t, 64, NTile,
                           pto::BLayout::RowMajor, 64, NTile,
                           pto::SLayout::ColMajor, 512>;
  using Global = pto::GlobalTensor<int8_t, pto::Shape<1, 1, 1, 16, 32>,
                                   pto::Stride<1, 1, 1, Inner, 1>>;
  Input first, second;
  Weight bank;
  pto::TASSIGN(first, slot * 1024);
  pto::TASSIGN(second, slot * 1024 + 512);
  pto::TASSIGN(bank, 32768 + slot * 64 * NTile);
  Global lhs(input + row * Inner + k);
  Global lhsNext(input + row * Inner + k + 32);
  pto::TLOAD(first, lhs);
  pto::TLOAD(second, lhsNext);
  for (unsigned stage = 0; stage < 2; ++stage) {
    copy_gm_to_cbuf(bank.data() + stage * NTile * 32,
                    weight + ((k / 32 + stage) * Width + col) * 32,
                    0, 1, NTile, 0, 0, (pad_t)0);
  }
}

// Weight descriptors expose physical [K/32, N/16, 16, 32] bytes, not logical ND.
template <unsigned Inner, unsigned Width, unsigned NTile = 256, bool DequantOutput = false, bool PackedScaleWords = false>
__aicore__ inline void BatchedNZProjectionWide(__gm__ int64_t* args) {
  static_assert(NTile % 16 == 0 && Width % NTile == 0);
  static_assert(Inner % 64 == 0);
  static_assert(!PackedScaleWords || DequantOutput);
  constexpr unsigned blocks = 8;
  const unsigned block = get_block_idx();
  if (block >= blocks) return;
  auto* x = reinterpret_cast<__gm__ Tensor*>(args[0]);
  auto* weight = reinterpret_cast<__gm__ Tensor*>(args[1]);
  auto* output = reinterpret_cast<__gm__ Tensor*>(args[DequantOutput ? 4 : 2]);
  auto* statusTensor = reinterpret_cast<__gm__ Tensor*>(args[DequantOutput ? 5 : 3]);
  if (statusTensor->ndims != 2 || statusTensor->shapes[0] != blocks || statusTensor->shapes[1] != 16) {
    trap();
    return;
  }
  auto* status = Data<int32_t>(statusTensor) + block * 16;
  if (x->ndims != 2 || x->shapes[0] != 32 || x->shapes[1] != Inner || weight->ndims != 2 ||
      weight->shapes[0] != Inner / 32 * Width || weight->shapes[1] != 32 || output->ndims != 2 ||
      output->shapes[0] != (DequantOutput ? 20 : 32) || output->shapes[1] != Width) {
    PublishStatus(status, 1);
    return;
  }
  __gm__ Tensor* bias = nullptr;
  __gm__ Tensor* scale = nullptr;
  if constexpr (DequantOutput) {
    bias = reinterpret_cast<__gm__ Tensor*>(args[2]);
    scale = reinterpret_cast<__gm__ Tensor*>(args[3]);
    if (bias->ndims != 2 || bias->shapes[0] != 1 || bias->shapes[1] != Width ||
        scale->ndims != 2 || scale->shapes[0] != 1 || scale->shapes[1] != (PackedScaleWords ? 2 * Width : Width)) {
      PublishStatus(status, 2);
      return;
    }
  }
  using MatA = pto::Tile<pto::TileType::Mat, int8_t, 16, 32, pto::BLayout::RowMajor, 16, 32,
                         pto::SLayout::RowMajor, 512>;
  using MatB = pto::Tile<pto::TileType::Mat, int8_t, 64, NTile, pto::BLayout::RowMajor, 64, NTile,
                         pto::SLayout::ColMajor, 512>;
  using Left = pto::Tile<pto::TileType::Left, int8_t, 16, 64, pto::BLayout::ColMajor, 16, 64,
                         pto::SLayout::RowMajor, 512>;
  using Right = pto::Tile<pto::TileType::Right, int8_t, 64, NTile, pto::BLayout::RowMajor, 64, NTile,
                          pto::SLayout::ColMajor, 512>;
  using Acc = pto::Tile<pto::TileType::Acc, int32_t, 16, NTile, pto::BLayout::ColMajor, 16, NTile,
                        pto::SLayout::RowMajor, 1024>;
  using Vec = pto::Tile<pto::TileType::Vec, int32_t, 16, 16>;
  using GlobalA = pto::GlobalTensor<int8_t, pto::Shape<1, 1, 1, 16, 32>, pto::Stride<1, 1, 1, Inner, 1>>;
  using GlobalOut = pto::GlobalTensor<int32_t, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<1, 1, 1, Width, 1>>;
  MatA ma, maNext;
  MatB mb;
  Left left;
  Right right;
  Acc acc;
  Vec values;
  pto::TASSIGN<0>(ma);
  pto::TASSIGN<512>(maNext);
  pto::TASSIGN<32768>(mb);
  pto::TASSIGN<0>(left);
  pto::TASSIGN<0>(right);
  pto::TASSIGN<0>(acc);
  pto::TASSIGN<0>(values);
  for (unsigned task = block; task < 2 * (Width / NTile); task += blocks) {
    const unsigned row = task / (Width / NTile) * 16;
    const unsigned col = task % (Width / NTile) * NTile;
    if constexpr (DequantOutput && NTile == 256 &&
                  ((Inner == 2048 && Width == 5120) || (Inner == 4096 && Width == 2048))) {
      static_assert(2 * 64 * NTile <= 65536);
      MatB mbNext;
      Left leftNext;
      Right rightNext;
      pto::TASSIGN<32768 + 64 * NTile>(mbNext);
      pto::TASSIGN<1024>(leftNext);
      pto::TASSIGN<64 * NTile>(rightNext);
      for (unsigned slot = 0; slot < 2; ++slot) {
        PrefetchProjectionStage<Inner, Width, NTile>(
            Data<int8_t>(x), Data<int8_t>(weight), row, col, slot * 64, slot);
        if (slot == 0) set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        else set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      }
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
      for (unsigned k = 0; k < Inner; k += 64) {
        const unsigned slot = (k / 64) & 1;
        auto* sourceB = slot == 0 ? mb.data() : mbNext.data();
        auto* leftSlot = slot == 0 ? left.data() : leftNext.data();
        auto* rightSlot = slot == 0 ? right.data() : rightNext.data();
        MatA inputSlot;
        pto::TASSIGN(inputSlot, slot * 1024);
        if (slot == 0) {
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
          wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        } else {
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
          wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
        }
        load_cbuf_to_ca(leftSlot, inputSlot.data(), 0, 2, 1, 0, 0, false, addr_cal_mode_t(0));
        for (unsigned stage = 0; stage < 2; ++stage) {
          load_cbuf_to_cb(rightSlot + stage * NTile * 32,
                          sourceB + stage * NTile * 32,
                          0, NTile / 16, 1, 0, 0, false, addr_cal_mode_t(0));
        }
        if (slot == 0) {
          set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
          set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
          wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        } else {
          set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
          set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
          wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        }
        if (k + 128 < Inner) {
          PrefetchProjectionStage<Inner, Width, NTile>(
              Data<int8_t>(x), Data<int8_t>(weight), row, col, k + 128, slot);
          if (slot == 0) set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
          else set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
        }
        if (slot == 0) wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        else wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
        mad(acc.data(), leftSlot, rightSlot, 16, 64, NTile, k == 0);
        if (slot == 0) set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        else set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
      }
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
      pipe_barrier(PIPE_ALL);
    } else {
      for (unsigned k = 0; k < Inner; k += 64) {
        GlobalA lhs(Data<int8_t>(x) + row * Inner + k);
        GlobalA lhsNext(Data<int8_t>(x) + row * Inner + k + 32);
        pto::TLOAD(ma, lhs);
        pto::TLOAD(maNext, lhsNext);
        for (unsigned stage = 0; stage < 2; ++stage) {
          copy_gm_to_cbuf(mb.data() + stage * NTile * 32,
                         Data<int8_t>(weight) + ((k / 32 + stage) * Width + col) * 32,
                         0, 1, NTile, 0, 0, (pad_t)0);
        }
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        load_cbuf_to_ca(left.data(), ma.data(), 0, 2, 1, 0, 0, false, addr_cal_mode_t(0));
        for (unsigned stage = 0; stage < 2; ++stage) {
          load_cbuf_to_cb(right.data() + stage * NTile * 32, mb.data() + stage * NTile * 32,
                          0, NTile / 16, 1, 0, 0, false, addr_cal_mode_t(0));
        }
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        mad(acc.data(), left.data(), right.data(), 16, 64, NTile, k == 0);
        pipe_barrier(PIPE_ALL);
      }
    }
    for (unsigned sub = 0; sub < NTile / 16; ++sub) {
      if constexpr (DequantOutput) {
        // Preserve the integer bias and native VDEQ16 narrowing. Process all
        // active rows together, without a diagnostic INT32 GM round trip.
        using Bias = pto::Tile<pto::TileType::Vec, int32_t, 1, 16>;
        using Scale = pto::Tile<pto::TileType::Vec, float, 1, 16>;
        using Words = pto::Tile<pto::TileType::Vec, int32_t, 1, 32>;
        using Halves = pto::Tile<pto::TileType::Vec, half, 16, 16>;
        using GlobalBias = pto::GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<1, 1, 1, 16, 1>>;
        using GlobalScale = pto::GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<1, 1, 1, 16, 1>>;
        Bias biases;
        Scale scales;
        Words words;
        Halves rounded;
        pto::TASSIGN<1024>(biases);
        pto::TASSIGN<1088>(scales);
        pto::TASSIGN<1152>(words);
        pto::TASSIGN<1280>(rounded);
        GlobalBias channelBias(Data<int32_t>(bias) + col + sub * 16);
        pto::TLOAD(biases, channelBias);
        if constexpr (PackedScaleWords) {
          using GlobalWords = pto::GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 32>,
                                               pto::Stride<1, 1, 1, 32, 1>>;
          GlobalWords channelWords(Data<int32_t>(scale) + 2 * (col + sub * 16));
          pto::TLOAD(words, channelWords);
        } else {
          GlobalScale channelScale(Data<float>(scale) + col + sub * 16);
          pto::TLOAD(scales, channelScale);
        }
        copy_matrix_cc_to_ubuf(values.data(), acc.data() + sub * 256, 0, 1, 1, 0, 0, CRMODE_NONE);
        pipe_barrier(PIPE_ALL);
        const unsigned validRows = row == 0 ? 16 : 4;
        set_mask_norm();
        set_vector_mask(0, 0xffff);
        vadd(values.data(), values.data(), biases.data(), static_cast<uint8_t>(validRows),
             1, 1, 1, 2, 2, 0);
        pipe_barrier(PIPE_V);
        set_vector_mask(UINT64_MAX, UINT64_MAX);
        if constexpr (!PackedScaleWords) {
          auto* scaleBits = reinterpret_cast<__ubuf__ uint32_t*>(scales.data());
          auto* packed = reinterpret_cast<__ubuf__ uint32_t*>(words.data());
          for (unsigned lane = 0; lane < 16; ++lane) {
            packed[2 * lane] = scaleBits[lane];
            packed[2 * lane + 1] = 0;
          }
        }
        pipe_barrier(PIPE_ALL);
        copy_matrix_ubuf_to_cc(acc.data() + sub * 256, values.data(), 0, 1, 1, 0, 0, CRMODE_NONE);
        pipe_barrier(PIPE_ALL);
        set_deqscale(reinterpret_cast<uint64_t>(words.data()) >> 5);
        copy_matrix_cc_to_ubuf(rounded.data(), acc.data() + sub * 256, 0, 1, 1, 0, 0, CRMODE_DEQSCALE_VDEQ16);
        pipe_barrier(PIPE_ALL);
        // A complete half row is one 32-byte DMA block. The second M tile
        // writes four rows only, so no padded row can overwrite the output.
        copy_ubuf_to_gm(Data<half>(output) + row * Width + col + sub * 16, rounded.data(),
                       0, validRows, 1, 0, Width / 16 - 1);
        pipe_barrier(PIPE_ALL);
        continue;
      }
      copy_matrix_cc_to_ubuf(values.data(), acc.data() + sub * 256, 0, 1, 1, 0, 0, CRMODE_NONE);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      GlobalOut destination(Data<int32_t>(output) + row * Width + col + sub * 16);
      pto::TSTORE(destination, values);
      pipe_barrier(PIPE_ALL);
    }
  }
  PublishStatus(status, 0);
}
