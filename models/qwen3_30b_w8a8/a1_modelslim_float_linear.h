/*
 * Copyright (c) PyPTO Contributors.
 * SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
 */
#pragma once
#include "a1_tensor.h"

// Prepack FP16 checkpoint weights as contiguous [2048,Width] exactly once.
// The default uses row zero; Batched fills up to16 independent CUBE rows.
template <unsigned Width, bool FloatOutput, unsigned Tokens = 1, bool Batched = false>
__aicore__ inline void ModelSlimFloatLinear(__gm__ int64_t* args) {
  const unsigned block = get_block_idx();
  if (block >= 8) return;
  auto* input = reinterpret_cast<__gm__ Tensor*>(args[0]);
  auto* weight = reinterpret_cast<__gm__ Tensor*>(args[1]);
  auto* output = reinterpret_cast<__gm__ Tensor*>(args[2]);
  auto* status = reinterpret_cast<__gm__ Tensor*>(args[3]);
  if (input->ndims != 2 || input->shapes[0] != Tokens || input->shapes[1] != 2048 ||
      weight->ndims != 2 || weight->shapes[0] != 2048 || weight->shapes[1] != Width ||
      output->ndims != 2 || output->shapes[0] != Tokens || output->shapes[1] != Width ||
      status->ndims != 2 || status->shapes[0] != 8 || status->shapes[1] != 16) {
    trap();
    return;
  }
  using Row = pto::Tile<pto::TileType::Vec, half, 1, 16>;
  using Padded = pto::Tile<pto::TileType::Vec, half, 16, 16>;
  using Mat = pto::Tile<pto::TileType::Mat, half, 16, 16, pto::BLayout::RowMajor,
                        16, 16, pto::SLayout::RowMajor, 512>;
  using Left = pto::Tile<pto::TileType::Left, half, 16, 16, pto::BLayout::ColMajor,
                         16, 16, pto::SLayout::RowMajor, 512>;
  using Right = pto::Tile<pto::TileType::Right, half, 16, 16, pto::BLayout::RowMajor,
                          16, 16, pto::SLayout::ColMajor, 512>;
  using Acc = pto::Tile<pto::TileType::Acc, float, 16, 16, pto::BLayout::ColMajor,
                        16, 16, pto::SLayout::RowMajor, 1024>;
  using Values = pto::Tile<pto::TileType::Vec, float, 16, 16>;
  using FloatRow = pto::Tile<pto::TileType::Vec, float, 1, 16>;
  using InputGlobal = pto::GlobalTensor<half, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<1, 1, 1, 2048, 1>>;
  using WeightGlobal = pto::GlobalTensor<half, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<1, 1, 1, Width, 1>>;
  using HalfGlobal = pto::GlobalTensor<half, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<1, 1, 1, Width, 1>>;
  using FloatGlobal = pto::GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<1, 1, 1, Width, 1>>;
  Row x, active;
  Padded padded, rounded;
  Mat ma, mb;
  Left left;
  Right right;
  Acc acc;
  Values values;
  FloatRow activeFloat;
  pto::TASSIGN<512>(x);
  pto::TASSIGN<512>(padded);
  pto::TASSIGN<1024>(values);
  pto::TASSIGN<1024>(activeFloat);
  pto::TASSIGN<2048>(rounded);
  pto::TASSIGN<2048>(active);
  pto::TASSIGN<0>(ma);
  pto::TASSIGN<32768>(mb);
  pto::TASSIGN<0>(left);
  pto::TASSIGN<0>(right);
  pto::TASSIGN<0>(acc);
  if constexpr (!Batched) {
    auto* bits = reinterpret_cast<__ubuf__ uint16_t*>(padded.data());
    for (unsigned element = 0; element < 256; ++element) bits[element] = 0;
    pipe_barrier(PIPE_ALL);
  }
  for (unsigned token = 0; token < Tokens; token += Batched ? 16 : 1) {
  for (unsigned column = block * 16; column < Width; column += 8 * 16) {
    if constexpr (Batched) {
      using ResidentWeight = pto::Tile<pto::TileType::Mat, half, 1024, 16,
          pto::BLayout::RowMajor, 1024, 16, pto::SLayout::RowMajor, 512>;
      using ResidentWeightGlobal = pto::GlobalTensor<half,
          pto::Shape<1, 1, 1, 1024, 16>, pto::Stride<1, 1, 1, Width, 1>>;
      using ResidentInput = pto::Tile<pto::TileType::Mat, half, 16, 1024,
          pto::BLayout::RowMajor, 16, 1024, pto::SLayout::RowMajor, 512>;
      using ResidentLeft = pto::Tile<pto::TileType::Left, half, 16, 1024,
          pto::BLayout::ColMajor, 16, 1024, pto::SLayout::RowMajor, 512>;
      using ResidentRight = pto::Tile<pto::TileType::Right, half, 1024, 16,
          pto::BLayout::RowMajor, 1024, 16, pto::SLayout::ColMajor, 512>;
      ResidentInput inputCache;
      ResidentWeight weightCache;
      ResidentLeft residentLeft;
      ResidentRight residentRight;
      pto::TASSIGN<0>(inputCache);
      pto::TASSIGN<32768>(weightCache);
      pto::TASSIGN<0>(residentLeft);
      pto::TASSIGN<0>(residentRight);
      const unsigned rows = Tokens - token < 16 ? Tokens - token : 16;
      for (unsigned base = 0; base < 2048; base += 1024) {
        // Each phase occupies32 KiB in L0A/L0B; issue one long-K MAD per phase.
        // Tail M lanes are never stored; read only the valid input rows.
        for (unsigned k = 0; k < 1024; k += 16) {
          copy_gm_to_cbuf(inputCache.data() + k / 16 * 256,
                         Data<half>(input) + token * 2048 + base + k,
                         0, rows, 1, 127, 0, (pad_t)0);
        }
        ResidentWeightGlobal weights(Data<half>(weight) + base * Width + column);
        pto::TLOAD(weightCache, weights);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        load_cbuf_to_ca(residentLeft.data(), inputCache.data(),
                        0, 64, 1, 0, 0, false, addr_cal_mode_t(0));
        load_cbuf_to_cb(residentRight.data(), weightCache.data(),
                        0, 64, 1, 0, 0, true, addr_cal_mode_t(0));
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        // Accumulate each resident K1024 operand pair before draining its buffers.
        mad(acc.data(), residentLeft.data(), residentRight.data(), 16, 1024, 16, base == 0);
        pipe_barrier(PIPE_ALL);
      }
    } else {
      for (unsigned k = 0; k < 2048; k += 16) {
        InputGlobal source(Data<half>(input) + token * 2048 + k);
        WeightGlobal weights(Data<half>(weight) + k * Width + column);
        pto::TLOAD(x, source);
        pto::TLOAD(mb, weights);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        pipe_barrier(PIPE_ALL);
        copy_ubuf_to_cbuf(ma.data(), padded.data(), 0, 1, 16, 0, 0);
        pipe_barrier(PIPE_ALL);
        pto::TMOV(left, ma);
        pto::TMOV(right, mb);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        if (k == 0) pto::TMATMUL(acc, left, right);
        else pto::TMATMUL_ACC(acc, acc, left, right);
        pipe_barrier(PIPE_ALL);
      }
    }
    pto::TMOV<Values, Acc, pto::AccToVecMode::SingleModeVec0>(values, acc);
    pipe_barrier(PIPE_V);
    pto::TCVT(rounded, values, pto::RoundMode::CAST_RINT);
    pipe_barrier(PIPE_V);
    if constexpr (Batched) {
      const unsigned rows = Tokens - token < 16 ? Tokens - token : 16;
      if constexpr (FloatOutput) {
        pto::TCVT(values, rounded, pto::RoundMode::CAST_NONE);
        pipe_barrier(PIPE_V);
      }
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      for (unsigned row = 0; row < rows; ++row) {
        if constexpr (FloatOutput) {
          FloatRow live;
          pto::TASSIGN(live, reinterpret_cast<uint64_t>(values.data() + row * 16));
          FloatGlobal target(Data<float>(output) + (token + row) * Width + column);
          pto::TSTORE(target, live);
        } else {
          Row live;
          pto::TASSIGN(live, reinterpret_cast<uint64_t>(rounded.data() + row * 16));
          HalfGlobal target(Data<half>(output) + (token + row) * Width + column);
          pto::TSTORE(target, live);
        }
      }
    } else {
      if constexpr (FloatOutput) {
        pto::TCVT(values, rounded, pto::RoundMode::CAST_NONE);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        FloatGlobal target(Data<float>(output) + token * Width + column);
        pto::TSTORE(target, activeFloat);
      } else {
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        HalfGlobal target(Data<half>(output) + token * Width + column);
        pto::TSTORE(target, active);
      }
    }
    pipe_barrier(PIPE_ALL);
  }
  }
  PublishStatus(Data<int32_t>(status) + block * 16, 0);
}
