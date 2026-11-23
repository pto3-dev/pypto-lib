/*
 * Copyright (c) PyPTO Contributors.
 * SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
 */
#pragma once
#include "a1_tensor.h"
#include "a1_grouped_channel_dequant.h"

// Resident weights: [expert,K/32,N/16,16,32]. Outputs keep original route order.
template <unsigned Inner, unsigned Width, bool Shared, unsigned KStep = 32, bool Pipeline = false, bool WideOutput = false, unsigned NColumns = 256, bool ChannelDequantized = false>
__aicore__ inline void BatchedNZExpertProjectionCached(__gm__ int64_t* args) {
  constexpr unsigned N_TILE = NColumns;
  constexpr unsigned M_TILE = 16;
  static_assert(Width % N_TILE == 0 && Inner % 32 == 0);
  static_assert(NColumns == 256 || NColumns == 512);
  static_assert(2 * KStep * N_TILE <= 65536, "Double-buffered Right tiles must fit A1 L0B");
  static_assert(KStep % 32 == 0 && Inner % KStep == 0);
  static_assert(!ChannelDequantized || (Shared && Pipeline && WideOutput && NColumns == 512));
  const unsigned block = get_block_idx();
  if (block >= 8) return;
  auto* x = reinterpret_cast<__gm__ Tensor*>(args[0]);
  auto* weight = reinterpret_cast<__gm__ Tensor*>(args[1]);
  auto* ids = reinterpret_cast<__gm__ Tensor*>(args[2]);
  auto* output = reinterpret_cast<__gm__ Tensor*>(args[3]);
  auto* status = reinterpret_cast<__gm__ Tensor*>(args[ChannelDequantized ? 5 : 4]);
  __gm__ Tensor* channelScale = nullptr;
  if constexpr (ChannelDequantized) {
    channelScale = reinterpret_cast<__gm__ Tensor*>(args[4]);
    if (channelScale->ndims != 2 || channelScale->shapes[0] != 128 || channelScale->shapes[1] != 2 * Width) {
      trap();
      return;
    }
  }
  if (x->ndims != 2 || x->shapes[0] != (Shared ? 20 : 160) || x->shapes[1] != Inner ||
      weight->ndims != 2 || weight->shapes[0] != 128 * Width * (Inner / 32) || weight->shapes[1] != 32 ||
      ids->ndims != 2 || ids->shapes[0] != 20 || ids->shapes[1] != 8 ||
      output->ndims != 2 || output->shapes[0] != 160 || output->shapes[1] != Width ||
      status->ndims != 2 || status->shapes[0] != 160 || status->shapes[1] != 16) {
    trap();
    return;
  }
  using A = pto::Tile<pto::TileType::Mat, int8_t, M_TILE, 32, pto::BLayout::RowMajor, M_TILE, 32,
                      pto::SLayout::RowMajor, 512>;
  using B = pto::Tile<pto::TileType::Mat, int8_t, KStep, N_TILE, pto::BLayout::RowMajor, KStep, N_TILE,
                      pto::SLayout::ColMajor, 512>;
  using L = pto::Tile<pto::TileType::Left, int8_t, M_TILE, KStep, pto::BLayout::ColMajor, M_TILE, KStep,
                      pto::SLayout::RowMajor, 512>;
  using R = pto::Tile<pto::TileType::Right, int8_t, KStep, N_TILE, pto::BLayout::RowMajor, KStep, N_TILE,
                      pto::SLayout::ColMajor, 512>;
  using C = pto::Tile<pto::TileType::Acc, int32_t, M_TILE, N_TILE, pto::BLayout::ColMajor, M_TILE, N_TILE,
                      pto::SLayout::RowMajor, 1024>;
  using V = pto::Tile<pto::TileType::Vec, int32_t, M_TILE, 16>;
  using Row = pto::Tile<pto::TileType::Vec, int32_t, 1, 16>;
  using Z = pto::Tile<pto::TileType::Vec, int8_t, M_TILE, 32>;
  using Routes = pto::Tile<pto::TileType::Vec, int32_t, 1, 160>;
  using IdGlobal = pto::GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 160>, pto::Stride<1, 1, 1, 160, 1>>;
  using Out = pto::GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<1, 1, 1, Width, 1>>;
  A a;
  B b;
  L l;
  R r;
  C c;
  V v;
  Z z;
  Routes allIds, selected;
  pto::TASSIGN<0>(a);
  pto::TASSIGN<32768>(b);
  pto::TASSIGN<0>(l);
  pto::TASSIGN<0>(r);
  pto::TASSIGN<0>(c);
  pto::TASSIGN<0>(v);
  pto::TASSIGN<1024>(z);
  pto::TASSIGN<2048>(allIds);
  pto::TASSIGN<2688>(selected);
  IdGlobal idSource(Data<int32_t>(ids));
  pto::TLOAD(allIds, idSource);
  set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
  for (unsigned route = block; route < 160; route += 8) {
    if (allIds.data()[route] < 0 || allIds.data()[route] >= 128)
      PublishStatus(Data<int32_t>(status) + route * 16, 2);
  }
  for (unsigned i = 0; i < M_TILE * 32; ++i) z.data()[i] = 0;
  pipe_barrier(PIPE_ALL);
  for (unsigned k = 0; k < Inner; k += 32) {
    copy_ubuf_to_cbuf(a.data() + k / 32 * M_TILE * 32, z.data(), 0, 1, M_TILE, 0, 0);
  }
  pipe_barrier(PIPE_ALL);
  using Counts = pto::Tile<pto::TileType::Vec, int32_t, 1, 128>;
  Counts counts;
  // UB [3328,3840) is disjoint from routing and accumulator staging.
  pto::TASSIGN<3328>(counts);
  for (unsigned expert = 0; expert < 128; ++expert) counts.data()[expert] = 0;
  for (unsigned route = 0; route < 160; ++route) {
    const int32_t expert = allIds.data()[route];
    if (expert >= 0 && expert < 128) ++counts.data()[expert];
  }
  unsigned groupIndex = 0;
  for (unsigned expert = 0; expert < 128; ++expert) {
    const unsigned count = counts.data()[expert];
    for (unsigned first = 0; first < count; first += M_TILE) {
#if defined(PYPTO_A1_EXPERT_COLUMN_PARTITION) && PYPTO_A1_EXPERT_COLUMN_PARTITION
      const unsigned group = groupIndex++;
      const unsigned firstColumnTile = (block + 8 - (group * (Width / N_TILE)) % 8) % 8;
      if (firstColumnTile >= Width / N_TILE) continue;
#else
      const unsigned owner = groupIndex++ % 8;
      if (owner != block) continue;
#endif
      unsigned selectedCount = 0;
      for (unsigned route = 0; route < 160; ++route) {
        if (allIds.data()[route] == static_cast<int32_t>(expert)) selected.data()[selectedCount++] = route;
      }
      const unsigned rows = count - first < M_TILE ? count - first : M_TILE;
      // Cache the group's complete activation once in <=32 KiB of L1.
      // Keep K32 physical fractals and unchanged INT32 accumulation order.
      // L1 input layout: [Inner/32, M_TILE, 32].
      for (unsigned lane = 0; lane < rows; ++lane) {
        const unsigned route = selected.data()[first + lane];
        const unsigned inputRow = Shared ? route / 8 : route;
        copy_gm_to_cbuf(a.data() + lane * 32, Data<int8_t>(x) + inputRow * Inner,
                       0, Inner / 32, 1, 0, M_TILE - 1, (pad_t)0);
      }
      pipe_barrier(PIPE_ALL);
#if defined(PYPTO_A1_EXPERT_COLUMN_PARTITION) && PYPTO_A1_EXPERT_COLUMN_PARTITION
      for (unsigned col = firstColumnTile * N_TILE; col < Width; col += 8 * N_TILE) {
#else
      for (unsigned col = 0; col < Width; col += N_TILE) {
#endif
        if constexpr (Pipeline) {
          static_assert(KStep == 64 && Inner / KStep >= 2);
          B bNext;
          L lNext;
          R rNext;
          pto::TASSIGN<32768 + KStep * N_TILE>(bNext);
          pto::TASSIGN<M_TILE * KStep>(lNext);
          pto::TASSIGN<KStep * N_TILE>(rNext);
          for (unsigned slot = 0; slot < 2; ++slot) {
            auto* target = slot == 0 ? b.data() : bNext.data();
            for (unsigned stage = 0; stage < KStep / 32; ++stage) {
              const uint64_t offset = static_cast<uint64_t>(expert) * Width * Inner +
                                      ((slot * KStep / 32 + stage) * Width + col) * 32;
              copy_gm_to_cbuf(target + stage * N_TILE * 32, Data<int8_t>(weight) + offset,
                             0, 1, N_TILE, 0, 0, (pad_t)0);
            }
            if (slot == 0) set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
            else set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
          }
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
          for (unsigned k = 0; k < Inner; k += KStep) {
            const unsigned slot = (k / KStep) & 1;
            auto* sourceB = slot == 0 ? b.data() : bNext.data();
            auto* left = slot == 0 ? l.data() : lNext.data();
            auto* right = slot == 0 ? r.data() : rNext.data();
            if (slot == 0) {
              wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
              wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
            } else {
              wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
              wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
            }
            load_cbuf_to_ca(left, a.data() + k / 32 * M_TILE * 32,
                            0, KStep / 32, 1, 0, 0, false, addr_cal_mode_t(0));
            for (unsigned stage = 0; stage < KStep / 32; ++stage) {
              load_cbuf_to_cb(right + stage * N_TILE * 32, sourceB + stage * N_TILE * 32,
                             0, N_TILE / 16, 1, 0, 0, false, addr_cal_mode_t(0));
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
            const unsigned next = k + 2 * KStep;
            if (next < Inner) {
              for (unsigned stage = 0; stage < KStep / 32; ++stage) {
                const uint64_t offset = static_cast<uint64_t>(expert) * Width * Inner +
                                        ((next / 32 + stage) * Width + col) * 32;
                copy_gm_to_cbuf(sourceB + stage * N_TILE * 32, Data<int8_t>(weight) + offset,
                               0, 1, N_TILE, 0, 0, (pad_t)0);
              }
              if (slot == 0) set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
              else set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
            }
            if (slot == 0) wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
            else wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
            mad(c.data(), left, right, M_TILE, KStep, N_TILE, k == 0);
            if (slot == 0) set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
            else set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
          }
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
          pipe_barrier(PIPE_ALL);
        } else {
          for (unsigned k = 0; k < Inner; k += KStep) {
            for (unsigned stage = 0; stage < KStep / 32; ++stage) {
              const uint64_t offset = static_cast<uint64_t>(expert) * Width * Inner + ((k / 32 + stage) * Width + col) * 32;
              copy_gm_to_cbuf(b.data() + stage * N_TILE * 32, Data<int8_t>(weight) + offset,
                             0, 1, N_TILE, 0, 0, (pad_t)0);
            }
            set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
            load_cbuf_to_ca(l.data(), a.data() + k / 32 * M_TILE * 32,
                            0, KStep / 32, 1, 0, 0, false, addr_cal_mode_t(0));
            for (unsigned stage = 0; stage < KStep / 32; ++stage) {
              load_cbuf_to_cb(r.data() + stage * N_TILE * 32, b.data() + stage * N_TILE * 32,
                             0, N_TILE / 16, 1, 0, 0, false, addr_cal_mode_t(0));
            }
            set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
            wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
            mad(c.data(), l.data(), r.data(), M_TILE, KStep, N_TILE, k == 0);
            pipe_barrier(PIPE_ALL);
          }
        }
        if constexpr (ChannelDequantized) {
          using Packed = pto::Tile<pto::TileType::Vec, int32_t, 1, 2 * N_TILE>;
          using PackedGlobal = pto::GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 2 * N_TILE>,
                                                pto::Stride<1, 1, 1, 2 * Width, 1>>;
          using Rounded = pto::Tile<pto::TileType::Vec, half, 1, M_TILE * N_TILE>;
          using HalfRow = pto::Tile<pto::TileType::Vec, half, 1, N_TILE>;
          using HalfOut = pto::GlobalTensor<half, pto::Shape<1, 1, 1, 1, N_TILE>,
                                            pto::Stride<1, 1, 1, Width, 1>>;
          Packed packed;
          Rounded rounded;
          HalfRow activeRow;
          pto::TASSIGN<4096>(packed);
          pto::TASSIGN<8192>(rounded);
          pto::TASSIGN<8192 + M_TILE * N_TILE * sizeof(half)>(activeRow);
          PackedGlobal table(Data<int32_t>(channelScale) + expert * 2 * Width + col * 2);
          pto::TLOAD(packed, table);
          pipe_barrier(PIPE_ALL);
          GroupedChannelDequant16<N_TILE>(c, packed, rounded);
          for (unsigned lane = 0; lane < rows; ++lane) {
            copy_ubuf_to_ubuf(activeRow.data(), rounded.data() + lane * 16, 0, N_TILE / 16, 1, 15, 0);
            pipe_barrier(PIPE_ALL);
            const unsigned route = selected.data()[first + lane];
            HalfOut destination(Data<half>(output) + route * Width + col);
            pto::TSTORE(destination, activeRow);
            pipe_barrier(PIPE_ALL);
          }
        } else if constexpr (WideOutput) {
          using FullAccumulator = pto::Tile<pto::TileType::Vec, int32_t, 1, M_TILE * N_TILE>;
          using FullRow = pto::Tile<pto::TileType::Vec, int32_t, 1, N_TILE>;
          using FullOut = pto::GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, N_TILE>,
                                            pto::Stride<1, 1, 1, Width, 1>>;
          FullAccumulator allRows;
          FullRow activeRow;
          pto::TASSIGN<4096>(allRows);
          pto::TASSIGN<4096 + M_TILE * N_TILE * sizeof(int32_t)>(activeRow);
          // Preserve the CUBE's integer NZ bits, then gather each complete row.
          copy_matrix_cc_to_ubuf(allRows.data(), c.data(), 0, 1, N_TILE / 16, 0, 0, CRMODE_NONE);
          pipe_barrier(PIPE_ALL);
          for (unsigned lane = 0; lane < rows; ++lane) {
            copy_ubuf_to_ubuf(activeRow.data(), allRows.data() + lane * 16,
                             0, N_TILE / 16, 2, 30, 0);
            pipe_barrier(PIPE_ALL);
            const unsigned route = selected.data()[first + lane];
            FullOut destination(Data<int32_t>(output) + route * Width + col);
            pto::TSTORE(destination, activeRow);
            pipe_barrier(PIPE_ALL);
          }
        } else {
          for (unsigned sub = 0; sub < N_TILE / 16; ++sub) {
            copy_matrix_cc_to_ubuf(v.data(), c.data() + sub * 256, 0, 1, 1, 0, 0, CRMODE_NONE);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            for (unsigned lane = 0; lane < rows; ++lane) {
              Row row;
              pto::TASSIGN(row, reinterpret_cast<uint64_t>(v.data() + lane * 16));
              const unsigned route = selected.data()[first + lane];
              Out destination(Data<int32_t>(output) + route * Width + col + sub * 16);
              pto::TSTORE(destination, row);
            }
            pipe_barrier(PIPE_ALL);
          }
        }
      }
#if defined(PYPTO_A1_EXPERT_COLUMN_PARTITION) && PYPTO_A1_EXPERT_COLUMN_PARTITION
      // One writer per route. Ordered-stream kernel completion joins all column owners
      // before the consumer kernel or the host's final status read.
      if (firstColumnTile == 0) {
#endif
      for (unsigned lane = 0; lane < rows; ++lane)
        PublishStatus(Data<int32_t>(status) + selected.data()[first + lane] * 16, 0);
#if defined(PYPTO_A1_EXPERT_COLUMN_PARTITION) && PYPTO_A1_EXPERT_COLUMN_PARTITION
      }
#endif
    }
  }
}
