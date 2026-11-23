/*
 * Copyright (c) PyPTO Contributors.
 * SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
 */
#pragma once
#include "a1_tensor.h"

// Resident weights: [expert,K/32,N/16,16,32]. Outputs keep original route order.
template <unsigned Inner, unsigned Width, bool Shared>
__aicore__ inline void BatchedNZExpertProjectionGrouped(__gm__ int64_t* args) {
  constexpr unsigned N_TILE = 256;
  constexpr unsigned M_TILE = 16;
  static_assert(Width % N_TILE == 0 && Inner % 32 == 0);
  const unsigned block = get_block_idx();
  if (block >= 8) return;
  auto* x = reinterpret_cast<__gm__ Tensor*>(args[0]);
  auto* weight = reinterpret_cast<__gm__ Tensor*>(args[1]);
  auto* ids = reinterpret_cast<__gm__ Tensor*>(args[2]);
  auto* output = reinterpret_cast<__gm__ Tensor*>(args[3]);
  auto* status = reinterpret_cast<__gm__ Tensor*>(args[4]);
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
  using B = pto::Tile<pto::TileType::Mat, int8_t, 32, N_TILE, pto::BLayout::RowMajor, 32, N_TILE,
                      pto::SLayout::ColMajor, 512>;
  using L = pto::Tile<pto::TileType::Left, int8_t, M_TILE, 32, pto::BLayout::ColMajor, M_TILE, 32,
                      pto::SLayout::RowMajor, 512>;
  using R = pto::Tile<pto::TileType::Right, int8_t, 32, N_TILE, pto::BLayout::RowMajor, 32, N_TILE,
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
  for (unsigned expert = block; expert < 128; expert += 8) {
    unsigned count = 0;
    for (unsigned route = 0; route < 160; ++route) {
      if (allIds.data()[route] == static_cast<int32_t>(expert)) selected.data()[count++] = route;
    }
    for (unsigned first = 0; first < count; first += M_TILE) {
      const unsigned rows = count - first < M_TILE ? count - first : M_TILE;
      copy_ubuf_to_cbuf(a.data(), z.data(), 0, 1, M_TILE, 0, 0);
      pipe_barrier(PIPE_ALL);
      for (unsigned col = 0; col < Width; col += N_TILE) {
        for (unsigned k = 0; k < Inner; k += 32) {
          for (unsigned lane = 0; lane < rows; ++lane) {
            const unsigned route = selected.data()[first + lane];
            const unsigned inputRow = Shared ? route / 8 : route;
            copy_gm_to_cbuf(a.data() + lane * 32, Data<int8_t>(x) + inputRow * Inner + k,
                           0, 1, 1, 0, 0, (pad_t)0);
          }
          const uint64_t offset = static_cast<uint64_t>(expert) * Width * Inner + (k / 32 * Width + col) * 32;
          copy_gm_to_cbuf(b.data(), Data<int8_t>(weight) + offset, 0, 1, N_TILE, 0, 0, (pad_t)0);
          set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
          wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
          load_cbuf_to_ca(l.data(), a.data(), 0, 1, 1, 0, 0, false, addr_cal_mode_t(0));
          load_cbuf_to_cb(r.data(), b.data(), 0, N_TILE / 16, 1, 0, 0, false, addr_cal_mode_t(0));
          set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
          wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
          mad(c.data(), l.data(), r.data(), M_TILE, 32, N_TILE, k == 0);
          pipe_barrier(PIPE_ALL);
        }
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
      for (unsigned lane = 0; lane < rows; ++lane)
        PublishStatus(Data<int32_t>(status) + selected.data()[first + lane] * 16, 0);
    }
  }
}
