/*
 * Copyright (c) PyPTO Contributors.
 * SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
 */
#pragma once
#include "a1_tensor.h"

// The initial A1 PTO ISA matrix helpers only expose N=16. Use explicit CUBE
// instructions for N=256 without relaxing their guards or changing epilogues.
// Resident bytes are [expert,K/32,N/16,16,32]; L0C is [N/16,M/16,16,16].
template <unsigned Inner, unsigned Width, bool Shared>
__aicore__ inline void BatchedNZExpertProjectionWide(__gm__ int64_t* args) {
  constexpr unsigned N = 256;
  static_assert(Width % N == 0 && Inner % 32 == 0);
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
  using A = pto::Tile<pto::TileType::Mat, int8_t, 16, 32, pto::BLayout::RowMajor, 16, 32,
                      pto::SLayout::RowMajor, 512>;
  using B = pto::Tile<pto::TileType::Mat, int8_t, 32, N, pto::BLayout::RowMajor, 32, N,
                      pto::SLayout::ColMajor, 512>;
  using L = pto::Tile<pto::TileType::Left, int8_t, 16, 32, pto::BLayout::ColMajor, 16, 32,
                      pto::SLayout::RowMajor, 512>;
  using R = pto::Tile<pto::TileType::Right, int8_t, 32, N, pto::BLayout::RowMajor, 32, N,
                      pto::SLayout::ColMajor, 512>;
  using C = pto::Tile<pto::TileType::Acc, int32_t, 16, N, pto::BLayout::ColMajor, 16, N,
                      pto::SLayout::RowMajor, 1024>;
  using V = pto::Tile<pto::TileType::Vec, int32_t, 16, 16>;
  using Row = pto::Tile<pto::TileType::Vec, int32_t, 1, 16>;
  using Z = pto::Tile<pto::TileType::Vec, int8_t, 16, 32>;
  using Out = pto::GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<1, 1, 1, Width, 1>>;
  A a;
  B b;
  L l;
  R r;
  C c;
  V v;
  Row row;
  Z z;
  pto::TASSIGN<0>(a);
  pto::TASSIGN<32768>(b);
  pto::TASSIGN<0>(l);
  pto::TASSIGN<0>(r);
  pto::TASSIGN<0>(c);
  pto::TASSIGN<0>(v);
  pto::TASSIGN<0>(row);
  pto::TASSIGN<1024>(z);
  for (unsigned i = 0; i < 512; ++i) z.data()[i] = 0;
  pipe_barrier(PIPE_ALL);
  copy_ubuf_to_cbuf(a.data(), z.data(), 0, 1, 16, 0, 0);
  pipe_barrier(PIPE_ALL);
  for (unsigned route = block; route < 160; route += 8) {
    const int expert = Data<int32_t>(ids)[route];
    if (expert < 0 || expert >= 128) {
      PublishStatus(Data<int32_t>(status) + route * 16, 2);
      continue;
    }
    for (unsigned col = 0; col < Width; col += N) {
      for (unsigned k = 0; k < Inner; k += 32) {
        copy_gm_to_cbuf(a.data(), Data<int8_t>(x) + (Shared ? route / 8 : route) * Inner + k,
                        0, 1, 1, 0, 0, (pad_t)0);
        const uint64_t offset = static_cast<uint64_t>(expert) * Width * Inner + (k / 32 * Width + col) * 32;
        copy_gm_to_cbuf(b.data(), Data<int8_t>(weight) + offset, 0, 1, N, 0, 0, (pad_t)0);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        load_cbuf_to_ca(l.data(), a.data(), 0, 1, 1, 0, 0, false, addr_cal_mode_t(0));
        load_cbuf_to_cb(r.data(), b.data(), 0, N / 16, 1, 0, 0, false, addr_cal_mode_t(0));
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        mad(c.data(), l.data(), r.data(), 16, 32, N, k == 0);
        pipe_barrier(PIPE_ALL);
      }
      for (unsigned sub = 0; sub < N / 16; ++sub) {
        copy_matrix_cc_to_ubuf(v.data(), c.data() + sub * 256, 0, 1, 1, 0, 0, CRMODE_NONE);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        Out destination(Data<int32_t>(output) + route * Width + col + sub * 16);
        pto::TSTORE(destination, row);
        pipe_barrier(PIPE_ALL);
      }
    }
    PublishStatus(Data<int32_t>(status) + route * 16, 0);
  }
}
