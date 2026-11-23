/*
 * Copyright (c) PyPTO Contributors.
 * SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
 */
#pragma once
#include "a1_tensor.h"

// Borrow [128,N,K] NZ storage. Select an expert from device top-8 IDs;
// no ND model-sized staging bank or host-side routing is needed.
template <unsigned Inner, unsigned Width, bool SharedTokenInput>
__aicore__ inline void BatchedNZExpertProjection(__gm__ int64_t* args) {
  const unsigned block = get_block_idx();
  if (block >= 8) return;
  auto* x = reinterpret_cast<__gm__ Tensor*>(args[0]);
  auto* weight = reinterpret_cast<__gm__ Tensor*>(args[1]);
  auto* ids = reinterpret_cast<__gm__ Tensor*>(args[2]);
  auto* output = reinterpret_cast<__gm__ Tensor*>(args[3]);
  auto* statusTensor = reinterpret_cast<__gm__ Tensor*>(args[4]);
  if (x->ndims != 2 || x->shapes[0] != (SharedTokenInput ? 20 : 160) || x->shapes[1] != Inner ||
      weight->ndims != 2 || weight->shapes[0] != 128 * Width * (Inner / 32) || weight->shapes[1] != 32 ||
      ids->ndims != 2 || ids->shapes[0] != 20 || ids->shapes[1] != 8 ||
      output->ndims != 2 || output->shapes[0] != 160 || output->shapes[1] != Width ||
      statusTensor->ndims != 2 || statusTensor->shapes[0] != 160 || statusTensor->shapes[1] != 16) {
    trap();
    return;
  }
  using MatA = pto::Tile<pto::TileType::Mat, int8_t, 16, 32, pto::BLayout::RowMajor, 16, 32,
                         pto::SLayout::RowMajor, 512>;
  using MatB = pto::Tile<pto::TileType::Mat, int8_t, 32, 16, pto::BLayout::RowMajor, 32, 16,
                         pto::SLayout::ColMajor, 512>;
  using Left = pto::Tile<pto::TileType::Left, int8_t, 16, 32, pto::BLayout::ColMajor, 16, 32,
                         pto::SLayout::RowMajor, 512>;
  using Right = pto::Tile<pto::TileType::Right, int8_t, 32, 16, pto::BLayout::RowMajor, 32, 16,
                          pto::SLayout::ColMajor, 512>;
  using Acc = pto::Tile<pto::TileType::Acc, int32_t, 16, 16, pto::BLayout::ColMajor, 16, 16,
                        pto::SLayout::RowMajor, 1024>;
  using Vec = pto::Tile<pto::TileType::Vec, int32_t, 16, 16>;
  using Row = pto::Tile<pto::TileType::Vec, int32_t, 1, 16>;
  using Pad = pto::Tile<pto::TileType::Vec, int8_t, 16, 32>;
  using GlobalB = pto::GlobalTensor<int8_t, pto::Shape<1, 1, 1, 32, 16>,
                                  pto::Stride<1, 1, 1, 1, 32>, pto::Layout::DN>;
  using GlobalOut = pto::GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<1, 1, 1, Width, 1>>;
  MatA ma;
  MatB mb;
  Left left;
  Right right;
  Acc acc;
  Vec values;
  Row active;
  Pad zero;
  pto::TASSIGN<0>(ma);
  pto::TASSIGN<32768>(mb);
  pto::TASSIGN<0>(left);
  pto::TASSIGN<0>(right);
  pto::TASSIGN<0>(acc);
  pto::TASSIGN<0>(values);
  pto::TASSIGN<0>(active);
  pto::TASSIGN<1024>(zero);
  for (unsigned i = 0; i < 512; ++i) zero.data()[i] = 0;
  pipe_barrier(PIPE_ALL);
  copy_ubuf_to_cbuf(ma.data(), zero.data(), 0, 1, 16, 0, 0);
  pipe_barrier(PIPE_ALL);
  for (unsigned route = block; route < 160; route += 8) {
    const int32_t expert = Data<int32_t>(ids)[route];
    auto* status = Data<int32_t>(statusTensor) + route * 16;
    if (expert < 0 || expert >= 128) {
      PublishStatus(status, 2);
      continue;
    }
    const unsigned inputRow = SharedTokenInput ? route / 8 : route;
    for (unsigned col = 0; col < Width; col += 16) {
      for (unsigned k = 0; k < Inner; k += 32) {
        copy_gm_to_cbuf(ma.data(), Data<int8_t>(x) + inputRow * Inner + k, 0, 1, 1, 0, 0, (pad_t)0);
        const uint64_t offset = static_cast<uint64_t>(expert) * Width * Inner + (k / 32 * Width + col) * 32;
        GlobalB rhs(Data<int8_t>(weight) + offset);
        pto::TLOAD(mb, rhs);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        pto::TMOV(left, ma);
        pto::TMOV(right, mb);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        if (k == 0) pto::TMATMUL(acc, left, right);
        else pto::TMATMUL_ACC(acc, acc, left, right);
        pipe_barrier(PIPE_ALL);
      }
      pto::TMOV<Vec, Acc, pto::AccToVecMode::SingleModeVec0>(values, acc);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      GlobalOut destination(Data<int32_t>(output) + route * Width + col);
      pto::TSTORE(destination, active);
      pipe_barrier(PIPE_ALL);
    }
    PublishStatus(status, 0);
  }
}
