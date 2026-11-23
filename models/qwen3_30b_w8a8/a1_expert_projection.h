/*
 * Copyright (c) PyPTO Contributors.
 * SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
 */
#pragma once
#include "a1_tensor.h"

template <unsigned Inner, unsigned Width>
__aicore__ inline void ExpertProjection(__gm__ int64_t* args) {
  constexpr unsigned blocks = 8;
  const unsigned block = get_block_idx();
  if (block >= blocks) return;
  const unsigned route = block;
  auto* x = reinterpret_cast<__gm__ Tensor*>(args[0]);
  auto* weight = reinterpret_cast<__gm__ Tensor*>(args[1]);
  auto* ids = reinterpret_cast<__gm__ Tensor*>(args[2]);
  auto* output = reinterpret_cast<__gm__ Tensor*>(args[3]);
  auto* statusTensor = reinterpret_cast<__gm__ Tensor*>(args[4]);
  if (statusTensor->ndims != 2 || statusTensor->shapes[0] != blocks || statusTensor->shapes[1] != 16) {
    trap();
    return;
  }
  auto* status = Data<int32_t>(statusTensor) + block * 16;
  if (x->ndims != 2 || x->shapes[0] != 128 || x->shapes[1] != Inner || weight->ndims != 2 ||
      weight->shapes[0] != 128 * Width || weight->shapes[1] != Inner || ids->ndims != 2 ||
      ids->shapes[0] != 1 || ids->shapes[1] != 8 || output->ndims != 2 || output->shapes[0] != 8 ||
      output->shapes[1] != Width) {
    PublishStatus(status, 1);
    return;
  }
  const int32_t expert = Data<int32_t>(ids)[route];
  if (expert < 0 || expert >= 128) {
    PublishStatus(status, 2);
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
  using GlobalA = pto::GlobalTensor<int8_t, pto::Shape<1, 1, 1, 16, 32>, pto::Stride<1, 1, 1, Inner, 1>>;
  using GlobalB =
      pto::GlobalTensor<int8_t, pto::Shape<1, 1, 1, 32, 16>, pto::Stride<1, 1, 1, 1, Inner>, pto::Layout::DN>;
  using GlobalOut = pto::GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<1, 1, 1, Width, 1>>;
  MatA ma;
  MatB mb;
  Left left;
  Right right;
  Acc acc;
  Vec values;
  Row activeRow;
  pto::TASSIGN<0>(ma);
  pto::TASSIGN<32768>(mb);
  pto::TASSIGN<0>(left);
  pto::TASSIGN<0>(right);
  pto::TASSIGN<0>(acc);
  pto::TASSIGN<0>(values);
  pto::TASSIGN<0>(activeRow);
  for (unsigned col = 0; col < Width; col += 16) {
    for (unsigned k = 0; k < Inner; k += 32) {
      GlobalA lhs(Data<int8_t>(x) + route * 16 * Inner + k);
      const uint64_t offset = (static_cast<uint64_t>(expert) * Width + col) * Inner + k;
      GlobalB rhs(Data<int8_t>(weight) + offset);
      pto::TLOAD(ma, lhs);
      pto::TLOAD(mb, rhs);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      pto::TMOV(left, ma);
      pto::TMOV(right, mb);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      if (k == 0)
        pto::TMATMUL(acc, left, right);
      else
        pto::TMATMUL_ACC(acc, acc, left, right);
      // Initial qualification serializes reuse; optimization is a separate gate.
      pipe_barrier(PIPE_ALL);
    }
    pto::TMOV<Vec, Acc, pto::AccToVecMode::SingleModeVec0>(values, acc);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    GlobalOut destination(Data<int32_t>(output) + route * Width + col);
    pto::TSTORE(destination, activeRow);
    pipe_barrier(PIPE_ALL);
  }
  PublishStatus(status, 0);
}
