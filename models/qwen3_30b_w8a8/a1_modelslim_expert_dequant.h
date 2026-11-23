/*
 * Copyright (c) PyPTO Contributors.
 * SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
 */
#pragma once
#include "a1_tensor.h"

// The 310P grouped-matmul kernel materializes its per-channel VDEQ16 result
// before multiplying by the token scale. Software FP32 multiplication is not
// interchangeable with this hardware boundary. The caller reserves UB8192..9856
// and L0C0..1024 for the duration of this helper.
template <typename IntegerTile, typename ScaleTile, typename HalfTile>
__aicore__ inline void ModelSlimChannelDequant(IntegerTile& input, ScaleTile& scale, HalfTile& output) {
  static_assert(IntegerTile::Rows == 1 && ScaleTile::Rows == 1 && HalfTile::Rows == 1);
  static_assert(IntegerTile::Cols == ScaleTile::Cols && IntegerTile::Cols == HalfTile::Cols);
  static_assert(IntegerTile::Cols % 16 == 0);
  using Matrix = pto::Tile<pto::TileType::Vec, int32_t, 16, 16>;
  using Rounded = pto::Tile<pto::TileType::Vec, half, 16, 16>;
  using Words = pto::Tile<pto::TileType::Vec, int32_t, 1, 32>;
  using Acc = pto::Tile<pto::TileType::Acc, int32_t, 16, 16, pto::BLayout::ColMajor,
                        16, 16, pto::SLayout::RowMajor, 1024>;
  Matrix matrix;
#ifdef PYPTO_A1_BATCHED_VECTOR_PADDING
  using ZeroStorage = pto::Tile<pto::TileType::Vec, float, 1, 256>;
  ZeroStorage zeros;
#endif
  Rounded rounded;
  Words words;
  Acc l0c;
  pto::TASSIGN<8192>(matrix);
#ifdef PYPTO_A1_BATCHED_VECTOR_PADDING
  pto::TASSIGN<8192>(zeros);
#endif
  pto::TASSIGN<9216>(rounded);
  pto::TASSIGN<9728>(words);
  pto::TASSIGN<0>(l0c);
  auto* scaleBits = reinterpret_cast<__ubuf__ uint32_t*>(scale.data());
  auto* packed = reinterpret_cast<__ubuf__ uint32_t*>(words.data());
  auto* result = reinterpret_cast<__ubuf__ uint16_t*>(output.data());
  auto* active = reinterpret_cast<__ubuf__ uint16_t*>(rounded.data());
  set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
#ifdef PYPTO_A1_BATCHED_VECTOR_PADDING
  // A positive FP32 zero clears the same INT32 padding bits without converting
  // active accumulators or scales. Only row zero changes between channel blocks.
  pto::TEXPANDS(zeros, 0.0f);
  pipe_barrier(PIPE_ALL);
#endif
  for (unsigned base = 0; base < IntegerTile::Cols; base += 16) {
#ifndef PYPTO_A1_BATCHED_VECTOR_PADDING
    for (unsigned element = 0; element < 256; ++element) matrix.data()[element] = 0;
#endif
    for (unsigned lane = 0; lane < 16; ++lane) {
      matrix.data()[lane] = input.data()[base + lane];
      packed[2 * lane] = scaleBits[base + lane];
      packed[2 * lane + 1] = 0;
    }
    pipe_barrier(PIPE_ALL);
    copy_matrix_ubuf_to_cc(l0c.data(), matrix.data(), 0, 1, 1, 0, 0, CRMODE_NONE);
    pipe_barrier(PIPE_ALL);
    set_deqscale(reinterpret_cast<uint64_t>(words.data()) >> 5);
    copy_matrix_cc_to_ubuf(rounded.data(), l0c.data(), 0, 1, 1, 0, 0, CRMODE_DEQSCALE_VDEQ16);
    pipe_barrier(PIPE_ALL);
    for (unsigned lane = 0; lane < 16; ++lane) result[base + lane] = active[lane];
  }
  set_flag(PIPE_S, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
}
