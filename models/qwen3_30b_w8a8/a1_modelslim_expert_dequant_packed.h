/*
 * Copyright (c) PyPTO Contributors.
 * SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
 */
#pragma once
#include "a1_tensor.h"

// Preserve hardware VDEQ16. Packed words contain [FP32 scale bits, zero] per channel.
// Scratch: UB32768..45056 and L0C0..8192; no overlap with retained SwiGLU values.
template <typename IntegerTile, typename PackedTile, typename HalfTile>
__aicore__ inline void PackedModelSlimChannelDequant128(
    IntegerTile& input, PackedTile& scales, HalfTile& output) {
  static_assert(IntegerTile::Rows == 1 && IntegerTile::Cols == 128);
  static_assert(PackedTile::Rows == 1 && PackedTile::Cols == 256);
  static_assert(HalfTile::Rows == 1 && HalfTile::Cols == 128);
  using Matrix = pto::Tile<pto::TileType::Vec, int32_t, 1, 2048>;
  using Zeros = pto::Tile<pto::TileType::Vec, float, 1, 2048>;
  using Rounded = pto::Tile<pto::TileType::Vec, half, 1, 2048>;
  using Acc = pto::Tile<pto::TileType::Acc, int32_t, 16, 128, pto::BLayout::ColMajor,
                        16, 128, pto::SLayout::RowMajor, 1024>;
  Matrix matrix;
  Zeros zeros;
  Rounded rounded;
  Acc l0c;
  pto::TASSIGN<32768>(matrix);
  pto::TASSIGN<32768>(zeros);
  pto::TASSIGN<40960>(rounded);
  pto::TASSIGN<0>(l0c);
  pto::TEXPANDS(zeros, 0.0f);
  pipe_barrier(PIPE_ALL);
  copy_ubuf_to_ubuf(matrix.data(), input.data(), 0, 8, 2, 0, 30);
  pipe_barrier(PIPE_ALL);
  copy_matrix_ubuf_to_cc(l0c.data(), matrix.data(), 0, 1, 8, 0, 0, CRMODE_NONE);
  pipe_barrier(PIPE_ALL);
  // Bind each 16-channel table explicitly instead of assuming table advancement
  // across a wide transfer. Keep the widened UB/L0C transfer and packing.
  for (unsigned column = 0; column < 8; ++column) {
    set_deqscale(reinterpret_cast<uint64_t>(scales.data() + column * 32) >> 5);
    copy_matrix_cc_to_ubuf(rounded.data() + column * 256, l0c.data() + column * 256,
                           0, 1, 1, 0, 0, CRMODE_DEQSCALE_VDEQ16);
    pipe_barrier(PIPE_ALL);
  }
  copy_ubuf_to_ubuf(output.data(), rounded.data(), 0, 8, 1, 15, 0);
  pipe_barrier(PIPE_ALL);
}
