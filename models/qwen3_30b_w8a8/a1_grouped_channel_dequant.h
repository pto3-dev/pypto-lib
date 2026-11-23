// Copyright (c) PyPTO Contributors.
// SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
#pragma once
#include "a1_tensor.h"

template <unsigned Width, typename AccumulatorTile, typename PackedTile, typename HalfTile>
__aicore__ inline void GroupedChannelDequant16(
    AccumulatorTile& accumulator, PackedTile& scales, HalfTile& output) {
  static_assert(Width % 16 == 0);
  static_assert(AccumulatorTile::Rows == 16 && AccumulatorTile::Cols == Width);
  static_assert(PackedTile::Rows == 1 && PackedTile::Cols == 2 * Width);
  static_assert(HalfTile::Rows == 1 && HalfTile::Cols == 16 * Width);
  for (unsigned column = 0; column < Width / 16; ++column) {
    set_deqscale(reinterpret_cast<uint64_t>(scales.data() + column * 32) >> 5);
    copy_matrix_cc_to_ubuf(output.data() + column * 256, accumulator.data() + column * 256,
                           0, 1, 1, 0, 0, CRMODE_DEQSCALE_VDEQ16);
    pipe_barrier(PIPE_ALL);
  }
}
