/*
 * Copyright (c) PyPTO Contributors.
 * SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
 */
#pragma once
#include <type_traits>

#include "a1_tensor.h"

// Model-private packer: input must already be finite integral FP16 in
// [-128,127], after the original FP32 RINT and FP16 materialization.
// This does not implement general TCVT overflow, saturation or NaN semantics.
template <typename QuantTile, typename HalfTile>
__aicore__ inline void PackBoundedIntegralInt8(QuantTile& dst, HalfTile& src) {
  static_assert(std::is_same_v<typename QuantTile::DType, int8_t>);
  static_assert(std::is_same_v<typename HalfTile::DType, half>);
  static_assert(QuantTile::Rows == 1 && HalfTile::Rows == 1);
  static_assert(QuantTile::Cols == HalfTile::Cols);
  static_assert(HalfTile::Numel % 128 == 0 && HalfTile::Numel / 128 <= 255);
  set_mask_norm();
  set_vector_mask(UINT64_MAX, UINT64_MAX);
  vconv_f162s8z(dst.data(), src.data(), static_cast<uint8_t>(HalfTile::Numel / 128),
                 1, 1, 4, 8);
}
