/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef GDN_OUTER_FP32_HPP
#define GDN_OUTER_FP32_HPP

#include <pto/pto-inst.hpp>

namespace pto::experimental {
// Isolated A2/A3 prototype. The caller supplies non-overlapping 2KiB scratch.
// This does not register a public PTO instruction or change the installed ISA.
template <bool Accumulate, typename Matrix, typename Row, typename Column, typename Scratch>
__tf__ PTO_INTERNAL void GdnOuterFp32Impl(
    typename Matrix::TileDType __in__ __out__ dst,
    typename Row::TileDType __in__ row,
    typename Column::TileDType __in__ column,
    typename Scratch::TileDType __out__ scratch)
{
    static_assert(Matrix::Rows == 64 && Matrix::Cols == 128);
    static_assert(Matrix::RowStride == 128 && Matrix::isRowMajor);
    static_assert(std::is_same_v<typename Matrix::DType, float>);
    static_assert(std::is_same_v<typename Row::DType, float>);
    static_assert(std::is_same_v<typename Column::DType, float>);
    static_assert(!Column::isRowMajor);  // Compact contiguous64-element vector.
    auto* dstPtr = (__ubuf__ float*)__cce_get_tile_ptr(dst);
    auto* rowPtr = (__ubuf__ float*)__cce_get_tile_ptr(row);
    auto* columnPtr = (__ubuf__ uint32_t*)__cce_get_tile_ptr(column);
    auto* scratchPtr = (__ubuf__ uint32_t*)__cce_get_tile_ptr(scratch);
    set_mask_norm();
    SetFullVecMaskByDType<float>();
    // The error vector is produced by the preceding VMULS. Explicitly wait
    // before the broadcast reads it; raw intrinsics do not insert this fence.
    pipe_barrier(PIPE_V);
    // Eight source scalars per repeat, each expanded to one32B block.
    vbrcb(scratchPtr, columnPtr, 1, 8, 8);
    pipe_barrier(PIPE_V);
    auto* broadcastPtr = (__ubuf__ float*)scratchPtr;
    for (unsigned col = 0; col < 128; col += 64) {
        if constexpr (Accumulate) {
            vmla(dstPtr + col, rowPtr + col, broadcastPtr, 64, 1, 1, 0, 16, 0, 1);
        } else {
            vmul(dstPtr + col, rowPtr + col, broadcastPtr, 64, 1, 1, 0, 16, 0, 1);
        }
    }
    pipe_barrier(PIPE_V);
}

template <bool Accumulate, typename Matrix, typename Row, typename Column, typename Scratch>
PTO_INTERNAL void GdnOuterFp32(Matrix& dst, Row& row, Column& column, Scratch& scratch)
{
    GdnOuterFp32Impl<Accumulate, Matrix, Row, Column, Scratch>(
        dst.data(), row.data(), column.data(), scratch.data());
}
} // namespace pto::experimental
#endif
