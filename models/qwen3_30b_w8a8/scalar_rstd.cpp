/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Scalar statistics boundary used by CANN 8.5.1 fused norm/quant. This is an
// owned AIV helper, not a call to a native normalization or quantization op.
// Args: FP32 mean-plus-epsilon Tensor*, FP32 reciprocal-standard-deviation Tensor*, valid rows.
#include <cstdint>
#include <cmath>
#include <pto/pto-inst.hpp>
#include "tensor.h"

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
  auto* input_tensor = reinterpret_cast<__gm__ Tensor*>(args[0]);
  auto* output_tensor = reinterpret_cast<__gm__ Tensor*>(args[1]);
  auto* input = reinterpret_cast<__gm__ float*>(input_tensor->buffer.addr) + input_tensor->start_offset;
  auto* output = reinterpret_cast<__gm__ float*>(output_tensor->buffer.addr) + output_tensor->start_offset;
  const uint32_t rows = static_cast<uint32_t>(args[2]);
#if !defined(__CPU_SIM) && !defined(__COSTMODEL)
  set_atomic_none();
  pipe_barrier(PIPE_ALL);
  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
  dsb((mem_dsb_t)0);
#endif
  // One logical block owns the whole <=16-row, 64-byte output cache line.
  for (uint32_t row = 0; row < rows; ++row) {
    const float value = input[row];
    output[row] = 1 / sqrt(value);
  }
#if !defined(__CPU_SIM) && !defined(__COSTMODEL)
  pipe_barrier(PIPE_ALL);
  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
  dsb((mem_dsb_t)0);
#endif
}
