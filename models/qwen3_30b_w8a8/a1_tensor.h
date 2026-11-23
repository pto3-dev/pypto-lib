/*
 * Copyright (c) PyPTO Contributors.
 * SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
 */
#pragma once
#include <pto/pto-inst.hpp>

#include "tensor.h"

#ifdef PYPTO_A1_MODEL_FP16
using A1ModelFloat = half;
#else
using A1ModelFloat = bfloat16_t;
#endif

template <typename T>
__aicore__ inline __gm__ T* Data(__gm__ Tensor* tensor) {
  return reinterpret_cast<__gm__ T*>(tensor->buffer.addr) + tensor->start_offset;
}

__aicore__ inline void PublishStatus(__gm__ int32_t* status, int32_t value) {
  status[0] = value;
  pipe_barrier(PIPE_ALL);
  dcci((__gm__ void*)status, cache_line_t::SINGLE_CACHE_LINE);
}
