/*
 * Copyright (c) PyPTO Contributors.
 * SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
 */
#define PYPTO_A1_BATCHED_VECTOR_PADDING 1
#include "a1_swiglu_quant.h"

extern "C" __global__ AICORE void SingleLaunchEntry(__gm__ int64_t* args) {
  ModelSlimSwiGLUQuant<true, true>(args);
}
