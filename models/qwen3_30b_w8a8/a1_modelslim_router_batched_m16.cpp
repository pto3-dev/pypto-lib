// Copyright (c) PyPTO Contributors.
// SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
#include "a1_modelslim_float_linear.h"
extern "C" __global__ AICORE void SingleLaunchEntry(__gm__ int64_t* args) {
  ModelSlimFloatLinear<128, false, 20, true>(args);
}
