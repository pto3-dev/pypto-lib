// Copyright (c) PyPTO Contributors.
// SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
#define PYPTO_A1_MODEL_FP16 1
#include "a1_head_norm_rope_direct.h"
extern "C" __global__ AICORE void SingleLaunchEntry(__gm__ int64_t* args) {
  A1DirectHeadNormRope<32>(args);
}
