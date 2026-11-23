// Copyright (c) PyPTO Contributors.
// SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
#define PYPTO_A1_BATCHED_VECTOR_PADDING 1
#include "a1_expert_combine.h"
extern "C" __global__ AICORE void SingleLaunchEntry(__gm__ int64_t* args) {
  ModelSlimExpertCombine<true, 2048, true>(args);
}
