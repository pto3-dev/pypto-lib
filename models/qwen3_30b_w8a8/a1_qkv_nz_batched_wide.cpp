// Copyright (c) PyPTO Contributors.
// SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
#include "a1_nz_projection_batched_wide.h"
extern "C" __global__ AICORE void SingleLaunchEntry(__gm__ int64_t* args) {
  BatchedNZProjectionWide<2048, 5120>(args);
}
