/*
 * Copyright (c) PyPTO Contributors.
 * SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
 */
#include "a1_expert_nz_grouped_cached.h"

extern "C" __global__ AICORE void SingleLaunchEntry(__gm__ int64_t* args) {
  BatchedNZExpertProjectionCached<768, 2048, false, 64, true>(args);
}
