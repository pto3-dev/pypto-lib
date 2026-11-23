/*
 * Copyright (c) PyPTO Contributors.
 * SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
 */
#include "a1_expert_nz_grouped.h"

extern "C" __global__ AICORE void SingleLaunchEntry(__gm__ int64_t* args) {
  BatchedNZExpertProjectionGrouped<768, 2048, false>(args);
}
