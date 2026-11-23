/*
 * Copyright (c) PyPTO Contributors.
 * SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
 */
#include "a1_route_input.h"
extern "C" __global__ AICORE void SingleLaunchEntry(__gm__ int64_t* args) { RouteInput<2048, 1>(args); }
