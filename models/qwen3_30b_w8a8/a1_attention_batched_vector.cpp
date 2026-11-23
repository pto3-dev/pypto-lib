/* Copyright (c) PyPTO Contributors.
 * SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0 */
#define PYPTO_A1_DIRECT_PAGES
#define PYPTO_A1_WIDE_QK
#define PYPTO_A1_WIDE_PV
#define PYPTO_A1_RETAIN_PROB_PADDING
#define PYPTO_A1_VECTOR_COPY
#define PYPTO_A1_PREFETCH_VALUES 1
#include "a1_attention_batched_modelslim.cpp"
