# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""One shared storage contract for model metadata, allocation and device views."""

PLAN_BYTES = 8192
FRAMES_OFFSET = PLAN_BYTES
FRAMES_BYTES = 3 * 2 * 512
EVENTS_OFFSET = FRAMES_OFFSET + FRAMES_BYTES
EVENTS_BYTES = 23 * 128
TRACE_OFFSET = EVENTS_OFFSET + EVENTS_BYTES
TRACE_BYTES_PER_LAYER = 23 * 128
ATTENTION_STORAGE_BYTES = TRACE_OFFSET + 16 * TRACE_BYTES_PER_LAYER
assert ATTENTION_STORAGE_BYTES == 61312
MLP_EVENTS_OFFSET = ATTENTION_STORAGE_BYTES
STORAGE_BYTES = ATTENTION_STORAGE_BYTES + 64 * 128
