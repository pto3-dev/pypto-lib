# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -------------------------------------------------------------------------------------------------------
"""Rank-local Decode dimensions for Qwen3-30B-A3B W8A8, TP=2."""

LAYERS = 48
HIDDEN = 2048
EXPERTS = 128
TOP_K = 8
INTERMEDIATE = 768
TP_SIZE = 2
LOCAL_INTERMEDIATE = INTERMEDIATE // TP_SIZE
W13_WIDTH = 2 * LOCAL_INTERMEDIATE
