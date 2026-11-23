# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Measured GDN numerical classes, separate from physical hardware quotas.

CANN9.1/910B4 payload332 key229571: M12/N4096/K5120, resident A,
B L1K256/L0K64,20 logical N groups of208. Static normal DSL334/335
matches all four actual model shards in336. Installed ShiftBlockAccess
400 positions checked337. Native row sweep366 matches the small path1..15
and large17..24; M16 is separate. Actual M16 payload367 has key230083,
same K256/K64 fields but unshifted traversal candidate (requires device
validation). Only the first12 input rows are real in366; remaining rows zero.
This source is a candidate, not full B1/B3 acceptance or910B3 certification.
"""
B2_N=16
B2_GROUP_SPAN=208
B2_HALF_GROUPS=10
B2_NO_SHIFT_BEGIN=18
B2_OUTER=20
B2_OFFSET_SCALE=2
B2_L1_K=256
B2_L0_K=64
PAYLOAD_SHA256='d925b5a87f67dad7e96f807c51c1574269a1cf1f10e85781dbca33d6e2a077ca'
ALIGNED16_KEY='230083'
ALIGNED16_PAYLOAD_SHA256='55c7320b6f8a670cf8720bae11bf3c1f44b160a2556aa1baf8b3d91384abbad3'
