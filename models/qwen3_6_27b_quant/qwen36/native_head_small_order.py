# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CANN9.1 M6/M12 head numerical specialization; hardware quotas are queried.

Real payload key229571, A fullK5120 resident, B L1K256/L0K64, N19x3328.
Actual model function replay306 matches all four real shards for6/12 rows.
Native sweep308 matches this shifted path for1..15 rows. Exactly16 rows
select key230083 (decoded non_factor_mn_flag1); unshifted prototype316
matches all four real M16 shards. Large17..24 uses the separate large path.
This is not a910B3 certification and not a physical19-core limit.
"""
SMALL_GROUP_SPAN=3328
SMALL_GROUPS=19
SMALL_HALF_GROUPS=SMALL_GROUPS//2
SMALL_OUTER=20
SMALL_OFFSET_SCALE=SMALL_OUTER//SMALL_HALF_GROUPS
SMALL_NO_SHIFT_BEGIN=SMALL_GROUPS-2
SMALL_L1_K=256
SMALL_L0_K=64
TILING_KEY='229571'
M12_PAYLOAD_SHA256='d5960fed3f232a1a5b4da35ed408ca70807c0e7133ac2f1cf8e05c60969e97f2'
M6_PAYLOAD_SHA256='8eb8ee4c996d4996eca3d29a4b3a0e9f3831028a4dfc82a38eb97c36b5c861b0'
ALIGNED_16_TILING_KEY='230083'
M16_PAYLOAD_SHA256='c7e9505f3e8dcc55f634ac4c819c84f1f12fa67a401d29f58236b3d180eb16f7'
