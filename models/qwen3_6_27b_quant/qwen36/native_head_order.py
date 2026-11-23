# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""LM-head numerical traversal derived from CANN9.1 key229585, 910B4.

Measured BF16 M24/N62080/K5120 payload: head282/DECODED_PAYLOAD.json.
The profiler timeline timed out; its already-written160-byte tiling payload
is valid diagnostic input, not performance evidence. Native unprofiled head
and current DSL were each bitwise calibrated to actual serving in head280b.

These20 logical N partitions define the reference addition order. They are
not a physical AIC quota: the kernel queries available_cluster_count().
Different hardware or native row/tiling specializations require validation.
"""
REFERENCE_M=24
REFERENCE_N=62080
REFERENCE_K=5120
HEAD_GROUP_SPAN=3200
HEAD_GROUPS=(REFERENCE_N+HEAD_GROUP_SPAN-1)//HEAD_GROUP_SPAN
HEAD_HALF_GROUPS=HEAD_GROUPS//2
HEAD_OUTER=10
HEAD_OFFSET_SCALE=HEAD_OUTER//HEAD_HALF_GROUPS
HEAD_NO_SHIFT_BEGIN=HEAD_GROUPS-2
TILING_KEY='229585'
TILING_PAYLOAD_SHA256='a3c39e75e5ffeaae7967d2ecba216a26fb6c805bcacd469a07febd65393e9cd7'

def validate():
    from .tuning import HEAD_N,HEAD_L1_K,HEAD_L0_K
    assert HEAD_GROUP_SPAN%HEAD_N==0
    assert(HEAD_GROUPS,HEAD_L1_K,HEAD_L0_K)==(20,512,128)
    assert HEAD_L1_K//HEAD_L0_K==4 and HEAD_OUTER*HEAD_L1_K==REFERENCE_K
    return dict(tiling_key=TILING_KEY,logical_reference_N_groups=HEAD_GROUPS,
                physical_AIC_quota='pl.system.available_cluster_count()',
                precision_specialization='910B4_CANN9.1_M24_N62080_K5120',
                B3_native_order_validated=False)
