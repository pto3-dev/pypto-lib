# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Numerical traversal specialization, separate from physical launch quotas.

Measured CANN9.1 key229585, 910B4, BF16 [24,5120]@[4096,5120].
The installed ShiftBlockAccess pass was checked for all19x10 outer positions;
normal DSL matches all4 real-rank fixtures and an independent cancellation
basis bitwise (native_order74b and native_order76b evidence).

These19 groups are reference numerical N partitions, not physical AIC count.
Kernel quotas still use pl.system.available_cluster_count(). A different
backend/native tiling or actual mathematical row shape needs revalidation.
"""
REFERENCE_M=24
REFERENCE_N=4096
REFERENCE_K=5120
N_GROUP_SPAN=224
N_GROUPS=(REFERENCE_N+N_GROUP_SPAN-1)//N_GROUP_SPAN
HALF_GROUPS=N_GROUPS//2
L1_K=512
L0_K=64
OUTER=REFERENCE_K//L1_K
INNER=L1_K//L0_K
OFFSET_SCALE=OUTER//HALF_GROUPS
# The source disables shifts for inward-overlapping last two reference groups.
NO_SHIFT_BEGIN=N_GROUPS-2
TILING_KEY='229585'
TILING_REFERENCE_SHA256='75a032a8d0e3938dd603480201519c825c07f48e7ccfc88d30cd031bbcd168e5'


def validate():
    assert (N_GROUPS,OUTER,INNER,NO_SHIFT_BEGIN)==(19,10,8,17)
    from .tuning import GDN_N,Z_N,GDN_L1_K,GDN_L0_K,Z_L1_K,Z_L0_K
    assert N_GROUP_SPAN%GDN_N==N_GROUP_SPAN%Z_N==0
    assert GDN_L1_K==Z_L1_K==L1_K
    assert GDN_L0_K==Z_L0_K==L0_K
    return dict(tiling_key=TILING_KEY,logical_reference_N_groups=N_GROUPS,
                N_group_span=N_GROUP_SPAN,L1_K=L1_K,L0_K=L0_K,
                physical_AIC_quota='pl.system.available_cluster_count()',
                precision_specialization='910B4_CANN9.1_M24_N4096_K5120',
                B3_native_order_validated=False)
