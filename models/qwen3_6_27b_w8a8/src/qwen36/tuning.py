# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compile-time tuning constants, separate from logical model geometry.

These defaults reproduce the measured L1K2560 candidate. Changing them is an
experiment, not an automatic B3/B4 preset. Dynamic launch limits still query
pl.system.available_cluster_count()/available_aiv_count() in the DSL wrappers.
The explicit native dispatch ABI guard must also accept the resulting graph.
"""

GATE_UP_N = 64
GATE_UP_L1_K = 2560
GATE_UP_L0_K = 256
GATE_UP_JOBS = 4352 // GATE_UP_N
DOWN_N = 64
DOWN_L1_K = 256
DOWN_L0_K = 256
DOWN_JOBS = 5120 // DOWN_N
GDN_N = 64
GDN_L1_K = 512
GDN_L0_K = 256
Z_N = 64
Z_L1_K = 256
Z_L0_K = 256
HEAD_N = 64
HEAD_L1_K = 512
HEAD_L0_K = 256


def validate():
    """Precheck divisibility and minimum/reference Mat working set, not DB proof."""
    from pypto import backend, ir

    b = backend.get_backend_instance(backend.BackendType.Ascend910B)
    capacities = {
        key: b.get_mem_size(getattr(ir.MemorySpace, key)) for key in ("Mat", "Left", "Right", "Acc")
    }
    if DOWN_L1_K != DOWN_L0_K:
        raise ValueError(
            "Down currently has one K loop; introduce and inspect a real nested pipeline before separating its K levels"
        )
    configs = [
        ("int8_gateup", 5120, GATE_UP_N, GATE_UP_L1_K, GATE_UP_L0_K, 1),
        ("int8_down", 4352, DOWN_N, DOWN_L1_K, DOWN_L0_K, 1),
        ("gdn", 5120, GDN_N, GDN_L1_K, GDN_L0_K, 2),
        ("z", 5120, Z_N, Z_L1_K, Z_L0_K, 2),
        ("head", 5120, HEAD_N, HEAD_L1_K, HEAD_L0_K, 2),
    ]
    rows = []
    for name, k, n, outer, inner, size in configs:
        if min(n, outer, inner) <= 0 or k % outer or outer % inner or inner % (32 if size == 1 else 16):
            raise ValueError((name, "Illegal/tail K requires explicit handling"))
        mat = 32 * k * size + 2 * outer * n * size
        if mat > capacities["Mat"]:
            raise ValueError((name, "Reference A + two B slots exceed queried L1", mat, capacities))
        if 32 * inner * size > capacities["Left"] or inner * n * size > capacities["Right"]:
            raise ValueError((name, "Single L0 tile exceeds queried capacity"))
        rows.append(dict(kernel=name, n=n, gm_to_l1_k=outer, l1_to_l0_k=inner, reference_mat_bytes=mat))
    return dict(
        capacity_bytes=capacities,
        reference_estimates=rows,
        note="Actual allocation and buffering must be checked in generated IR/C++; these are prechecks only",
    )
