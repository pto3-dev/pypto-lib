# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""SPMD-local A reuse; launch geometry and downstream readiness are preserved."""

import json
from pathlib import Path
from pypto import backend, ir


def projection_plan(input_dim, output_dim, *, n_tile, k_tile, inner_k, outer_stage=2):
    selected = backend.get_backend_instance(backend.get_backend_type())
    capacity = selected.get_mem_size(ir.MemorySpace.Mat)
    # Descriptor alignment is a backend resource, not a hardcoded chip budget.
    alignments = [
        mem.alignment
        for die in selected.soc.die_counts
        for cluster in die.cluster_counts
        for core in cluster.core_counts
        for mem in core.mems
        if mem.mem_type == ir.MemorySpace.Mat
    ]
    if not capacity or not alignments:
        return dict(eligible=False, reason="unknown_l1_resources")
    alignment = max(alignments)
    aligned = lambda size: ((size + alignment - 1) // alignment) * alignment
    activation = aligned(16 * input_dim * 2)
    weights = outer_stage * aligned(n_tile * k_tile * 2)
    # GetCoreCount is a conservative compilation resource bound; actual launch
    # width STILL comes from pl.system.available_cluster_count() in the kernel.
    slots = selected.get_core_count(ir.CoreType.CUBE)
    jobs = output_dim // n_tile
    legal = input_dim % k_tile == 0 and k_tile % inner_k == 0 and output_dim % n_tile == 0
    eligible = legal and slots > 0 and jobs >= 2 * slots and activation + weights <= capacity
    return dict(
        eligible=eligible,
        reason="eligible" if eligible else "capacity_or_parallelism",
        l1_capacity_bytes=capacity,
        alignment_bytes=alignment,
        activation_bytes=activation,
        weight_slots_bytes=weights,
        conservative_l1_bytes=activation + weights,
        backend_slots=slots,
        independent_jobs=jobs,
        n_tile=n_tile,
        k_tile=k_tile,
        inner_k=inner_k,
        launch_geometry_changed=False,
        dependency_boundary_changed=False,
    )


def selected_projection_keys():
    path = Path(__file__).with_name("l1_reuse_selection.json")
    return set(json.loads(path.read_text())["selected"])
