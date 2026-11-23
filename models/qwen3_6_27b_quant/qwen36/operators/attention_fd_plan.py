# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Device metadata planner for Qwen GQA6/1 and1..4 verifier requests.

Only integer shape metadata is inspected. Physical limits come from system
queries. The greedy weighted512-stack policy follows the verified9.1 Host
reference; no captured partition table or tensor value participates in planning.
"""
import pypto.language as pl
from qwen36.sequence_config import SEQUENCE

BLOCKS = (SEQUENCE + 511) // 512


@pl.function(type=pl.FunctionType.InCore)
def fd_plan_kernel(lengths: pl.Tensor[[24], pl.INT32],
                   ends: pl.Tensor[[4], pl.INT32],
                   clusters: pl.Scalar[pl.INT32],
                   plan: pl.Out[pl.Tensor[[4, BLOCKS], pl.INT32]],
                   safe_lengths: pl.Out[pl.Tensor[[24], pl.INT32]],
                   summary: pl.Out[pl.Tensor[[1, 8], pl.INT32]],
                   safe_matrix: pl.Tensor[[1, 24], pl.INT32]):
    cluster_count = pl.cast(clusters, pl.INT64)
    ids = pl.tile.full([4, BLOCKS], pl.INT32, -1)
    safe = pl.tile.full([1, 24], pl.INT32, 0)
    info = pl.tile.full([1, 8], pl.INT32, 0)
    total = pl.cast(0, pl.INT64)
    maximum = pl.cast(0, pl.INT64)
    requests = pl.cast(0, pl.INT64)
    for request in pl.range(4):
        queries = pl.cast(pl.read(ends, [request]), pl.INT64) - request * 6
        if queries > 0:
            requests = requests + 1
            length = pl.cast(pl.read(lengths, [request * 6 + queries - 1]), pl.INT64)
            total = total + queries * length * 6
            maximum = pl.max(maximum, length)
            for query in pl.range(queries):
                value = pl.read(lengths, [request * 6 + query])
                pl.tile.write(safe, [0, request * 6 + query], value)
    budget = (total + cluster_count - 1) // cluster_count
    use_fd = maximum >= 1024 and ((requests * 5 <= cluster_count * 2) or
        (requests * 5 <= cluster_count * 4 and maximum >= cluster_count * 512))
    remaining = budget
    core = pl.cast(0, pl.INT64)
    for request in pl.range(4):
        queries = pl.cast(pl.read(ends, [request]), pl.INT64) - request * 6
        if queries > 0:
            length = pl.cast(pl.read(lengths, [request * 6 + queries - 1]), pl.INT64)
            count = (length + 511) // 512
            part = pl.cast(-1, pl.INT64)
            previous_core = pl.cast(-1, pl.INT64)
            for block in pl.range(count):
                if use_fd:
                    if remaining <= 0:
                        core = core + 1
                        remaining = budget
                    if previous_core != core:
                        part = part + 1
                        previous_core = core
                    cost = queries * 6 * pl.min(512, length - block * 512)
                    remaining = remaining - cost
                    pl.tile.write(ids, [request, block], pl.cast(part, pl.INT32))
                else:
                    pl.tile.write(ids, [request, block], pl.cast(0, pl.INT32))
    pl.tile.write(info, [0, 0], pl.cast(cluster_count, pl.INT32))
    pl.tile.write(info, [0, 1], pl.cast(budget, pl.INT32))
    pl.tile.write(info, [0, 2], pl.cast(core + 1, pl.INT32))
    pl.tile.write(info, [0, 3], pl.cast(use_fd, pl.INT32))
    pl.tile.write(info, [0, 4], pl.cast(requests, pl.INT32))
    pl.tile.write(info, [0, 5], pl.cast(total, pl.INT32))
    pl.tile.write(info, [0, 6], pl.cast(maximum, pl.INT32))
    pl.store(ids, [0, 0], plan)
    pl.store(safe, [0, 0], safe_matrix)
    pl.store(info, [0, 0], summary)
    return plan, safe_lengths, summary


@pl.function(type=pl.FunctionType.Inline)
def fd_plan(lengths: pl.Tensor[[24], pl.INT32], ends: pl.Tensor[[4], pl.INT32],
            plan: pl.Out[pl.Tensor[[4, BLOCKS], pl.INT32]],
            safe_lengths: pl.Out[pl.Tensor[[24], pl.INT32]],
            summary: pl.Out[pl.Tensor[[1, 8], pl.INT32]]):
    safe_matrix = pl.reshape(safe_lengths, [1, 24])
    clusters = pl.system.available_cluster_count()
    # Exactly one planning job, not a hardcoded physical resource limit.
    with pl.spmd(1):
        plan, safe_lengths, summary = fd_plan_kernel(lengths, ends, clusters,
            plan, safe_lengths, summary, safe_matrix)
    return plan, safe_lengths, summary
