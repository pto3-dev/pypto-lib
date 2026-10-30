# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Six-stage attention plan assembler; binary assets are reproducible.

This is metadata serialization, not Python model/source generation. Leaf IDs,
argument slots and geometry are explicit; Simpler validates per-lane order and
cross-lane cycles. Runtime grid is selected from available_cluster_count().
"""

import json, struct
from pathlib import Path
from dataclasses import replace
from simpler_setup.device_program import Invocation, LeafSignature, compile_device_program
from qwen36.device_plan_layout import SOFTMAX_JOBS


def make_plan(
    ids,
    bindings,
    root_count,
    grid,
    scalar_constants=(),
    tensor_count=None,
    softmax_jobs=SOFTMAX_JOBS,
    request_partition=False,
):
    nq, ka, sc, sm, pv, rg = ids
    nodes = []

    def vector(name, fid, count, args, after=()):
        for block in range(count):
            owner = block % grid
            lane = "aiv" + str((block // grid) % 2)
            nodes.append(Invocation(f"{name}{block}", fid, lane, owner, block, count, args, after))

    vector("nq", nq, 24, bindings[0])
    vector("ka", ka, 24, bindings[1])
    ready = tuple(n.name for n in nodes)
    if request_partition and grid < 4:
        raise ValueError("Request partition needs at least one core per token")
    for block in range(grid):
        nodes.append(
            Invocation(
                f"sc{block}",
                sc,
                "aic",
                block,
                block,
                grid,
                bindings[2],
                (f"nq{block % 4}", f"ka{block % 4}") if request_partition else ready,
            )
        )
    scores = tuple(f"sc{block}" for block in range(grid))
    soft_count = min(2 * grid, softmax_jobs)
    if request_partition:
        if soft_count != 24:
            raise ValueError("Request-local softmax requires one row per block")
        for block in range(soft_count):
            deps = tuple(f"sc{b}" for b in range(grid) if b % 4 == block // 6)
            nodes.append(
                Invocation(
                    f"sm{block}",
                    sm,
                    "aiv" + str((block // grid) % 2),
                    block % grid,
                    block,
                    soft_count,
                    bindings[3],
                    deps,
                )
            )
    else:
        vector("sm", sm, soft_count, bindings[3], scores)
    soft = tuple(f"sm{block}" for block in range(soft_count))
    for block in range(grid):
        nodes.append(
            Invocation(
                f"pv{block}",
                pv,
                "aic",
                block,
                block,
                grid,
                bindings[4],
                tuple(f"sm{r}" for r in range((block % 4) * 6, (block % 4 + 1) * 6))
                if request_partition
                else soft,
            )
        )
    if request_partition:
        for row in range(24):
            deps = tuple(f"pv{b}" for b in range(grid) if b % 4 == row // 6)
            nodes.append(
                Invocation(
                    f"rg{row}", rg, "aiv" + str((row // grid) % 2), row % grid, row, 24, bindings[5], deps
                )
            )
    else:
        vector("rg", rg, min(2 * grid, 48), bindings[5], tuple(f"pv{block}" for block in range(grid)))
    sig = [
        LeafSignature(fid, lane, len(args))
        for fid, lane, args in [
            (nq, "aiv", bindings[0]),
            (ka, "aiv", bindings[1]),
            (sc, "aic", bindings[2]),
            (sm, "aiv", bindings[3]),
            (pv, "aic", bindings[4]),
            (rg, "aiv", bindings[5]),
        ]
    ]
    plan = compile_device_program(
        nodes,
        sig,
        grid=grid,
        available_clusters=grid,
        root_arg_count=root_count,
        epoch_arg=root_count - 1,
        max_wait_cycles=50_000_000,
        record_trace=False,
    )
    image = bytearray(plan.image)
    h = struct.unpack_from("<16I", image, 0)
    waited = {struct.unpack_from("<I", image, h[11] + 4 * i)[0] for i in range(h[6])}
    for i in range(h[4]):
        struct.pack_into("<I", image, h[9] + 64 * i + 40, int(i not in waited))
    # Version 2 folds statically proven model geometry into unsigned immediates.
    # This keeps the callable prefix within its 32-direction ABI, without
    # changing any numerical leaf entry or expanding a shared SDK ABI.
    if tensor_count is not None:
        original_epoch = root_count - 1
        for index in range(h[5]):
            offset = h[10] + 4 * index
            binding = struct.unpack_from("<I", image, offset)[0]
            if binding >= 4 + tensor_count:
                slot = binding - (4 + tensor_count)
                if slot >= len(scalar_constants) or not 0 <= scalar_constants[slot] < 2**31:
                    raise ValueError("Invalid constant binding")
                struct.pack_into("<I", image, offset, (1 << 31) | scalar_constants[slot])
        struct.pack_into("<I", image, 4, 2)
        struct.pack_into("<I", image, 12 * 4, 4 + tensor_count)
        struct.pack_into("<I", image, 13 * 4, 5 + tensor_count)
    return replace(plan, image=bytes(image))


def build(grid):
    spec = json.loads(Path(__file__).with_name("attention_plan.json").read_text())
    return make_plan(
        spec["leaf_ids"],
        spec["bindings"],
        spec["root_arg_count"],
        grid,
        scalar_constants=spec["scalar_constants"],
        tensor_count=spec["tensor_count"],
        softmax_jobs=spec["softmax_jobs"],
        request_partition=False,
    ).image
