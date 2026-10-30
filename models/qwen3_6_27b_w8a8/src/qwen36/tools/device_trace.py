# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Optional terminal-only decoding of the existing device-plan trace protocol.

No instruction or graph dependency changes. The plan flag enables three
get_sys_cnt_aicore samples per leaf. These diagnostic costs must stay out of
formal ITL measurements. Callers must supply the measured clock frequency.
"""

import struct
from qwen36.device_plan_layout import MAX_NODES, TRACE_BYTES_PER_LAYER

MAGIC = 0x44505247


def instrument_plan(data):
    image = bytearray(data)
    header = struct.unpack_from("<16I", image)
    if header[0] != MAGIC or header[1] != 2 or header[2] != len(image):
        raise ValueError("Unexpected plan ABI")
    struct.pack_into("<I", image, 15 * 4, 1)
    return bytes(image)


def decode(data, plan, clock_freq_hz, layer_group, expected_epoch, names):
    if clock_freq_hz <= 0:
        raise ValueError("Missing actual trace clock frequency")
    h = struct.unpack_from("<16I", plan)
    if h[0] != MAGIC or h[1] != 2 or h[4] > MAX_NODES:
        raise ValueError("Invalid plan")
    if len(data) < TRACE_BYTES_PER_LAYER:
        raise ValueError("Truncated trace allocation")
    records = []
    for i in range(h[4]):
        node = struct.unpack_from("<16I", plan, h[9] + 64 * i)
        waiting, entered, returned, physical, epoch = struct.unpack_from("<5q", data, 128 * i)
        if epoch != expected_epoch:
            raise ValueError(f"Missing/stale node trace {i}: epoch{epoch}, expected{expected_epoch}")
        if not waiting <= entered <= returned:
            raise ValueError("Non-monotonic device samples")
        factor = 1e6 / clock_freq_hz
        records.append(
            dict(
                layer_group=layer_group,
                node=i,
                kernel=names[str(node[0])],
                func_id=node[0],
                lane=["aic", "aiv0", "aiv1"][node[1]],
                owner=node[2],
                logical_block=node[3],
                logical_count=node[4],
                physical_core_id=physical,
                epoch=epoch,
                wait_begin_ticks=waiting,
                kernel_enter_ticks=entered,
                kernel_return_ticks=returned,
                wait_and_argument_preparation_us=(entered - waiting) * factor,
                leaf_entry_to_return_us=(returned - entered) * factor,
                note="Wait interval includes argument-frame setup; leaf return drains its own engines; samples do not include event publication after return",
            )
        )
    return records


def export_terminal(executor, variant_build, folder):
    """Read all16 retired attention traces after completion; never timed decode."""
    import json
    import torch
    from pathlib import Path
    from qwen36.device_plan_layout import TRACE_OFFSET

    raw = json.loads((folder / "chip_swimlane_records.json").read_text())
    meta = raw["metadata"]
    grid = sum(kind == "aic" for kind in meta["core_types"])
    plan = instrument_plan((variant_build / f"production_attention_grid{grid}.bin").read_bytes())
    manifest = json.loads(
        (Path(__file__).resolve().parents[1] / "native_backend/link_manifest.json").read_text()
    )
    names = {str(row["func_id"]): row["name"].format(rank=executor.rank) for row in manifest["kernels"]}
    storage = executor.exported["device_program_storage"].view(torch.uint8).flatten()
    begin = TRACE_OFFSET
    size = 16 * TRACE_BYTES_PER_LAYER
    data = storage[begin : begin + size].cpu().numpy().tobytes()
    epoch = int(executor.control_i32[1].cpu())
    rows = []
    for group in range(16):
        segment = data[group * TRACE_BYTES_PER_LAYER : (group + 1) * TRACE_BYTES_PER_LAYER]
        rows.extend(decode(segment, plan, meta["clock_freq_hz"], group, epoch * 16 + group + 1, names))
    groups = []
    for group in range(16):
        subset = [r for r in rows if r["layer_group"] == group]
        stats = {}
        for kernel in sorted({r["kernel"] for r in subset}):
            ks = [r for r in subset if r["kernel"] == kernel]
            stats[kernel] = dict(
                nodes=len(ks),
                max_leaf_us=max(r["leaf_entry_to_return_us"] for r in ks),
                max_wait_and_preparation_us=max(r["wait_and_argument_preparation_us"] for r in ks),
                first_enter_ticks=min(r["kernel_enter_ticks"] for r in ks),
                last_return_ticks=max(r["kernel_return_ticks"] for r in ks),
            )
        groups.append(
            dict(
                layer_group=group,
                leafs=stats,
                envelope_us=(
                    max(r["kernel_return_ticks"] for r in subset) - min(r["wait_begin_ticks"] for r in subset)
                )
                * 1e6
                / meta["clock_freq_hz"],
            )
        )
    payload = dict(
        rank=executor.rank,
        grid=grid,
        clock_freq_hz=meta["clock_freq_hz"],
        counter_epoch=epoch,
        nodes=len(rows),
        all16_layers_validated=True,
        groups=groups,
        records=rows,
        scope="Terminal diagnostic only, plan flags1. Per-leaf timestamps include profiling overhead. Wait+frame preparation is separate from leaf entry/return; event publication is outside both.",
    )
    path = folder / "attention_leaf_trace.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return dict(path=str(path), nodes=len(rows), all16_layers_validated=True)
