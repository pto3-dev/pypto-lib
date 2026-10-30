# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Audit actual fused code, resources, and SPMD selection without a device launch."""

import collections, hashlib, json, re
from pathlib import Path
from pypto import backend, ir

N = Path(__file__).resolve().parents[1]
backend.set_backend_type(backend.BackendType.Ascend910B)
device = backend.get_backend_instance(backend.BackendType.Ascend910B)
limits = {
    s.name: device.get_mem_size(s)
    for s in [
        ir.MemorySpace.Vec,
        ir.MemorySpace.Mat,
        ir.MemorySpace.Left,
        ir.MemorySpace.Right,
        ir.MemorySpace.Acc,
    ]
}
p = N / "build/fused/ptoas/fused_row.pto"
t = p.read_text()
constants = {m[0]: int(m[1]) for m in re.findall(r"(%\w+) = arith.constant (\d+) : i64", t)}
allocs = []
for m in re.finditer(r"(%\w+) = pto.alloc_tile addr = (%\w+).*?dtype=(\w+), rows=(\d+), cols=(\d+)", t):
    name, addr, dtype, rows, cols = m.groups()
    size = int(rows) * int(cols) * dict(f32=4, i32=4, bf16=2, f16=2, i8=1)[dtype]
    allocs.append(
        dict(name=name, address=constants[addr], bytes=size, dtype=dtype, rows=int(rows), cols=int(cols))
    )
peak = max(a["address"] + a["bytes"] for a in allocs)
assert peak <= limits["Vec"]
loads = [s for s in t.splitlines() if "pto.tload " in s]
stores = [s for s in t.splitlines() if "pto.tstore " in s]
assert len(loads) == 2 and len(stores) == 3, (len(loads), len(stores))
audit = dict(
    backend="Ascend910B",
    limits_bytes=limits,
    ub_static_high_water_bytes=peak,
    allocations=allocs,
    tload_lexical_count=2,
    tstore_lexical_count=3,
    load_scope="28 sequential [6,256] partials loads and one [1,1536] gate load per token; no reread from gated output",
    read_bytes_saved_per_layer=24 * 1536 * 2,
    read_bytes_saved_per_target_rank=16 * 24 * 1536 * 2,
    diagnostic_gated_store_retained=True,
    source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
)
(N / "evidence/fused_buffer_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
items = json.loads((N / "evidence/spmd_inventory.json").read_text())
for row in items:
    f = row["function"]
    module = Path(row["module"]).name
    if module == "attention.py" and f == "reduce_gate":
        policy = "replace_boundary"
        reason = "Fuse six heads of one token with dynamic quantization; independent static tile leaf, not a second SPMD quant stage."
    elif module in ("projection_bf16.py", "quantized.py", "down_prefetch.py"):
        policy = "retain_weight_streaming_spmd"
        reason = "Keep A resident and stream weight tiles within a core. Split request rows may reload weights; quantization needs a complete feature row. Reassess aligned producer/consumer tiles before splitting."
    elif module == "gdn.py" and f == "recurrent":
        policy = "next_candidate"
        reason = "Consider full-head recurrent plus gated norm only after UB/state/rounding and 192-to48 job balance audit; not enabled."
    elif module == "norm_pointwise.py" and f == "norm_gdn":
        policy = "next_candidate"
        reason = "Full128-value head is reduction boundary; predecessor currently emits32-value value partitions. Must align ownership before fusion."
    elif module in ("collectives.py", "collective_lanes.py", "collective_even_lanes.py"):
        policy = "retain_existing_fusion_protocol"
        reason = "Preserve rank-ordered reduction/notification protocol. Already combines add/norm/quant where active; tile-level release requires explicit remote ready dependencies."
    elif module == "embedding.py":
        policy = "retain_sharded_spmd"
        reason = "Sharded vocabulary lookup plus cross-rank reduction; local fusion cannot remove the remote dependency."
    elif module == "attention.py" and f in ("compute_scores", "softmax_scores", "compute_partial_values"):
        policy = "retain_tiled_static_plan"
        reason = "Running implementation uses explicit per-lane leaf invocations. True early consumption needs a finer producer completion boundary, not only deleting outer spmd."
    else:
        policy = "audit_callsite_before_change"
        reason = "Pointwise/small reduction can fuse when ownership and complete reduction axis align. Some source helpers are unused in the linked quantized target; do not count import as active execution."
    row.update(
        policy=policy,
        reason=reason,
        active_status="Source inventory, not active-task count; actual linked child IDs are in build/target_gate_quant_fused images and resolved kernel_config.py",
    )
(N / "evidence/spmd_decisions.json").write_text(json.dumps(items, indent=2) + "\n")
print(
    json.dumps(
        dict(
            ub_bytes=peak,
            limit=limits["Vec"],
            functions=len(items),
            policies=dict(collections.Counter(x["policy"] for x in items)),
        ),
        indent=2,
    )
)
