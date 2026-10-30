# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Check every supported static grid preserves all unrelated leaf dependencies."""

import json, struct, importlib.util
from pathlib import Path

N = Path(__file__).resolve().parents[1]
B = N.parent / "attention_dsl_20261029/build/target_dsl_explicit_v1"
C = N / "build/target_gate_quant_fused"


def decode(b):
    h = struct.unpack_from("<16I", b)
    steps = [struct.unpack_from("<16I", b, h[9] + i * 64) for i in range(h[4])]
    names = [(s[0], s[3]) for s in steps]
    result = {}
    for s in steps:
        bind = struct.unpack_from("<" + "I" * s[6], b, h[10] + 4 * s[5])
        deps = [names[struct.unpack_from("<I", b, h[11] + 4 * j)[0]] for j in range(s[7], s[7] + s[8])]
        result[s[0], s[3]] = dict(lane=s[1], owner=s[2], logical_count=s[4], args=list(bind), waits=deps)
    return h, result


rows = []
for rank in range(4):
    root = C / f"next_levels/rank{rank}"
    spec = importlib.util.spec_from_file_location("candidate_plan", root / "native_backend/attention_plan.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for p in (B / f"next_levels/rank{rank}").glob("production_attention_grid*.bin"):
        old, nodes = decode(p.read_bytes())
        new, actual = decode((root / p.name).read_bytes())
        assert new[12:14] == (30, 31)
        assert module.build(old[3]) == (root / p.name).read_bytes()
        for key, value in nodes.items():
            if key[0] != 30:
                assert actual[key] == value, (rank, old[3], key)
        fused = {k: v for k, v in actual.items() if k[0] == 30}
        assert len(fused) == 24
        for (fid, row), v in fused.items():
            assert v["args"] == [26, 27, 10, 28, 29] and v["logical_count"] == 24
            assert set(v["waits"]) <= {(29, j) for j in range(old[3])}
            # A same-lane predecessor may replace a cross-lane wait, but PV is AIC.
            assert len(v["waits"]) == old[3]
        rows.append(dict(rank=rank, grid=old[3], old_nodes=old[4], new_nodes=new[4]))
    text = (N / f"src/runtime/orchestration_rank{rank}.cpp").read_text()
    assert "rt_submit_aiv_task(31," not in text
    assert (
        "production_args.add_inout(q_inline372)" in text and "production_args.add_inout(xs_inline373)" in text
    )
(N / "evidence/plan_validation.json").write_text(
    json.dumps(dict(status="passed", checked_plans=len(rows), rows=rows), indent=2) + "\n"
)
print("ALL_96_STATIC_PLANS_VALIDATED", flush=True)
