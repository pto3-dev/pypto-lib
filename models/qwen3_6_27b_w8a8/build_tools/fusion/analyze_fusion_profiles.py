# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run native scheduler/early-dispatch toolkit and compare internal stage envelopes."""

from pathlib import Path
import json, statistics, subprocess, sys

N = Path(__file__).resolve().parents[1]
H = N / "runtime_validation"
G = H / "evidence/gate_quant_fused_02"
sys.path.insert(0, str(H / "src"))
from qwen36.tools.analyze_profiles import prepare

folders = []
for rank in [1, 3]:
    for label in ["reference", "candidate"]:
        source = G / f"kernel_pair/rank{rank}/profile/{label}"
        dest = N / f"evidence/toolkit/{label}/rank{rank}"
        prepare(source, dest, rank)
        folders.append(dest)
node = "/root/.vscode-server/bin/07f806f999227108933c2e30515b26eecc1fda74/node"
script = N.parent / "direct_w8a8_20261027/tools/toolkit_analysis.cjs"
with (N / "evidence/toolkit.log").open("w") as log:
    subprocess.run(
        [node, str(script), *map(str, folders)], check=True, stdout=log, stderr=subprocess.STDOUT, timeout=180
    )
records = []
for rank in range(4):
    row = dict(rank=rank)
    for label in ["reference", "candidate"]:
        p = G / f"kernel_pair/rank{rank}/profile/{label}"
        d = json.loads((p / "attention_leaf_trace.json").read_text())
        hz = d["clock_freq_hz"]
        groups = []
        for group in range(16):
            rs = [r for r in d["records"] if r["layer_group"] == group]
            byid = {fid: [r for r in rs if r["func_id"] == fid] for fid in range(25, 31)}
            ends = {fid: max(r["kernel_return_ticks"] for r in x) for fid, x in byid.items()}
            starts = {fid: min(r["kernel_enter_ticks"] for r in x) for fid, x in byid.items()}
            factor = 1e6 / hz
            groups.append(
                dict(
                    layer_group=group,
                    attention_envelope_us=(max(ends.values()) - min(r["wait_begin_ticks"] for r in rs))
                    * factor,
                    qk_span_us=(ends[27] - starts[27]) * factor,
                    pv_span_us=(ends[29] - starts[29]) * factor,
                    softmax_span_us=(ends[28] - starts[28]) * factor,
                    reduce_or_fused_span_us=(ends[30] - starts[30]) * factor,
                    pv_to_end_us=(ends[30] - ends[29]) * factor,
                    reduce_or_fused_nodes=len(byid[30]),
                )
            )
        row[label] = dict(
            groups=groups,
            sums_us={
                k: sum(g[k] for g in groups)
                for k in [
                    "attention_envelope_us",
                    "qk_span_us",
                    "pv_span_us",
                    "softmax_span_us",
                    "reduce_or_fused_span_us",
                    "pv_to_end_us",
                ]
            },
            note="Reference attention ends before its separate quant task; candidate attention includes quant. These envelopes have unequal boundaries and must not alone determine fusion gain.",
        )
        tf = N / f"evidence/toolkit/{label}/rank{rank}/toolkit_deep_analysis.json"
        if tf.exists():
            td = json.loads(tf.read_text())
            row[label]["toolkit"] = {
                k: td[k] for k in ["makespanUs", "computeTotalUs", "stallTotalUs", "stallByKind"]
            }
    records.append(row)
summary = dict(
    status="passed",
    timed_run_scope="Unprofiled8 consecutive prepared replays per window; two AB/BA windows",
    ranks=[],
    profiles=records,
    production_itl_updated=False,
)
for rank in range(4):
    d = json.loads((G / f"kernel_pair/rank{rank}/result.json").read_text())
    r = statistics.mean(d["samples_ms"]["reference"])
    c = statistics.mean(d["samples_ms"]["candidate"])
    summary["ranks"].append(
        dict(
            rank=rank,
            reference_ms=r,
            candidate_ms=c,
            gain_ms=r - c,
            gain_percent=100 * (1 - c / r),
            all_comparisons_byte_equal=all(
                not t["different"]
                for key in ["checks", "stateful_replay_checks", "ragged_checks"]
                for t in d[key]
            ),
            cache_bytes=d["compared_cache_bytes"],
        )
    )
(N / "evidence/fusion_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary["ranks"], indent=2), flush=True)
