# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Summarize complete runs, retaining failures and comparison limitations."""

import json
import statistics
from pathlib import Path

P = Path(__file__).resolve().parents[1]
W = P.parent
BASE = W / "mtp_subbatch_graph_20261029/evidence/formal_subbatches_10k_01"
LATER = W / "baseline_matched_subbatches_20261030/evidence/matched_native_pto_10k_01"


def load(path):
    return json.loads(path.read_text())


def period(folder):
    records = []
    for path in (folder / "cpu_verifier_trace").glob("*.json"):
        records.extend(load(path)["records"])
    records.sort(key=lambda r: r["monotonic_ns"])
    values = []
    for a, b in zip(records, records[1:]):
        if (
            len(a["req_ids"]) == len(b["req_ids"]) == 4
            and set(a["req_ids"]) == set(b["req_ids"])
            and all(n > 0 for n in a["valid_output_lengths"])
            and all(n > 0 for n in b["valid_output_lengths"])
        ):
            values.append((b["monotonic_ns"] - a["monotonic_ns"]) / 1e6)
    return (
        dict(
            samples=len(values),
            mean_ms=statistics.mean(values),
            median_ms=statistics.median(values),
            over55ms=sum(x > 55 for x in values),
            over65ms=sum(x > 65 for x in values),
        )
        if values
        else None
    )


def row(folder):
    result = load(folder / "official_result.json")
    valid = (
        result["completed"] == 4
        and not result.get("failed", 0)
        and result["input_lens"] == [32768] * 4
        and result["output_lens"] == [10240] * 4
    )
    status = load(folder / "status.json")
    cmd = status["server_command"]
    cfg = json.loads(cmd[cmd.index("--additional-config") + 1])["qwen38_pto_mpmd"]
    out = dict(
        folder=str(folder),
        valid_complete_10k=valid,
        output_lengths=result["output_lens"],
        devices=status["devices"],
        target_build=cfg["build_root"],
        compact_inputs=cfg.get("compact_inputs", "off"),
    )
    if not valid:
        out["reason"] = "EngineCore exited unexpectedly; truncated SSE outputs; no performance promotion"
        return out
    summary = load(folder / "measurement/summary.json")
    out.update(
        tpot_ms=summary["average_tpot_ms"],
        stable_sse=summary["client_receive_itl"]["stable_sse_batch_gap"],
        stable_token_amortized=summary["client_receive_itl"]["stable_token_weighted_amortized_interval"],
        acceptance=summary["mtp"],
        ttft_ms=summary["ttft_ms"],
        input_hash=summary["input_token_sha256"],
        output_hash=summary["output_token_sha256"],
        common_b4_cpu_period=period(folder),
    )
    stats = load(folder / "worker_stats.json")
    out["hccl"] = [x.get("deterministic_native_tp", x.get("native_hccl_policy")) for x in stats]
    return out


def main():
    rows = {"historical16": row(BASE / "pto"), "later17": row(LATER / "pto")}
    for key in ["off", "both"]:
        rows["fixed_hccl_" + key] = row(P / "evidence/fixed_numeric_pipeline_10k_01" / key / "pto")
    for name in [
        "fast_fused_compact_10k_01",
        "fast_fused_compact_10k_02",
        "fast_fused_compact_10k_03",
        "fast_fused_compact_10k_04",
        "fast_fused_compact_10k_05",
        "fast_fused_compact_10k_06",
        "historical16_control_10k_02",
    ]:
        folder = P / "evidence" / name / "pto"
        if (folder / "official_result.json").exists() and (
            (folder / "measurement/summary.json").exists()
            or load(folder / "status.json").get("status") == "failed"
        ):
            rows[name] = row(folder)
    a, b = rows["fixed_hccl_off"], rows["fixed_hccl_both"]
    comparison = dict(
        same_input_ids=a["input_hash"] == b["input_hash"],
        same_output_ids=a["output_hash"] == b["output_hash"],
        observed_tpot_reduction_ms=a["tpot_ms"] - b["tpot_ms"],
        limitations=[
            "Admission order and resulting prefill chunks differ.",
            "HCCL repeatability alone does not fix execution-shape differences.",
            "Observed cross-run TPOT difference is not an isolated causal speedup.",
        ],
    )
    recovered = rows.get("fast_fused_compact_10k_06", {})
    goal_met = bool(
        recovered.get("valid_complete_10k")
        and recovered.get("tpot_ms", 1e9) <= rows["historical16"]["tpot_ms"]
    )
    out = dict(
        goal_scope="Restore measured historical TPOT on frozen public surrogate fixture; native long-output equivalence tracked separately",
        goal_tpot_ms=rows["historical16"]["tpot_ms"],
        rows=rows,
        fixed_hccl_comparison=comparison,
        completed_goal=goal_met,
        performance_goal_met=goal_met,
        long_output_native_numerics_closed=False,
        current_verified_run="fast_fused_compact_10k_06" if goal_met else None,
        authorized_shm_reclaim=json.loads((P / "evidence/SHM_AUTHORIZED_RECLAIM.json").read_text()),
        failure_cause="fast03: Python faulthandler captured SIGBUS in shm_broadcast.py enqueue:741; /dev/shm64MiB, free667648B; owned1MiB posix_fallocate returned ENOSPC. fast01 unrecorded signal is not retroactively asserted.",
        profiling="Formal runs do not enable tensor audits, swimlane or PMU; retired-state validation occurs only after timing.",
    )
    (P / "evidence/RECOVERY_STATUS.json").write_text(json.dumps(out, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: {n: v for n, v in r.items() if n in ["valid_complete_10k", "tpot_ms", "output_lengths"]}
                for k, r in rows.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
