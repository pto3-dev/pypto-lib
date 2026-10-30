# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Analyze the already collected terminal profiles; no device execution."""

from collections import defaultdict
import json
from pathlib import Path
import statistics
import subprocess
from qwen36.tools.analyze_profiles import prepare

P = Path(__file__).resolve().parents[1]
RUN = P / "evidence/fast_fused_compact_10k_04/pto/kernel_pair"
OUT = P / "evidence/fast04_retired_analysis"


def main():
    OUT.mkdir(exist_ok=False)
    folders = []
    summary = []
    for rank in [1, 3]:
        for label in ["reference", "candidate"]:
            source = RUN / f"rank{rank}/profile" / label
            dest = OUT / label / f"rank{rank}"
            prepare(source, dest, rank)
            folders.append(dest)
            raw = json.loads((source / "chip_swimlane_records.json").read_text())
            deps = json.loads((source / "deps.json").read_text())
            kernels = {r["task_id"]: tuple(r["kernel_ids"]) for r in deps["tasks"]}
            tasks = defaultdict(list)
            by_kind = defaultdict(list)
            missing = set()
            for r in raw["aicore_tasks"]:
                tasks[str(r[1])].append(r)
            scale = 1e6 / raw["metadata"]["clock_freq_hz"]
            for task, rs in tasks.items():
                if task not in kernels:
                    missing.add(task)
                    continue
                kind = ",".join(str(k) for k in kernels[task])
                by_kind[kind].append((max(r[4] for r in rs) - min(r[3] for r in rs)) * scale)
            summary.append(
                dict(
                    rank=rank,
                    label=label,
                    unmatched_task_ids=sorted(missing),
                    kernels={
                        k: dict(
                            tasks=len(v),
                            median_worker_envelope_us=statistics.median(v),
                            sum_worker_envelopes_us=sum(v),
                        )
                        for k, v in by_kind.items()
                    },
                    note="Diagnostic outer worker envelopes include internal waiting and overlap; sums are not disjoint critical-path components.",
                )
            )
    (OUT / "outer_envelopes.json").write_text(json.dumps(summary, indent=2) + "\n")
    node = "/root/.vscode-server/bin/07f806f999227108933c2e30515b26eecc1fda74/node"
    tool = P.parent / "direct_w8a8_20261027/tools/toolkit_analysis.cjs"
    with (OUT / "toolkit.log").open("w") as log:
        subprocess.run(
            [node, str(tool), *map(str, folders)],
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=180,
        )
    print("ANALYZED", OUT)


if __name__ == "__main__":
    main()
