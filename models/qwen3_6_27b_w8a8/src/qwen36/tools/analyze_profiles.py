# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Lossless scheduler aliases and toolkit analysis, never a build dependency."""

import argparse, hashlib, json, subprocess
from pathlib import Path

W = Path(__file__).resolve().parents[3]
NODE = "/root/.vscode-server/bin/07f806f999227108933c2e30515b26eecc1fda74/node"


def prepare(source, output, rank):
    output.mkdir(parents=True, exist_ok=False)
    file = source / "chip_swimlane_records.json"
    raw = json.loads(file.read_text())
    workers = raw["aicore_tasks"]
    scheduler = raw["scheduler_tasks"]["records"]
    epochs = {row[6] for row in workers}
    if len(epochs) != 1:
        raise ValueError("Expected one diagnostic epoch")
    worker_keys = {(row[0], row[2], row[6]) for row in workers}
    sched_keys = {(row[0], row[1], row[4]) for row in scheduler}
    if len(worker_keys) != len(workers) or worker_keys != sched_keys:
        raise ValueError("Scheduler/worker join is not bijective")
    streams = sorted(raw["scheduler_records"]["streams"], key=lambda row: row["scheduler_id"])
    if any(row["capture"]["dropped"] or row["capture"]["truncated"] for row in streams):
        raise ValueError("Incomplete scheduler capture")
    raw["aicpu_tasks"] = scheduler
    raw["aicpu_scheduler_phases"] = [row["records"] for row in streams]
    (output / "chip_swimlane_toolkit_compat.json").write_text(json.dumps(raw))
    deps = json.loads((source / "deps.json").read_text())
    (output / "deps.json").write_text(json.dumps(deps))
    manifest = json.loads((W / "src/qwen36/native_backend/link_manifest.json").read_text())
    names = {str(row["func_id"]): row["name"].format(rank=rank) for row in manifest["kernels"]}
    used = {str(k) for t in deps["tasks"] for k in t["kernel_ids"] if k >= 0}
    if used - set(names):
        raise ValueError("Unregistered func_id in profile")
    (output / "name_map.json").write_text(
        json.dumps(
            dict(callable_id_to_name=names, source=str(W / "src/qwen36/native_backend/link_manifest.json")),
            indent=2,
        )
        + "\n"
    )
    mapping = dict(
        source=str(file),
        source_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
        timestamps_changed=False,
        run_epoch=next(iter(epochs)),
        worker_records=len(workers),
        scheduler_records=len(scheduler),
        aliases={
            "aicpu_tasks": "scheduler_tasks.records",
            "aicpu_scheduler_phases": "scheduler_records.streams sorted by scheduler_id, records unchanged",
        },
        dropped_records=0,
    )
    (output / "toolkit_compat_mapping.json").write_text(json.dumps(mapping, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    folders = []
    for label in ["current", "paired"]:
        for rank in range(4):
            source = args.run / "profile" / label / f"rank{rank}"
            output = args.output / label / f"rank{rank}"
            prepare(source, output, rank)
            folders.append(output)
    with (args.output / "toolkit.log").open("w") as log:
        subprocess.run(
            [NODE, str(W / "tools/toolkit_analysis.cjs"), *map(str, folders)],
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=180,
            check=True,
        )
    records = []
    for folder in folders:
        d = json.loads((folder / "toolkit_deep_analysis.json").read_text())
        records.append(
            dict(
                folder=str(folder),
                makespan_us=d["makespanUs"],
                compute_us=d["computeTotalUs"],
                stall_us=d["stallTotalUs"],
            )
        )
    (args.output / "summary.json").write_text(json.dumps(records, indent=2) + "\n")
    print("TOOLKIT_DONE", json.dumps(records), flush=True)


if __name__ == "__main__":
    main()
