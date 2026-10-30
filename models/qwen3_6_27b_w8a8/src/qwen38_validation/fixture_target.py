# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bounded diagnostic target launches, export fixtures and all-layer leaf traces.

No timed inference path changes, no new drafter, no new scheduler protocol.
The normal serving executor and already-validated target images are unchanged.
"""

import json
import time
from pathlib import Path

import torch

from .byte_compare import equal as byte_equal
from .kernel_pair import clone_four
from .fixture_export import export
from qwen38_runtime.artifacts import ArtifactBundle
from qwen38_runtime.runtime_factory import create_target_runtime
from qwen38_runtime.borrowed_runtime import checked
from qwen38_runtime.stream_completion import synchronize_stream
from qwen38_runtime.device_plan_layout import PLAN_STRIDE, STORAGE_BYTES
from qwen36.tools.device_trace import instrument_plan, export_terminal


@torch.inference_mode()
def validate(executor):
    if not (executor.settings.get("attention_fixture_only") and executor.settings.get("terminal_profile")):
        raise RuntimeError("Explicit terminal fixture collection required")
    if executor.policy.execution.value != "direct":
        raise ValueError("Production target mode must remain DIRECT")
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.npu.stream(executor.capture_stream):
            return _run(executor)
    finally:
        torch.set_num_threads(threads)


def _run(executor):
    folder = Path(executor.settings["session_dir"]).parent / "kernel_pair" / f"rank{executor.rank}"
    folder.mkdir(parents=True, exist_ok=False)
    stream = torch.npu.current_stream()
    synchronize_stream(stream.npu_stream)
    started = time.monotonic()

    def stage(name):
        (folder / "stage.json").write_text(
            json.dumps(dict(rank=executor.rank, stage=name, elapsed_s=time.monotonic() - started)) + "\n"
        )

    stage("retire_runtime")
    bindings = tuple(executor.borrowed_runtime.retained)
    contexts = [executor.borrowed_comm.contexts[n] for n in ["logit_data", "logit_signal", "data", "signal"]]
    executor.borrowed_runtime.close()
    executor.borrowed_runtime = None
    executor.bridge = None
    roots = {
        "reference": Path(executor.settings["build_root"]),
        "candidate": Path(executor.settings["kernel_pair_build"]),
    }
    bundles = {label: ArtifactBundle.load(root, executor.rank) for label, root in roots.items()}
    assert bundles["reference"].params == bundles["candidate"].params
    stage("snapshot_and_clone")
    pools = {n: t.cpu() for n, t in executor.exported.items() if n.startswith("cache_pool")}
    case = clone_four(executor, pools)
    (folder / "case.json").write_text(json.dumps(case, indent=2) + "\n")
    excluded = {"control_i32", "logit_data", "logit_signal", "data", "signal", "device_program_storage"}
    scratch = {
        p["name"]: executor.exported[p["name"]].cpu()
        for p in bundles["reference"].params
        if p["direction"] in ("InOut", "Out")
        and p["name"] not in excluded
        and not p["name"].startswith("cache_pool")
    }
    plans = {}
    for label, bundle in bundles.items():
        host = torch.zeros(STORAGE_BYTES, dtype=torch.uint8)
        for i, plan in enumerate(bundle.plans):
            traced = instrument_plan(plan)
            host[i * PLAN_STRIDE : i * PLAN_STRIDE + len(traced)] = torch.frombuffer(
                bytearray(traced), dtype=torch.uint8
            )
        plans[label] = host

    def barrier():
        if executor.borrowed_comm.lib.comm_barrier(executor.borrowed_comm.handle):
            raise RuntimeError("Fixture barrier failed")

    def restore(label):
        synchronize_stream(stream.npu_stream)
        barrier()
        for name, host in pools.items():
            executor.exported[name].copy_(host)
        for name, host in scratch.items():
            executor.exported[name].copy_(host)
        executor.exported["device_program_storage"].copy_(plans[label])
        executor.exported["signal"].zero_()
        executor.exported["logit_signal"].zero_()
        executor.control_i32[1].fill_(1)
        synchronize_stream(stream.npu_stream)
        barrier()

    expected = None
    checks = []
    fixture = None
    heap = int(executor.settings.get("diagnostic_ring_heap_mib", 64)) * 1024 * 1024
    try:
        for event_type in [2, 8]:
            for label, bundle in bundles.items():
                stage(f"{label}_pmu{event_type}")
                runtime = create_target_runtime(
                    bundle,
                    executor.policy.runtime_dir,
                    int(executor.hidden.device.index),
                    aicpu_threads=0,
                    ring_task_window=4096,
                    ring_heap=heap,
                    ring_dep_pool=16384,
                )
                try:
                    runtime.bind(bindings, contexts)
                    restore(label)
                    dest = folder / "profile" / f"{label}_pmu{event_type}"
                    dest.mkdir(parents=True)
                    runtime.config.enable_scope_stats = 1
                    runtime.config.enable_chip_swimlane = 4
                    runtime.config.enable_dep_gen = 1
                    runtime.config.enable_pmu = event_type
                    runtime.config.output_prefix = str(dest).encode()
                    runtime.prepare()
                    barrier()
                    checked(
                        runtime.lib.simpler_launch_run_on_stream(
                            runtime.ctx, runtime.run, int(stream.npu_stream), None
                        ),
                        "Fixture profile launch",
                    )
                    synchronize_stream(stream.npu_stream)
                    checked(runtime.lib.simpler_wait_run(runtime.ctx, runtime.run), "Fixture profile wait")
                    checked(
                        runtime.lib.simpler_finalize_run(runtime.ctx, runtime.run), "Fixture profile finalize"
                    )
                    runtime.prepared = False
                    trace = export_terminal(executor, bundle.root / f"next_levels/rank{executor.rank}", dest)
                    actual = dict(hidden=executor.hidden.cpu(), logits=executor.logits.cpu())
                    actual.update({n: executor.exported[n].cpu() for n in pools})
                    if expected is None:
                        expected = actual
                        stage("export_layer63_fixture")
                        fixture = export(executor, actual, folder / "fixture", case, bundle)
                    different = [n for n in actual if not byte_equal(actual[n], expected[n])]
                    checks.append(
                        dict(label=label, pmu_event_type=event_type, different=different, trace=trace)
                    )
                    if different:
                        raise ValueError("Profile changed target outputs " + str(different))
                    (folder / "partial.json").write_text(
                        json.dumps(dict(checks=checks, fixture=fixture), indent=2) + "\n"
                    )
                finally:
                    synchronize_stream(stream.npu_stream)
                    runtime.close()
    finally:
        executor.borrowed_comm.close()
        executor.closed = True
    result = dict(
        status="passed",
        rank=executor.rank,
        fixture=fixture,
        checks=checks,
        comparison_tensors=len(expected),
        complete_physical_cache_pools=len(pools),
        compared_cache_bytes=sum(t.numel() * t.element_size() for t in pools.values()),
        source_images={n: b.hashes["callable.image"] for n, b in bundles.items()},
        scope="Four ordinary diagnostic target executions (2 policies x2 event groups), traced all16 attention layers; no unprofiled timing or ITL claim. One checkpoint initialization only.",
    )
    (folder / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    stage("passed")
    return result
