# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bounded fixed-state repeated DIRECT target submissions, no serving metric."""

import json, time, os
from pathlib import Path
import torch
from .kernel_pair import clone_four
from qwen38_runtime.stream_completion import synchronize_stream


@torch.inference_mode()
def validate(executor):
    with torch.npu.stream(executor.capture_stream):
        return run(executor)


def run(executor):
    if executor.policy.execution.value != "direct":
        raise ValueError("DIRECT stress only")
    folder = Path(executor.settings["session_dir"]).parent / "direct_stress" / f"rank{executor.rank}"
    folder.mkdir(parents=True, exist_ok=False)
    stream = torch.npu.current_stream()
    synchronize_stream(stream.npu_stream)
    # Serving owns another caller stream. Retire that completed lifecycle,
    # then create the diagnostic lifecycle on this stream; never weaken the
    # one-workspace/one-stream guard.
    from qwen38_runtime.borrowed_runtime import BorrowedRuntime
    from qwen38_runtime.resources import TargetSubmission
    from qwen38_runtime.artifacts import ArtifactBundle
    from qwen38_runtime.device_plan_layout import STORAGE_BYTES, PLAN_STRIDE

    bindings = tuple(executor.borrowed_runtime.retained)
    contexts = [executor.borrowed_comm.contexts[n] for n in ("logit_data", "logit_signal", "data", "signal")]
    executor.borrowed_runtime.close()
    executor.borrowed_runtime = None
    executor.bridge = None
    bundle = ArtifactBundle.load(Path(executor.settings["build_root"]), executor.rank)
    host = torch.zeros(STORAGE_BYTES, dtype=torch.uint8)
    for index, plan in enumerate(bundle.plans):
        host[index * PLAN_STRIDE : index * PLAN_STRIDE + len(plan)] = torch.frombuffer(
            bytearray(plan), dtype=torch.uint8
        )
    executor.exported["device_program_storage"].copy_(host)
    runtime = BorrowedRuntime(
        executor.policy.runtime_dir,
        int(executor.hidden.device.index),
        bundle.image,
        aicpu_threads=0,
        ring_task_window=4096,
        ring_heap=int(executor.settings.get("diagnostic_ring_heap_mib", 32)) * 1024 * 1024,
        ring_dep_pool=16384,
    )
    runtime.bind(bindings, contexts)
    runtime.prepare_execution("direct")
    executor.borrowed_runtime = runtime
    executor.bridge = TargetSubmission(runtime, executor.control_i32[1:2])
    pools = {name: value.cpu() for name, value in executor.exported.items() if name.startswith("cache_pool")}
    case = clone_four(executor, pools)
    for name, value in pools.items():
        executor.exported[name].copy_(value)
    del pools
    synchronize_stream(stream.npu_stream)
    if executor.borrowed_comm.lib.comm_barrier(executor.borrowed_comm.handle):
        raise RuntimeError("stress start barrier")
    steps = int(executor.settings["runtime_stress_steps"])
    start = time.monotonic()
    checkpoint = dict(status="running", rank=executor.rank, steps_requested=steps, case=case, pid=os.getpid())

    def save():
        (folder / "progress.json").write_text(json.dumps(checkpoint, indent=2) + "\n")

    save()
    try:
        for index in range(steps):
            executor.bridge.enqueue()
            if (index + 1) % 32 == 0 or index + 1 == steps:
                synchronize_stream(stream.npu_stream)
                checkpoint.update(steps_completed=index + 1, elapsed_seconds=time.monotonic() - start)
                save()
    except BaseException as error:
        checkpoint.update(status="failed", error=repr(error))
        save()
        raise
    result = dict(
        checkpoint,
        status="passed",
        comparison_tensors=0,
        scope="Fixed metadata repeated target execution, synthetic four-active slots. No production performance or numerical-equivalence claim.",
    )
    (folder / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    executor.close()
    return result
