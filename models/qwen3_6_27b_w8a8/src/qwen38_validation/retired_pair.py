# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exact image + same retired state: integrated policy vs historical FULL ABI.

User requests have finished. Caches/workspaces are restored outside timed spans;
no request continues from these diagnostic outputs. No model import/generator.
"""

import json, os
from pathlib import Path
import torch
from .legacy_runtime import BorrowedGraphRuntime, DeviceTensorBinding
from qwen38_runtime.config import Execution
from qwen38_runtime.stream_completion import synchronize_stream

LEGACY = Path(
    "/home/g00893677/pto-dev/qwen3_8_27b-perf/work/inprocess_native_nz_20261023/simpler/build/lib/a2a3"
)


@torch.inference_mode()
def validate(executor):
    if executor.settings.get("attention_fixture_only"):
        from .fixture_target import validate as fixture

        return fixture(executor)
    if executor.settings.get("runtime_stress_steps"):
        from .direct_stress import validate as stress

        return stress(executor)
    if executor.settings.get("kernel_pair_build"):
        from .kernel_pair import validate as pair

        return pair(executor)
    if not executor.settings.get("terminal_profile"):
        raise RuntimeError("Retired-state validation was not enabled")
    with torch.npu.stream(executor.capture_stream):
        return _validate_on_stream(executor)


def _validate_on_stream(executor):
    root = Path(executor.settings["session_dir"]).parent / "retired_pair"
    folder = root / f"rank{executor.rank}"
    folder.mkdir(parents=True, exist_ok=False)
    current = executor.borrowed_runtime
    stream = torch.npu.current_stream()
    synchronize_stream(stream.npu_stream)
    for ref in executor.caller_graphs:
        graph = ref()
        if graph is not None:
            graph.reset()
    executor.caller_graphs.clear()
    executor.graph_entries.clear()
    build = Path(executor.settings["build_root"]) / "next_levels" / f"rank{executor.rank}"
    params = json.loads((build / "compiled_meta.json").read_text())["params"]
    metadata = executor.native_metadata.cpu()
    slots = executor.control_i64[48:72].cpu().tolist()
    selections = []
    for name, tensor in executor.cache_views.items():
        layer = int(name[1 : name.index("_")])
        indices = slots if name.endswith(("keys", "values")) else metadata[layer, :48].tolist()
        indices = sorted({int(i) for i in indices if 0 <= int(i) < tensor.shape[0]})
        if indices:
            selections.append((name, tensor, torch.tensor(indices, dtype=torch.int64, device=tensor.device)))
    cache_before = {name: tensor.index_select(0, index).cpu() for name, tensor, index in selections}
    excluded = {"control_i32", "logit_data", "logit_signal", "data", "signal"}
    mutable = {
        p["name"]: executor.exported[p["name"]]
        for p in params
        if p["direction"] in ("InOut", "Out")
        and p["name"] not in excluded
        and not p["name"].startswith("cache_pool")
    }
    scratch_before = {name: tensor.cpu() for name, tensor in mutable.items()}

    def barrier(label):
        rc = executor.borrowed_comm.lib.comm_barrier(executor.borrowed_comm.handle)
        if rc:
            raise RuntimeError(f"Validation barrier {label}: {rc}")

    def restore(reinitialize_plan=False):
        synchronize_stream(stream.npu_stream)
        barrier("before restore")
        for name, tensor, index in selections:
            tensor.index_copy_(0, index, cache_before[name].to(tensor.device))
        for name, tensor in mutable.items():
            if name != "device_program_storage":
                tensor.copy_(scratch_before[name].to(tensor.device))
        if reinitialize_plan:
            from qwen38_runtime.artifacts import ArtifactBundle
            from qwen38_runtime.device_plan_layout import PLAN_STRIDE, STORAGE_BYTES

            bundle = ArtifactBundle.load(Path(executor.settings["build_root"]), executor.rank)
            storage = executor.exported["device_program_storage"]
            host = torch.zeros(STORAGE_BYTES, dtype=torch.uint8)
            for index, plan in enumerate(bundle.plans):
                host[index * PLAN_STRIDE : index * PLAN_STRIDE + len(plan)] = torch.frombuffer(
                    bytearray(plan), dtype=torch.uint8
                )
            storage.copy_(host.to(storage.device))
        executor.exported["signal"].zero_()
        executor.exported["logit_signal"].zero_()
        executor.control_i32[1].zero_()
        synchronize_stream(stream.npu_stream)
        barrier("after restore")

    def outputs():
        d = dict(hidden=executor.hidden.cpu(), logits=executor.logits.cpu())
        d.update({name: tensor.index_select(0, index).cpu() for name, tensor, index in selections})
        return d

    bindings = [
        DeviceTensorBinding(v.tensor, v.shape) if hasattr(v, "tensor") and hasattr(v, "shape") else v
        for v in current.retained
    ]
    contexts = [
        executor.borrowed_comm.contexts[name] for name in ["logit_data", "logit_signal", "data", "signal"]
    ]
    from qwen38_runtime.borrowed_runtime import BorrowedRuntime
    from qwen38_runtime.resources import TargetSubmission

    image = (build / "callable.image").read_bytes()
    # Simpler device initialization installs per-process device dispatch state.
    # Do not keep two independently initialized runtime contexts live together.
    integrated_bindings = tuple(current.retained)
    current.close()
    executor.borrowed_runtime = None
    executor.bridge = None
    timings = {"integrated": [], "legacy_full": []}
    expected = None
    checks = []

    def stage(label):
        (folder / "stage.json").write_text(json.dumps(dict(stage=label, rank=executor.rank)) + "\n")

    for label in ["legacy_full", "integrated"]:
        stage(label + "_initialize")
        if label == "legacy_full":
            runtime = BorrowedGraphRuntime(
                LEGACY,
                int(executor.hidden.device.index),
                image,
                aicpu_threads=0,
                ring_task_window=4096,
                ring_heap=32 * 1024 * 1024,
                ring_dep_pool=16384,
            )
            runtime.bind(bindings, contexts)
            runtime.prepare_graph()
        else:
            runtime = BorrowedRuntime(
                executor.policy.runtime_dir,
                int(executor.hidden.device.index),
                image,
                aicpu_threads=0,
                ring_task_window=4096,
                ring_heap=32 * 1024 * 1024,
                ring_dep_pool=16384,
            )
            runtime.bind(integrated_bindings, contexts)
            runtime.prepare_execution(executor.policy.execution.value)
            executor.borrowed_runtime = runtime
            executor.bridge = TargetSubmission(runtime, executor.control_i32[1:2])
        graph = None
        try:
            if label == "legacy_full" or executor.policy.execution is Execution.FULL:
                restore(reinitialize_plan=True)
                stage(label + "_capture")
                graph = torch.npu.NPUGraph()
                capture = stream
                with torch.npu.graph(graph, stream=capture):
                    executor.control_i32[1].add_(1)
                    runtime.enqueue_graph(capture.npu_stream)
                synchronize_stream(capture.npu_stream)
            else:
                restore(reinitialize_plan=True)
            for iteration in range(2):
                restore()
                stage(label + "_sample" + str(iteration))
                begin = torch.npu.Event(enable_timing=True)
                end = torch.npu.Event(enable_timing=True)
                begin.record(stream)
                if graph is not None:
                    graph.replay()
                else:
                    executor.bridge.enqueue()
                end.record(stream)
                synchronize_stream(stream.npu_stream)
                timings[label].append(begin.elapsed_time(end))
                actual = outputs()
                if expected is None:
                    expected = actual
                different = [
                    name
                    for name in actual
                    if not torch.equal(actual[name].view(torch.uint8), expected[name].view(torch.uint8))
                ]
                checks.append(dict(label=label, iteration=iteration, different=different))
                if different:
                    raise ValueError(
                        f"Integrated {executor.policy.execution.value} state differs: {different}"
                    )
        finally:
            synchronize_stream(stream.npu_stream)
            if graph is not None:
                graph.reset()
            if label == "legacy_full":
                runtime.close()
    result = dict(
        status="passed",
        rank=executor.rank,
        pid=os.getpid(),
        execution=executor.policy.execution.value,
        comparison_tensors=len(expected),
        checks=checks,
        samples_ms=timings,
        active_sequences=sum(slots[index] >= 0 for index in range(0, 24, 6)),
        selected_cache_rows={name: int(index.numel()) for name, tensor, index in selections},
        scope="Same image, restored retired state; sequential runtime lifetimes, historical FULL then integrated. Target only; two samples diagnostic, not a formal performance baseline.",
    )
    (folder / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    stage("passed")
    return result
