# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Same-PID full target execution hosted by each native vLLM TP rank."""

import json, os, time
from pathlib import Path
import torch
from qwen36.runtime.borrowed_graph_runtime import BorrowedGraphRuntime, DeviceTensorBinding
from qwen36.runtime.borrowed_comm import BorrowedCommWindows
from qwen36.runtime.native_nz_weights import bank_native_weights
from qwen36.model_layout import WeightBankSpec, H, KG, VG

W = Path(__file__).resolve().parents[1]
RUNTIME_BASE = Path("/home/g00893677/pto-dev/qwen3_8_27b-perf/work/inprocess_native_nz_20261023")


class NativeGraphSubmission:
    def __init__(self, runtime, epoch):
        self.runtime = runtime
        self.epoch = epoch

    def enqueue(self):
        self.epoch.add_(1)
        self.runtime.enqueue_graph(torch.npu.current_stream().npu_stream)


def prepare_weights(executor):
    if executor.shared_weights is not None:
        return
    start = time.perf_counter()
    spec = WeightBankSpec()
    executor.shared_weights, executor.nz_bank_owners, executor.shared_weight_info = bank_native_weights(
        executor.causal, spec
    )
    executor.shared_weight_info["initialization_ms"] = (time.perf_counter() - start) * 1000
    print("NATIVE_NZ_SHARED_WEIGHTS_READY", executor.rank, executor.shared_weight_info, flush=True)


def native_cache_bindings(executor, context):
    from vllm_ascend.pypto.qwen3_8_cache import full_attention_views, gdn_state_views

    rows = block_size = slots = None
    for index, layer in enumerate(executor.causal.model.layers):
        if index % 4 == 3:
            native = layer.self_attn.attn.kv_cache[context.virtual_engine]
            key, value = full_attention_views(native, H)
            size = native[0].shape[1]
            if rows is not None and (rows, block_size) != (key.shape[0], size):
                raise ValueError("Native full cache dimensions differ")
            rows, block_size = key.shape[0], size
            executor.cache_views.update({f"l{index}_keys": key, f"l{index}_values": value})
        else:
            conv, state = gdn_state_views(layer.linear_attn.kv_cache[context.virtual_engine], KG, VG)
            if slots is not None and slots != conv.shape[0]:
                raise ValueError("Native GDN dimensions differ")
            slots = conv.shape[0]
            executor.cache_views.update({f"l{index}_conv_state": conv, f"l{index}_recurrent_state": state})
    identities = {}
    pools = []
    layout = []
    for name, value in executor.cache_views.items():
        storage = value.untyped_storage()
        identity = storage.data_ptr()
        if identity not in identities:
            pool = f"cache_pool{len(pools)}"
            identities[identity] = pool
            pools.append([pool, storage.nbytes()])
            executor.exported[pool] = torch.empty(0, dtype=torch.bfloat16, device=value.device).set_(
                storage, 0, (storage.nbytes() // 2,), (1,)
            )
        pool = identities[identity]
        layout.append(
            [
                name,
                pool,
                value.data_ptr() - identity,
                list(value.shape),
                "FP32" if value.dtype == torch.float32 else "BF16",
            ]
        )
    return dict(
        max_seq=executor.max_seq,
        cache_rows=rows,
        cache_block_size=block_size,
        state_slots=slots,
        native_cache_pools=pools,
        native_cache_layout=layout,
    )


def prepare_graph(executor, context, device):
    if executor.bridge is not None:
        return
    if torch.npu.is_current_stream_capturing():
        raise RuntimeError("Borrowed execution must be prepared before capture")
    prepare_weights(executor)
    executor._ensure_outputs(device)
    dimensions = native_cache_bindings(executor, context)
    build_root = Path(executor.settings["build_root"])
    report = Path(executor.settings["session_dir"]) / f"rank{executor.rank}_dimensions.json"
    report.write_text(json.dumps(dimensions, indent=2) + "\n")
    if executor.settings.get("geometry_only"):
        # Initialization-only rendezvous: keep the first rank alive until every
        # peer has persisted its geometry. This never executes in benchmark runs.
        deadline = time.monotonic() + 60
        while not all((report.parent / f"rank{r}_dimensions.json").is_file() for r in range(4)):
            if time.monotonic() > deadline:
                raise TimeoutError("Peers did not finish geometry collection within60s")
            time.sleep(0.2)
        raise RuntimeError(f"Geometry-only initialization complete; actual native cache saved to {report}")
    build_contract = json.loads((build_root / "build_contract.json").read_text())
    executor.communication_protocol = build_contract.get("communication_protocol", "group_counter_v1")
    executor.communication_max_steps = build_contract.get("communication_max_steps", 67108864)
    if build_contract.get("compile_smoke"):
        raise ValueError("Compile-smoke geometry cannot be used for inference")
    if build_contract.get("model") != "Qwen3.6 mixed W8A8":
        raise ValueError("Wrong build contract")
    expected = json.loads((build_root / f"cache_dimensions_rank{executor.rank}.json").read_text())
    if dimensions != expected:
        report = Path(executor.settings["session_dir"]) / f"rank{executor.rank}_dimensions.json"
        report.write_text(json.dumps(dimensions, indent=2) + "\n")
        raise ValueError(f"Native cache geometry differs from precompiled graph; actual saved to {report}")
    cols = (executor.max_seq + dimensions["cache_block_size"] - 1) // dimensions["cache_block_size"]
    executor.table_cols = cols
    executor.control_i64 = torch.zeros(72, dtype=torch.int64, device=device)
    executor.control_i64[48:72].fill_(-1)
    executor.control_i32 = torch.tensor(
        [executor.rank, 0, 0, 0]
        + [0, 6, 12, 18]
        + [24] * 21
        + [0] * 24
        + [0] * 24
        + [0] * (24 * cols)
        + [1] * 24,
        dtype=torch.int32,
        device=device,
    )
    executor.native_metadata = torch.zeros((64, 24 * (3 + cols)), dtype=torch.int32, device=device)
    executor.native_metadata[:, :48].fill_(-1)
    executor.native_metadata[:, 48:72].fill_(1)
    executor.exported.update(executor.shared_weights)
    executor.exported.update(
        control_i64=executor.control_i64,
        control_i32=executor.control_i32,
        native_metadata=executor.native_metadata,
        global_logits=executor.logits,
        hidden_result=executor.hidden,
    )
    rotary = executor.causal.model.layers[3].self_attn.rotary_emb.cos_sin_cache[: executor.max_seq]
    executor.exported.update(
        cos=rotary[:, :32].to(dtype=torch.bfloat16).contiguous(),
        sin=rotary[:, 32:].to(dtype=torch.bfloat16).contiguous(),
    )
    build = build_root / "next_levels" / f"rank{executor.rank}"
    metadata = json.loads((build / "compiled_meta.json").read_text())
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "int32": torch.int32,
        "int64": torch.int64,
        "uint8": torch.uint8,
        "int8": torch.int8,
    }
    for param in metadata["params"]:
        name = param["name"]
        if name in executor.exported or name in ("logit_data", "logit_signal", "data", "signal"):
            continue
        shape = param["shape"]
        dtype = dtype_map[param["dtype"]]
        if name == "softmax_columns":
            value = torch.arange(1024, dtype=dtype, device=device).reshape(shape)
        elif name == "device_program_storage":
            from qwen36.device_plan_layout import MAX_CLUSTERS, PLAN_STRIDE, STORAGE_BYTES

            host = torch.zeros(STORAGE_BYTES, dtype=torch.uint8)
            for grid in range(1, MAX_CLUSTERS + 1):
                image = (build / f"production_attention_grid{grid}.bin").read_bytes()
                host[(grid - 1) * PLAN_STRIDE : (grid - 1) * PLAN_STRIDE + len(image)] = torch.frombuffer(
                    bytearray(image), dtype=torch.uint8
                )
            value = host.to(device)
        else:
            value = torch.zeros(shape, dtype=dtype, device=device)
        executor.exported[name] = value
    executor.capture_stream = torch.npu.Stream(device=device)
    torch.npu.synchronize()
    runtime = BorrowedGraphRuntime(
        RUNTIME_BASE / "simpler/build/lib/a2a3",
        int(device.index),
        (build / "callable.image").read_bytes(),
        aicpu_threads=0,
        ring_task_window=4096,
        ring_heap=32 * 1024 * 1024,
        ring_dep_pool=16384,
    )
    executor.borrowed_runtime = runtime
    windows = [
        (param["name"], tuple(param["shape"]), dtype_map[param["dtype"]])
        for param in metadata["params"]
        if param["name"] in ("logit_data", "logit_signal", "data", "signal")
    ]
    executor.borrowed_comm = BorrowedCommWindows(
        runtime.lib,
        executor.rank,
        4,
        executor.capture_stream,
        Path(executor.settings["session_dir"]) / "rootinfo",
        windows,
    )
    executor.exported.update(executor.borrowed_comm.buffers)
    tensors = []
    for param in metadata["params"]:
        value = executor.exported[param["name"]]
        if param.get("layout") == "NZ":
            shape = param["shape"]
            batch = shape[0] if len(shape) == 3 else 1
            n, k = shape[-2:]
            block = 32 if param["dtype"] == "int8" else 16
            if k % block or n % 16:
                raise ValueError(f"Unaligned NZ argument {param['name']}")
            value = DeviceTensorBinding(value, (batch, k // block, n // 16, 16, block))
        tensors.append(value)
    runtime.bind(
        tensors,
        [executor.borrowed_comm.contexts[name] for name in ("logit_data", "logit_signal", "data", "signal")],
    )
    runtime.prepare_graph()
    executor.bridge = NativeGraphSubmission(runtime, executor.control_i32[1:2])
    executor.replay_window = 1
    executor.zero_row = torch.zeros(24 * (3 + cols), dtype=torch.int32, device=device)
    executor.one = torch.ones(24, dtype=torch.int32, device=device)
    executor.topology = dict(
        native_pid=os.getpid(),
        runtime_pid=os.getpid(),
        rank=executor.rank,
        extra_l3_process=False,
        extra_l2_process=False,
    )
    executor.generation = str(build.parent.parent)
    info = dict(
        rank=executor.rank,
        pid=os.getpid(),
        borrowed_context=True,
        outer_request_done_bridge=False,
        cache_contents_copied=False,
        argument_count=len(tensors),
        runtime_config=dict(
            aicpu_threads="architecture default",
            ring_task_window=4096,
            ring_heap=32 * 1024 * 1024,
            ring_dep_pool=16384,
        ),
        cache_dimensions=dimensions,
    )
    (Path(executor.settings["session_dir"]) / f"rank{executor.rank}_prepared.json").write_text(
        json.dumps(info, indent=2) + "\n"
    )
    print("NATIVE_NZ_FULL_GRAPH_PREPARED", executor.rank, os.getpid(), flush=True)


def close(executor):
    runtime = getattr(executor, "borrowed_runtime", None)
    if runtime is None:
        return
    torch.npu.synchronize()
    for reference in executor.caller_graphs:
        graph = reference()
        if graph is not None:
            graph.reset()
    executor.caller_graphs.clear()
    executor.bridge = None
    runtime.close()
    executor.borrowed_comm.close()
    executor.borrowed_runtime = None
    print("NATIVE_NZ_FULL_GRAPH_CLOSED", executor.rank, os.getpid(), flush=True)


def _terminal_state_selection(executor):
    metadata = executor.native_metadata.cpu()
    kv_slots = executor.control_i64[48:72].cpu().tolist()
    views = []
    for name, tensor in executor.cache_views.items():
        layer = int(name[1 : name.index("_")])
        candidates = kv_slots if name.endswith(("keys", "values")) else metadata[layer, :48].tolist()
        indices = sorted(set(int(i) for i in candidates if 0 <= int(i) < tensor.shape[0]))
        if not indices:
            continue
        device_indices = torch.tensor(indices, dtype=torch.int64, device=tensor.device)
        views.append((name, tensor, device_indices))
    return views


@torch.inference_mode()
def profile_terminal_step(executor):
    """Paired terminal-state diagnostic, outside the reported request timing.

    Both variants receive identical cache bytes and numerical inputs. Only
    communication epochs advance between them. These caches are retired after
    this check, so no request resumes from either diagnostic output.
    """
    if not executor.settings.get("terminal_profile"):
        raise ValueError("Terminal profiling was not selected for this run")
    runtime = executor.borrowed_runtime
    torch.npu.synchronize()
    for reference in executor.caller_graphs:
        graph = reference()
        if graph is not None:
            graph.reset()
    executor.caller_graphs.clear()
    from qwen36.runtime.borrowed_graph_runtime import checked

    checked(runtime.lib.simpler_finalize_run(runtime.ctx, runtime.run), "release captured profile binding")
    runtime.prepared = runtime.graph_ready = False
    executor.bridge = None
    torch.npu.empty_cache()
    generation = Path(executor.generation)
    current_meta = json.loads(
        (generation / "next_levels" / f"rank{executor.rank}" / "compiled_meta.json").read_text()
    )
    paired = executor.settings.get("paired_build_root")
    if paired:
        other = Path(paired)
        other_meta = json.loads(
            (other / "next_levels" / f"rank{executor.rank}" / "compiled_meta.json").read_text()
        )
        if current_meta["params"] != other_meta["params"]:
            raise ValueError("Paired callable ABI differs")
        runtime.register_callable(
            1, (other / "next_levels" / f"rank{executor.rank}" / "callable.image").read_bytes()
        )
        order = [("current", 0), ("paired", 1)]
    else:
        order = [("current", 0)]
    selections = _terminal_state_selection(executor)
    before = {name: tensor.index_select(0, index).cpu() for name, tensor, index in selections}
    excluded = {"control_i32", "logit_data", "logit_signal", "data", "signal"}
    mutable = {
        param["name"]: executor.exported[param["name"]]
        for param in current_meta["params"]
        if param["direction"] in ("InOut", "Out")
        and param["name"] not in excluded
        and not param["name"].startswith("cache_pool")
    }
    scratch_before = {name: tensor.cpu() for name, tensor in mutable.items()}
    checkpoint_bytes = sum(
        value.numel() * value.element_size()
        for value in list(before.values()) + list(scratch_before.values())
    )
    arguments = runtime.retained
    contexts = [
        executor.borrowed_comm.contexts[name] for name in ("logit_data", "logit_signal", "data", "signal")
    ]
    output_root = Path(executor.settings["session_dir"]).parent / "profile"
    records = []
    reference = None
    for label, callable_id in order:
        for name, tensor, index in selections:
            tensor.index_copy_(0, index, before[name].to(tensor.device))
        for name, tensor in mutable.items():
            tensor.copy_(scratch_before[name].to(tensor.device))
        # Plan func_ids are local to the callable image. ABI-compatible images
        # can assign different IDs, so the plan bytes must follow the variant.
        variant_root = generation if callable_id == 0 else other
        if "device_program_storage" in executor.exported:
            from qwen36.device_plan_layout import MAX_CLUSTERS, PLAN_STRIDE

            storage = executor.exported["device_program_storage"].view(torch.uint8).flatten()
            variant_build = variant_root / "next_levels" / f"rank{executor.rank}"
            for grid in range(1, MAX_CLUSTERS + 1):
                plan = (variant_build / f"production_attention_grid{grid}.bin").read_bytes()
                if executor.settings.get("terminal_device_trace"):
                    from qwen36.tools.device_trace import instrument_plan

                    plan = instrument_plan(plan)
                start = (grid - 1) * PLAN_STRIDE
                storage[start : start + len(plan)].copy_(
                    torch.frombuffer(bytearray(plan), dtype=torch.uint8).to(storage.device)
                )
        torch.npu.synchronize()
        folder = output_root / label / f"rank{executor.rank}"
        folder.mkdir(parents=True, exist_ok=True)
        runtime.config.enable_chip_swimlane = 4
        runtime.config.enable_dep_gen = 1
        runtime.config.enable_pmu = 2
        runtime.config.output_prefix = str(folder).encode()
        # Retired-state protocol comparison: counters and epoch must start
        # together. This is outside request timing and never drains decode.
        checked(
            executor.borrowed_comm.lib.comm_barrier(executor.borrowed_comm.handle), "terminal before reset"
        )
        executor.exported["signal"].zero_()
        executor.exported["logit_signal"].zero_()
        executor.control_i32[1].fill_(1)
        torch.npu.synchronize()
        checked(
            executor.borrowed_comm.lib.comm_barrier(executor.borrowed_comm.handle), "terminal after reset"
        )
        runtime.bind(arguments, contexts, callable_id=callable_id)
        runtime.eager(torch.npu.current_stream().npu_stream)
        torch.npu.synchronize()
        runtime.finish_eager()
        current = dict(hidden=executor.hidden.cpu(), logits=executor.logits.cpu())
        current.update({name: tensor.index_select(0, index).cpu() for name, tensor, index in selections})
        if reference is None:
            reference = current
            different = []
        else:
            different = [
                name
                for name, value in current.items()
                if not torch.equal(value.view(torch.uint8), reference[name].view(torch.uint8))
            ]
        details = {
            name: dict(
                max_abs_error=float(
                    (current[name].float() - reference[name].float()).abs().nan_to_num().max()
                ),
                different_bytes=int(
                    (current[name].view(torch.uint8) != reference[name].view(torch.uint8)).sum()
                ),
                nan_count=int(torch.isnan(current[name]).sum()) if current[name].is_floating_point() else 0,
            )
            for name in different
        }
        row = dict(
            build_root=str(generation if callable_id == 0 else other),
            difference_details=details,
            rank=executor.rank,
            pid=os.getpid(),
            label=label,
            output=str(folder),
            swimlane_level=4,
            dependency_graph=True,
            pmu=True,
            terminal_only=True,
            checkpoint_bytes=checkpoint_bytes,
            restored_mutable_workspaces=len(mutable),
            comparison_tensors=len(current),
            bitwise_different=different,
        )
        if executor.settings.get("terminal_device_trace"):
            from qwen36.tools.device_trace import export_terminal

            row["attention_leaf_trace"] = export_terminal(executor, variant_build, folder)
        (folder / "diagnostic_scope.json").write_text(json.dumps(row, indent=2) + "\n")
        records.append(row)
    if executor.settings.get("terminal_graph_timing") and paired:
        # A few complete target replays, no PMU/swimlane and no drafter.
        # Identical retired state is restored before each timed replay. Signals
        # and epoch are reset together after all ranks finish the previous one.
        samples = {"current": [], "paired": []}
        runtime.config.enable_chip_swimlane = 0
        runtime.config.enable_dep_gen = 0
        runtime.config.enable_pmu = 0
        for sample in range(3):
            for label, callable_id in order if sample % 2 == 0 else order[::-1]:
                for name, tensor, index in selections:
                    tensor.index_copy_(0, index, before[name].to(tensor.device))
                for name, tensor in mutable.items():
                    tensor.copy_(scratch_before[name].to(tensor.device))
                variant_build = (
                    (generation if callable_id == 0 else other) / "next_levels" / f"rank{executor.rank}"
                )
                from qwen36.device_plan_layout import MAX_CLUSTERS, PLAN_STRIDE

                storage = executor.exported["device_program_storage"].view(torch.uint8).flatten()
                for grid in range(1, MAX_CLUSTERS + 1):
                    plan = (variant_build / f"production_attention_grid{grid}.bin").read_bytes()
                    start = (grid - 1) * PLAN_STRIDE
                    storage[start : start + len(plan)].copy_(
                        torch.frombuffer(bytearray(plan), dtype=torch.uint8).to(storage.device)
                    )
                torch.npu.synchronize()
                checked(
                    executor.borrowed_comm.lib.comm_barrier(executor.borrowed_comm.handle),
                    "timing before reset",
                )
                executor.exported["signal"].zero_()
                executor.exported["logit_signal"].zero_()
                executor.control_i32[1].zero_()
                torch.npu.synchronize()
                checked(
                    executor.borrowed_comm.lib.comm_barrier(executor.borrowed_comm.handle),
                    "timing after reset",
                )
                runtime.bind(arguments, contexts, callable_id=callable_id)
                runtime.prepare_graph()
                graph = torch.npu.NPUGraph()
                stream = executor.capture_stream
                try:
                    with torch.npu.graph(graph, stream=stream):
                        executor.control_i32[1].add_(1)
                        runtime.enqueue_graph(stream.npu_stream)
                    # Capture executes on this backend. Restore the same retired
                    # bytes once more, then time only the captured target replay.
                    stream.synchronize()
                    for name, tensor, index in selections:
                        tensor.index_copy_(0, index, before[name].to(tensor.device))
                    for name, tensor in mutable.items():
                        if name != "device_program_storage":
                            tensor.copy_(scratch_before[name].to(tensor.device))
                    torch.npu.synchronize()
                    checked(
                        executor.borrowed_comm.lib.comm_barrier(executor.borrowed_comm.handle),
                        "timing capture complete",
                    )
                    executor.exported["signal"].zero_()
                    executor.exported["logit_signal"].zero_()
                    executor.control_i32[1].zero_()
                    torch.npu.synchronize()
                    checked(
                        executor.borrowed_comm.lib.comm_barrier(executor.borrowed_comm.handle), "timing ready"
                    )
                    begin = torch.npu.Event(enable_timing=True)
                    end = torch.npu.Event(enable_timing=True)
                    with torch.npu.stream(stream):
                        begin.record(stream)
                        graph.replay()
                        end.record(stream)
                    stream.synchronize()
                    samples[label].append(begin.elapsed_time(end))
                    actual = dict(hidden=executor.hidden.cpu(), logits=executor.logits.cpu())
                    actual.update(
                        {name: tensor.index_select(0, index).cpu() for name, tensor, index in selections}
                    )
                    mismatched = [
                        name
                        for name, value in actual.items()
                        if not torch.equal(value.view(torch.uint8), reference[name].view(torch.uint8))
                    ]
                    if mismatched:
                        raise ValueError(
                            f"Timed {label} replay differs from same-state reference: {mismatched}"
                        )
                finally:
                    stream.synchronize()
                    graph.reset()
                    checked(runtime.lib.simpler_finalize_run(runtime.ctx, runtime.run), "release timed graph")
                    runtime.prepared = runtime.graph_ready = False
        result = dict(
            rank=executor.rank,
            samples_ms=samples,
            bitwise_equal=True,
            comparison_tensors=len(reference),
            scope="captured complete target only; no drafter or input packing; fixed retired state; interleaved3x; no profiler",
        )
        (output_root / f"rank{executor.rank}_timing.json").write_text(json.dumps(result, indent=2) + "\n")
    (output_root / f"rank{executor.rank}_paired.json").write_text(json.dumps(records, indent=2) + "\n")
    return records
