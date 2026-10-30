# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared target resources, independent of direct/FULL execution policy."""

import json, os, time
from pathlib import Path
import torch
from qwen38_runtime.borrowed_runtime import DeviceTensorBinding
from .runtime_factory import create_target_runtime
from qwen38_runtime.borrowed_comm import BorrowedCommWindows
from qwen38_runtime.native_nz_weights import bank_native_weights
from qwen38_runtime.model_layout import WeightBankSpec, H, KG, VG


class TargetSubmission:
    def __init__(self, runtime, epoch):
        self.runtime = runtime
        self.epoch = epoch

    def enqueue(self):
        self.epoch.add_(1)
        self.runtime.enqueue(torch.npu.current_stream().npu_stream)


def prepare_weights(executor):
    if executor.shared_weights is not None:
        return
    start = time.perf_counter()
    spec = WeightBankSpec()
    executor.shared_weights, executor.nz_bank_owners, executor.shared_weight_info = bank_native_weights(
        executor.causal,
        spec,
        strategy=executor.policy.weight_strategy,
        validate_weight_bytes=executor.settings.get("validate_weight_bytes", False),
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


def prepare_resources(executor, context, device):
    if executor.bridge is not None:
        return
    if executor.closed:
        raise RuntimeError("Target resources are closed")
    if torch.npu.is_current_stream_capturing():
        raise RuntimeError("Resources must be prepared outside stream capture")
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
    from .artifacts import ArtifactBundle

    bundle = ArtifactBundle.load(build_root, executor.rank)
    executor.artifact_hashes = bundle.hashes
    build_contract = bundle.contract
    ends_contract = build_contract.get("ragged_valid_ends")
    if executor.ragged_enabled and ends_contract != "starts5_8_v1":
        raise ValueError("Ragged runtime requires matched valid-end kernels")
    executor.communication_max_steps = int(build_contract.get("communication_max_steps", 67108864))
    if executor.communication_max_steps <= 0:
        raise ValueError("Invalid communication epoch budget")
    if build_contract.get("compile_smoke"):
        raise ValueError("Compile-smoke geometry cannot be used for inference")
    # Legacy artifact labels are provenance, not the configured model target.
    bundle.validate_geometry(dimensions)
    expected = bundle.geometry
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
    if ends_contract == "starts5_8_v1":
        executor.canonical_ends = torch.tensor([6, 12, 18, 24], dtype=torch.int32, device=device)
        executor.control_i32[9:13].copy_(executor.canonical_ends)
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
    metadata = {"params": bundle.params}
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
            from qwen38_runtime.device_plan_layout import MAX_CLUSTERS, PLAN_STRIDE, STORAGE_BYTES

            host = torch.zeros(STORAGE_BYTES, dtype=torch.uint8)
            for grid in range(1, MAX_CLUSTERS + 1):
                image = bundle.plans[grid - 1]
                host[(grid - 1) * PLAN_STRIDE : (grid - 1) * PLAN_STRIDE + len(image)] = torch.frombuffer(
                    bytearray(image), dtype=torch.uint8
                )
            value = host.to(device)
        else:
            value = torch.zeros(shape, dtype=dtype, device=device)
        executor.exported[name] = value
    executor.capture_stream = torch.npu.Stream(device=device)
    torch.npu.synchronize()
    ring_heap_mib = int(executor.settings.get("diagnostic_ring_heap_mib", 32))
    if not 16 <= ring_heap_mib <= 256:
        raise ValueError("Runtime heap must be16..256MiB")
    ring_heap_bytes = ring_heap_mib * 1024 * 1024
    runtime = create_target_runtime(
        bundle,
        executor.policy.runtime_dir,
        int(device.index),
        aicpu_threads=0,
        ring_task_window=4096,
        ring_heap=ring_heap_bytes,
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
    runtime.prepare_execution(executor.policy.execution.value)
    executor.bridge = TargetSubmission(runtime, executor.control_i32[1:2])
    executor.replay_window = 1
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
        execution=executor.policy.execution.value,
        outer_request_done_bridge=False,
        cache_contents_copied=False,
        argument_count=len(tensors),
        runtime_config=dict(
            aicpu_threads="architecture default",
            ring_task_window=4096,
            ring_heap=ring_heap_bytes,
            ring_dep_pool=16384,
        ),
        cache_dimensions=dimensions,
    )
    (Path(executor.settings["session_dir"]) / f"rank{executor.rank}_prepared.json").write_text(
        json.dumps(info, indent=2) + "\n"
    )
    print("NATIVE_NZ_TARGET_PREPARED", executor.rank, os.getpid(), flush=True)


def close(executor):
    runtime = getattr(executor, "borrowed_runtime", None)
    if runtime is None:
        return
    # FULL replay may run on a different caller stream from capture.
    # Direct execution is drained by the bounded C++ finalizer itself.
    caller = getattr(executor, "last_replay_stream", None)
    if caller is not None:
        from .stream_completion import synchronize_stream

        synchronize_stream(caller, executor.settings.get("stream_timeout_ms", 30000))
    for reference in executor.caller_graphs:
        graph = reference()
        if graph is not None:
            graph.reset()
    executor.caller_graphs.clear()
    executor.graph_entries.clear()
    executor.bridge = None
    runtime.close()
    executor.borrowed_comm.close()
    executor.borrowed_runtime = None
    print("NATIVE_NZ_TARGET_CLOSED", executor.rank, os.getpid(), flush=True)
