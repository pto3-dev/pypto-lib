# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Retired, same-state DIRECT artifact pair; no production hooks changed.

Optional four-active case clones an actual retired prefix into non-overlapping
native pool regions. It is a synthetic bandwidth/correctness diagnostic, not
four independent production prompts or an ITL benchmark. Pools are compared in
full, including KV, GDN and convolution aliases. Runtime lifetimes are sequential.
"""

import json, os, time
from pathlib import Path
import torch
from .byte_compare import equal as byte_equal, STATS as compare_stats
from qwen38_runtime.artifacts import ArtifactBundle
from qwen38_runtime.borrowed_runtime import checked
from qwen38_runtime.runtime_factory import create_target_runtime
from qwen38_runtime.resources import TargetSubmission
from qwen38_runtime.stream_completion import synchronize_stream
from qwen38_runtime.device_plan_layout import PLAN_STRIDE, STORAGE_BYTES


def pool_view(host, device_view):
    offset = (device_view.data_ptr() - device_view.untyped_storage().data_ptr()) // device_view.element_size()
    return torch.empty(0, dtype=device_view.dtype).set_(
        host.untyped_storage(), offset, tuple(device_view.shape), tuple(device_view.stride())
    )


def clone_four(executor, pools):
    """Mutate CPU copies only; source tensors are saved before aliased writes."""
    slots = executor.control_i64[48:72].cpu()
    active = [r for r in range(4) if int(slots[r * 6]) >= 0]
    if not active:
        raise ValueError("No real retired sequence to clone")
    src = active[0]
    ci64 = executor.control_i64.cpu()
    meta = executor.native_metadata.cpu()
    positions = ci64[24 + src * 6 : 24 + (src + 1) * 6].clone()
    ids = ci64[src * 6 : (src + 1) * 6].clone()
    rows_per_request = executor.max_seq
    # Native pool aliases: state regions consume three layers * B4 * MTP6.
    state_regions = 3 * 4 * 6
    first = executor.cache_views["l0_recurrent_state"]
    keys = executor.cache_views["l3_keys"]
    rows_per_state = keys.shape[0] // first.shape[0]
    if keys.shape[0] % first.shape[0] or rows_per_request % rows_per_state:
        raise ValueError("Unrepresentable native pool packing")
    kv_base = state_regions * rows_per_state
    if kv_base + 4 * rows_per_request > keys.shape[0]:
        raise ValueError("Four independent KV areas exceed actual native pool")
    row_bytes = keys[0].numel() * keys.element_size()
    if first[0].numel() * first.element_size() != rows_per_state * row_bytes:
        raise ValueError("Unexpected state/KV alias block size")
    if int(positions.min()) < 0 or int(positions.max()) >= rows_per_request:
        raise ValueError("Position exceeds artifact")
    provenance = []
    for group in range(16):
        first_layer = group * 4
        key = executor.cache_views[f"l{first_layer + 3}_keys"]
        identity = key.untyped_storage().data_ptr()
        name = next(
            n
            for n, t in executor.exported.items()
            if n.startswith("cache_pool") and t.untyped_storage().data_ptr() == identity
        )
        host = pools[name]
        layer_views = {
            n: pool_view(host, v)
            for n, v in executor.cache_views.items()
            if first_layer <= int(n[1 : n.index("_")]) <= first_layer + 3
        }
        kv = layer_views[f"l{first_layer + 3}_keys"]
        vv = layer_views[f"l{first_layer + 3}_values"]
        old = meta[first_layer + 3].clone()
        visible = int(old[48 + src * 6 + 5])
        pages = (visible + 127) // 128
        if not 0 < visible <= rows_per_request:
            raise ValueError("Invalid retired context length")
        source_pages = old[72:].reshape(24, -1)[src * 6, :pages].long()
        source_rows = (source_pages[:, None] * 128 + torch.arange(128)).reshape(-1)
        if bool((source_rows < 0).any()) or int(source_rows.max()) >= kv.shape[0]:
            raise ValueError("Invalid physical source page")
        source_key = kv.index_select(0, source_rows)
        source_value = vv.index_select(0, source_rows)
        saved = []
        for local in range(3):
            layer = first_layer + local
            st = int(meta[layer, src])
            conv = int(meta[layer, 28 + src])
            accepted = int(meta[layer, 48 + src])
            if min(st, conv) < 0 or not 1 <= accepted <= 6:
                raise ValueError("Invalid source GDN state")
            saved.append(
                (
                    layer_views[f"l{layer}_recurrent_state"][st].clone(),
                    layer_views[f"l{layer}_conv_state"][conv].clone(),
                    accepted,
                )
            )
        for local, (state, conv, accepted) in enumerate(saved):
            layer = first_layer + local
            s = layer_views[f"l{layer}_recurrent_state"]
            c = layer_views[f"l{layer}_conv_state"]
            if s.shape[0] < state_regions:
                raise ValueError("Insufficient actual state capacity")
            starts = local * 24 + torch.arange(4) * 6
            for req, start in enumerate(starts.tolist()):
                s[start : start + 6].copy_(state.expand(6, *state.shape))
                c[start : start + 6].copy_(conv.expand(6, *conv.shape))
            row = meta[layer]
            row[:48].fill_(-1)
            row[:4].copy_(starts)
            row[4:24].copy_((starts[:, None] + torch.arange(1, 6)[None, :]).transpose(0, 1).reshape(-1))
            row[24:28].copy_(starts)
            row[28:32].copy_(starts)
            row[48:52].fill_(accepted)
        for req in range(4):
            dest = kv_base + req * rows_per_request
            kv[dest : dest + len(source_rows)].copy_(source_key)
            vv[dest : dest + len(source_rows)].copy_(source_value)
        row = meta[first_layer + 3]
        row[48:72].copy_((positions + 1).repeat(4))
        tables = row[72:].reshape(24, -1)
        for req in range(4):
            table = torch.arange(tables.shape[1]) + kv_base // 128 + req * (rows_per_request // 128)
            tables[req * 6 : (req + 1) * 6].copy_(table.expand(6, -1))
        provenance.append(
            dict(
                group=group,
                source_request=src,
                source_visible=visible,
                pool=name,
                state_regions=[0, state_regions],
                kv_row_begin=kv_base,
                kv_row_end=kv_base + 4 * rows_per_request,
                rows_per_state=rows_per_state,
            )
        )
    for req in range(4):
        ci64[req * 6 : (req + 1) * 6].copy_((ids + req * 13) % 248320)
        ci64[24 + req * 6 : 24 + (req + 1) * 6].copy_(positions)
        ci64[48 + req * 6 : 48 + (req + 1) * 6].copy_(positions + kv_base + req * rows_per_request)
    executor.control_i64.copy_(ci64.to(executor.hidden.device))
    executor.native_metadata.copy_(meta.to(executor.hidden.device))
    executor.control_i32[29:77].copy_(executor.native_metadata[0, :48])
    n = 24 * executor.table_cols
    executor.control_i32[77 : 77 + n].copy_(executor.native_metadata[3, 72:])
    executor.control_i32[77 + n : 101 + n].copy_(executor.native_metadata[3, 48:72])
    assert executor.control_i32[9:13].cpu().tolist() == [6, 12, 18, 24], (
        "B4 clone requires all24 current valid rows"
    )
    return dict(
        kind="synthetic_four_independent_cache_regions_from_one_real_retired_prefix",
        requests=4,
        target_rows=24,
        source_request=src,
        input_token_ids=ci64[:24].tolist(),
        positions=ci64[24:48].tolist(),
        groups=provenance,
        production_representative=False,
    )


@torch.inference_mode()
def validate(executor):
    if not executor.settings.get("terminal_profile") or not executor.settings.get("kernel_pair_build"):
        raise RuntimeError("Explicit diagnostic pair required")
    if executor.policy.execution.value != "direct":
        raise ValueError("This experiment compares DIRECT with DIRECT")
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.npu.stream(executor.capture_stream):
            return _run(executor)
    finally:
        torch.set_num_threads(old_threads)


def _run(executor):
    folder = Path(executor.settings["session_dir"]).parent / "kernel_pair" / f"rank{executor.rank}"
    folder.mkdir(parents=True, exist_ok=False)
    stream = torch.npu.current_stream()
    synchronize_stream(stream.npu_stream)
    ring_heap_mib = int(executor.settings.get("diagnostic_ring_heap_mib", 32))
    if not 16 <= ring_heap_mib <= 256:
        raise ValueError("Diagnostic heap must be16..256MiB")
    ring_heap_bytes = ring_heap_mib * 1024 * 1024
    (folder / "heap_config.json").write_text(
        json.dumps(
            dict(
                ring_heap_bytes=ring_heap_bytes,
                source="existing diagnostic_ring_heap_mib CLI setting",
                applies_to="all reference/candidate runtime instances",
            )
        )
        + "\n"
    )
    stage_begin = time.monotonic()

    def stage(label):
        row = dict(stage=label, rank=executor.rank, elapsed_s=time.monotonic() - stage_begin)
        (folder / "stage.json").write_text(json.dumps(row) + "\n")
        with (folder / "stages.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")

    stage("retire_original_runtime")
    bindings = tuple(executor.borrowed_runtime.retained)
    contexts = [executor.borrowed_comm.contexts[n] for n in ("logit_data", "logit_signal", "data", "signal")]
    executor.borrowed_runtime.close()
    executor.borrowed_runtime = None
    executor.bridge = None
    builds = {
        "reference": Path(executor.settings["build_root"]),
        "candidate": Path(executor.settings["kernel_pair_build"]),
    }
    extras = executor.settings.get("kernel_pair_extra_builds", {})
    if set(extras) & set(builds):
        raise ValueError("Duplicate pair labels")
    builds.update({label: Path(root) for label, root in extras.items()})
    if len(builds) > 4:
        raise ValueError("At most four independent candidates per bounded gate")
    bundles = {label: ArtifactBundle.load(root, executor.rank) for label, root in builds.items()}
    if any(
        b.params != bundles["reference"].params or b.geometry != bundles["reference"].geometry
        for b in bundles.values()
    ):
        raise ValueError("Kernel pair ABI/geometry differs")
    stage("snapshot_pools")
    pools = {n: t.cpu() for n, t in executor.exported.items() if n.startswith("cache_pool")}
    case = (
        clone_four(executor, pools)
        if executor.settings.get("clone_four_active")
        else dict(
            kind="actual_retired_state",
            production_representative=False,
            requests=sum(int(v) >= 0 for v in executor.control_i64[48:72:6].cpu()),
        )
    )
    (folder / "case.json").write_text(json.dumps(case, indent=2) + "\n")
    excluded = {"control_i32", "logit_data", "logit_signal", "data", "signal", "device_program_storage"}
    scratch = {
        p["name"]: executor.exported[p["name"]].cpu()
        for p in bundles["reference"].params
        if p["direction"] in ("InOut", "Out")
        and p["name"] not in excluded
        and not p["name"].startswith("cache_pool")
    }
    plan_hosts = {}
    for label, bundle in bundles.items():
        host = torch.zeros(STORAGE_BYTES, dtype=torch.uint8)
        for i, plan in enumerate(bundle.plans):
            host[i * PLAN_STRIDE : i * PLAN_STRIDE + len(plan)] = torch.frombuffer(
                bytearray(plan), dtype=torch.uint8
            )
        plan_hosts[label] = host

    def barrier():
        if executor.borrowed_comm.lib.comm_barrier(executor.borrowed_comm.handle):
            raise RuntimeError("Diagnostic communication barrier failed")

    def restore(label):
        synchronize_stream(stream.npu_stream)
        barrier()
        for name, host in pools.items():
            executor.exported[name].copy_(host)
        for name, host in scratch.items():
            executor.exported[name].copy_(host)
        executor.exported["device_program_storage"].copy_(plan_hosts[label])
        executor.exported["signal"].zero_()
        executor.exported["logit_signal"].zero_()
        executor.control_i32[1].zero_()
        synchronize_stream(stream.npu_stream)
        barrier()

    def outputs():
        d = dict(hidden=executor.hidden.cpu(), logits=executor.logits.cpu())
        d.update({n: executor.exported[n].cpu() for n in pools})
        return d

    def collect_profiles():
        labels = list(bundles)
        for label in labels:
            runtime = create_target_runtime(
                bundles[label],
                executor.policy.runtime_dir,
                int(executor.hidden.device.index),
                aicpu_threads=0,
                ring_task_window=4096,
                ring_heap=ring_heap_bytes,
                ring_dep_pool=16384,
            )
            try:
                stage(label + "_profile")
                runtime.bind(bindings, contexts)
                restore(label)
                dest = folder / "profile" / label
                dest.mkdir(parents=True)
                runtime.config.enable_scope_stats = 1
                runtime.config.enable_chip_swimlane = 4
                runtime.config.enable_dep_gen = 1
                runtime.config.enable_pmu = int(executor.settings.get("kernel_pair_profile_pmu", 2))
                runtime.config.output_prefix = str(dest).encode()
                if executor.settings.get("profile_internal_attention"):
                    from qwen36.tools.device_trace import instrument_plan

                    host = plan_hosts[label].clone()
                    for i, plan in enumerate(bundles[label].plans):
                        traced = instrument_plan(plan)
                        host[i * PLAN_STRIDE : i * PLAN_STRIDE + len(traced)] = torch.frombuffer(
                            bytearray(traced), dtype=torch.uint8
                        )
                    executor.exported["device_program_storage"].copy_(host)
                executor.control_i32[1].fill_(1)
                synchronize_stream(stream.npu_stream)
                runtime.prepare()
                barrier()
                checked(
                    runtime.lib.simpler_launch_run_on_stream(
                        runtime.ctx, runtime.run, int(stream.npu_stream), None
                    ),
                    "diagnostic profile launch",
                )
                synchronize_stream(stream.npu_stream)
                checked(runtime.lib.simpler_wait_run(runtime.ctx, runtime.run), "diagnostic profile wait")
                checked(
                    runtime.lib.simpler_finalize_run(runtime.ctx, runtime.run), "diagnostic profile finalize"
                )
                runtime.prepared = False
                if executor.settings.get("profile_internal_attention"):
                    from qwen36.tools.device_trace import export_terminal

                    export_terminal(executor, builds[label] / f"next_levels/rank{executor.rank}", dest)
            finally:
                synchronize_stream(stream.npu_stream)
                runtime.close()

    if executor.settings.get("kernel_pair_profile_only", False):
        collect_profiles()
        executor.borrowed_comm.close()
        executor.closed = True
        result = dict(
            status="passed",
            numerical_gate_run=False,
            profile_pmu=int(executor.settings.get("kernel_pair_profile_pmu", 2)),
            rank=executor.rank,
            artifact_sha256={n: b.hashes["callable.image"] for n, b in bundles.items()},
            scope="Profiling-only reuse of previously numerically checked images; no performance or numerical gate inferred from completion.",
        )
        (folder / "host_comparison_stats.json").write_text(json.dumps(compare_stats, indent=2) + "\n")
        (folder / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        stage("passed")
        return result
    stress_steps = int(executor.settings.get("pair_stress_steps", 0))
    if stress_steps:
        stage("fixed_state_stress_initialize")
        runtime = create_target_runtime(
            bundles["reference"],
            executor.policy.runtime_dir,
            int(executor.hidden.device.index),
            aicpu_threads=0,
            ring_task_window=4096,
            ring_heap=ring_heap_bytes,
            ring_dep_pool=16384,
        )
        try:
            runtime.bind(bindings, contexts)
            runtime.prepare_execution("direct")
            submit = TargetSubmission(runtime, executor.control_i32[1:2])
            restore("reference")
            start = time.monotonic()
            for index in range(stress_steps):
                submit.enqueue()
                if (index + 1) % 32 == 0 or index + 1 == stress_steps:
                    synchronize_stream(stream.npu_stream)
                    (folder / "stress.json").write_text(
                        json.dumps(
                            dict(
                                status="passed" if index + 1 == stress_steps else "running",
                                steps_completed=index + 1,
                                steps_requested=stress_steps,
                                ring_heap_bytes=ring_heap_bytes,
                                elapsed_seconds=time.monotonic() - start,
                            )
                        )
                        + "\n"
                    )
        finally:
            synchronize_stream(stream.npu_stream)
            runtime.close()
        # Restore original physical pools and metadata before all numerical and
        # timing comparisons. Stress-mutated state never becomes the oracle.
        restore("reference")
    expected = None
    checks = []
    timings = {label: [] for label in builds}
    submits = []
    labels = list(builds)
    for order in [labels, list(reversed(labels)), labels]:
        for label in order:
            stage(label + "_initialize")
            runtime = create_target_runtime(
                bundles[label],
                executor.policy.runtime_dir,
                int(executor.hidden.device.index),
                aicpu_threads=0,
                ring_task_window=4096,
                ring_heap=ring_heap_bytes,
                ring_dep_pool=16384,
            )
            try:
                runtime.bind(bindings, contexts)
                runtime.prepare_execution("direct")
                submit = TargetSubmission(runtime, executor.control_i32[1:2])
                restore(label)
                stage(label + "_warmup")
                submit.enqueue()
                synchronize_stream(stream.npu_stream)
                restore(label)
                stage(label + "_sample")
                begin = torch.npu.Event(enable_timing=True)
                end = torch.npu.Event(enable_timing=True)
                barrier()
                host_begin = time.monotonic_ns()
                begin.record(stream)
                submit.enqueue()
                end.record(stream)
                synchronize_stream(stream.npu_stream)
                timings[label].append(begin.elapsed_time(end))
                submits.append(dict(label=label, host_submit_ns=host_begin))
                actual = outputs()
                if expected is None:
                    expected = actual
                different = [n for n in actual if not byte_equal(actual[n], expected[n])]
                checks.append(dict(label=label, iteration=len(timings[label]) - 1, different=different))
                if different:
                    raise ValueError("Kernel same-state differences " + str(different))
                (folder / "timing_partial.json").write_text(
                    json.dumps(
                        dict(
                            status="same_state_samples",
                            checks=checks,
                            samples_ms=timings,
                            submit_times=submits,
                            complete_physical_cache_pools=len(pools),
                            compared_cache_bytes=sum(t.numel() * t.element_size() for t in pools.values()),
                            case=case,
                        ),
                        indent=2,
                    )
                    + "\n"
                )
            finally:
                synchronize_stream(stream.npu_stream)
                runtime.close()
    # Stateful replay gate: the SAME prepared graph executes two successive
    # epochs without resetting external cache or rebuilding its task metadata.
    sequence_expected = None
    sequence_checks = []
    for label in builds:
        stage(label + "_stateful_replay")
        runtime = create_target_runtime(
            bundles[label],
            executor.policy.runtime_dir,
            int(executor.hidden.device.index),
            aicpu_threads=0,
            ring_task_window=4096,
            ring_heap=ring_heap_bytes,
            ring_dep_pool=16384,
        )
        try:
            runtime.bind(bindings, contexts)
            runtime.prepare_execution("direct")
            submit = TargetSubmission(runtime, executor.control_i32[1:2])
            restore(label)
            barrier()
            expected_steps = [] if sequence_expected is None else sequence_expected
            for replay in range(2):
                submit.enqueue()
                synchronize_stream(stream.npu_stream)
                actual = outputs()
                if sequence_expected is None:
                    expected_steps.append(actual)
                different = [n for n in actual if not byte_equal(actual[n], expected_steps[replay][n])]
                # This model's single reusable event slab is last written by
                # attention group 15. Unused terminal slots remain zero.
                signals = (
                    executor.exported["device_program_storage"][12718080:12761088]
                    .cpu()
                    .view(torch.int64)[::16]
                )
                observed = torch.unique(signals[signals != 0]).tolist()
                expected_epoch = (replay + 1) * 16 + 16
                record = dict(
                    label=label,
                    replay=replay,
                    different=different,
                    expected_epoch=expected_epoch,
                    observed_epochs=observed,
                )
                sequence_checks.append(record)
                (folder / "stateful_replay_checks.json").write_text(
                    json.dumps(sequence_checks, indent=2) + "\n"
                )
                if different or observed != [expected_epoch]:
                    raise ValueError("Stateful graph replay mismatch " + str(record))
            if sequence_expected is None:
                sequence_expected = expected_steps
        finally:
            synchronize_stream(stream.npu_stream)
            runtime.close()
    del sequence_expected, expected_steps, actual
    # One additional correctness-only boundary case, outside every timing.
    # Its active rows keep the same IDs/positions; padding repeats the last
    # real input, but has no KV slot or GDN/conv state transition.
    ragged_checks = []
    ragged_lengths = executor.settings.get("kernel_pair_ragged_lengths")
    if ragged_lengths:
        if len(ragged_lengths) != 4 or any(not 1 <= n <= 6 for n in ragged_lengths):
            raise ValueError("Invalid ragged diagnostic lengths")
        stage("ragged_metadata")
        canonical64 = executor.control_i64.cpu()
        canonical32 = executor.control_i32.cpu()
        canonical_meta = executor.native_metadata.cpu()
        ci64 = canonical64.clone()
        meta = canonical_meta.clone()
        ci32 = canonical32.clone()
        for req, nvalid in enumerate(ragged_lengths):
            begin = req * 6
            for index in range(begin + nvalid, begin + 6):
                ci64[index] = ci64[begin + nvalid - 1]
                ci64[24 + index] = ci64[24 + begin + nvalid - 1]
                ci64[48 + index] = -1
            ci32[9 + req] = begin + nvalid
        for layer in range(3, 64, 4):
            meta[layer, 48:72].copy_((ci64[24:48] + 1).to(torch.int32))
        n = 24 * executor.table_cols
        ci32[29:77].copy_(meta[0, :48])
        ci32[77 : 77 + n].copy_(meta[3, 72:])
        ci32[77 + n : 101 + n].copy_(meta[3, 48:72])
        executor.control_i64.copy_(ci64.to(executor.hidden.device))
        executor.control_i32.copy_(ci32.to(executor.hidden.device))
        executor.native_metadata.copy_(meta.to(executor.hidden.device))
        expected_ragged = None
        try:
            for label in builds:
                stage(label + "_ragged")
                runtime = create_target_runtime(
                    bundles[label],
                    executor.policy.runtime_dir,
                    int(executor.hidden.device.index),
                    aicpu_threads=0,
                    ring_task_window=4096,
                    ring_heap=ring_heap_bytes,
                    ring_dep_pool=16384,
                )
                try:
                    runtime.bind(bindings, contexts)
                    runtime.prepare_execution("direct")
                    submit = TargetSubmission(runtime, executor.control_i32[1:2])
                    restore(label)
                    barrier()
                    submit.enqueue()
                    synchronize_stream(stream.npu_stream)
                    actual = outputs()
                    if expected_ragged is None:
                        expected_ragged = actual
                    different = [
                        name for name in actual if not byte_equal(actual[name], expected_ragged[name])
                    ]
                    ragged_checks.append(
                        dict(
                            label=label,
                            query_lengths=ragged_lengths,
                            different=different,
                            full_physical_pools=len(pools),
                            scope="same padded rows and complete physical cache; correctness only",
                        )
                    )
                    (folder / "ragged_checks.json").write_text(json.dumps(ragged_checks, indent=2) + "\n")
                    if different:
                        raise ValueError("Ragged full-target differences " + str(different))
                finally:
                    synchronize_stream(stream.npu_stream)
                    runtime.close()
        finally:
            executor.control_i64.copy_(canonical64.to(executor.hidden.device))
            executor.control_i32.copy_(canonical32.to(executor.hidden.device))
            executor.native_metadata.copy_(canonical_meta.to(executor.hidden.device))
            synchronize_stream(stream.npu_stream)
            barrier()
    # Profiler is outside all timed samples. Reset the exact same state and keep
    # scheduler/early-dispatch plus PMU records for the complete target callable.
    collect_profiles()
    executor.borrowed_comm.close()
    executor.closed = True
    result = dict(
        status="passed",
        stateful_replay_checks=sequence_checks,
        ragged_checks=ragged_checks,
        rank=executor.rank,
        pid=os.getpid(),
        execution="direct",
        ring_heap_bytes=ring_heap_bytes,
        comparison_tensors=len(expected),
        complete_physical_cache_pools=len(pools),
        compared_cache_bytes=sum(t.numel() * t.element_size() for t in pools.values()),
        checks=checks,
        samples_ms=timings,
        submit_times=submits,
        case=case,
        artifact_sha256={n: b.hashes["callable.image"] for n, b in bundles.items()},
        artifact_hashes_by_variant={n: b.hashes for n, b in bundles.items()},
        scope="Same input and full native pool state; sequential runtime lifetimes. Cloned B4 case is synthetic, not production ITL. One equal untimed warmup per runtime; snapshots, restore, barriers and profiler outside timed samples.",
    )
    (folder / "host_comparison_stats.json").write_text(json.dumps(compare_stats, indent=2) + "\n")
    (folder / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    stage("passed")
    return result
