# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Formal performance candidate: default HCCL, compact inputs, validated fusion.

HCCL determinism is diagnostic-only and is deliberately not enabled here.
"""

import argparse, hashlib, json, os, shutil, sys
from pathlib import Path

P = Path(__file__).resolve().parents[1]
W = P.parent
K = W / "kernel_tuning_20261027"
sys.path.insert(0, str(K / "tools"))
sys.path.insert(0, str(P / "tools"))
import recovery_stage as runner
from qwen36.tools.idle_guard import wait_for_free_group, wait_for_idle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="fast_fused_compact_10k_01")
    parser.add_argument("--devices")
    parser.add_argument("--idle-timeout", type=int, default=180)
    parser.add_argument("--skip-terminal-validation", action="store_true")
    options = parser.parse_args()
    if Path(options.name).name != options.name:
        raise ValueError("Invalid evidence name")
    build = W / "spmd_fusion_20261029/build/target_gate_quant_fused"
    from qwen38_runtime.artifacts import ArtifactBundle

    base = W / "attention_native_fixture_20261029/build/target_key_value"
    for rank in range(4):
        a, b = ArtifactBundle.load(base, rank), ArtifactBundle.load(build, rank)
        if a.params != b.params or a.geometry != b.geometry:
            raise ValueError("Target ABI/cache mismatch")
    out = P / "evidence" / options.name
    out.mkdir(exist_ok=False)
    snapshot = out / "source_snapshot"
    snapshot.mkdir()
    for name in ["src", "integration", "bootstrap", "reference", "lib", "vendor_overlay"]:
        shutil.copytree(P / name, snapshot / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(
        runner.W / "src", snapshot / "model_src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    shutil.copy2(K / "tools/response_audit.py", snapshot / "src/qwen38_runtime/request_audit.py")
    shutil.copy2(__file__, snapshot / "run_used.py")
    shutil.copy2(Path(runner.__file__), snapshot / "recovery_stage_used.py")
    shutil.copy2(P / "env.sh", snapshot / "env_used.sh")
    (out / "source_manifest.json").write_text(
        json.dumps(
            {
                str(f.relative_to(snapshot)): hashlib.sha256(f.read_bytes()).hexdigest()
                for f in snapshot.rglob("*")
                if f.is_file()
            },
            indent=2,
        )
        + "\n"
    )
    runner.D, runner.F = P, out
    runner.DATA = W / "prompt_suite_20261028/engineering_context_seed1024_32k_10k_v1"
    shutil.copy2(runner.DATA / "manifest.json", out / "dataset_manifest.json")
    original = runner.command

    def command(*args, **kwargs):
        cmd = original(*args, **kwargs)
        i = cmd.index("--additional-config") + 1
        cfg = json.loads(cmd[i])
        cfg["qwen38_pto_mpmd"].update(
            mtp_async_graph="merged_subbatches",
            mtp_graph_validate_calls=0,
            diagnostic_ring_heap_mib=64,
            metadata_preprocess="adaptive_graph",
            gdn_metadata="raw_canonical",
            spec_metadata_cache="uniform_v1",
            spec_metadata_validate_calls=0,
            compact_inputs="both",
            compact_validate_calls=0,
            compact_record_calls=False,
            terminal_profile=not options.skip_terminal_validation,
            kernel_pair_build=str(base),
            kernel_pair_reuse_steps=8,
            clone_four_active=False,
            kernel_pair_profile_pmu=0,
        )
        cfg["qwen38_pto_mpmd"].pop("native_tp_reduce", None)
        cmd[i] = json.dumps(cfg)
        i = cmd.index("--compilation-config") + 1
        cfg = json.loads(cmd[i])
        cfg.update(mode=3, cudagraph_capture_sizes=[6, 12, 18, 24])
        cmd[i] = json.dumps(cfg)
        i = cmd.index("--speculative-config") + 1
        cfg = json.loads(cmd[i])
        cfg["enforce_eager"] = False
        cmd[i] = json.dumps(cfg)
        cmd[cmd.index("--worker-extension-cls") + 1] = (
            "qwen38_runtime.fast_recovery_worker.FastRecoveryWorker"
        )
        return cmd

    runner.command = command
    # Prefer the historical1/2/3/4 group when available; protect external jobs.
    if options.devices:
        cards = list(map(int, options.devices.split(",")))
        if len(cards) != 4 or len(set(cards)) != 4 or any(c not in range(8) for c in cards):
            raise ValueError("Invalid TP4 devices")
        wait_for_idle(cards, timeout=options.idle_timeout, stable_seconds=5)
    else:
        cards = wait_for_free_group(candidates=tuple(range(8)), timeout=options.idle_timeout)
    status = dict(
        status="running",
        pid=os.getpid(),
        devices=cards,
        formal=True,
        inputs=32768,
        outputs=10240,
        requests=4,
        mtp=5,
        request_rate="inf",
        profile=False,
        target=str(build),
        native_hccl="default unchanged",
        candidate_changes=[
            "compact device input preparation",
            "validated DSL attention fusion and AIV waiter quota",
        ],
        scope="Performance recovery candidate; does not isolate independent fusion benefit",
    )

    def save():
        (out / "status.json").write_text(json.dumps(status, indent=2) + "\n")

    save()
    try:
        runner.stage(
            "pto",
            cards,
            snapshot,
            backend="pypto",
            build=build,
            output_len=10240,
            progress_timeout=180,
            terminal_validate=not options.skip_terminal_validation,
        )
        status.update(status="complete")
    except BaseException as e:
        status.update(status="failed", error=repr(e))
        raise
    finally:
        save()
    print("FAST_FUSED_FORMAL_COMPLETE", out, flush=True)


if __name__ == "__main__":
    main()
