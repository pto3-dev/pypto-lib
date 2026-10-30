# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Build a reproducible vLLM command with independent backend/execution policy."""

from pathlib import Path
import argparse, json, sys


def command(
    model,
    build,
    session,
    port,
    *,
    backend="pypto",
    execution="direct",
    weight_strategy="device_preloaded_nd_to_nz",
    kv_blocks=None,
    unsupported_decode="error",
    model_target="Qwen3.6-27B",
    full_sync="required",
):
    settings = dict(
        mtp_async_graph="merged_subbatches",
        backend=backend,
        execution=execution,
        graph=execution == "full",
        phase_nz=False,
        mtp=True,
        terminal_profile=False,
        session_dir=str(session / "runtime"),
        build_root=str(build),
        unsupported_decode=unsupported_decode,
        model_target=model_target,
        full_sync=full_sync,
        checkpoint_provenance=str(model),
        weight_strategy=weight_strategy,
    )
    cmd = [
        sys.executable,
        "-m",
        "qwen38_runtime.http_server",
        "--stats-dir",
        str(session),
        "serve",
        str(model),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--served-model-name",
        "qwen36-mixed",
        "--dtype",
        "bfloat16",
        "--quantization",
        "ascend",
        "--tensor-parallel-size",
        "4",
        "--pipeline-parallel-size",
        "1",
        "--max-model-len",
        "43008",
        "--max-num-seqs",
        "4",
        "--max-num-batched-tokens",
        "2048",
        "--async-scheduling",
        "--enable-chunked-prefill",
        "--gpu-memory-utilization",
        "0.96",
        "--no-enable-prefix-caching",
        "--language-model-only",
        "--seed",
        "1024",
        "--compilation-config",
        json.dumps(dict(cudagraph_mode="FULL_DECODE_ONLY", mode=3, cudagraph_capture_sizes=[6, 12, 18, 24])),
        "--speculative-config",
        json.dumps(dict(method="mtp", num_speculative_tokens=5)),
        "--worker-extension-cls",
        "qwen38_runtime.native_nz_worker.NativeNzWorker",
        "--additional-config",
        json.dumps({"qwen38_pto_mpmd": settings}),
    ]
    if kv_blocks is not None:
        cmd += ["--num-gpu-blocks-override", str(kv_blocks)]
    return cmd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--build-root", type=Path, required=True)
    p.add_argument("--session-dir", type=Path, required=True)
    p.add_argument("--port", type=int, default=18638)
    p.add_argument("--backend", choices=["native", "pypto"], default="pypto")
    p.add_argument("--execution", choices=["direct", "full"], default="direct")
    p.add_argument(
        "--weight-strategy",
        choices=[
            "native_nz",
            "batched_nd_to_nz",
            "direct_cpu_nz",
            "preloaded_nd_to_nz",
            "device_preloaded_nd_to_nz",
        ],
        default="device_preloaded_nd_to_nz",
    )
    p.add_argument("--kv-blocks", type=int)
    p.add_argument("--unsupported-decode", choices=["error", "native"], default="error")
    p.add_argument("--model-target", default="Qwen3.6-27B")
    p.add_argument("--full-sync", choices=["required", "stream"], default="required")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    cmd = command(
        a.model.resolve(),
        a.build_root.resolve(),
        a.session_dir.resolve(),
        a.port,
        backend=a.backend,
        execution=a.execution,
        weight_strategy=a.weight_strategy,
        kv_blocks=a.kv_blocks,
        unsupported_decode=a.unsupported_decode,
        model_target=a.model_target,
        full_sync=a.full_sync,
    )
    if a.dry_run:
        print(json.dumps(cmd, indent=2))
        return
    import os

    a.session_dir.mkdir(parents=True, exist_ok=False)
    (a.session_dir / "runtime").mkdir()
    (a.session_dir / "command.json").write_text(json.dumps(cmd, indent=2) + "\n")
    os.environ["VLLM_ASCEND_ENABLE_NZ"] = "2"
    os.execv(sys.executable, cmd)


if __name__ == "__main__":
    main()
