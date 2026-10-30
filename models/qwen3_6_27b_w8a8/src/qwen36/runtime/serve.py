# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Explicit supported runtime: TP4, native MTP5 drafter, complete PyPTO target."""

import json, sys


def command(model, build, session, port, paired_build=None):
    settings = dict(
        graph=True,
        phase_nz=False,
        mtp=True,
        terminal_profile=True,
        geometry_only=False,
        session_dir=str(session / "runtime"),
        build_root=str(build),
    )
    if paired_build is not None:
        settings["paired_build_root"] = str(paired_build)
    return [
        sys.executable,
        "-m",
        "qwen36.runtime.http_server",
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
        json.dumps(dict(cudagraph_mode="FULL_DECODE_ONLY", cudagraph_capture_sizes=[24])),
        "--speculative-config",
        json.dumps(dict(method="mtp", num_speculative_tokens=5)),
        "--worker-extension-cls",
        "qwen36.runtime.native_nz_worker.NativeNzWorker",
        "--additional-config",
        json.dumps({"qwen38_pto_mpmd": settings}),
        "--num-gpu-blocks-override",
        "192",
    ]


def main():
    """Launch one server after an idle-card check; never launches benchmark requests."""
    import argparse, os, socket
    from pathlib import Path
    from qwen36.tools.idle_guard import wait_for_free_group, wait_for_idle

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--build", type=Path)
    p.add_argument("--session", type=Path, required=True)
    p.add_argument("--backend", choices=["pto", "native"], default="pto")
    p.add_argument(
        "--devices", help="Four idle physical card IDs, comma-separated; default: choose any idle four"
    )
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--idle-timeout", type=int, default=1200)
    a = p.parse_args()
    if a.backend == "pto" and (a.build is None or not (a.build / "build_contract.json").is_file()):
        p.error("PTO requires a linked --build directory")
    if a.session.exists():
        p.error("Use a new --session directory; existing evidence is never overwritten")
    if a.devices:
        cards = list(map(int, a.devices.split(",")))
        if len(cards) != 4 or len(set(cards)) != 4:
            p.error("--devices must contain four distinct physical card IDs")
        wait_for_idle(cards, timeout=a.idle_timeout, stable_seconds=5)
    else:
        cards = wait_for_free_group(timeout=a.idle_timeout)
    with socket.socket() as channel:
        channel.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        channel.bind(("127.0.0.1", a.port))
        port = channel.getsockname()[1]
    a.session.mkdir(parents=True)
    (a.session / "runtime").mkdir()
    cmd = command(a.model, a.build, a.session, port)
    at = cmd.index("--additional-config") + 1
    if a.backend == "native":
        cmd[at] = "{}"
        cmd[cmd.index("--worker-extension-cls") + 1] = (
            "qwen36.runtime.native_baseline_worker.NativeBenchmarkWorker"
        )
    else:
        settings = json.loads(cmd[at])
        settings["qwen38_pto_mpmd"]["terminal_profile"] = False
        cmd[at] = json.dumps(settings)
    env = os.environ.copy()
    env.update(
        ASCEND_RT_VISIBLE_DEVICES=",".join(map(str, cards)),
        VLLM_ASCEND_ENABLE_NZ="2",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTORCH_NPU_ALLOC_CONF="expandable_segments:False",
        ASCEND_PROCESS_LOG_PATH=str(a.session / "ascend"),
    )
    for key in list(env):
        if key.startswith(("VLLM_ASCEND_PYPTO_QWEN", "PYPTO_MTP_RUNTIME_RESERVE")):
            env.pop(key, None)
    (a.session / "launch.json").write_text(
        json.dumps(
            dict(devices=cards, backend=a.backend, url=f"http://127.0.0.1:{port}", command=cmd), indent=2
        )
        + "\n"
    )
    print(
        "QWEN36_SERVER",
        json.dumps(dict(devices=cards, url=f"http://127.0.0.1:{port}", session=str(a.session))),
        flush=True,
    )
    os.execvpe(cmd[0], cmd, env)


if __name__ == "__main__":
    main()
