# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""One native/PTO frozen-input10K pair on the same initially idle four cards.

No parameter sweep, no profiler inside timed requests, no implicit retry, and no
reuse of a completed result directory. A short device gate must already pass.
"""

import argparse, hashlib, json, os, shutil, signal, socket, subprocess, sys, time, urllib.request
from pathlib import Path
import psutil
from qwen36.runtime.serve import command
from qwen36.tools.idle_guard import wait_for_free_group, wait_for_idle

W = Path(__file__).resolve().parents[3]
ROOT = W.parents[1]


def run_stage(folder, backend, request, build, cards, snapshot):
    folder.mkdir(exist_ok=False)
    (folder / "runtime").mkdir()
    owned = {}
    server = client = None
    status = dict(
        status="waiting_same_cards",
        backend=backend,
        devices=cards,
        request=str(request),
        build=str(build) if backend == "pto" else None,
    )

    def save():
        (folder / "status.json").write_text(json.dumps(status, indent=2) + "\n")

    def harvest():
        if server:
            try:
                for p in psutil.Process(server.pid).children(recursive=True):
                    owned[p.pid] = p
            except psutil.NoSuchProcess:
                pass

    save()
    try:
        wait_for_idle(cards, timeout=1200, stable_seconds=5)
        env = os.environ.copy()
        env.update(
            ASCEND_RT_VISIBLE_DEVICES=",".join(map(str, cards)),
            VLLM_ASCEND_ENABLE_NZ="2",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            PYTORCH_NPU_ALLOC_CONF="expandable_segments:False",
            ASCEND_PROCESS_LOG_PATH=str(folder / "ascend"),
            QWEN36_DIRECT_ROOT=str(snapshot),
            QWEN36_CPU_MTP_TRACE=str(folder / "cpu_verifier_trace"),
            QWEN36_BENCH_REQUEST_AUDIT=str(folder / "request_audit"),
        )
        paths = env["PYTHONPATH"].split(":")
        env["PYTHONPATH"] = ":".join(
            str(snapshot / p.relative_to(W)) if (p := Path(value)).is_relative_to(W) else value
            for value in paths
        )
        for key in list(env):
            if key.startswith(("VLLM_ASCEND_PYPTO_QWEN", "PYPTO_MTP_RUNTIME_RESERVE")):
                env.pop(key, None)
        # Independent ephemeral ports avoid TIME_WAIT on the prior server.
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        url = f"http://127.0.0.1:{port}"
        cmd = command(ROOT / "checkpoints/public-inner-surrogate", build, folder, port)
        at = cmd.index("--additional-config") + 1
        settings = json.loads(cmd[at])
        settings["qwen38_pto_mpmd"]["terminal_profile"] = False
        if backend == "native":
            settings = {}
            cmd[cmd.index("--worker-extension-cls") + 1] = (
                "qwen36.runtime.native_baseline_worker.NativeBenchmarkWorker"
            )
        cmd[at] = json.dumps(settings)
        status.update(status="starting", server_command=cmd)
        save()
        with (folder / "server.log").open("w") as log:
            server = subprocess.Popen(
                cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
        status["server_pid"] = server.pid
        save()
        deadline = time.monotonic() + 1200
        while time.monotonic() < deadline:
            harvest()
            if server.poll() is not None:
                raise RuntimeError("Server exited during initialization")
            t = (folder / "server.log").read_text(errors="replace")
            if any(
                x in t
                for x in [
                    "WorkerProc failed to start.",
                    "EngineCore failed to start.",
                    "Free memory on device (",
                ]
            ):
                raise RuntimeError("Fatal initialization failure; see server.log")
            try:
                with urllib.request.urlopen(url + "/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except (OSError, TimeoutError):
                pass
            time.sleep(2)
        else:
            raise TimeoutError("Initialization1200s deadline")
        with urllib.request.urlopen(url + "/metrics", timeout=10) as r:
            (folder / "metrics_before.prom").write_bytes(r.read())
        status["status"] = "generation_10k"
        save()
        with (folder / "client.log").open("w") as log:
            client = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "qwen36.tools.benchmark_stream",
                    "--request",
                    str(request),
                    "--url",
                    url + "/v1/completions",
                    "--output",
                    str(folder / "measurement"),
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        deadline = time.monotonic() + 1800
        while client.poll() is None:
            harvest()
            if server.poll() is not None:
                raise RuntimeError("Server exited during generation")
            if time.monotonic() > deadline:
                raise TimeoutError("10K client1800s deadline")
            time.sleep(2)
        if client.returncode:
            raise RuntimeError("Generation client failed; raw events retained")
        with urllib.request.urlopen(url + "/metrics", timeout=10) as r:
            (folder / "metrics_after.prom").write_bytes(r.read())
        status["status"] = "collecting"
        save()
        control = json.loads((folder / "lifecycle_control.json").read_text())
        with socket.socket(socket.AF_UNIX) as channel:
            channel.settimeout(300)
            channel.connect(control["socket"])
            channel.sendall(b'{"command":"collect-and-close"}\n')
            reply = json.loads(channel.makefile("rb").readline())
        if not reply.get("ok"):
            raise RuntimeError(reply)
        summary = json.loads((folder / "measurement/summary.json").read_text())
        status.update(
            status="complete",
            average_tpot_ms=summary["average_tpot_ms"],
            ttft_ms=summary["ttft_ms"],
            lifecycle=reply,
        )
        save()
    except BaseException as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise
    finally:
        harvest()
        for p in [client, server]:
            if p is not None and p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    p.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait(timeout=5)
        for p in owned.values():
            try:
                p.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(list(owned.values()), timeout=5)
        for p in alive:
            try:
                p.kill()
            except psutil.NoSuchProcess:
                pass
        status["owned_descendant_pids"] = sorted(owned)
        save()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--candidate-build", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--idle-timeout", type=int, default=1200)
    parser.add_argument("--resume", action="store_true")
    a = parser.parse_args()
    gate = json.loads((a.gate / "status.json").read_text())
    timing = json.loads((a.gate / "target_timing_comparison.json").read_text())
    if (
        gate["status"] != "passed"
        or gate["build"] != str(a.candidate_build)
        or not timing["all_replays_bitwise_equal"]
        or timing["improvement_percent"] <= 0
    ):
        raise ValueError("Passed same-state/device timing gate required")
    template = json.loads(a.request.read_text())
    if (
        template["max_tokens"] != 10240
        or len(template["prompt"]) != 4
        or any(len(p) != 32768 for p in template["prompt"])
    ):
        raise ValueError("Exactly4x32768 input/10240 output required")
    if Path(a.name).name != a.name:
        raise ValueError("Name must be single component")
    folder = W / "evidence" / a.name
    snapshot = folder / "source_snapshot"
    frozen = folder / "frozen_request.json"
    if a.resume:
        status = json.loads((folder / "status.json").read_text())
        if frozen.read_bytes() != a.request.read_bytes() or status["gate"] != str(a.gate):
            raise ValueError("Resume must preserve exact frozen request and device gate")
        status.update(status="resuming", pid=os.getpid())
    else:
        folder.mkdir(exist_ok=False)
        snapshot.mkdir()
        for name in ["src", "integration", "bootstrap"]:
            shutil.copytree(W / name, snapshot / name, ignore=shutil.ignore_patterns("__pycache__"))
        (folder / "source_manifest.json").write_text(
            json.dumps(
                {
                    str(p.relative_to(snapshot)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in snapshot.rglob("*")
                    if p.is_file()
                },
                indent=2,
            )
            + "\n"
        )
        shutil.copy2(a.request, frozen)
        status = dict(
            status="waiting_four_idle_cards",
            pid=os.getpid(),
            gate=str(a.gate),
            request_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),
            source_snapshot=str(snapshot),
            native_mtp_drafter=True,
        )

    def save():
        (folder / "status.json").write_text(json.dumps(status, indent=2) + "\n")

    save()
    try:
        cards = status.get("devices") if a.resume else None
        if cards is None:
            cards = wait_for_free_group(timeout=a.idle_timeout)
        status["devices"] = cards
        for backend in ["native", "pto"]:
            stage = folder / backend
            if a.resume and stage.exists():
                previous = json.loads((stage / "status.json").read_text())
                if previous["status"] == "complete":
                    continue
                if (stage / "measurement").exists():
                    raise ValueError(
                        "Existing measured stage requires explicit recovery; refuse repeating it"
                    )
                stage.rename(folder / f"{backend}_unmeasured_startup_{int(time.time())}")
            status["status"] = "running_" + backend
            save()
            run_stage(stage, backend, frozen, a.candidate_build, cards, snapshot)
        if "error" in status:
            status.setdefault("recovered_startup_errors", []).append(status.pop("error"))
        status["status"] = "complete"
        save()
        print("FORMAL_PAIR_DONE", str(folder), flush=True)
    except BaseException as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise


if __name__ == "__main__":
    main()
