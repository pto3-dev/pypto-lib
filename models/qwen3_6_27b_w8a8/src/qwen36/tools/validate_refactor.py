# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""One short same-state refactor check. Never starts a second server or a 10K run."""

import argparse, hashlib, json, os, shutil, signal, socket, subprocess, sys, time, urllib.request
from pathlib import Path
import psutil
from qwen36.runtime.serve import command
from qwen36.tools.idle_guard import wait_for_free_group

W = Path(__file__).resolve().parents[3]
ROOT = W.parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--idle-timeout", type=int, default=1200)
    p.add_argument("--build", type=Path, default=W / "build/behavior_reference")
    p.add_argument("--terminal-graph-timing", action="store_true")
    p.add_argument("--host-trace", action="store_true")
    p.add_argument("--device-trace", action="store_true")
    p.add_argument("--paired-build", type=Path, default=ROOT / "build/norm_quant_l1k2560_candidate")
    a = p.parse_args()
    if Path(a.name).name != a.name:
        raise ValueError("Name must be a single component")
    folder = W / "evidence" / a.name
    folder.mkdir(exist_ok=False)
    (folder / "runtime").mkdir()
    status = dict(
        status="waiting_four_idle_cards",
        pid=os.getpid(),
        scope="64-token correctness only; profiler runs after generation",
        build=str(a.build),
        paired_build=str(a.paired_build),
    )

    def save():
        (folder / "status.json").write_text(json.dumps(status, indent=2) + "\n")

    save()
    server = client = None
    owned = {}

    def harvest():
        if server:
            try:
                for proc in psutil.Process(server.pid).children(recursive=True):
                    owned[proc.pid] = proc
            except psutil.NoSuchProcess:
                pass

    try:
        cards = wait_for_free_group(timeout=a.idle_timeout)
        status["devices"] = cards
        source = folder / "source_snapshot"
        source.mkdir()
        for name in ["src", "integration", "bootstrap"]:
            shutil.copytree(W / name, source / name, ignore=shutil.ignore_patterns("__pycache__"))
        hashes = {
            str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source.rglob("*")
            if p.is_file()
        }
        (folder / "source_manifest.json").write_text(json.dumps(hashes, indent=2) + "\n")
        status["source_snapshot"] = str(source)
        env = os.environ.copy()
        env.update(
            ASCEND_RT_VISIBLE_DEVICES=",".join(map(str, cards)),
            VLLM_ASCEND_ENABLE_NZ="2",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            PYTORCH_NPU_ALLOC_CONF="expandable_segments:False",
            ASCEND_PROCESS_LOG_PATH=str(folder / "ascend"),
        )
        if a.host_trace:
            env["QWEN36_HOST_TRACE"] = str(folder / "host_trace")
        env["QWEN36_DIRECT_ROOT"] = str(source)
        paths = env["PYTHONPATH"].split(":")
        env["PYTHONPATH"] = ":".join(
            str(source / p.relative_to(W)) if (p := Path(value)).is_relative_to(W) else value
            for value in paths
        )
        for key in list(env):
            if key.startswith(("VLLM_ASCEND_PYPTO_QWEN", "PYPTO_MTP_RUNTIME_RESERVE")):
                env.pop(key, None)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        url = f"http://127.0.0.1:{port}"
        cmd = command(ROOT / "checkpoints/public-inner-surrogate", a.build, folder, port, a.paired_build)
        if a.terminal_graph_timing or a.device_trace:
            at = cmd.index("--additional-config") + 1
            settings = json.loads(cmd[at])
            settings["qwen38_pto_mpmd"]["terminal_graph_timing"] = a.terminal_graph_timing
            settings["qwen38_pto_mpmd"]["terminal_device_trace"] = a.device_trace
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
                raise RuntimeError("Server exited during initialization; see server.log")
            startup_log = (folder / "server.log").read_text(errors="replace")
            if any(
                marker in startup_log
                for marker in [
                    "WorkerProc failed to start.",
                    "EngineCore failed to start.",
                    "Free memory on device (",
                ]
            ):
                raise RuntimeError("Fatal worker initialization failure detected promptly; see server.log")
            try:
                with urllib.request.urlopen(url + "/health", timeout=2) as response:
                    if response.status == 200:
                        break
            except (OSError, TimeoutError):
                pass
            time.sleep(2)
        else:
            raise TimeoutError("Initialization exceeded1200s")
        status["status"] = "correctness_request"
        save()
        with (folder / "client.log").open("w") as log:
            client = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "qwen36.tools.benchmark_stream",
                    "--request",
                    str(W / "reference/correctness_64.json"),
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
        deadline = time.monotonic() + 600
        while client.poll() is None:
            harvest()
            if server.poll() is not None:
                raise RuntimeError("Server exited during request")
            if time.monotonic() > deadline:
                raise TimeoutError("Short request exceeded600s")
            time.sleep(1)
        if client.returncode:
            counts = [0] * 4
            event_file = folder / "measurement/events.jsonl"
            if event_file.exists():
                for line in event_file.read_text().splitlines():
                    event = json.loads(line)
                    counts[event["request_index"]] += sum(
                        len(c.get("token_ids") or []) for c in event["response"].get("choices", [])
                    )
            if counts != [64] * 4:
                raise RuntimeError("Short request failed; see client.log")
            status["client_warning"] = (
                "All four64-token outputs arrived; metrics client failed; correctness diagnostic continues"
            )
        status["status"] = "paired_terminal_state_check"
        save()
        control = json.loads((folder / "lifecycle_control.json").read_text())
        with socket.socket(socket.AF_UNIX) as channel:
            channel.settimeout(480)
            channel.connect(control["socket"])
            channel.sendall(b'{"command":"profile-and-close"}\n')
            reply = json.loads(channel.makefile("rb").readline())
        if not reply.get("ok"):
            raise RuntimeError(reply)
        records = json.loads((folder / "terminal_profile.json").read_text())
        if len(records) != 4 or any(
            len(rows) != 2 or any(row["bitwise_different"] for row in rows) for rows in records
        ):
            raise ValueError("Paired state comparison failed")
        status.update(
            status="passed",
            comparison_tensors_per_rank=[rows[-1]["comparison_tensors"] for rows in records],
            lifecycle=reply,
        )
    except BaseException as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        harvest()
        for process in [client, server]:
            if process is not None and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
        for proc in owned.values():
            try:
                proc.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(list(owned.values()), timeout=5)
        for proc in alive:
            try:
                proc.kill()
            except psutil.NoSuchProcess:
                pass
        status["owned_descendant_pids"] = sorted(owned)
        save()
        print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
