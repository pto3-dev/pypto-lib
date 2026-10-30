# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Official bench serve, immutable RandomDataset fixture, native/PTO once each."""

import argparse, hashlib, json, os, re, shutil, signal, socket, subprocess, sys, time, urllib.request
from pathlib import Path
import psutil
from qwen38_runtime.serve import command
from qwen36.tools.idle_guard import wait_for_free_group, wait_for_idle

R = Path("/home/g00893677/pto-dev/qwen3_6_27b-inner-w8a8")
K = R / "work/kernel_tuning_20261027"
D = K.parent / "direct_weight_load_20261027"
W = K.parent / "direct_w8a8_20261027"
F = K / "evidence/native_pto_frozen_random_10k_20261027_01"
DATA = K / "datasets/random_seed1024_32k_10k_v1"
MODEL = R / "checkpoints/public-inner-surrogate"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def stage(
    label,
    cards,
    snapshot,
    *,
    backend=None,
    build=None,
    output_len=10240,
    input_len=32768,
    progress_timeout=90,
    terminal_validate=False,
):
    folder = F / label
    folder.mkdir()
    (folder / "runtime").mkdir()
    owned = {}
    server = client = None
    status = dict(
        status="waiting_same_cards", backend=label, devices=cards, fixture_sha256=sha(DATA / "manifest.json")
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
        # Deployment reserve for this TP4 benchmark, not a model/kernel capacity.
        # Sparse POSIX shm ftruncate can succeed even when future page writes SIGBUS.
        shm = os.statvfs("/dev/shm")
        free = shm.f_bavail * shm.f_frsize
        reserve = 32 * 1024 * 1024
        status["host_shared_memory"] = dict(
            total_bytes=shm.f_blocks * shm.f_frsize, free_bytes=free, benchmark_reserve_bytes=reserve
        )
        save()
        if free < reserve:
            raise RuntimeError(
                f"/dev/shm available {free} bytes below {reserve}-byte benchmark reserve; refusing GPU startup after confirmed shared-memory SIGBUS"
            )
        wait_for_idle(cards, timeout=180, stable_seconds=5)
        env = os.environ.copy()
        env.update(
            PYTHONFAULTHANDLER="1",
            ASCEND_RT_VISIBLE_DEVICES=",".join(map(str, cards)),
            QWEN38_RUNTIME_ROOT=str(snapshot),
            VLLM_ASCEND_ENABLE_NZ="2",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            PYTORCH_NPU_ALLOC_CONF="expandable_segments:False",
            ASCEND_PROCESS_LOG_PATH=str(folder / "ascend"),
            QWEN36_CPU_MTP_TRACE=str(folder / "cpu_verifier_trace"),
            QWEN36_BENCH_REQUEST_AUDIT=str(folder / "request_audit"),
        )
        env["PYTHONPATH"] = ":".join(
            str(snapshot / p.relative_to(D))
            if (p := Path(v)).is_relative_to(D)
            else (str(snapshot / "model_src") if p == W / "src" else v)
            for v in env["PYTHONPATH"].split(":")
        )
        for key in list(env):
            if key.startswith(("VLLM_ASCEND_PYPTO_QWEN", "PYPTO_MTP_RUNTIME_RESERVE")):
                env.pop(key, None)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        url = f"http://127.0.0.1:{port}"
        selected_backend = backend or ("native" if label == "native" else "pypto")
        cmd = command(
            MODEL,
            build or W / "build/comm_lanes_selfcontained",
            folder,
            port,
            backend=selected_backend,
            execution="direct",
            weight_strategy="native_nz" if selected_backend == "native" else "device_preloaded_nd_to_nz",
            kv_blocks=192,
            unsupported_decode="native",
        )
        bench = [
            sys.executable,
            "-m",
            "vllm.entrypoints.cli.main",
            "bench",
            "serve",
            "--backend",
            "openai",
            "--base-url",
            url,
            "--endpoint",
            "/v1/completions",
            "--model",
            str(MODEL),
            "--tokenizer",
            str(MODEL),
            "--served-model-name",
            "qwen36-mixed",
            "--dataset-name",
            "custom",
            "--dataset-path",
            str(DATA / "custom.jsonl"),
            "--disable-shuffle",
            "--skip-chat-template",
            "--custom-output-len",
            str(output_len),
            "--num-prompts",
            "4",
            "--max-concurrency",
            "4",
            "--seed",
            "1024",
            "--ignore-eos",
            "--temperature",
            "1.0",
            "--top-p",
            "0.95",
            "--top-k",
            "20",
            "--extra-body",
            json.dumps(dict(seed=1024, return_token_ids=True)),
            "--save-result",
            "--save-detailed",
            "--request-rate",
            "inf",
            "--num-warmups",
            "0",
            "--ready-check-timeout-sec",
            "0",
            "--percentile-metrics",
            "ttft,tpot,itl,e2el",
            "--metric-percentiles",
            "50,90,99",
            "--result-dir",
            str(folder),
            "--result-filename",
            "official_result.json",
        ]
        status.update(status="starting", server_command=cmd, bench_command=bench)
        save()
        with (folder / "server.log").open("w") as log:
            server = subprocess.Popen(
                cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
        status["server_pid"] = server.pid
        save()
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            harvest()
            if server.poll() is not None:
                raise RuntimeError("Server exited during initialization")
            text = (folder / "server.log").read_text(errors="replace")
            if any(
                m in text
                for m in [
                    "WorkerProc failed to start.",
                    "EngineCore failed to start.",
                    "Free memory on device (",
                ]
            ):
                raise RuntimeError("Fatal initialization failure")
            try:
                with urllib.request.urlopen(url + "/health", timeout=2) as reply:
                    if reply.status == 200:
                        break
            except (OSError, TimeoutError):
                pass
            time.sleep(2)
        else:
            raise TimeoutError("Initialization900s deadline")
        with urllib.request.urlopen(url + "/metrics", timeout=10) as reply:
            (folder / "metrics_before.prom").write_bytes(reply.read())
        status["status"] = "generation_" + str(output_len)
        save()
        print("BENCH_START", label, cards, flush=True)
        with (folder / "bench.log").open("w") as log:
            client = subprocess.Popen(
                bench, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
        deadline = time.monotonic() + 2400
        last_progress = time.monotonic()
        log_offset = 0
        while client.poll() is None:
            harvest()
            if server.poll() is not None:
                raise RuntimeError("Server exited during generation")
            if time.monotonic() > deadline:
                raise TimeoutError("Benchmark2400s deadline")
            # Observe only new existing logger output; no metrics RPC, device sync or
            # instrumentation in the model's measured path. Heartbeats with zero rates
            # cannot indefinitely extend the deadline for a stalled engine.
            with (folder / "server.log").open() as observed:
                observed.seek(log_offset)
                new_log = observed.read()
                log_offset = observed.tell()
            rates = re.findall(r"Avg (?:prompt|generation) throughput: ([0-9.]+) tokens/s", new_log)
            if any(float(rate) > 0 for rate in rates):
                last_progress = time.monotonic()
            if progress_timeout and time.monotonic() - last_progress > progress_timeout:
                status["progress_watchdog"] = dict(
                    timeout_seconds=progress_timeout,
                    last_progress_monotonic=last_progress,
                    detected_monotonic=time.monotonic(),
                )
                save()
                raise TimeoutError(f"No reported prompt/generation progress for {progress_timeout}s")
            time.sleep(2)
        if client.returncode:
            raise RuntimeError("Official bench client failed")
        result = json.loads((folder / "official_result.json").read_text())
        if (
            result["completed"] != 4
            or result.get("failed", 0)
            or result["input_lens"] != [input_len] * 4
            or result["output_lens"] != [output_len] * 4
        ):
            raise ValueError("Official result shape/completion mismatch")
        with urllib.request.urlopen(url + "/metrics", timeout=10) as reply:
            (folder / "metrics_after.prom").write_bytes(reply.read())
        status["status"] = "collecting"
        save()
        ctl = json.loads((folder / "lifecycle_control.json").read_text())
        with socket.socket(socket.AF_UNIX) as ch:
            ch.settimeout(600 if terminal_validate else 180)
            ch.connect(ctl["socket"])
            ch.sendall(
                (
                    json.dumps(
                        {"command": "validate-and-close" if terminal_validate else "collect-and-close"}
                    )
                    + "\n"
                ).encode()
            )
            reply = json.loads(ch.makefile("rb").readline())
        if not reply.get("ok"):
            raise RuntimeError(reply)
        audits = [
            json.loads(p.read_text()) for p in sorted((folder / "request_audit").glob("request_*.json"))
        ]
        assert len(audits) == 4 and all(a["http_status"] == 200 for a in audits)
        status.update(status="complete", official_mean_tpot_ms=result["mean_tpot_ms"], lifecycle=reply)
        save()
        print("BENCH_COMPLETE", label, flush=True)
    except BaseException as e:
        status.update(status="failed", error=f"{type(e).__name__}: {e}")
        save()
        raise
    finally:
        harvest()
        for proc in [client, server]:
            if proc and proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=5)
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


def main():
    global F, DATA
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    if Path(args.name).name != args.name:
        raise ValueError("Invalid evidence name")
    F = K / "evidence" / args.name
    DATA = args.dataset.resolve()
    if not DATA.is_relative_to(K / "datasets"):
        raise ValueError("Use a frozen fixture under this work directory")
    manifest = json.loads((DATA / "manifest.json").read_text())
    if len(manifest["requests"]) != 4 or any(r["input_tokens"] != 32768 for r in manifest["requests"]):
        raise ValueError("Invalid fixture shapes")
    F.mkdir(exist_ok=False)
    snapshot = F / "source_snapshot"
    snapshot.mkdir()
    for name in ["src", "integration", "bootstrap", "reference", "lib"]:
        shutil.copytree(D / name, snapshot / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(K / "runtime_fixed_lib/a2a3", snapshot / "lib/a2a3", dirs_exist_ok=True)
    shutil.copytree(W / "src", snapshot / "model_src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(K / "tools/response_audit.py", snapshot / "src/qwen38_runtime/request_audit.py")
    shutil.copy2(Path(__file__), snapshot / "run_frozen_bench_pair.py")
    (F / "source_manifest.json").write_text(
        json.dumps(
            {str(p.relative_to(snapshot)): sha(p) for p in snapshot.rglob("*") if p.is_file()}, indent=2
        )
        + "\n"
    )
    shutil.copy2(DATA / "manifest.json", F / "dataset_manifest.json")
    cards = wait_for_free_group(timeout=180)
    status = dict(
        status="starting", pid=os.getpid(), devices=cards, fixture=str(DATA), official_client=True, mtp=5
    )

    def save():
        (F / "status.json").write_text(json.dumps(status, indent=2) + "\n")

    save()
    try:
        for label in ["native", "pto"]:
            status["status"] = "running_" + label
            save()
            stage(label, cards, snapshot)
        status["status"] = "complete"
        save()
        print("PAIR_COMPLETE", F, flush=True)
    except BaseException as e:
        status.update(status="failed", error=f"{type(e).__name__}: {e}")
        save()
        raise


if __name__ == "__main__":
    main()
