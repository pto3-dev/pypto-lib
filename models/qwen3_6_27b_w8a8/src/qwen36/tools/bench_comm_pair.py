# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bounded TP4 quota-only collective+Norm+Quant comparison.

Independent shared-memory windows, fresh epochs between variants, one graph per
quota, no change to the model/default protocol. Timings cover the full collective
and math, taking max rank per block. CPU barriers/reset are outside timing.
"""

import argparse, ctypes, datetime, json, os, signal, socket, statistics, subprocess, sys, time
from pathlib import Path

W = Path(__file__).resolve().parents[3]
BINARY = Path(
    "/home/g00893677/pto-dev/qwen3_8_27b-perf/work/inprocess_native_nz_20261023/simpler/build/lib/a2a3"
)


def child(folder, rank, port):
    experiment = json.loads((folder / "experiment.json").read_text())
    labels = ["reference", "candidate"]
    import torch
    import torch.distributed as dist
    from qwen36.runtime.borrowed_comm import BorrowedCommWindows
    from qwen36.runtime.borrowed_graph_runtime import BorrowedGraphRuntime, checked

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.npu.set_device(rank)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=4,
        timeout=datetime.timedelta(seconds=180),
    )
    stream = torch.npu.Stream()
    device = torch.device(f"npu:{rank}")
    random = torch.Generator().manual_seed(1024)
    residual = (torch.randn((24, 5120), generator=random) * 0.25).bfloat16().to(device)
    weight = (torch.randn((1, 5120), generator=random) * 0.02).bfloat16().to(device)
    random.manual_seed(2048 + rank)
    x = torch.randn((24, 5120), generator=random).to(device)
    metadata = torch.zeros(4, dtype=torch.int32, device=device)
    torch.npu.synchronize()
    runtimes = {}
    graphs = {}
    windows = None
    timings = {label: [] for label in labels}
    correct = {}
    outputs = {}
    reference = None
    try:
        for quota in labels:
            runtime = BorrowedGraphRuntime(
                BINARY, rank, (folder / f"{quota}.image").read_bytes(), aicpu_threads=0
            )
            runtimes[quota] = runtime
        windows = BorrowedCommWindows(
            runtimes["reference"].lib,
            rank,
            4,
            stream,
            folder / "hccl_root",
            [("data", (3096, 5120), torch.bfloat16), ("signal", (516, 1), torch.int32)],
        )

        def reset():
            stream.synchronize()
            dist.barrier()
            with torch.npu.stream(stream):
                windows.buffers["signal"].zero_()
                metadata.zero_()
            stream.synchronize()
            dist.barrier()

        for quota in labels:
            outputs[quota] = [
                torch.empty((24, 5120), dtype=torch.bfloat16, device=device),
                torch.empty((24, 5120), dtype=torch.bfloat16, device=device),
            ]
            if experiment.get("kind") != "even":
                outputs[quota] += [
                    torch.empty((24, 5120), dtype=torch.int8, device=device),
                    torch.empty((24, 1), dtype=torch.float32, device=device),
                ]
            runtime = runtimes[quota]
            runtime.bind(
                [
                    x,
                    residual,
                    weight,
                    metadata,
                    windows.buffers["data"],
                    windows.buffers["signal"],
                    *outputs[quota],
                ],
                scalars=[windows.contexts["data"], windows.contexts["signal"]],
            )
            runtime.prepare_graph()
            reset()
            graph = torch.npu.NPUGraph()
            graphs[quota] = graph
            with torch.npu.graph(graph, stream=stream):
                metadata[1].add_(1)
                runtime.enqueue_graph(stream.npu_stream)
            with torch.npu.stream(stream):
                graph.replay()
            stream.synchronize()
            values = [o.cpu() for o in outputs[quota]]
            if reference is None:
                reference = values
            differences = [
                i
                for i, (a, b) in enumerate(zip(reference, values))
                if not torch.equal(a.view(torch.uint8), b.view(torch.uint8))
            ]
            if differences:
                raise ValueError(f"Rank{rank} quota{quota} math outputs differ {differences}")
            correct[str(quota)] = True
        repeats = 8 if experiment.get("operations_per_replay", 1) > 1 else 64
        for block in range(3):
            for quota in labels if block % 2 == 0 else labels[::-1]:
                reset()
                with torch.npu.stream(stream):
                    for _ in range(3):
                        graphs[quota].replay()
                stream.synchronize()
                dist.barrier()
                begin = torch.npu.Event(enable_timing=True)
                end = torch.npu.Event(enable_timing=True)
                with torch.npu.stream(stream):
                    begin.record(stream)
                    for _ in range(repeats):
                        graphs[quota].replay()
                    end.record(stream)
                stream.synchronize()
                timings[str(quota)].append(
                    begin.elapsed_time(end) * 1000 / repeats / experiment.get("operations_per_replay", 1)
                )
                for i, (expected, actual) in enumerate(zip(reference, outputs[quota])):
                    if not torch.equal(expected.view(torch.uint8), actual.cpu().view(torch.uint8)):
                        raise ValueError(f"Output{i} changed in timed block")
                dist.barrier()
        for quota in labels:
            reset()
            runtime = runtimes[quota]
            graphs[quota].reset()
            checked(runtime.lib.simpler_finalize_run(runtime.ctx, runtime.run), "release graph for profile")
            runtime.prepared = runtime.graph_ready = False
            destination = folder / quota / f"rank{rank}"
            destination.mkdir(parents=True, exist_ok=False)
            runtime.config.enable_chip_swimlane = 4
            runtime.config.enable_dep_gen = 1
            runtime.config.enable_pmu = 2
            runtime.config.output_prefix = str(destination).encode()
            with torch.npu.stream(stream):
                metadata[1].fill_(1)
            stream.synchronize()
            dist.barrier()
            runtime.eager(stream.npu_stream)
            stream.synchronize()
            runtime.finish_eager()
            dist.barrier()
        result = dict(
            status="passed",
            rank=rank,
            logical_device=rank,
            timings_us=timings,
            bitwise_equal=correct,
            replays_per_sample=repeats,
        )
        (folder / f"rank{rank}_result.json").write_text(json.dumps(result, indent=2) + "\n")
        print("COMM_RANK_RESULT", json.dumps(result), flush=True)
    finally:
        stream.synchronize()
        for graph in graphs.values():
            graph.reset()
        for runtime in runtimes.values():
            runtime.close()
        if windows:
            windows.close()
        dist.destroy_process_group()


def parent(args):
    from pypto import ir

    labels = ["reference", "candidate"]
    from qwen36.tools.idle_guard import wait_for_free_group

    folder = W / "evidence" / args.name
    folder.mkdir(exist_ok=False)
    status = dict(
        status="compiling_images",
        pid=os.getpid(),
        scope="isolated TP4 collective+Norm+Quant, not whole-model ITL",
    )

    def save():
        (folder / "status.json").write_text(json.dumps(status, indent=2) + "\n")

    save()
    experiment = dict(
        reference="comm_probe_series24" if args.lane else "comm_probe_24",
        candidate="comm_probe_laneseries24" if args.lane else "comm_probe_12",
        operations_per_replay=64 if args.lane else 1,
        candidate_protocol="lane odd-operation frontier" if args.lane else "original group barrier",
        candidate_width=24 if args.lane else 12,
    )
    if args.even:
        experiment = dict(
            reference="even_probe_reference",
            candidate="even_probe_lane",
            kind="even",
            operations_per_replay=64,
            candidate_protocol="even lane frontier;48 AddNorm+16 Add",
            candidate_width=24,
        )
    (folder / "experiment.json").write_text(json.dumps(experiment, indent=2) + "\n")
    for quota in labels:
        program = ir.CompiledProgram.from_dir(W / "build" / experiment[quota], platform="a2a3")
        chip = program.chip_callable
        (folder / f"{quota}.image").write_bytes(ctypes.string_at(chip.buffer_ptr(), chip.buffer_size()))
    status["status"] = "waiting_four_idle_cards"
    save()
    cards = wait_for_free_group(timeout=args.timeout)
    status.update(status="running", devices=cards)
    save()
    with socket.socket() as channel:
        channel.bind(("127.0.0.1", 0))
        port = channel.getsockname()[1]
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(map(str, cards))
    processes = []
    logs = []
    try:
        for rank in range(4):
            log = (folder / f"rank{rank}.log").open("w")
            logs.append(log)
            processes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "qwen36.tools.bench_comm_pair",
                        "--child",
                        str(rank),
                        "--folder",
                        str(folder),
                        "--port",
                        str(port),
                    ],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )
        deadline = time.monotonic() + 360
        while any(p.poll() is None for p in processes):
            failed = [rank for rank, p in enumerate(processes) if p.poll() not in (None, 0)]
            if failed:
                raise RuntimeError(f"Child rank{failed} failed; inspect rank logs")
            if time.monotonic() > deadline:
                raise TimeoutError("Comm probe bounded360s deadline")
            time.sleep(1)
        if any(p.returncode for p in processes):
            raise RuntimeError("Child failed")
        ranks = [json.loads((folder / f"rank{rank}_result.json").read_text()) for rank in range(4)]
        samples = {
            str(quota): [max(r["timings_us"][str(quota)][i] for r in ranks) for i in range(3)]
            for quota in labels
        }
        medians = {k: statistics.median(v) for k, v in samples.items()}
        result = dict(
            status="passed",
            experiment=experiment,
            devices=cards,
            rows=24,
            shape=[24, 5120],
            samples_max_rank_us=samples,
            median_max_rank_us=medians,
            improvement_percent=(1 - medians["candidate"] / medians["reference"]) * 100,
            bitwise_equal=True,
            rounds=3,
            replays_per_sample=8 if experiment["operations_per_replay"] > 1 else 64,
            operations_per_replay=experiment["operations_per_replay"],
            reset="all ranks barrier, zero signal+epoch, barrier before each variant; outside timed interval",
            timing="full graph collective+Norm+Quant, max of rank event intervals; queueing and skew included",
            formal_model_performance=False,
        )
        (folder / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        status.update(result)
        save()
        print("COMM_PAIR_RESULT", json.dumps(result), flush=True)
    except BaseException as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise
    finally:
        for p in processes:
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait(timeout=5)
        for log in logs:
            log.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name")
    parser.add_argument("--lane", action="store_true")
    parser.add_argument("--even", action="store_true")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--child", type=int)
    parser.add_argument("--folder", type=Path)
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    if args.child is not None:
        child(args.folder, args.child, args.port)
    else:
        if not args.name or Path(args.name).name != args.name:
            raise ValueError("Single-component --name required")
        parent(args)


if __name__ == "__main__":
    main()
