# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bounded QK-only diagnosis: actual B4 vs one active request, PMU0 vs PMU2.

No model benchmark, speculation or inter-rank communication. This is one TP4
rank's exact QK shape, conditioning cache before measurements. Full output math
is unchanged; selected visible columns are checked against CPU FP32 dot. Masked padded
columns are unspecified intermediate storage and are not a numerical contract.
"""

import argparse, ctypes, json, statistics, time
from pathlib import Path
import torch
from pypto import ir
from qwen36.runtime.borrowed_graph_runtime import BorrowedGraphRuntime, checked
from qwen36.tools.idle_guard import _query, wait_for_idle

W = Path(__file__).resolve().parents[3]
BINARY = Path(
    "/home/g00893677/pto-dev/qwen3_8_27b-perf/work/inprocess_native_nz_20261023/simpler/build/lib/a2a3"
)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--name", required=True)
    a.add_argument("--active", type=int, choices=[1, 4], required=True)
    a.add_argument("--pmu", type=int, choices=[0, 2], required=True)
    args = a.parse_args()
    if Path(args.name).name != args.name:
        raise ValueError("Single-component name required")
    folder = W / "evidence" / args.name
    folder.mkdir(exist_ok=False)
    status = dict(status="waiting_idle_card", pid=__import__("os").getpid())

    def save():
        (folder / "status.json").write_text(json.dumps(status, indent=2) + "\n")

    save()
    card = None
    deadline = time.monotonic() + 300
    while card is None and time.monotonic() < deadline:
        for c in range(8):
            try:
                if "No process in device." in _query(c, "proc-mem"):
                    wait_for_idle([c], timeout=10, stable_seconds=2)
                    card = c
                    break
            except (RuntimeError, OSError):
                pass
        if card is None:
            time.sleep(2)
    if card is None:
        raise TimeoutError("No idle card")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.npu.set_device(card)
    status.update(status="running", device=card)
    save()
    program = ir.CompiledProgram.from_dir(W / "build/qk_b4_diagnosis", platform="a2a3")
    chip = program.chip_callable
    image = ctypes.string_at(chip.buffer_ptr(), chip.buffer_size())
    rng = torch.Generator().manual_seed(20261027)
    qcpu = (torch.randn((24, 6, 256), generator=rng) * 0.1).bfloat16()
    cachecpu = (torch.randn((294912, 1, 256), generator=rng) * 0.1).bfloat16()
    tablecpu = torch.stack([torch.arange(336, dtype=torch.int32) + (t // 6) * 336 for t in range(24)])
    q = qcpu.npu()
    cache = cachecpu.npu()
    table = tablecpu.npu()
    lengths = torch.zeros(24, dtype=torch.int32, device=q.device)
    out = torch.empty((144, 43008), dtype=torch.float32, device=q.device)
    eviction = torch.ones(512 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    stream = torch.npu.Stream()
    results = []
    try:
        for active in [args.active]:
            runtime = BorrowedGraphRuntime(BINARY, card, image, aicpu_threads=0)
            graph = torch.npu.NPUGraph()
            try:
                with torch.npu.stream(stream):
                    lengths.zero_()
                    lengths[: active * 6].fill_(32768)
                stream.synchronize()
                runtime.bind([q, cache, table, lengths, out])
                runtime.prepare_graph()
                with torch.npu.graph(graph, stream=stream):
                    runtime.enqueue_graph(stream.npu_stream)
                with torch.npu.stream(stream):
                    graph.replay()
                stream.synchronize()
                samples = []
                for _ in range(3):
                    with torch.npu.stream(stream):
                        eviction.add_(1)
                        begin = torch.npu.Event(enable_timing=True)
                        end = torch.npu.Event(enable_timing=True)
                        begin.record(stream)
                        graph.replay()
                        end.record(stream)
                    stream.synchronize()
                    samples.append(begin.elapsed_time(end) * 1000)
                (folder / "timing_partial.json").write_text(
                    json.dumps(
                        dict(
                            active_requests=active, samples_us=samples, median_us=statistics.median(samples)
                        ),
                        indent=2,
                    )
                    + "\n"
                )
                actual = out.cpu()
                errors = []
                for row in [0, 5, 35, 36, 71, 72, 107, 108, 143]:
                    token = row // 6
                    request = token // 6
                    if request >= active:
                        continue
                    for col in [0, 127, 128, 16383, 32767]:
                        physical = int(tablecpu[token, col // 128]) * 128 + col % 128
                        expected = (
                            (qcpu[token, row % 6].float() * cachecpu[physical, 0].float()).sum()
                            if request < active and col < 32768
                            else torch.tensor(0.0)
                        )
                        diff = abs(float(actual[row, col]) - float(expected))
                        errors.append(diff)
                        if diff > 1e-4 + abs(float(expected)) * 1e-4:
                            raise ValueError(
                                f"QK sample differs row{row} col{col}: {actual[row, col]} vs {expected}"
                            )
                graph.reset()
                checked(runtime.lib.simpler_finalize_run(runtime.ctx, runtime.run), "retire QK graph")
                runtime.prepared = runtime.graph_ready = False
                profiles = []
                for pmu in [args.pmu]:
                    dest = folder / f"active{active}_pmu{pmu}"
                    dest.mkdir()
                    with torch.npu.stream(stream):
                        eviction.add_(1)
                    stream.synchronize()
                    runtime.config.enable_chip_swimlane = 4
                    runtime.config.enable_dep_gen = 1
                    runtime.config.enable_pmu = pmu
                    runtime.config.output_prefix = str(dest).encode()
                    runtime.eager(stream.npu_stream)
                    stream.synchronize()
                    runtime.finish_eager()
                    d = json.loads((dest / "chip_swimlane_records.json").read_text())
                    factor = 1e6 / d["metadata"]["clock_freq_hz"]
                    rs = d["aicore_tasks"]
                    per = [dict(worker_core=r[0], duration_us=(r[4] - r[3]) * factor) for r in rs]
                    profiles.append(
                        dict(
                            pmu=pmu,
                            worker_records=len(rs),
                            envelope_us=(max(r[4] for r in rs) - min(r[3] for r in rs)) * factor,
                            per_core=per,
                        )
                    )
                results.append(
                    dict(
                        active_requests=active,
                        logical_shape=[24, 6, 32768, 256],
                        captured_qk_samples_us=samples,
                        median_us=statistics.median(samples),
                        sampled_output_max_abs_error=max(errors),
                        sampled_elements=len(errors),
                        padded_scores="Unspecified before causal softmax; not required to be zero",
                        profiles=profiles,
                    )
                )
            finally:
                stream.synchronize()
                graph.reset()
                runtime.close()
        result = dict(
            status="passed",
            device=card,
            scope="Exact one TP4-rank QK kernel; actual24 rows with either1 or4 live32K requests. No static outer plan, communication, verifier/drafter or whole-model ITL.",
            conditioning="512MiB read/write before every timing/profile, outside interval",
            cases=results,
        )
        (folder / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        status.update(status="passed")
        save()
        print(json.dumps(result), flush=True)
    except BaseException as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise


if __name__ == "__main__":
    main()
