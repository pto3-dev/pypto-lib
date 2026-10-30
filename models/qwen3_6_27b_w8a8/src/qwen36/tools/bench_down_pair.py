# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Three interleaved conditioned whole-interval timings, one probe per variant.

Real public INT8 down weights; fixed24 input rows; complete quant+down interval.
No changes to the main model/default profile. Each graph owns a separate borrowed
runtime; no overlapping execution or extra communication service is introduced.
"""

import argparse, ctypes, hashlib, json, statistics, time
from pathlib import Path
import torch
from safetensors import safe_open
from pypto import ir
from qwen36.runtime.borrowed_graph_runtime import BorrowedGraphRuntime, DeviceTensorBinding, checked
from qwen36.tools.idle_guard import _query, wait_for_idle

W = Path(__file__).resolve().parents[3]
ROOT = W.parents[1]
BINARY = Path(
    "/home/g00893677/pto-dev/qwen3_8_27b-perf/work/inprocess_native_nz_20261023/simpler/build/lib/a2a3"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    output = W / "evidence" / args.name
    output.mkdir(exist_ok=False)
    images = {}
    for label in ["reference", "candidate"]:
        root = W / "build" / f"down_probe_{label}"
        program = ir.CompiledProgram.from_dir(root, platform="a2a3")
        chip = program.chip_callable
        images[label] = ctypes.string_at(chip.buffer_ptr(), chip.buffer_size())
    model = ROOT / "checkpoints/public-inner-surrogate"
    index = json.loads((model / "quant_model_weights.safetensors.index.json").read_text())["weight_map"]
    stem = "model.language_model.layers.11.mlp.down_proj"

    def tensor(name):
        with safe_open(model / index[name], framework="pt", device="cpu") as file:
            return file.get_tensor(name)

    full = tensor(stem + ".weight")
    wc = full[:, :4352].T.contiguous()
    scale = tensor(stem + ".weight_scale").flatten().float()
    if bool(tensor(stem + ".weight_offset").ne(0).any()):
        raise ValueError("Nonzero offset unsupported")
    provenance = dict(
        checkpoint=str(model),
        layer=11,
        tp=4,
        rank=0,
        public_surrogate=True,
        weight_sha256=hashlib.sha256(wc.numpy().tobytes()).hexdigest(),
        scale_dtype="FP32 checkpoint scale; old/new use exactly the same values",
    )
    deadline = time.monotonic() + args.timeout
    card = None
    while time.monotonic() < deadline and card is None:
        for candidate in range(8):
            try:
                if "No process in device." in _query(candidate, "proc-mem"):
                    wait_for_idle([candidate], timeout=10, stable_seconds=2)
                    card = candidate
                    break
            except (RuntimeError, OSError):
                pass
        if card is None:
            time.sleep(2)
    if card is None:
        raise TimeoutError("No idle card for bounded microbenchmark")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.npu.set_device(card)
    rng = torch.Generator().manual_seed(20261027)
    source = torch.randn((24, 4352), generator=rng).bfloat16()
    source[0] = 0
    source[1] *= 1e-8
    x = source.npu()
    raw = wc.reshape(272, 16, 160, 32).permute(2, 0, 1, 3).contiguous().flatten().repeat(16).npu()
    ws = scale.repeat(16, 1).npu()
    shape = (16, 160, 272, 16, 32)
    outs = {label: torch.empty((24, 5120), dtype=torch.float32, device=x.device) for label in images}
    eviction = torch.ones(512 * 1024 * 1024, dtype=torch.uint8, device=x.device)
    torch.npu.synchronize()
    stream = torch.npu.Stream()
    runtimes = {}
    graphs = {}
    timing = {label: [] for label in images}
    try:
        for label, data in images.items():
            runtime = BorrowedGraphRuntime(BINARY, card, data, aicpu_threads=0)
            runtimes[label] = runtime
            runtime.bind([x, DeviceTensorBinding(raw, shape), ws, outs[label]])
            runtime.prepare_graph()
            graph = torch.npu.NPUGraph()
            graphs[label] = graph
            with torch.npu.graph(graph, stream=stream):
                runtime.enqueue_graph(stream.npu_stream)
            with torch.npu.stream(stream):
                graph.replay()
            stream.synchronize()
        reference = outs["reference"].cpu()
        if not torch.equal(reference.view(torch.uint8), outs["candidate"].cpu().view(torch.uint8)):
            raise ValueError("Down candidate differs from reference")
        for sample in range(3):
            labels = ["reference", "candidate"] if sample % 2 == 0 else ["candidate", "reference"]
            for label in labels:
                with torch.npu.stream(stream):
                    eviction.add_(1)
                    begin = torch.npu.Event(enable_timing=True)
                    end = torch.npu.Event(enable_timing=True)
                    begin.record(stream)
                    graphs[label].replay()
                    end.record(stream)
                stream.synchronize()
                timing[label].append(begin.elapsed_time(end) * 1000)
                if not torch.equal(reference.view(torch.uint8), outs[label].cpu().view(torch.uint8)):
                    raise ValueError("Output changed during interleaved timing")
        profiles = {}
        for label in images:
            graph = graphs[label]
            runtime = runtimes[label]
            graph.reset()
            checked(
                runtime.lib.simpler_finalize_run(runtime.ctx, runtime.run),
                "release graph for terminal profile",
            )
            runtime.prepared = runtime.graph_ready = False
            folder = output / label
            folder.mkdir()
            runtime.config.enable_chip_swimlane = 4
            runtime.config.enable_dep_gen = 1
            runtime.config.enable_pmu = 2
            runtime.config.output_prefix = str(folder).encode()
            runtime.eager(stream.npu_stream)
            stream.synchronize()
            runtime.finish_eager()
            raw_profile = json.loads((folder / "chip_swimlane_records.json").read_text())
            rows = raw_profile["aicore_tasks"]
            factor = 1e6 / raw_profile["metadata"]["clock_freq_hz"]
            profiles[label] = dict(
                envelope_us=(max(r[4] for r in rows) - min(r[3] for r in rows)) * factor,
                worker_records=len(rows),
            )
        medians = {label: statistics.median(v) for label, v in timing.items()}
        result = dict(
            status="passed",
            device=card,
            provenance=provenance,
            rows=24,
            shape=[24, 4352, 5120],
            bitwise_equal=True,
            samples_us=timing,
            median_us=medians,
            improvement_percent=(1 - medians["candidate"] / medians["reference"]) * 100,
            profile=profiles,
            scope="complete captured quant+down interval; not whole-model ITL; separate identical borrowed runtimes",
            conditioning="512MiB read/write before each timing, outside events",
            formal_model_performance=False,
        )
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print("DOWN_PAIR_RESULT", json.dumps(result), flush=True)
    finally:
        stream.synchronize()
        for graph in graphs.values():
            graph.reset()
        for runtime in runtimes.values():
            runtime.close()


if __name__ == "__main__":
    main()
