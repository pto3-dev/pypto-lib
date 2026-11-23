"""Exact finite-input dynamic-quant regression for the standalone A1 candidate."""

import json
import os
import statistics
from contextlib import ExitStack
from pathlib import Path

import pytest
import torch


def test_moe_input_retained_row_graph(tmp_path, *, control_source=None, candidate_source=None):
    import torch_npu
    from models.qwen3_30b_w8a8 import moe_batched_modelslim_a1
    from simpler_setup import A1BorrowedLaunch, KernelCompiler

    isa = os.environ.get("PTO_ISA_ROOT")
    if not isa:
        pytest.skip("PTO_ISA_ROOT is required")
    torch.set_num_threads(2)
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    model = Path(moe_batched_modelslim_a1.__file__).parent
    compiler = KernelCompiler("a1")
    sources = (
        Path(control_source) if control_source else model / "a1_modelslim_dynamic_quant.cpp",
        Path(candidate_source) if candidate_source else model / "a1_modelslim_dynamic_quant_retained.cpp",
    )
    images = [
        compiler.compile_single_launch(str(source), isa, extra_include_dirs=[str(model)])
        for source in sources
    ]
    generator = torch.Generator().manual_seed(3103275)
    report = {}
    for rows, width in ((1, 2048), (8, 2048), (20, 2048), (160, 768)):
        x = torch.zeros((rows, width), dtype=torch.float16, device="npu:0")
        quant = [torch.zeros_like(x, dtype=torch.int8) for _ in images]
        scales = [torch.zeros((rows, 16), dtype=torch.float32, device="npu:0") for _ in images]
        statuses = [torch.full((rows, 16), -7, dtype=torch.int32, device="npu:0") for _ in images]
        with ExitStack() as owners:
            graphs = []
            for arm, image in enumerate(images):
                owner = owners.enter_context(A1BorrowedLaunch(image))
                call = owner.prepare(x, quant[arm], scales[arm], statuses[arm], blocks=8)
                call()
                torch.npu.synchronize()
                graph = torch.npu.NPUGraph()
                owners.callback(graph.reset)
                with torch.npu.graph(graph):
                    for _ in range(8):
                        call()
                graphs.append(graph)
            for case in range(7):
                values = torch.randn((rows, width), generator=generator).half()
                if case == 0:
                    values.zero_()
                elif case == 1:
                    values.mul_(1e-6)
                elif case == 2:
                    values[:, :4] = torch.tensor([65504, -65504, 0, -0.0]).half()
                elif case == 3:
                    values.zero_()
                    values[:, 127::128] = 1
                    values[:, 63::128] = -1
                elif case == 4:
                    # Exact scale=1 exercises both signs at every RINT tie.
                    values.copy_((torch.arange(width) % 254 - 127).half().add_(0.5).expand(rows, -1))
                    values[:, 0] = 127
                elif case == 5:
                    encodings = torch.arange(rows * width) % 0x7C00
                    encodings += (torch.arange(rows * width) % 2) * 0x8000
                    values = encodings.to(torch.int16).view(torch.float16).reshape(rows, width)
                x.copy_(values)
                for graph in graphs:
                    graph.replay()
                assert torch.equal(quant[0].cpu(), quant[1].cpu()), (rows, width, case)
                assert torch.equal(scales[0].cpu().view(torch.int32), scales[1].cpu().view(torch.int32))
                expected_scale = values.float().abs().amax(dim=-1) * (1.0 / 127.0)
                assert torch.equal(scales[1].cpu()[:, 0], expected_scale), (rows, width, case, "CPU scale")
                assert all(not torch.count_nonzero(status.cpu()[:, 0]) for status in statuses)
            times = []
            for graph in graphs:
                for _ in range(5):
                    graph.replay()
                samples = []
                for _ in range(30):
                    start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                    start.record()
                    graph.replay()
                    end.record()
                    end.synchronize()
                    samples.append(start.elapsed_time(end) / 8.0)
                times.append(samples)
            report[f"{rows}x{width}"] = {
                "exact": True,
                "captured_repetitions": 8,
                "control_ms": times[0],
                "candidate_ms": times[1],
                "sources": [str(source) for source in sources],
                "control_median_ms": statistics.median(times[0]),
                "candidate_median_ms": statistics.median(times[1]),
            }
    (tmp_path / "input-retained-row-times.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
