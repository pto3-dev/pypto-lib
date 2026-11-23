# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""Exact B20 static-quantization graph replay against the native NPU quantizer."""

import json
import os
import statistics
from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize("width", (2048, 4096))
@pytest.mark.parametrize("implementation", ("scalar_dsl", "vector_dsl", "native_vector"))
def test_batched_modelslim_static_quant_graph(width, implementation, tmp_path):
    torch_npu = pytest.importorskip("torch_npu")
    isa = os.environ.get("PTO_ISA_ROOT")
    if not isa:
        pytest.skip("PTO_ISA_ROOT is required for the allocated 310P device")
    from simpler_setup import A1BorrowedLaunch

    from models.qwen3_30b_w8a8.compile_modelslim_a1 import compile_modelslim_program
    from models.qwen3_30b_w8a8.quant_batched_modelslim_a1 import make_a1_batched_modelslim_static_quant
    from models.qwen3_30b_w8a8.quant_batched_modelslim_a1_vector import make_a1_batched_modelslim_vector_quant
    from simpler_setup import KernelCompiler

    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    if implementation == "native_vector":
        model = Path(__file__).parents[1] / "models/qwen3_30b_w8a8"
        image = KernelCompiler("a1").compile_single_launch(
            str(model / "a1_static_quant_batched_vector.cpp"), isa, build_dir=str(tmp_path)
        )
        order = ("x", "reciprocal", "offset", "quant")
    else:
        factory = (make_a1_batched_modelslim_static_quant if implementation == "scalar_dsl"
                   else make_a1_batched_modelslim_vector_quant)
        image, order = compile_modelslim_program(factory(width), tmp_path, Path(isa))
    assert set(order) == {"x", "reciprocal", "offset", "quant"}
    generator = torch.Generator().manual_seed(310320 + width)
    x = torch.zeros((20, width), dtype=torch.float16, device="npu:0")
    reciprocal = torch.ones((1, width), dtype=torch.float16, device="npu:0")
    offset = torch.zeros_like(reciprocal)
    quant = torch.zeros((20, width), dtype=torch.int8, device="npu:0")
    tensors = {"x": x, "reciprocal": reciprocal, "offset": offset, "quant": quant}
    with A1BorrowedLaunch(image, device_id=0) as launch:
        call = launch.prepare(*(tensors[name] for name in order), blocks=8)
        call()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        try:
            with torch.npu.graph(graph):
                call()
            for replay in range(3):
                values = (torch.randn((20, width), generator=generator) * 100).half()
                values[0, :8] = torch.tensor([-256, -128.5, -127.5, -0.5, 0.5, 1.5, 127.5, 256]).half()
                inverse = (torch.rand((1, width), generator=generator) + 0.25).half()
                zero_point = (torch.randn((1, width), generator=generator) * 3).half()
                if replay == 0:
                    inverse[:, :8] = 1
                    zero_point[:, :8] = 0
                x.copy_(values)
                reciprocal.copy_(inverse)
                offset.copy_(zero_point)
                expected = torch_npu.npu_quantize(x, reciprocal, offset, torch.qint8, -1, False).cpu()
                graph.replay()
                assert torch.equal(quant.cpu(), expected)
            benchmark_path = os.environ.get("PYPTO_A1_QUANT_BENCH_OUTPUT")
            if benchmark_path and implementation != "scalar_dsl":
                graph.reset()
                with torch.npu.graph(graph):
                    for _ in range(100):
                        call()
                for _ in range(5):
                    graph.replay()
                torch.npu.synchronize()
                start = torch.npu.Event(enable_timing=True)
                end = torch.npu.Event(enable_timing=True)
                samples = []
                for _ in range(100):
                    start.record()
                    graph.replay()
                    end.record()
                    end.synchronize()
                    samples.append(start.elapsed_time(end) * 1000 / 100)
                destination = Path(benchmark_path)
                results = json.loads(destination.read_text()) if destination.exists() else {}
                results[f"{implementation}_{width}"] = {
                    "median_us": statistics.median(samples), "samples_us": samples,
                    "captured_calls": 100, "warmup": 5, "rounds": 100,
                    "precision": "exact_native_npu_quantize_three_replays",
                    "scope": "isolated_operator_graph_device_events",
                }
                destination.write_text(json.dumps(results, indent=2) + "\n")
        finally:
            graph.reset()


@pytest.mark.parametrize("rows,width", ((20, 2048), (160, 768)))
def test_modelslim_dynamic_quant_graph(rows, width, tmp_path):
    torch_npu = pytest.importorskip("torch_npu")
    isa = os.environ.get("PTO_ISA_ROOT")
    if not isa:
        pytest.skip("PTO_ISA_ROOT is required for the allocated 310P device")
    from simpler_setup import A1BorrowedLaunch

    from models.qwen3_30b_w8a8.compile_modelslim_a1 import compile_modelslim_program, compile_modelslim_source
    from models.qwen3_30b_w8a8.quant_dynamic_modelslim_a1_vector import make_a1_modelslim_dynamic_quant

    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    native_image = compile_modelslim_source("a1_modelslim_dynamic_quant_retained.cpp", tmp_path / "native", Path(isa))
    program = make_a1_modelslim_dynamic_quant(rows, width)
    dsl_image, order = compile_modelslim_program(program, tmp_path / "dsl", Path(isa))
    assert set(order) == {"x", "quant", "scale", "status"}
    generator = torch.Generator().manual_seed(310330 + width)
    x = torch.zeros((rows, width), dtype=torch.float16, device="npu:0")
    outputs = []
    for _ in range(2):
        outputs.append({
            "x": x,
            "quant": torch.full((rows, width), -91, dtype=torch.int8, device="npu:0"),
            "scale": torch.full((rows, 16), -23.0, dtype=torch.float32, device="npu:0"),
            "status": torch.full((rows, 16), 17, dtype=torch.int32, device="npu:0"),
        })
    with A1BorrowedLaunch(native_image, device_id=0) as native, A1BorrowedLaunch(dsl_image, device_id=0) as dsl:
        native_order = ("x", "quant", "scale", "status")
        calls = [native.prepare(*(outputs[0][name] for name in native_order), blocks=8),
                 dsl.prepare(*(outputs[1][name] for name in order), blocks=8)]
        for call in calls:
            call()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        try:
            with torch.npu.graph(graph):
                for call in calls:
                    call()
            for replay, multiplier in enumerate((0.0, 1e-6, 0.01, 1.0, 1000.0)):
                values = (torch.randn((rows, width), generator=generator) * multiplier).half()
                if replay == 3:
                    values[0, :8] = torch.tensor([-127, -126.5, -1.5, -0.5, 0.5, 1.5, 126.5, 127]).half()
                    values[0, 8:] = 0
                if replay == 4:
                    values[-1, :4] = torch.tensor([-65504.0, 65504.0, -0.0, 0.0]).half()
                expected_scale = values.float().abs().amax(dim=1) * torch.tensor(1.0 / 127.0)
                inverse = torch.where(expected_scale > 1e-7, expected_scale.reciprocal(), 0.0)
                expected = (values.float() * inverse[:, None]).round().int().float().half().to(torch.int8)
                x.copy_(values)
                graph.replay()
                for tensors in outputs:
                    assert torch.equal(tensors["quant"].cpu(), expected)
                    assert torch.equal(tensors["scale"][:, 0].cpu(), expected_scale)
                    assert torch.equal(tensors["scale"][:, 1:].cpu(), torch.full((rows, 15), -23.0))
                    assert torch.equal(tensors["status"][:, 0].cpu(), torch.zeros(rows, dtype=torch.int32))
                    assert torch.equal(tensors["status"][:, 1:].cpu(), torch.full((rows, 15), 17, dtype=torch.int32))
            benchmark_path = os.environ.get("PYPTO_A1_DYNAMIC_QUANT_BENCH_OUTPUT")
            if benchmark_path:
                for implementation, call in zip(("native_retained", "vector_dsl"), calls, strict=True):
                    graph.reset()
                    with torch.npu.graph(graph):
                        for _ in range(100):
                            call()
                    for _ in range(5):
                        graph.replay()
                    torch.npu.synchronize()
                    start = torch.npu.Event(enable_timing=True)
                    end = torch.npu.Event(enable_timing=True)
                    samples = []
                    for _ in range(100):
                        start.record()
                        graph.replay()
                        end.record()
                        end.synchronize()
                        samples.append(start.elapsed_time(end) * 1000 / 100)
                    destination = Path(benchmark_path)
                    results = json.loads(destination.read_text()) if destination.exists() else {}
                    results[f"{implementation}_{rows}_{width}"] = {
                        "median_us": statistics.median(samples), "samples_us": samples,
                        "captured_calls": 100, "warmup": 5, "rounds": 100,
                        "precision": "exact_native_retained_and_fp32_golden_five_replays",
                        "scope": "isolated_operator_graph_device_events",
                    }
                    destination.write_text(json.dumps(results, indent=2) + "\n")
        finally:
            graph.reset()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
