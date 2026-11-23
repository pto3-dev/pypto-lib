# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""B1/B20 Norm and AddRMSNorm graph replay against native 310P operations."""

import os
import importlib.util
from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize("rows,fused_add,resident_row", [
    (1, False, False), (1, True, False), (20, False, False), (20, True, False), (20, True, True),
])
def test_modelslim_norm_graph(rows, fused_add, resident_row, tmp_path):
    torch_npu = pytest.importorskip("torch_npu")
    isa = os.environ.get("PTO_ISA_ROOT")
    if not isa:
        pytest.skip("PTO_ISA_ROOT is required for the allocated 310P device")
    from simpler_setup import A1BorrowedLaunch

    from models.qwen3_30b_w8a8.compile_modelslim_a1 import compile_modelslim_program
    from models.qwen3_30b_w8a8.norm_modelslim_a1 import make_a1_modelslim_norm

    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    program = make_a1_modelslim_norm(fused_add=fused_add, rows=rows,
                                    balanced_reduction=rows == 20 and not fused_add, resident_row=resident_row)
    image, order = compile_modelslim_program(program, tmp_path, Path(isa))
    tensors = {name: torch.zeros((rows, 2048), dtype=torch.float16, device="npu:0")
               for name in ("x", "residual", "y", "residual_out")}
    tensors.update({name: torch.zeros((1, 2048), dtype=torch.float16, device="npu:0")
                    for name in ("weight", "bias")})
    generator = torch.Generator().manual_seed(310320)
    with A1BorrowedLaunch(image, device_id=0) as launch:
        call = launch.prepare(*(tensors[name] for name in order), blocks=1 if rows == 1 else 8)
        call()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        try:
            with torch.npu.graph(graph):
                call()
            for magnitude in (0.125, 1.0, 8.0):
                for name in ("x", "residual"):
                    tensors[name].copy_((torch.randn((rows, 2048), generator=generator) * magnitude).half())
                tensors["weight"].copy_((torch.randn((1, 2048), generator=generator) * 0.25 + 1).half())
                tensors["bias"].copy_((torch.randn((1, 2048), generator=generator) * 0.125).half())
                x, residual = tensors["x"], tensors["residual"]
                gamma = tensors["weight"].flatten()
                if fused_add:
                    normalized, _, skip = torch_npu.npu_add_rms_norm(x, residual, gamma, 1e-6)
                else:
                    normalized, _ = torch_npu.npu_rms_norm(x, gamma, 1e-6)
                    skip = x.clone()
                expected = (normalized + tensors["bias"]).cpu()
                expected_skip = skip.cpu()
                graph.replay()
                actual = tensors["y"].cpu()
                assert torch.equal(tensors["residual_out"].cpu(), expected_skip)
                if not torch.equal(actual, expected):
                    fixture = tmp_path / "norm-mismatch.pt"
                    torch.save({"actual": actual, "expected": expected,
                                **{name: t.cpu() for name, t in tensors.items()}}, fixture)
                    pytest.fail(f"rows={rows}, fused={fused_add}, mismatches={torch.count_nonzero(actual != expected)}, fixture={fixture}")
        finally:
            graph.reset()


def test_b20_fused_norm_static_quant_graph(tmp_path):
    """Keep the materialized FP16 norm boundary inside one quantizing kernel."""
    torch_npu = pytest.importorskip("torch_npu")
    isa = os.environ.get("PTO_ISA_ROOT")
    if not isa:
        pytest.skip("PTO_ISA_ROOT is required for the allocated 310P device")
    from simpler_setup import A1BorrowedLaunch
    from models.qwen3_30b_w8a8.compile_modelslim_a1 import compile_modelslim_program
    source = Path(__file__).with_name("kernels") / "port910_norm_quant_candidate.py"
    spec = importlib.util.spec_from_file_location("models.qwen3_30b_w8a8._norm_quant_candidate", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    image, order = compile_modelslim_program(module.make_a1_modelslim_norm_quant(fused_add=True, rows=20),
                                           tmp_path / "norm_quant", Path(isa))
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    tensors = {name: torch.zeros((20, 2048), dtype=torch.float16, device="npu:0")
               for name in ("x", "residual", "y", "residual_out")}
    tensors.update({name: torch.zeros((1, 2048), dtype=torch.float16, device="npu:0")
                    for name in ("weight", "bias", "reciprocal", "offset")})
    tensors["quant"] = torch.zeros((20, 2048), dtype=torch.int8, device="npu:0")
    generator = torch.Generator().manual_seed(310326)
    with A1BorrowedLaunch(image) as owner:
        call = owner.prepare(*(tensors[name] for name in order), blocks=8)
        call()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        try:
            with torch.npu.graph(graph):
                call()
            for magnitude in (0.125, 1.0, 8.0):
                for name in ("x", "residual"):
                    tensors[name].copy_((torch.randn((20, 2048), generator=generator) * magnitude).half())
                gamma = (torch.randn((1, 2048), generator=generator) * 0.25 + 1).half()
                beta = (torch.randn((1, 2048), generator=generator) * 0.125).half()
                gamma[:, :8] = 0
                beta[:, :8] = torch.tensor([-256, -128.5, -127.5, -0.5, 0.5, 1.5, 127.5, 256]).half()
                reciprocal = (torch.rand((1, 2048), generator=generator) * 255 + 0.25).half()
                offset = (torch.randn((1, 2048), generator=generator) * 3).half()
                reciprocal[:, :8], offset[:, :8] = 1, 0
                for name, value in (("weight", gamma), ("bias", beta), ("reciprocal", reciprocal), ("offset", offset)):
                    tensors[name].copy_(value)
                native, _, residual = torch_npu.npu_add_rms_norm(tensors["x"], tensors["residual"],
                                                               tensors["weight"].flatten(), 1e-6)
                native = native + tensors["bias"]
                expected_quant = torch_npu.npu_quantize(native, tensors["reciprocal"], tensors["offset"],
                                                        torch.qint8, -1, False).cpu()
                graph.replay()
                assert torch.equal(tensors["y"].cpu(), native.cpu())
                assert torch.equal(tensors["residual_out"].cpu(), residual.cpu())
                assert torch.equal(tensors["quant"].cpu(), expected_quant)
        finally:
            graph.reset()



def test_b20_post_norm_graph_native_exact(tmp_path, candidate_source=None, control_source=None):
    """Compare full-row loading with native biased AddRMSNorm without changing reduction order."""
    import json
    import statistics
    from contextlib import ExitStack

    torch_npu = pytest.importorskip("torch_npu")
    isa = os.environ.get("PTO_ISA_ROOT")
    if not isa:
        pytest.skip("PTO_ISA_ROOT is required for the allocated 310P device")
    from simpler_setup import A1BorrowedLaunch, KernelCompiler
    from models.qwen3_30b_w8a8.compile_modelslim_a1 import compile_modelslim_program
    from models.qwen3_30b_w8a8.norm_modelslim_a1 import make_a1_modelslim_norm

    def factory(source, name):
        if source is None:
            return make_a1_modelslim_norm
        spec = importlib.util.spec_from_file_location(name, source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.make_a1_modelslim_norm

    torch.set_num_threads(2)
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    inputs = {name: torch.zeros((20, 2048), dtype=torch.float16, device="npu:0")
              for name in ("x", "residual")}
    inputs.update({name: torch.zeros((1, 2048), dtype=torch.float16, device="npu:0")
                   for name in ("weight", "bias")})
    storage = [{name: torch.full((32, 2048), -91, dtype=torch.float16, device="npu:0")
                for name in ("y", "residual_out")} for _ in range(2)]
    outputs = [{name: value[:20] for name, value in arm.items()} for arm in storage]
    if candidate_source is None:
        candidate_source = Path(__file__).parents[1] / "models/qwen3_30b_w8a8/a1_post_norm_batched.cpp"
    generator = torch.Generator().manual_seed(3103291)
    with ExitStack() as owners:
        graphs = []
        for arm, source in enumerate((control_source, candidate_source)):
            if source is not None and Path(source).suffix == ".cpp":
                model = Path(__file__).parents[1] / "models/qwen3_30b_w8a8"
                image = KernelCompiler("a1").compile_single_launch(
                    str(source), isa, extra_include_dirs=[str(model)])
                order = ("x", "residual", "weight", "bias", "y", "residual_out")
            else:
                build = factory(source, f"_post_norm_arm_{arm}")
                image, order = compile_modelslim_program(build(fused_add=True, rows=20),
                                                        tmp_path / f"arm-{arm}", Path(isa))
            owner = owners.enter_context(A1BorrowedLaunch(image))
            bindings = inputs | outputs[arm]
            call = owner.prepare(*(bindings[name] for name in order), blocks=8)
            call()
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            owners.callback(graph.reset)
            with torch.npu.graph(graph):
                call()
            graphs.append(graph)
        for case, magnitude in enumerate((0.0, 1e-4, 0.125, 1.0, 8.0, 64.0, 1.0, 1.0, 1.0)):
            values = {name: (torch.randn((20, 2048), generator=generator) * magnitude).half()
                      for name in ("x", "residual")}
            if case == 6:
                values["residual"] = -values["x"]
                values["residual"][:, ::64] += 2**-10
            elif case == 7:
                values["x"].zero_()
                values["residual"].zero_()
                for row in range(20):
                    values["x"][row, (row * 127) % 2048] = row + 1
            elif case == 8:
                values["x"][:, 63::64] = 32
                values["residual"][:, 64::64] = -16
            values["weight"] = (torch.randn((1, 2048), generator=generator) * 0.25 + 1).half()
            values["bias"] = (torch.randn((1, 2048), generator=generator) * 0.125).half()
            for name, value in values.items():
                inputs[name].copy_(value)
            native, _, skip = torch_npu.npu_add_rms_norm(inputs["x"], inputs["residual"],
                                                        inputs["weight"].flatten(), 1e-6)
            expected = {"y": (native + inputs["bias"]).cpu(), "residual_out": skip.cpu()}
            for arm, graph in enumerate(graphs):
                graph.replay()
                for name, value in expected.items():
                    actual = outputs[arm][name].cpu()
                    if not torch.equal(actual.view(torch.int16), value.view(torch.int16)):
                        fixture = tmp_path / f"norm-mismatch-{case}-{arm}-{name}.pt"
                        torch.save({"actual": actual, "expected": value, "inputs": values}, fixture)
                    assert torch.equal(actual.view(torch.int16), value.view(torch.int16)), (case, arm, name)
                    assert torch.all(storage[arm][name][20:].cpu() == -91), (case, arm, name, "guard")
        for graph in graphs:
            for _ in range(5):
                graph.replay()
        samples = [[], []]
        for sample in range(30):
            for arm in ((0, 1) if sample % 2 == 0 else (1, 0)):
                begin = torch.npu.Event(enable_timing=True)
                end = torch.npu.Event(enable_timing=True)
                begin.record()
                for _ in range(20):
                    graphs[arm].replay()
                end.record()
                end.synchronize()
                samples[arm].append(begin.elapsed_time(end) / 20)
        report = {"native_exact_cases": 9, "guard_rows": 12,
                  "samples_ms": samples, "median_ms": [statistics.median(v) for v in samples],
                  "replays_per_sample": 20, "alternating": True}
        (tmp_path / "post-norm-times.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
