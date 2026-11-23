# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""Compare full B20 resident head DSL with the selected direct C++ bodies."""

import importlib.util
import os
from contextlib import ExitStack
from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize("heads,kind", [(32, "q"), (4, "k")])
def test_b20_resident_head_dsl_graph(heads, kind, tmp_path):
    torch_npu = pytest.importorskip("torch_npu")
    from simpler_setup import A1BorrowedLaunch, KernelCompiler

    from models.qwen3_30b_w8a8.compile_modelslim_a1 import compile_modelslim_program

    isa = Path(os.environ["PTO_ISA_ROOT"])
    candidate = (
        Path(__file__).with_name("kernels") / "qwen30_direct_head_dsl_candidate.py"
    )
    spec = importlib.util.spec_from_file_location(
        "qwen30_direct_head_candidate", candidate
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    program = module.make_a1_resident_head_norm_rope(heads)
    image, order = compile_modelslim_program(program, tmp_path / "dsl", isa)
    assert order == ("x", "weight", "cos_sin", "norm", "rotated")
    source = (
        Path(__file__).parents[1]
        / "models/qwen3_30b_w8a8"
        / f"a1_head_norm_rope_direct_{kind}.cpp"
    )
    native = KernelCompiler("a1").compile_single_launch(str(source), str(isa))
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    rows = 20 * max(heads, 8)
    inputs = {
        "x": torch.zeros((rows, 128), dtype=torch.float16, device="npu:0"),
        "weight": torch.zeros((1, 128), dtype=torch.float16, device="npu:0"),
        "cos_sin": torch.zeros((20, 128), dtype=torch.float16, device="npu:0"),
    }
    generator = torch.Generator().manual_seed(310014 + heads)
    with ExitStack() as owners:
        graphs, buffers, guards = {}, {}, {}
        for name, body in [("dsl", image), ("selected_cpp", native)]:
            storage = {
                label: torch.full(
                    (rows * 128 + 32,), 0.25, dtype=torch.float16, device="npu:0"
                )
                for label in ["norm", "rotated"]
            }
            values = {
                **inputs,
                **{
                    label: value[16:-16].view(rows, 128)
                    for label, value in storage.items()
                },
            }
            owner = owners.enter_context(A1BorrowedLaunch(body, device_id=0))
            call = owner.prepare(*(values[label] for label in order), blocks=8)
            call()
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            owners.callback(graph.reset)
            with torch.npu.graph(graph):
                call()
            graphs[name], buffers[name], guards[name] = graph, values, storage
        for magnitude in [0.125, 1.0, 8.0, 0.0]:
            raw = (
                torch.randn((20, max(heads, 8), 128), generator=generator) * magnitude
            ).half()
            if heads == 4:
                raw[:, 4:] = 0
            weight = (torch.randn((1, 128), generator=generator) * 0.25 + 1).half()
            phase = (torch.randn((20, 128), generator=generator) * 0.25).half()
            inputs["x"].copy_(raw.view(rows, 128))
            inputs["weight"].copy_(weight)
            inputs["cos_sin"].copy_(phase)
            for storage in guards.values():
                for value in storage.values():
                    value.fill_(0.25)
            for graph in graphs.values():
                graph.replay()
            for label in ["norm", "rotated"]:
                actual = buffers["dsl"][label].cpu().view(torch.int16)
                expected = buffers["selected_cpp"][label].cpu().view(torch.int16)
                if not torch.equal(actual, expected):
                    torch.save(
                        {
                            "actual_bits": actual,
                            "selected_bits": expected,
                            "input": raw,
                            "weight": weight,
                            "phase": phase,
                        },
                        tmp_path / f"mismatch-{kind}-{label}-{magnitude}.pt",
                    )
                assert torch.equal(actual, expected), (kind, label, magnitude)
                for storage in guards.values():
                    assert torch.equal(
                        storage[label][:16].cpu(),
                        torch.full((16,), 0.25, dtype=torch.float16),
                    )
                    assert torch.equal(
                        storage[label][-16:].cpu(),
                        torch.full((16,), 0.25, dtype=torch.float16),
                    )
            assert torch.equal(
                inputs["x"].cpu().view(torch.int16),
                raw.view(rows, 128).view(torch.int16),
            )
