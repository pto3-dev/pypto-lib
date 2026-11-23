# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""Formal public-DSL SwiGLU versus the retained A1 grouped kernel."""

import os
from contextlib import ExitStack
from pathlib import Path

import pytest
import torch


def test_a1_swiglu_dsl_exact_ordinary_and_graph(tmp_path):
    torch_npu = pytest.importorskip("torch_npu")
    isa = os.environ.get("PTO_ISA_ROOT")
    if not isa:
        pytest.skip("PTO_ISA_ROOT is required for an allocated A1 device")
    from models.qwen3_30b_w8a8.compile_modelslim_a1 import (
        compile_modelslim_program,
        compile_modelslim_source,
    )
    from models.qwen3_30b_w8a8.swiglu_quant_batched_a1 import swiglu_quant_b20
    from simpler_setup import A1BorrowedLaunch

    order = ("input", "input_scale", "ids", "weight_scale", "quant", "scale", "status")
    candidate, labels = compile_modelslim_program(
        swiglu_quant_b20, tmp_path / "dsl", Path(isa)
    )
    assert tuple(labels) == order
    reference = compile_modelslim_source(
        "a1_swiglu_quant_batched_grouped_dequant.cpp", tmp_path / "cpp", Path(isa)
    )
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    generator = torch.Generator().manual_seed(310027)
    cpu = {
        "input": (torch.randn((160, 1536), generator=generator) * 3).half(),
        "input_scale": torch.rand((20, 16), generator=generator) + 0.1,
        "ids": torch.randint(0, 128, (20, 8), generator=generator, dtype=torch.int32),
        "weight_scale": torch.randint(
            -100, 100, (128, 3072), generator=generator, dtype=torch.int32
        ),
    }
    inputs = {name: value.npu() for name, value in cpu.items()}
    outputs, storages, calls, graphs = {}, {}, {}, {}
    with ExitStack() as owners:
        for variant, image in [("cpp", reference), ("dsl", candidate)]:
            owner = owners.enter_context(A1BorrowedLaunch(image, device_id=0))
            outputs[variant], storages[variant] = {}, {}
            for name, shape, dtype, sentinel in [
                ("quant", (160, 768), torch.int8, -67),
                ("scale", (160, 16), torch.float32, -19),
                ("status", (160, 16), torch.int32, -7),
            ]:
                storage = torch.full(
                    (torch.Size(shape).numel() + 64,),
                    sentinel,
                    dtype=dtype,
                    device="npu:0",
                )
                storages[variant][name] = (storage, sentinel)
                outputs[variant][name] = storage[32:-32].reshape(shape)
            arguments = inputs | outputs[variant]
            call = owner.prepare(*(arguments[name] for name in order), blocks=8)
            calls[variant] = call
            call()
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            owners.callback(graph.reset)
            with torch.npu.graph(graph):
                call()
            graphs[variant] = graph

        for state in range(4):
            values = {name: value.clone() for name, value in cpu.items()}
            if state == 1:
                values["input"].zero_()
            if state >= 2:
                for lane in range(8):
                    values["ids"][(lane * 3 + state) % 20, lane] = (
                        -1 if lane % 2 else 128
                    )
            for name, value in values.items():
                inputs[name].copy_(value)
            for replay in [False, True]:
                for variant in calls:
                    for storage, sentinel in storages[variant].values():
                        storage.fill_(sentinel)
                    graphs[variant].replay() if replay else calls[variant]()
                torch.npu.synchronize()
                actual = {
                    variant: {name: value.cpu() for name, value in result.items()}
                    for variant, result in outputs.items()
                }
                for name in ["quant", "scale", "status"]:
                    assert torch.equal(
                        actual["cpp"][name].view(torch.uint8),
                        actual["dsl"][name].view(torch.uint8),
                    )
                for variant, result in actual.items():
                    for lane in range(8):
                        invalid = (
                            (values["ids"][:, lane] < 0)
                            | (values["ids"][:, lane] >= 128)
                        ).nonzero()
                        first = int(invalid[0, 0]) if invalid.numel() else 20
                        assert torch.all(result["status"][lane : first * 8 : 8, 0] == 0)
                        if first < 20:
                            assert result["status"][first * 8 + lane, 0] == 2
                            assert torch.all(
                                result["quant"][first * 8 + lane :: 8] == -67
                            )
                            assert torch.all(
                                result["scale"][first * 8 + lane :: 8, 0] == -19
                            )
                    assert torch.all(result["scale"][:, 1:] == -19)
                    assert torch.all(result["status"][:, 1:] == -7)
                    for storage, sentinel in storages[variant].values():
                        view = storage.cpu()
                        assert torch.all(view[:32] == sentinel) and torch.all(
                            view[-32:] == sentinel
                        )
                    if state == 1:
                        assert torch.all(result["quant"] == 0) and torch.all(
                            result["scale"][:, 0] == 0
                        )
                for name, value in values.items():
                    assert torch.equal(
                        inputs[name].cpu().view(torch.uint8), value.view(torch.uint8)
                    )
