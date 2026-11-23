# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""Native-bit expert combine/next-Norm under changing input graph replay."""

import importlib.util
import os
from contextlib import ExitStack
from pathlib import Path

import pytest
import torch


def test_a1_expert_combine_norm_dsl(tmp_path):
    torch_npu = pytest.importorskip("torch_npu")
    isa = os.environ.get("PTO_ISA_ROOT")
    if not isa:
        pytest.skip("PTO_ISA_ROOT is required")
    from simpler_setup import A1BorrowedLaunch
    from simpler_setup.elf_parser import allocated_section_fingerprints

    from models.qwen3_30b_w8a8.compile_modelslim_a1 import (
        compile_modelslim_program,
        compile_modelslim_source,
    )

    root = Path(__file__).resolve().parents[1]
    candidate_path = (
        Path(__file__).parent / "kernels/qwen30_combine_norm_dsl_candidate.py"
    )
    spec = importlib.util.spec_from_file_location(
        "combine_norm_candidate", candidate_path
    )
    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)
    dsl, order = compile_modelslim_program(
        candidate.resident_expert_combine_norm, tmp_path / "dsl", Path(isa)
    )
    expected_order = (
        "acc",
        "input_scale",
        "ids",
        "weight_scale",
        "route_weights",
        "expert_y",
        "y",
        "status",
        "residual",
        "gamma",
        "beta",
        "norm_y",
        "residual_out",
    )
    assert tuple(order) == expected_order
    native_build = tmp_path / "native"
    native = compile_modelslim_source(
        "a1_expert_combine_norm_batched.cpp", native_build, Path(isa)
    )
    selected = torch.load(
        root.parent / "artifacts/qwen30/dsl-direct-head-candidate/images.pt",
        map_location="cpu",
        weights_only=False,
    )["native"]["combine_norm"]
    assert allocated_section_fingerprints(native) == allocated_section_fingerprints(
        selected
    )
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    generator = torch.Generator().manual_seed(310016)
    source = {
        "acc": torch.randint(
            -(1 << 25), 1 << 25, (160, 2048), generator=generator, dtype=torch.int32
        ),
        "input_scale": torch.full((160, 16), 2.0**-8),
        "ids": torch.randint(0, 128, (20, 8), generator=generator, dtype=torch.int32),
        "weight_scale": torch.rand((128, 2048), generator=generator) * (2.0**-16),
        "route_weights": torch.softmax(
            torch.randn((20, 8), generator=generator), dim=-1
        )
        .half()
        .float(),
        "residual": (torch.randn((20, 2048), generator=generator) * 0.25).half(),
        "gamma": (torch.randn((1, 2048), generator=generator) * 0.125 + 1).half(),
        "beta": (torch.randn((1, 2048), generator=generator) * 0.125).half(),
    }
    tensors = {name: value.npu() for name, value in source.items()}
    outputs, guards, calls = {}, {}, []
    with ExitStack() as owners:
        for name, image in [("native", native), ("dsl", dsl)]:
            buffers = {}
            guards[name] = {}
            for key, shape in [
                ("expert_y", (160, 2048)),
                ("y", (20, 2048)),
                ("norm_y", (20, 2048)),
                ("residual_out", (20, 2048)),
            ]:
                guard = torch.full(
                    (shape[0] * shape[1] + 32,),
                    -19,
                    dtype=torch.float16,
                    device="npu:0",
                )
                buffers[key] = guard[16:-16].reshape(shape)
                guards[name][key] = guard
            buffers["status"] = torch.full(
                (8, 16), -7, dtype=torch.int32, device="npu:0"
            )
            owner = owners.enter_context(A1BorrowedLaunch(image, device_id=0))
            call = owner.prepare(
                *({**tensors, **buffers}[key] for key in order), blocks=8
            )
            calls.append(call)
            outputs[name] = buffers

        def check(invalid=False):
            torch.npu.synchronize()
            for key in ["expert_y", "y", "norm_y", "residual_out"]:
                native_output = outputs["native"][key].cpu().view(torch.int16)
                dsl_output = outputs["dsl"][key].cpu().view(torch.int16)
                assert torch.equal(native_output, dsl_output), (
                    key,
                    (native_output != dsl_output).sum().item(),
                )
                if invalid:
                    assert torch.equal(
                        outputs["dsl"][key].cpu(),
                        torch.full_like(outputs["dsl"][key].cpu(), -19),
                    )
                for variant in guards:
                    guard = guards[variant][key].cpu()
                    assert torch.equal(
                        guard[:16], torch.full((16,), -19, dtype=torch.float16)
                    )
                    assert torch.equal(
                        guard[-16:], torch.full((16,), -19, dtype=torch.float16)
                    )
            for variant in outputs:
                status = outputs[variant]["status"].cpu()
                assert torch.equal(
                    status[:, 0],
                    torch.full((8,), 2 if invalid else 0, dtype=torch.int32),
                )
                assert torch.equal(
                    status[:, 1:], torch.full((8, 15), -7, dtype=torch.int32)
                )
            for key, value in source.items():
                assert torch.equal(
                    tensors[key].cpu().view(torch.uint8), value.view(torch.uint8)
                ), key

        for call in calls:
            call()
        check()
        graph = torch.npu.NPUGraph()
        try:
            with torch.npu.graph(graph):
                for call in calls:
                    call()
            for state in range(4):
                source["acc"] = torch.randint(
                    -(1 << 26),
                    1 << 26,
                    (160, 2048),
                    generator=generator,
                    dtype=torch.int32,
                )
                if state == 1:
                    source["acc"].zero_()
                if state == 2:
                    source["acc"][:, :4] = torch.tensor(
                        [-2147483648, -16777217, 16777217, 2147483647],
                        dtype=torch.int32,
                    )
                    source["weight_scale"] *= 2.0**-6
                source["ids"] = torch.randint(
                    0, 128, (20, 8), generator=generator, dtype=torch.int32
                )
                source["route_weights"] = (
                    torch.softmax(torch.randn((20, 8), generator=generator), dim=-1)
                    .half()
                    .float()
                )
                source["residual"] = (
                    torch.randn((20, 2048), generator=generator) * (0.125 + state)
                ).half()
                for key, value in source.items():
                    tensors[key].copy_(value)
                graph.replay()
                check()
            for invalid_id in [-1, 128]:
                source["ids"][0, 0] = invalid_id
                tensors["ids"].copy_(source["ids"])
                for variant in guards:
                    for guard in guards[variant].values():
                        guard.fill_(-19)
                    outputs[variant]["status"].fill_(-7)
                graph.replay()
                check(invalid=True)
        finally:
            graph.reset()


def test_a1_expert_combine_dsl(tmp_path):
    torch_npu = pytest.importorskip('torch_npu')
    isa = os.environ.get('PTO_ISA_ROOT')
    if not isa:
        pytest.skip('PTO_ISA_ROOT is required')
    from simpler_setup import A1BorrowedLaunch
    from simpler_setup.elf_parser import allocated_section_fingerprints
    from models.qwen3_30b_w8a8.compile_modelslim_a1 import compile_modelslim_program, compile_modelslim_source
    root = Path(__file__).resolve().parents[1]
    from models.qwen3_30b_w8a8.combine_modelslim_a1 import resident_expert_combine
    dsl, order = compile_modelslim_program(resident_expert_combine, tmp_path / 'dsl', Path(isa))
    expected_order = ('acc', 'input_scale', 'ids', 'weight_scale', 'route_weights', 'expert_y', 'y', 'status')
    assert tuple(order) == expected_order
    native_build = tmp_path / 'native'
    native = compile_modelslim_source('a1_expert_combine_batched.cpp', native_build, Path(isa))
    selected = torch.load(root.parent / 'artifacts/qwen30/dsl-combine-norm-candidate/images.pt', map_location='cpu', weights_only=False)['moe']['combine']
    assert allocated_section_fingerprints(native) == allocated_section_fingerprints(selected)
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    generator = torch.Generator().manual_seed(310016)
    source = {'acc': torch.randint(-(1 << 25), 1 << 25, (160, 2048), generator=generator, dtype=torch.int32), 'input_scale': torch.full((160, 16), 2.0 ** (-8)), 'ids': torch.randint(0, 128, (20, 8), generator=generator, dtype=torch.int32), 'weight_scale': torch.rand((128, 2048), generator=generator) * 2.0 ** (-16), 'route_weights': torch.softmax(torch.randn((20, 8), generator=generator), dim=-1).half().float()}
    tensors = {name: value.npu() for name, value in source.items()}
    outputs, guards, calls = ({}, {}, [])
    with ExitStack() as owners:
        for name, image in [('native', native), ('dsl', dsl)]:
            buffers = {}
            guards[name] = {}
            for key, shape in [('expert_y', (160, 2048)), ('y', (20, 2048))]:
                guard = torch.full((shape[0] * shape[1] + 32,), -19, dtype=torch.float16, device='npu:0')
                buffers[key] = guard[16:-16].reshape(shape)
                guards[name][key] = guard
            buffers['status'] = torch.full((8, 16), -7, dtype=torch.int32, device='npu:0')
            owner = owners.enter_context(A1BorrowedLaunch(image, device_id=0))
            call = owner.prepare(*({**tensors, **buffers}[key] for key in order), blocks=8)
            calls.append(call)
            outputs[name] = buffers

        def check(invalid=False):
            torch.npu.synchronize()
            for key in ['expert_y', 'y']:
                native_output = outputs['native'][key].cpu().view(torch.int16)
                dsl_output = outputs['dsl'][key].cpu().view(torch.int16)
                assert torch.equal(native_output, dsl_output), (key, (native_output != dsl_output).sum().item())
                if invalid:
                    assert torch.equal(outputs['dsl'][key].cpu(), torch.full_like(outputs['dsl'][key].cpu(), -19))
                for variant in guards:
                    guard = guards[variant][key].cpu()
                    assert torch.equal(guard[:16], torch.full((16,), -19, dtype=torch.float16))
                    assert torch.equal(guard[-16:], torch.full((16,), -19, dtype=torch.float16))
            for variant in outputs:
                status = outputs[variant]['status'].cpu()
                assert torch.equal(status[:, 0], torch.full((8,), 2 if invalid else 0, dtype=torch.int32))
                assert torch.equal(status[:, 1:], torch.full((8, 15), -7, dtype=torch.int32))
            for key, value in source.items():
                assert torch.equal(tensors[key].cpu().view(torch.uint8), value.view(torch.uint8)), key
        for call in calls:
            call()
        check()
        graph = torch.npu.NPUGraph()
        try:
            with torch.npu.graph(graph):
                for call in calls:
                    call()
            for state in range(4):
                source['acc'] = torch.randint(-(1 << 26), 1 << 26, (160, 2048), generator=generator, dtype=torch.int32)
                if state == 1:
                    source['acc'].zero_()
                if state == 2:
                    source['acc'][:, :4] = torch.tensor([-2147483648, -16777217, 16777217, 2147483647], dtype=torch.int32)
                    source['weight_scale'] *= 2.0 ** (-6)
                source['ids'] = torch.randint(0, 128, (20, 8), generator=generator, dtype=torch.int32)
                source['route_weights'] = torch.softmax(torch.randn((20, 8), generator=generator), dim=-1).half().float()
                for key, value in source.items():
                    tensors[key].copy_(value)
                graph.replay()
                check()
            for invalid_id in [-1, 128]:
                source['ids'][0, 0] = invalid_id
                tensors['ids'].copy_(source['ids'])
                for variant in guards:
                    for guard in guards[variant].values():
                        guard.fill_(-19)
                    outputs[variant]['status'].fill_(-7)
                graph.replay()
                check(invalid=True)
        finally:
            graph.reset()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
