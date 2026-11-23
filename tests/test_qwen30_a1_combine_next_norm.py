# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""Exact graph replay of Combine and the next layer's biased AddRMSNorm."""

import json
import os
import statistics
from contextlib import ExitStack
from pathlib import Path

import pytest
import torch


def test_combine_next_norm_graph(tmp_path):
    """Retiling must preserve ordered expert sums and the next biased AddRMSNorm."""
    import torch_npu
    from simpler_setup import A1BorrowedLaunch
    from models.qwen3_30b_w8a8.compile_modelslim_a1 import compile_modelslim_program, compile_modelslim_source
    from models.qwen3_30b_w8a8.norm_modelslim_a1 import make_a1_modelslim_norm

    isa = os.environ.get("PTO_ISA_ROOT")
    if not isa:
        pytest.skip("PTO_ISA_ROOT is required")
    torch.set_num_threads(2)
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    images = [compile_modelslim_source(name, tmp_path / str(index), Path(isa)) for index, name in enumerate((
        "a1_expert_combine_batched.cpp", "a1_expert_combine_norm_batched.cpp"))]
    norm_image, norm_order = compile_modelslim_program(make_a1_modelslim_norm(fused_add=True, rows=20),
                                                     tmp_path / "norm", Path(isa))
    generator = torch.Generator().manual_seed(3103278)
    acc = torch.zeros((160, 2048), dtype=torch.int32, device="npu:0")
    activation = torch.rand((160, 16), generator=generator).mul(0.001).add(0.0001).npu()
    ids = (torch.arange(160).reshape(20, 8) % 128).int().npu()
    channel = torch.rand((128, 2048), generator=generator).mul(0.002).add(0.0001).npu()
    route_weight = torch.softmax(torch.randn((20, 8), generator=generator), dim=-1).npu()
    skip = torch.zeros((20, 2048), dtype=torch.float16, device="npu:0")
    gamma = torch.rand((1, 2048), generator=generator).add(0.5).half().npu()
    beta = torch.randn((1, 2048), generator=generator).half().npu()
    expert = [torch.zeros((160, 2048), dtype=torch.float16, device="npu:0") for _ in range(2)]
    output = [torch.zeros_like(skip) for _ in range(2)]
    norm = [torch.zeros_like(skip) for _ in range(2)]
    skip_out = [torch.zeros_like(skip) for _ in range(2)]
    status = [torch.full((8, 16), -7, dtype=torch.int32, device="npu:0") for _ in range(2)]
    with ExitStack() as owners:
        graphs = []
        for arm in range(2):
            owner = owners.enter_context(A1BorrowedLaunch(images[arm]))
            tensors = [acc, activation, ids, channel, route_weight, expert[arm], output[arm], status[arm]]
            if arm == 1:
                tensors += [skip, gamma, beta, norm[arm], skip_out[arm]]
            calls = [owner.prepare(*tensors, blocks=8)]
            if arm == 0:
                norm_owner = owners.enter_context(A1BorrowedLaunch(norm_image))
                bindings = {"x": output[arm], "residual": skip, "weight": gamma, "bias": beta,
                            "y": norm[arm], "residual_out": skip_out[arm]}
                calls.append(norm_owner.prepare(*(bindings[name] for name in norm_order), blocks=8))
            for call in calls:
                call()
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            owners.callback(graph.reset)
            with torch.npu.graph(graph):
                for call in calls:
                    call()
            graphs.append(graph)
        for case in range(4):
            values = torch.randint(-1000000, 1000000, (160, 2048), dtype=torch.int32, generator=generator)
            if case == 0:
                values.zero_()
            acc.copy_(values)
            skip.copy_(torch.randn((20, 2048), generator=generator).half())
            for graph in graphs:
                graph.replay()
            for left, right in zip((expert[0], output[0], norm[0], skip_out[0]),
                                   (expert[1], output[1], norm[1], skip_out[1]), strict=True):
                assert torch.equal(left.cpu(), right.cpu()), (case, torch.count_nonzero(left.cpu() != right.cpu()))
            scales_cpu = channel.cpu()[ids.cpu().flatten().long()]
            dequant = (values.float() * scales_cpu * activation.cpu()[:, :1]).half()
            total = torch.zeros((20, 2048))
            for route in range(8):
                total += dequant.reshape(20, 8, 2048)[:, route].float() * route_weight.cpu()[:, route:route + 1]
            assert torch.equal(expert[1].cpu(), dequant)
            assert torch.equal(output[1].cpu(), total.half())
            assert all(not torch.count_nonzero(value.cpu()[:, 0]) for value in status)
        times = []
        for graph in graphs:
            for _ in range(5):
                graph.replay()
            samples = []
            for _ in range(30):
                begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                begin.record()
                graph.replay()
                end.record()
                end.synchronize()
                samples.append(begin.elapsed_time(end))
            times.append(samples)
        (tmp_path / "combine-norm-times.json").write_text(json.dumps({
            "control_ms": times[0], "fused_ms": times[1], "exact": True,
            "control_median_ms": statistics.median(times[0]),
            "candidate_median_ms": statistics.median(times[1])}, indent=2))
