# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""Portable positive FP32 multiply/add case straddling an FP16 midpoint."""

import os
from pathlib import Path

import pytest
import torch


def test_attention_total_rounding_matches_native():
    isa = os.environ.get("PTO_ISA_ROOT")
    if not isa:
        pytest.skip("PTO_ISA_ROOT is required for the allocated 310P device")
    torch_npu = pytest.importorskip("torch_npu")
    from simpler_setup import A1BorrowedLaunch, KernelCompiler

    from models.qwen3_30b_w8a8 import moe_batched_modelslim_a1

    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    root = Path(moe_batched_modelslim_a1.__file__).parent
    image = KernelCompiler("a1").compile_single_launch(
        str(Path(__file__).with_name("kernels") / "a1_attention_total_rounding.cpp"), isa,
        extra_include_dirs=[str(root)])
    # Generated independently with seed3103, not copied from checkpoint weights.
    values = torch.tensor([1.261507272720337, 0.8293904066085815, 0.123151496052742], dtype=torch.float32)
    inputs = values[:, None].expand(3, 8).contiguous().npu()
    output = torch.zeros((2, 8), dtype=torch.float32, device="npu:0")
    expected = inputs[0] * inputs[1]
    expected = expected + inputs[2]
    with A1BorrowedLaunch(image) as owner:
        call = owner.prepare(inputs, output)
        call()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        try:
            with torch.npu.graph(graph):
                call()
            for _ in range(3):
                graph.replay()
                actual = output.cpu()
                assert torch.equal(actual[1], expected.cpu())
                assert torch.equal(actual[1].half(), expected.half().cpu())
                assert float(actual[1, 0]) == 1.16943359375
                assert float(actual[1, 0].half()) == 1.169921875
            print("ROUNDING_CONTROL", actual[:, 0].tolist(), actual[:, 0].half().tolist(), flush=True)
        finally:
            graph.reset()
