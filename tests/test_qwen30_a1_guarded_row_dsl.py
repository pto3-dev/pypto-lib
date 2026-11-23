# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""Guarded indirect DMA: device-produced indices, graph replay and invalid IDs."""

import importlib.util
import os
from pathlib import Path

import pytest
import torch


def test_a1_guarded_expert_row_dma(tmp_path):
    torch_npu = pytest.importorskip("torch_npu")
    isa = os.environ.get("PTO_ISA_ROOT")
    if not isa:
        pytest.skip("PTO_ISA_ROOT is required")
    from simpler_setup import A1BorrowedLaunch

    from models.qwen3_30b_w8a8.compile_modelslim_a1 import compile_modelslim_program

    path = Path(__file__).parent / "kernels/qwen30_guarded_row_dsl_candidate.py"
    spec = importlib.util.spec_from_file_location("guarded_row_candidate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    image, order = compile_modelslim_program(
        module.guarded_expert_row, tmp_path, Path(isa)
    )
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    indices = torch.zeros((8, 16), dtype=torch.int32, device="npu:0")
    source = torch.arange(128 * 64, dtype=torch.float32).reshape(128, 64)
    table = source.npu()
    guarded = torch.full((8 * 64 + 32,), -19.0, device="npu:0")
    output = guarded[16:-16].reshape(8, 64)
    tensors = {"indices": indices, "table": table, "output": output}
    with A1BorrowedLaunch(image, device_id=0) as launch:
        call = launch.prepare(*(tensors[name] for name in order), blocks=8)
        call()
        torch.npu.synchronize()
        assert torch.equal(output.cpu(), source[0].expand(8, -1))
        graph = torch.npu.NPUGraph()
        try:
            with torch.npu.graph(graph):
                indices.add_(1)
                table.add_(1)
                call()
            for values in [
                [-1, 0, 1, 63, 126, 127, 128, 129],
                [127, -2, 64, 0, 15, 31, 95, 126],
                [-2147483648, 2147483646, -8, -1, 127, 128, 7, 63],
                [3, 127, 0, 64, 18, 42, 91, 111],
            ]:
                metadata = torch.zeros((8, 16), dtype=torch.int32)
                metadata[:, 0] = torch.tensor(values)
                indices.copy_(metadata)
                table.copy_(source)
                guarded.fill_(-19)
                graph.replay()
                actual = guarded.cpu()
                expected = torch.full((8, 64), -19.0)
                for row, value in enumerate(values):
                    expert = value + 1
                    if 0 <= expert < 128:
                        expected[row] = source[expert] + 1
                assert torch.equal(actual[16:-16].reshape(8, 64), expected)
                assert torch.equal(actual[:16], torch.full((16,), -19.0))
                assert torch.equal(actual[-16:], torch.full((16,), -19.0))
                assert torch.equal(table.cpu(), source + 1)
        finally:
            graph.reset()
