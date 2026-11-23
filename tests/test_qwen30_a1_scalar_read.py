# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""A1 scalar metadata reads after device producers and changing graph replays."""

import os
from pathlib import Path

import pypto.language as pl
import pytest
import torch


@pytest.mark.parametrize("dtype", [torch.int32, torch.float32, torch.float16])
def test_a1_scalar_read_after_device_producer(dtype, tmp_path):
    torch_npu = pytest.importorskip("torch_npu")
    isa = os.environ.get("PTO_ISA_ROOT")
    if not isa:
        pytest.skip("PTO_ISA_ROOT is required")
    from simpler_setup import A1BorrowedLaunch

    from models.qwen3_30b_w8a8.compile_modelslim_a1 import compile_modelslim_program

    element = {torch.int32: pl.INT32, torch.float32: pl.FP32, torch.float16: pl.FP16}[dtype]

    @pl.jit.incore
    def indexed_read(
        indices: pl.Tensor[[8, 16], pl.INT32],
        values: pl.Tensor[[32, 16], element],
        output: pl.InOut[pl.Tensor[[8, 32], element]],
    ):
        block = pl.tile.get_block_idx()
        index = pl.tensor.read(indices, [block, 0])
        value = pl.tensor.read(values, [index, 0])
        pl.tensor.write(output, [block, 0], value)
        return output

    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    image, order = compile_modelslim_program(indexed_read, tmp_path, Path(isa))
    assert set(order) == {"indices", "values", "output"}
    sources = list(tmp_path.rglob("*.cpp"))
    assert any("pto::A1LoadScalar" in p.read_text() for p in sources)
    indices = torch.zeros((8, 16), device="npu:0", dtype=torch.int32)
    values = torch.zeros((32, 16), device="npu:0", dtype=dtype)
    # Scalar stores own complete cache lines; this test isolates scalar loads.
    output = torch.full((8, 32), -19, device="npu:0", dtype=dtype)
    tensors = {"indices": indices, "values": values, "output": output}
    base = torch.arange(512).reshape(32, 16).to(dtype)
    with A1BorrowedLaunch(image, device_id=0) as launch:
        call = launch.prepare(*(tensors[name] for name in order), blocks=8)
        values.copy_(base)
        call()
        torch.npu.synchronize()
        assert torch.equal(output[:, 0].cpu(), base[0, 0].expand(8))
        graph = torch.npu.NPUGraph()
        try:
            with torch.npu.graph(graph):
                for _ in range(4):
                    values.add_(1)
                    indices.add_(1)
                    call()
            for replay in range(16):
                metadata = torch.zeros((8, 16), dtype=torch.int32)
                metadata[:, 0] = (torch.arange(8) * 3 + replay) % 24
                original = base + replay * 2
                indices.copy_(metadata)
                values.copy_(original)
                output.fill_(-19)
                graph.replay()
                expected = original[(metadata[:, 0] + 4).long(), 0] + 4
                actual = output.cpu()
                assert torch.equal(actual[:, 0], expected), (dtype, replay, actual[:, 0], expected)
                assert torch.equal(actual[:, 1:], torch.full((8, 31), -19, dtype=dtype))
        finally:
            graph.reset()
