# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""Aligned A1 Vec extraction with source lifetime and bit-preserving replay checks."""

import os
from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("rows,width,first_row,result_rows", [(1, 2048, 0, 1), (4, 256, 1, 2)])
def test_a1_vec_extract_preserves_bits_and_source(dtype, rows, width, first_row, result_rows, tmp_path):
    torch_npu = pytest.importorskip("torch_npu")
    isa = os.environ.get("PTO_ISA_ROOT")
    if not isa:
        pytest.skip("PTO_ISA_ROOT is required")
    import pypto.language as pl
    from simpler_setup import A1BorrowedLaunch

    from models.qwen3_30b_w8a8.compile_modelslim_a1 import compile_modelslim_program

    element_type = pl.FP16 if dtype == torch.float16 else pl.FP32
    storage_type = torch.int16 if dtype == torch.float16 else torch.int32
    bits = 16 if dtype == torch.float16 else 32
    COLUMN_TILE = 64

    @pl.jit.incore
    def extract_windows(
        source: pl.Tensor[[rows, width], element_type],
        output: pl.Out[pl.Tensor[[result_rows, width], element_type]],
        preserved: pl.Out[pl.Tensor[[rows, width], element_type]],
    ):
        resident = pl.tile.load(source, [0, 0], [rows, width])
        for column in pl.range(0, width, COLUMN_TILE):
            window = pl.tile.extract(resident, first_row, column, [result_rows, COLUMN_TILE], target_memory=pl.Mem.Vec)
            pl.tile.store(window, [0, column], output)
        pl.tile.store(resident, [0, 0], preserved)
        return output, preserved

    image, order = compile_modelslim_program(extract_windows, tmp_path / "program", Path(isa))
    assert order == ("source", "output", "preserved")
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    source = torch.empty((rows, width), dtype=dtype, device="npu:0")
    storage = torch.full((result_rows * width + 32,), 0.25, dtype=dtype, device="npu:0")
    output = storage[16:-16].view(result_rows, width)
    preserved = torch.empty_like(source)
    generator = torch.Generator().manual_seed(310013 + bits + rows)
    with A1BorrowedLaunch(image, device_id=0) as owner:
        call = owner.prepare(source, output, preserved, blocks=1)
        source.zero_()
        call()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        try:
            with torch.npu.graph(graph):
                call()
            for replay in range(4):
                raw = torch.randint(-(2 ** (bits - 1)), 2 ** (bits - 1), (rows, width),
                                    dtype=storage_type, generator=generator)
                if replay == 0:
                    patterns = ([0, -32768, 31744, -1024, 32257, 1, -1, 15360]
                                if bits == 16 else
                                [0, -(2**31), 2139095040, -8388608, 2143289345, 1, -1, 1065353216])
                    raw.view(-1)[:8] = torch.tensor(patterns, dtype=storage_type)
                source.copy_(raw.view(dtype))
                assert torch.equal(source.cpu().view(storage_type), raw)
                storage.fill_(0.25)
                graph.replay()
                assert torch.equal(output.cpu().view(storage_type), raw[first_row:first_row + result_rows])
                assert torch.equal(preserved.cpu().view(storage_type), raw)
                guard = torch.full((16,), 0.25, dtype=dtype).view(storage_type)
                assert torch.equal(storage[:16].cpu().view(storage_type), guard)
                assert torch.equal(storage[-16:].cpu().view(storage_type), guard)
        finally:
            graph.reset()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])



@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("width", [64, 128, 2048])
@pytest.mark.parametrize("operation", ["sum", "max"])
def test_a1_scalar_only_row_reduction(dtype, width, operation, tmp_path):
    torch_npu = pytest.importorskip("torch_npu")
    isa = os.environ.get("PTO_ISA_ROOT")
    if not isa:
        pytest.skip("PTO_ISA_ROOT is required")
    import pypto.language as pl
    from simpler_setup import A1BorrowedLaunch

    from models.qwen3_30b_w8a8.compile_modelslim_a1 import compile_modelslim_program

    element_type = pl.FP16 if dtype == torch.float16 else pl.FP32
    OUTPUT_TILE = 16 if dtype == torch.float16 else 8
    IS_SUM = operation == "sum"

    @pl.jit.incore
    def scalar_reduction(
        source: pl.Tensor[[1, width], element_type],
        output: pl.Out[pl.Tensor[[1, OUTPUT_TILE], element_type]],
    ):
        values = pl.tile.load(source, [0, 0], [1, width])
        scratch = pl.tile.create([1, width], dtype=element_type)
        if IS_SUM:
            reduced = pl.tile.row_sum(values, scratch)
        else:
            reduced = pl.tile.row_max(values, scratch)
        scalar = pl.tile.read(reduced, [0, 0])
        result = pl.tile.full([1, OUTPUT_TILE], dtype=element_type, value=0.0)
        pl.tile.write(result, [0, 0], scalar)
        pl.tile.store(result, [0, 0], output)
        return output

    image, order = compile_modelslim_program(scalar_reduction, tmp_path / "program", Path(isa))
    assert order == ("source", "output")
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    source = torch.zeros((1, width), dtype=dtype, device="npu:0")
    storage = torch.full((OUTPUT_TILE + 32,), 0.25, dtype=dtype, device="npu:0")
    output = storage[16:-16].view(1, OUTPUT_TILE)
    with A1BorrowedLaunch(image, device_id=0) as owner:
        call = owner.prepare(source, output, blocks=1)
        call()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        try:
            with torch.npu.graph(graph):
                call()
            for value in [0.0, 1.0, -1.0, 0.5]:
                source.fill_(value)
                storage.fill_(0.25)
                graph.replay()
                expected = torch.zeros((1, OUTPUT_TILE), dtype=dtype)
                expected[0, 0] = width * value if operation == "sum" else value
                assert torch.equal(output.cpu(), expected)
                assert torch.equal(storage[:16].cpu(), torch.full((16,), 0.25, dtype=dtype))
                assert torch.equal(storage[-16:].cpu(), torch.full((16,), 0.25, dtype=dtype))
        finally:
            graph.reset()
