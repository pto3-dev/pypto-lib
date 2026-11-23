# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""Exact INT32 bias addition through A1 DSL and changing graph inputs."""

import os
from pathlib import Path

import pypto.language as pl
import pytest
import torch


@pytest.mark.parametrize("rows,cols", [(1, 8), (4, 16), (20, 16), (20, 64), (20, 128), (1, 256)])
def test_a1_integer_bias_dsl(rows, cols, tmp_path):
    torch_npu = pytest.importorskip("torch_npu")
    isa = os.environ.get("PTO_ISA_ROOT")
    if not isa:
        pytest.skip("PTO_ISA_ROOT is required")
    from models.qwen3_30b_w8a8.compile_modelslim_a1 import compile_modelslim_program
    from simpler_setup import A1BorrowedLaunch

    @pl.jit.incore
    def add_bias(
        src: pl.Tensor[[rows, cols], pl.INT32],
        bias: pl.Tensor[[1, cols], pl.INT32],
        output: pl.InOut[pl.Tensor[[rows, cols], pl.INT32]],
    ):
        values = pl.tile.load(src, [0, 0], [rows, cols])
        channels = pl.tile.load(bias, [0, 0], [1, cols])
        expanded = pl.tile.col_expand(values, channels)
        pl.tile.store(pl.tile.add(values, expanded), [0, 0], output)
        return output

    image, order = compile_modelslim_program(add_bias, tmp_path, Path(isa))
    generated = "".join(p.read_text() for p in tmp_path.rglob("*.pto"))
    assert "pto.tadd" in generated and "pto.tcolexpand" in generated
    assert "pto.tcvt" not in generated
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    src = torch.zeros((rows, cols), dtype=torch.int32, device="npu:0")
    bias = torch.zeros((1, cols), dtype=torch.int32, device="npu:0")
    storage = torch.full((rows * cols + 32,), -70001, dtype=torch.int32, device="npu:0")
    output = storage[16:-16].view(rows, cols)
    tensors = {"src": src, "bias": bias, "output": output}
    limits = torch.tensor([0, 1, -1, 2**24 + 1, -(2**24 + 1), 2**31 - 1, -(2**31), 127],
                          dtype=torch.int32)
    generator = torch.Generator().manual_seed(310010 + rows + cols)
    with A1BorrowedLaunch(image, device_id=0) as launch:
        call = launch.prepare(*(tensors[name] for name in order), blocks=1)
        call()
        torch.npu.synchronize()
        assert torch.equal(output.cpu(), torch.zeros_like(output.cpu()))
        graph = torch.npu.NPUGraph()
        try:
            with torch.npu.graph(graph):
                call()
            for replay in range(5):
                x = torch.randint(-(2**31), 2**31, (rows, cols), generator=generator, dtype=torch.int64)
                b = torch.randint(-(2**31), 2**31, (1, cols), generator=generator, dtype=torch.int64)
                x[0, :8] = limits.long()
                b[0, :8] = torch.roll(limits, replay).long()
                expected = (x + b).to(torch.int32)
                src.copy_(x.to(torch.int32))
                bias.copy_(b.to(torch.int32))
                storage.fill_(-70001)
                graph.replay()
                assert torch.equal(output.cpu(), expected), (rows, cols, replay)
                guard = torch.full((16,), -70001, dtype=torch.int32)
                assert torch.equal(storage[:16].cpu(), guard)
                assert torch.equal(storage[-16:].cpu(), guard)
        finally:
            graph.reset()


@pytest.mark.parametrize("rows,k", [(16, 2048), (4, 2048), (16, 4096), (4, 4096)])
def test_a1_nz_matmul_integer_bias_dsl(rows, k, tmp_path):
    torch_npu = pytest.importorskip("torch_npu")
    isa = os.environ.get("PTO_ISA_ROOT")
    if not isa:
        pytest.skip("PTO_ISA_ROOT is required")
    from models.qwen3_30b_w8a8.compile_modelslim_a1 import compile_modelslim_program
    from simpler_setup import A1BorrowedLaunch

    columns = 256

    @pl.jit.incore
    def projection_bias(
        src: pl.Tensor[[rows, k], pl.INT8],
        weights: pl.Tensor[[1, columns, k], pl.INT8, pl.NZ],
        bias: pl.Tensor[[1, columns], pl.INT32],
        output: pl.InOut[pl.Tensor[[rows, columns], pl.INT32]],
    ):
        activation = pl.tile.load(src, [0, 0], [16, 64], valid_shape=[rows, 64], target_memory=pl.Mem.Mat)
        resident = pl.tile.load(weights, [0, 0, 0], [1, columns, 64], target_memory=pl.Mem.Mat)
        matrix = pl.tile.reshape(resident, [columns, 64])
        right_view = pl.tile.transpose_view(matrix)
        left = pl.tile.extract(activation, 0, 0, [16, 64], target_memory=pl.Mem.Left)
        right = pl.tile.extract(right_view, 0, 0, [64, columns], target_memory=pl.Mem.Right)
        accumulator = pl.tile.matmul(left, right)
        for k0 in pl.range(64, k, 64):
            next_activation = pl.tile.load(
                src, [0, k0], [16, 64], valid_shape=[rows, 64], target_memory=pl.Mem.Mat
            )
            next_resident = pl.tile.load(weights, [0, 0, k0], [1, columns, 64], target_memory=pl.Mem.Mat)
            next_matrix = pl.tile.reshape(next_resident, [columns, 64])
            next_view = pl.tile.transpose_view(next_matrix)
            next_left = pl.tile.extract(next_activation, 0, 0, [16, 64], target_memory=pl.Mem.Left)
            next_right = pl.tile.extract(next_view, 0, 0, [64, columns], target_memory=pl.Mem.Right)
            accumulator = pl.tile.matmul_acc(accumulator, next_left, next_right)
        for col in pl.range(0, columns, 16):
            values = pl.tile.extract(accumulator, 0, col, [16, 16], target_memory=pl.Mem.Vec)
            channels = pl.tile.load(bias, [0, col], [1, 16])
            expanded = pl.tile.col_expand(values, channels)
            pl.tile.store(pl.tile.add(values, expanded), [0, col], output)
        return output

    image, order = compile_modelslim_program(projection_bias, tmp_path, Path(isa))
    generated = "".join(p.read_text() for p in tmp_path.rglob("*.pto"))
    assert "pto.tmatmul.acc" in generated and "pto.tcolexpand" in generated
    assert "pto.tadd" in generated and "pto.tcvt" not in generated
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    src = torch.zeros((rows, k), dtype=torch.int8, device="npu:0")
    weights = torch.zeros((1, k // 32, columns // 16, 16, 32), dtype=torch.int8, device="npu:0")
    weights = torch_npu.npu_format_cast(weights, 2)
    assert torch_npu.get_npu_format(weights) == 2
    bias = torch.zeros((1, columns), dtype=torch.int32, device="npu:0")
    storage = torch.full((rows * columns + 32,), -70001, dtype=torch.int32, device="npu:0")
    output = storage[16:-16].view(rows, columns)
    tensors = {"src": src, "weights": weights, "bias": bias, "output": output}
    generator = torch.Generator().manual_seed(310010 + rows + k)
    with A1BorrowedLaunch(image, device_id=0) as launch:
        call = launch.prepare(*(tensors[name] for name in order), blocks=1)
        call()
        torch.npu.synchronize()
        assert torch.equal(output.cpu(), torch.zeros((rows, columns), dtype=torch.int32))
        graph = torch.npu.NPUGraph()
        try:
            with torch.npu.graph(graph):
                call()
            for replay in range(5):
                x = torch.randint(-128, 128, (rows, k), generator=generator, dtype=torch.int8)
                w = torch.randint(-128, 128, (columns, k), generator=generator, dtype=torch.int8)
                b = torch.randint(-(2**31), 2**31, (1, columns), generator=generator, dtype=torch.int64)
                x[0].fill_(127)
                w[0].fill_(127)
                b[0, :4] = torch.tensor([1, 2**24 + 1, 2**31 - 1, -(2**31)], dtype=torch.int64)
                expected = (x.long() @ w.long().T + b).to(torch.int32)
                assert (x[0].long() @ w[0].long()).item() > 2**24
                packed = w.reshape(columns // 16, 16, k // 32, 32).permute(2, 0, 1, 3).unsqueeze(0).contiguous()
                src.copy_(x)
                weights.copy_(packed)
                bias.copy_(b.to(torch.int32))
                storage.fill_(-70001)
                graph.replay()
                assert torch.equal(output.cpu(), expected), (rows, k, replay)
                guard = torch.full((16,), -70001, dtype=torch.int32)
                assert torch.equal(storage[:16].cpu(), guard)
                assert torch.equal(storage[-16:].cpu(), guard)
        finally:
            graph.reset()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
