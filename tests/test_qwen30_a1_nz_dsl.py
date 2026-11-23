# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""Canonical NZ resident weights through DSL, PTOAS, CCE and changing ACL replay."""

import os
from pathlib import Path

import pypto.language as pl
import pytest
import torch


@pytest.mark.parametrize(
    "integer,rows,columns,k,full_n,full_k,column,inner,expert,accumulate,chunked",
    [
        (True, 16, 16, 32, 64, 96, 16, 32, 1, False, False),
        (True, 16, 256, 64, 768, 256, 256, 64, 1, False, False),
        (True, 32, 512, 128, 1024, 256, 512, 64, 0, True, False),
        (False, 16, 16, 16, 64, 96, 16, 32, 1, False, False),
        (False, 16, 128, 128, 256, 256, 128, 64, 1, False, False),
        (False, 32, 128, 64, 256, 128, 128, 32, 0, True, False),
        (True, 16, 256, 128, 768, 256, 256, 64, 1, False, True),
        (False, 32, 128, 128, 256, 256, 128, 64, 1, False, True),
    ],
)
def test_a1_canonical_nz_dsl(
    integer, rows, columns, k, full_n, full_k, column, inner, expert, accumulate, chunked, tmp_path
):
    torch_npu = pytest.importorskip("torch_npu")
    isa = os.environ.get("PTO_ISA_ROOT")
    if not isa:
        pytest.skip("PTO_ISA_ROOT is required")
    from simpler_setup import A1BorrowedLaunch

    from models.qwen3_30b_w8a8.compile_modelslim_a1 import compile_modelslim_program

    element = pl.INT8 if integer else pl.FP16
    result = pl.INT32 if integer else pl.FP32
    c0 = 32 if integer else 16

    @pl.jit.incore
    def nz_projection(
        lhs: pl.Tensor[[rows, full_k], element],
        weights: pl.Tensor[[2, full_n, full_k], element, pl.NZ],
        output: pl.InOut[pl.Tensor[[rows, columns], result]],
    ):
        left_source = pl.tile.load(lhs, [0, inner], [rows, k], target_memory=pl.Mem.Mat)
        weight_source = pl.tile.load(
            weights, [expert, column, inner], [1, columns, k], target_memory=pl.Mem.Mat
        )
        weight_matrix = pl.tile.reshape(weight_source, [columns, k])
        right_view = pl.tile.transpose_view(weight_matrix)
        left = pl.tile.extract(left_source, 0, 0, [rows, k], target_memory=pl.Mem.Left)
        right = pl.tile.extract(right_view, 0, 0, [k, columns], target_memory=pl.Mem.Right)
        accumulator = pl.tile.matmul(left, right)
        if accumulate:
            accumulator = pl.tile.matmul_acc(accumulator, left, right)
        for col in pl.range(0, columns, 16):
            fragment = pl.tile.extract(accumulator, 0, col, [rows, 16], target_memory=pl.Mem.Vec)
            pl.tile.store(fragment, [0, col], output)
        return output

    @pl.jit.incore
    def nz_projection_chunked(
        lhs: pl.Tensor[[rows, full_k], element],
        weights: pl.Tensor[[2, full_n, full_k], element, pl.NZ],
        output: pl.InOut[pl.Tensor[[rows, columns], result]],
    ):
        accumulator = pl.tile.create([rows, columns], result, target_memory=pl.Mem.Acc)
        for k0 in pl.range(inner, inner + k, 64):
            activation = pl.tile.load(lhs, [0, k0], [rows, 64], target_memory=pl.Mem.Mat)
            resident = pl.tile.load(
                weights, [expert, column, k0], [1, columns, 64], target_memory=pl.Mem.Mat
            )
            matrix = pl.tile.reshape(resident, [columns, 64])
            view = pl.tile.transpose_view(matrix)
            a = pl.tile.extract(activation, 0, 0, [rows, 64], target_memory=pl.Mem.Left)
            b = pl.tile.extract(view, 0, 0, [64, columns], target_memory=pl.Mem.Right)
            accumulator = pl.tile.matmul_acc(accumulator, a, b, init_cond=(k0 == inner))
        for col in pl.range(0, columns, 16):
            fragment = pl.tile.extract(accumulator, 0, col, [rows, 16], target_memory=pl.Mem.Vec)
            pl.tile.store(fragment, [0, col], output)
        return output

    assert not chunked or not accumulate
    program = nz_projection_chunked if chunked else nz_projection
    image, order = compile_modelslim_program(program, tmp_path, Path(isa))
    assert set(order) == {"lhs", "weights", "output"}
    sources = list(tmp_path.rglob("*.cpp"))
    assert any("pto::Layout::NZ" in source.read_text() for source in sources)
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    dtype = torch.int8 if integer else torch.float16
    result_dtype = torch.int32 if integer else torch.float32
    lhs = torch.zeros((rows, full_k), dtype=dtype, device="npu:0")
    weights = torch.zeros((2, full_k // c0, full_n // 16, 16, c0), dtype=dtype, device="npu:0")
    weights = torch_npu.npu_format_cast(weights, 2)
    assert torch_npu.get_npu_format(weights) == 2
    storage = torch.full((rows * columns + 32,), -70001, dtype=result_dtype, device="npu:0")
    output = storage[16:-16].view(rows, columns)
    tensors = {"lhs": lhs, "weights": weights, "output": output}
    generator = torch.Generator().manual_seed(310750 + rows + columns + k + expert)
    with A1BorrowedLaunch(image, device_id=0) as launch:
        call = launch.prepare(*(tensors[name] for name in order), blocks=1)
        call()
        torch.npu.synchronize()
        assert torch.equal(output.cpu(), torch.zeros((rows, columns), dtype=result_dtype))
        graph = torch.npu.NPUGraph()
        try:
            with torch.npu.graph(graph):
                call()
            for replay in range(5):
                x = torch.randint(-3, 4, (rows, full_k), generator=generator).to(dtype)
                w = torch.randint(-3, 4, (2, full_n, full_k), generator=generator).to(dtype)
                if integer and replay == 4:
                    x[0, inner:inner + 4] = torch.tensor([-128, 127, -1, 1], dtype=dtype)
                    w[expert, column, inner:inner + 4] = torch.tensor([127, -128, -1, 1], dtype=dtype)
                if not integer:
                    x = (x / 8).half()
                    w = (w / 8).half()
                packed = w.reshape(2, full_n // 16, 16, full_k // c0, c0).permute(0, 3, 1, 2, 4).contiguous()
                expected = (
                    x[:, inner:inner + k].double()
                    @ w[expert, column:column + columns, inner:inner + k].double().T
                    * (2 if accumulate else 1)
                ).to(result_dtype)
                lhs.copy_(x)
                weights.copy_(packed)
                storage.fill_(-70001)
                graph.replay()
                assert torch.equal(output.cpu(), expected), (integer, rows, columns, k, replay)
                guards = storage.cpu()
                assert torch.equal(guards[:16], torch.full((16,), -70001, dtype=result_dtype))
                assert torch.equal(guards[-16:], torch.full((16,), -70001, dtype=result_dtype))
        finally:
            graph.reset()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
