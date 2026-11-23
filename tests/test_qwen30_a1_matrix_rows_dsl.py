# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""A1 matrix valid rows through DSL, NZ weights and changing graph replay."""

import os
from pathlib import Path

import pypto.language as pl
import pytest
import torch


@pytest.mark.parametrize(
    "integer,rows,physical_rows,columns,k,accumulate,chunked",
    [
        (True, 20, 32, 256, 64, False, False),
        (True, 4, 16, 16, 32, True, False),
        (True, 31, 32, 512, 128, True, False),
        (False, 20, 32, 128, 64, False, False),
        (False, 1, 16, 16, 16, True, False),
        (False, 17, 32, 128, 128, True, False),
        (True, 20, 32, 256, 128, False, True),
        (False, 20, 32, 128, 128, False, True),
    ],
)
def test_a1_matrix_valid_rows_dsl(integer, rows, physical_rows, columns, k, accumulate, chunked, tmp_path):
    torch_npu = pytest.importorskip("torch_npu")
    isa = os.environ.get("PTO_ISA_ROOT")
    if not isa:
        pytest.skip("PTO_ISA_ROOT is required")
    from simpler_setup import A1BorrowedLaunch

    from models.qwen3_30b_w8a8.compile_modelslim_a1 import compile_modelslim_program

    element = pl.INT8 if integer else pl.FP16
    result = pl.INT32 if integer else pl.FP32
    c0 = 32 if integer else 16
    full_n, full_k, column, inner = 1024, 256, 256, 64
    step = 64 if chunked else k

    @pl.jit.incore
    def projection(
        lhs: pl.Tensor[[rows, full_k], element],
        weights: pl.Tensor[[2, full_n, full_k], element, pl.NZ],
        output: pl.InOut[pl.Tensor[[rows, columns], result]],
    ):
        source = pl.tile.load(
            lhs, [0, inner], [physical_rows, step], valid_shape=[rows, step], target_memory=pl.Mem.Mat
        )
        resident = pl.tile.load(
            weights, [1, column, inner], [1, columns, step], target_memory=pl.Mem.Mat
        )
        matrix = pl.tile.reshape(resident, [columns, step])
        right_view = pl.tile.transpose_view(matrix)
        left = pl.tile.extract(source, 0, 0, [physical_rows, step], target_memory=pl.Mem.Left)
        right = pl.tile.extract(right_view, 0, 0, [step, columns], target_memory=pl.Mem.Right)
        accumulator = pl.tile.matmul(left, right)
        if accumulate:
            accumulator = pl.tile.matmul_acc(accumulator, left, right)
        if chunked:
            for k0 in pl.range(inner + step, inner + k, step):
                activation = pl.tile.load(
                    lhs, [0, k0], [physical_rows, step], valid_shape=[rows, step], target_memory=pl.Mem.Mat
                )
                resident_next = pl.tile.load(
                    weights, [1, column, k0], [1, columns, step], target_memory=pl.Mem.Mat
                )
                matrix_next = pl.tile.reshape(resident_next, [columns, step])
                view_next = pl.tile.transpose_view(matrix_next)
                left_next = pl.tile.extract(activation, 0, 0, [physical_rows, step], target_memory=pl.Mem.Left)
                right_next = pl.tile.extract(view_next, 0, 0, [step, columns], target_memory=pl.Mem.Right)
                accumulator = pl.tile.matmul_acc(accumulator, left_next, right_next)
        for col in pl.range(0, columns, 16):
            fragment = pl.tile.extract(accumulator, 0, col, [physical_rows, 16], target_memory=pl.Mem.Vec)
            pl.tile.store(fragment, [0, col], output)
        return output

    image, order = compile_modelslim_program(projection, tmp_path, Path(isa))
    assert set(order) == {"lhs", "weights", "output"}
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
    generator = torch.Generator().manual_seed(310820 + rows + columns + k)
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
                    w[1, column, inner:inner + 4] = torch.tensor([127, -128, -1, 1], dtype=dtype)
                if not integer:
                    x, w = (x / 8).half(), (w / 8).half()
                packed = w.reshape(2, full_n // 16, 16, full_k // c0, c0).permute(0, 3, 1, 2, 4).contiguous()
                expected = (x[:, inner:inner + k].double()
                            @ w[1, column:column + columns, inner:inner + k].double().T
                            * (2 if accumulate else 1)).to(result_dtype)
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
