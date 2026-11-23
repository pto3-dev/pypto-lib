# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""Wide-N A1 GEMM through DSL, PTOAS, native execution and changing ACL replay."""

import os
from pathlib import Path

import pypto.language as pl
import pytest
import torch


@pytest.mark.parametrize("integer,rows,columns,accumulate", [
    (True, 16, 16, False),
    (True, 16, 256, False),
    (True, 32, 512, True),
    (False, 16, 16, False),
    (False, 16, 256, False),
    (False, 32, 512, True),
])
def test_a1_wide_n_dsl(integer, rows, columns, accumulate, tmp_path):
    torch_npu = pytest.importorskip("torch_npu")
    isa = os.environ.get("PTO_ISA_ROOT")
    if not isa:
        pytest.skip("PTO_ISA_ROOT is required")
    from simpler_setup import A1BorrowedLaunch

    from models.qwen3_30b_w8a8.compile_modelslim_a1 import compile_modelslim_program

    k = 32 if integer else 16
    element = pl.INT8 if integer else pl.FP16
    result = pl.INT32 if integer else pl.FP32

    @pl.jit.incore
    def wide_n(
        lhs: pl.Tensor[[rows, k], element],
        rhs: pl.Tensor[[columns, k], element],
        output: pl.InOut[pl.Tensor[[rows, columns], result]],
    ):
        left = pl.load(lhs, [0, 0], [rows, k], target_memory=pl.MemorySpace.Mat)
        source = pl.load(rhs, [0, 0], [columns, k], target_memory=pl.MemorySpace.Mat)
        right = pl.tile.transpose_view(source)
        accumulator = pl.matmul(left, right)
        if accumulate:
            accumulator = pl.matmul_acc(accumulator, left, right)
        for column in pl.range(0, columns, 16):
            fragment = pl.tile.extract(
                accumulator, 0, column, [rows, 16], target_memory=pl.MemorySpace.Vec
            )
            pl.store(fragment, [0, column], output)
        return output

    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    image, order = compile_modelslim_program(wide_n, tmp_path, Path(isa))
    assert set(order) == {"lhs", "rhs", "output"}
    generated = list(tmp_path.rglob("*.cpp"))
    assert any("TEXTRACT(" in path.read_text() for path in generated)
    input_dtype = torch.int8 if integer else torch.float16
    output_dtype = torch.int32 if integer else torch.float32
    lhs = torch.zeros((rows, k), dtype=input_dtype, device="npu:0")
    rhs = torch.zeros((columns, k), dtype=input_dtype, device="npu:0")
    storage = torch.full((rows * columns + 32,), -70001, dtype=output_dtype, device="npu:0")
    output = storage[16:-16].view(rows, columns)
    tensors = {"lhs": lhs, "rhs": rhs, "output": output}
    generator = torch.Generator().manual_seed(310320 + rows + columns + accumulate)
    with A1BorrowedLaunch(image, device_id=0) as launch:
        call = launch.prepare(*(tensors[name] for name in order), blocks=1)
        call()
        torch.npu.synchronize()
        assert torch.equal(output.cpu(), torch.zeros((rows, columns), dtype=output_dtype))
        graph = torch.npu.NPUGraph()
        try:
            with torch.npu.graph(graph):
                call()
            for replay in range(5):
                x = torch.randint(-3, 4, (rows, k), generator=generator).to(input_dtype)
                w = torch.randint(-3, 4, (columns, k), generator=generator).to(input_dtype)
                if integer and replay == 4:
                    x[0, :4] = torch.tensor([-128, 127, -1, 1], dtype=input_dtype)
                    w[0, :4] = torch.tensor([127, -128, -1, 1], dtype=input_dtype)
                if not integer:
                    x = (x / 8).half()
                    w = (w / 8).half()
                expected = (x.double() @ w.double().T * (2 if accumulate else 1)).to(output_dtype)
                lhs.copy_(x)
                rhs.copy_(w)
                storage.fill_(-70001)
                graph.replay()
                actual = output.cpu()
                assert torch.equal(actual, expected), (integer, rows, columns, accumulate, replay)
                guards = storage.cpu()
                assert torch.equal(guards[:16], torch.full((16,), -70001, dtype=output_dtype))
                assert torch.equal(guards[-16:], torch.full((16,), -70001, dtype=output_dtype))
        finally:
            graph.reset()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
