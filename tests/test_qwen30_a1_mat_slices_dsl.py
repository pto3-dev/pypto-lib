# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""K-axis Mat slices through the complete A1 DSL and graph-replay path."""

import os
import json
import re
from pathlib import Path

import pypto.language as pl
import pytest
import torch


def _check_pipeline_storage(directory, rows, columns, element_bytes):
    """Check real loop buffers and preload ordering, not measured execution overlap."""
    path = next(p for p in directory.rglob("projection.pto") if p.parent.name == "ptoas")
    text = path.read_text()
    constants = {name: int(value) for name, value in re.findall(
        r"(%[\w]+) = arith.constant (-?\d+) : i64", text
    )}
    allocations = {}
    for name, address, height, width in re.findall(
        r"(%[\w]+) = pto.alloc_tile addr = (%[\w]+)[^\n]*?loc=mat,[^\n]*?rows=(\d+), cols=(\d+)", text
    ):
        allocations[name] = (constants[address], int(height), int(width))
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if "scf.for %k0" in line)
    depth = 0
    end = None
    for i in range(start, len(lines)):
        depth += lines[i].count("{") - lines[i].count("}")
        if depth == 0:
            end = i
            break
    assert end is not None
    loop = lines[start:end + 1]
    loads = []
    for i, line in enumerate(loop):
        if "pto.tload" in line:
            target = re.search(r"outs\((%\w+) :", line).group(1)
            address, height, width = allocations[target]
            loads.append({"line": i, "address": address, "rows": height, "columns": width,
                          "bytes": height * width * element_bytes})
    assert len(loads) == 4
    assert sorted(load["rows"] for load in loads) == sorted([rows, rows, columns, columns])
    for i, left in enumerate(loads):
        for right in loads[i + 1:]:
            assert left["address"] + left["bytes"] <= right["address"] or (
                right["address"] + right["bytes"] <= left["address"]
            ), (left, right)
    first_mad = next(i for i, line in enumerate(loop) if "pto.tmatmul" in line)
    assert max(load["line"] for load in loads) < first_mad
    emitted = "\n".join(p.read_text() for p in directory.rglob("*.cpp"))
    assert "PIPE_MTE2, PIPE_MTE1" in emitted and "PIPE_M, PIPE_MTE1" in emitted
    proof = {"source": str(path), "two_stage_loads": loads,
             "all_stage_loads_before_first_mad": True, "event_edges_present": True,
             "measured_compute_dma_overlap": False}
    (directory / "pipeline-proof.json").write_text(json.dumps(proof, indent=2) + "\n")


@pytest.mark.parametrize(
    "integer,rows,physical_rows,columns,offset,k,pipeline_k",
    [(True, 20, 32, 256, 64, 64, 0), (True, 32, 32, 512, 32, 64, 0),
     (True, 4, 16, 16, 32, 32, 0), (False, 20, 32, 128, 64, 64, 0),
     (False, 32, 32, 256, 16, 64, 0), (False, 1, 16, 16, 16, 16, 0),
     (True, 20, 32, 256, 0, 64, 384), (True, 20, 32, 256, 0, 64, 512),
     (False, 20, 32, 128, 0, 64, 384), (False, 20, 32, 128, 0, 64, 512),
     (True, 20, 32, 256, 0, 64, 2048), (True, 20, 32, 256, 0, 64, 4096),
     (False, 20, 32, 128, 0, 64, 2048)],
)
def test_a1_mat_k_slices_dsl(integer, rows, physical_rows, columns, offset, k, pipeline_k, tmp_path):
    torch_npu = pytest.importorskip("torch_npu")
    isa = os.environ.get("PTO_ISA_ROOT")
    if not isa:
        pytest.skip("PTO_ISA_ROOT is required")
    from simpler_setup import A1BorrowedLaunch

    from models.qwen3_30b_w8a8.compile_modelslim_a1 import compile_modelslim_program

    element = pl.INT8 if integer else pl.FP16
    result = pl.INT32 if integer else pl.FP32
    c0 = 32 if integer else 16
    pipelined = bool(pipeline_k)
    full_n = 5120 if integer and pipeline_k == 2048 else 2048 if pipeline_k == 4096 else 1024
    full_k, column, inner = pipeline_k or 256, 256, 0 if pipelined else 64

    @pl.jit.incore
    def projection(
        lhs: pl.Tensor[[rows, full_k], element],
        weights: pl.Tensor[[2, full_n, full_k], element, pl.NZ],
        output: pl.InOut[pl.Tensor[[rows, columns], result]],
    ):
        source = pl.tile.load(lhs, [0, inner], [physical_rows, 128],
                              valid_shape=[rows, 128], target_memory=pl.Mem.Mat)
        resident = pl.tile.load(weights, [1, column, inner], [1, columns, 128], target_memory=pl.Mem.Mat)
        matrix = pl.tile.reshape(resident, [columns, 128])
        view = pl.tile.transpose_view(matrix)
        left = pl.tile.extract(source, 0, offset, [physical_rows, k], target_memory=pl.Mem.Left)
        right = pl.tile.extract(view, offset, 0, [k, columns], target_memory=pl.Mem.Right)
        accumulator = pl.tile.matmul(left, right)
        if pipelined:
            second_left = pl.tile.extract(source, 0, 64, [physical_rows, 64], target_memory=pl.Mem.Left)
            second_right = pl.tile.extract(view, 64, 0, [64, columns], target_memory=pl.Mem.Right)
            accumulator = pl.tile.matmul_acc(accumulator, second_left, second_right)
            for k0 in pl.pipeline(128, full_k, 128, stage=2):
                activation = pl.tile.load(lhs, [0, k0], [physical_rows, 128],
                                          valid_shape=[rows, 128], target_memory=pl.Mem.Mat)
                resident_next = pl.tile.load(weights, [1, column, k0], [1, columns, 128],
                                             target_memory=pl.Mem.Mat)
                matrix_next = pl.tile.reshape(resident_next, [columns, 128])
                view_next = pl.tile.transpose_view(matrix_next)
                a0 = pl.tile.extract(activation, 0, 0, [physical_rows, 64], target_memory=pl.Mem.Left)
                b0 = pl.tile.extract(view_next, 0, 0, [64, columns], target_memory=pl.Mem.Right)
                accumulator = pl.tile.matmul_acc(accumulator, a0, b0)
                a1 = pl.tile.extract(activation, 0, 64, [physical_rows, 64], target_memory=pl.Mem.Left)
                b1 = pl.tile.extract(view_next, 64, 0, [64, columns], target_memory=pl.Mem.Right)
                accumulator = pl.tile.matmul_acc(accumulator, a1, b1)
        else:
            accumulator = pl.tile.matmul_acc(accumulator, left, right)
        for col in pl.range(0, columns, 16):
            fragment = pl.tile.extract(accumulator, 0, col, [physical_rows, 16], target_memory=pl.Mem.Vec)
            pl.tile.store(fragment, [0, col], output)
        return output

    image, order = compile_modelslim_program(projection, tmp_path, Path(isa))
    assert set(order) == {"lhs", "weights", "output"}
    if pipelined and full_k >= 2048:
        _check_pipeline_storage(tmp_path, physical_rows, columns, 1 if integer else 2)
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
    generator = torch.Generator().manual_seed(310900 + rows + columns + offset + k)
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
                start = inner + offset
                if integer and replay == 4:
                    x[0, start:start + 4] = torch.tensor([-128, 127, -1, 1], dtype=dtype)
                    w[1, column, start:start + 4] = torch.tensor([127, -128, -1, 1], dtype=dtype)
                if not integer:
                    x, w = (x / 8).half(), (w / 8).half()
                packed = w.reshape(2, full_n // 16, 16, full_k // c0, c0).permute(0, 3, 1, 2, 4).contiguous()
                if pipelined:
                    expected = (x.double() @ w[1, column:column + columns].double().T).to(result_dtype)
                else:
                    expected = (x[:, start:start + k].double()
                                @ w[1, column:column + columns, start:start + k].double().T * 2).to(result_dtype)
                lhs.copy_(x)
                weights.copy_(packed)
                storage.fill_(-70001)
                graph.replay()
                assert torch.equal(output.cpu(), expected), (integer, rows, columns, offset, k, replay)
                guards = storage.cpu()
                assert torch.equal(guards[:16], torch.full((16,), -70001, dtype=result_dtype))
                assert torch.equal(guards[-16:], torch.full((16,), -70001, dtype=result_dtype))
        finally:
            graph.reset()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
