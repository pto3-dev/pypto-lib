# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""Native packed-channel dequantization through DSL, including every FP16 bit."""

import os
from contextlib import ExitStack
from pathlib import Path

import pypto.language as pl
import pytest
import torch


@pytest.mark.parametrize("rows,physical", [(1, 16), (4, 16), (16, 16), (20, 32), (32, 32)])
def test_a1_channel_dequant_dsl(rows, physical, tmp_path):
    torch_npu = pytest.importorskip("torch_npu")
    isa = os.environ.get("PTO_ISA_ROOT")
    if not isa:
        pytest.skip("PTO_ISA_ROOT is required")
    from models.qwen3_30b_w8a8.compile_modelslim_a1 import compile_modelslim_program
    from simpler_setup import A1BorrowedLaunch, KernelCompiler

    @pl.jit.incore
    def dequant(
        src: pl.Tensor[[rows, 16], pl.INT32],
        scales: pl.Tensor[[1, 16], pl.INT64],
        output: pl.InOut[pl.Tensor[[rows, 16], pl.FP16]],
    ):
        values = pl.tile.load(src, [0, 0], [physical, 16], valid_shape=[rows, 16])
        packed = pl.tile.load(scales, [0, 0], [1, 16])
        result = pl.tile.dequantize_channel(values, packed)
        pl.tile.store(result, [0, 0], output)
        return output

    image, order = compile_modelslim_program(dequant, tmp_path / "dsl", Path(isa))
    generated = "".join(p.read_text() for p in (tmp_path / "dsl").rglob("*.pto"))
    assert "pto.tdequant.channel" in generated
    assert "pto.tcvt" not in generated and "pto.tmul" not in generated
    generated_cpp = "".join(p.read_text() for p in (tmp_path / "dsl").rglob("*.cpp"))
    assert "TMOV_FP(" in generated_cpp and "TileType::Acc" in generated_cpp
    tests = Path(__file__).parent
    (tmp_path / "reference").mkdir()
    reference = KernelCompiler("a1").compile_single_launch(
        str(tests / "kernels/a1_grouped_channel_dequant_reference.cpp"), isa,
        build_dir=str(tmp_path / "reference"),
        extra_include_dirs=[str(tests.parent / "models/qwen3_30b_w8a8")],
    )
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    source = torch.zeros((rows, 16), dtype=torch.int32, device="npu:0")
    scales = torch.empty((1, 16), dtype=torch.int64, device="npu:0")
    storage = torch.full((rows * 16 + 32,), -99.0, dtype=torch.float16, device="npu:0")
    output = storage[16:-16].view(rows, 16)
    native_source = torch.zeros((128, 512), dtype=torch.int32, device="npu:0")
    native_scales = torch.zeros((8, 512, 2), dtype=torch.int32, device="npu:0")
    native_rows = torch.zeros((8, 16), dtype=torch.int32, device="npu:0")
    native_output = torch.full((128, 512), -99.0, dtype=torch.float16, device="npu:0")
    native_status = torch.full((8, 16), -7, dtype=torch.int32, device="npu:0")
    counts = torch.zeros((8, 16), dtype=torch.int32)
    for group in range((rows + 15) // 16):
        counts[group, 0] = min(16, rows - 16 * group)
    native_rows.copy_(counts)
    unit_words = torch.zeros((1, 16, 2), dtype=torch.int32)
    unit_words[:, :, 0].copy_(torch.ones((1, 16)).view(torch.int32))
    scales.copy_(unit_words.view(torch.int64).reshape(1, 16))
    native_scales[:, :16].copy_(unit_words.expand(8, -1, -1))
    tensors = {"src": source, "scales": scales, "output": output}
    generator = torch.Generator().manual_seed(311011 + rows)
    with ExitStack() as owners:
        native_owner = owners.enter_context(A1BorrowedLaunch(reference, device_id=0))
        native_call = native_owner.prepare(
            native_source, native_scales.reshape(8, 1024), native_rows,
            native_output, native_status, blocks=8,
        )
        dsl_owner = owners.enter_context(A1BorrowedLaunch(image, device_id=0))
        dsl_call = dsl_owner.prepare(*(tensors[name] for name in order), blocks=1)
        calls = [native_call, dsl_call]
        graphs = []
        for call in calls:
            call()
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            owners.callback(graph.reset)
            with torch.npu.graph(graph):
                call()
            graphs.append(graph)
        for replay in range(6):
            integers = torch.randint(-(2**30), 2**30, (rows, 16), generator=generator, dtype=torch.int32)
            channels = torch.exp2(torch.randint(-30, 3, (1, 16), generator=generator).float())
            channels *= torch.rand((1, 16), generator=generator).add(0.5)
            if replay == 0:
                integers.zero_()
            elif replay == 1:
                integers[:, :5] = torch.tensor([-(2**31), 2**31 - 1, -1, 0, 1], dtype=torch.int32)
            elif replay == 2:
                channels[:, :4] = torch.tensor([0.0, -0.0, 1e-30, -1e-30])
            elif replay == 3:
                channels[:, :4] = torch.tensor([float("inf"), -float("inf"), float("nan"), -1.0])
            elif replay == 4:
                integers[:, :4] = torch.tensor([2**24 + 1, -(2**24 + 1), 2**25 - 1, -(2**25 - 1)])
            words = torch.zeros((1, 16, 2), dtype=torch.int32)
            words[:, :, 0].copy_(channels.contiguous().view(torch.int32))
            source.copy_(integers)
            scales.copy_(words.view(torch.int64).reshape(1, 16))
            native_source[:rows, :16].copy_(integers)
            native_scales[:, :16].copy_(words.expand(8, -1, -1))
            storage.fill_(-99.0)
            native_output.fill_(-99.0)
            native_status.fill_(-7)
            for graph in graphs:
                graph.replay()
            actual = output.cpu().view(torch.int16)
            expected = native_output[:rows, :16].cpu().view(torch.int16)
            if not torch.equal(actual, expected):
                torch.save({"source": integers, "scale_words": words, "actual_bits": actual,
                            "native_bits": expected}, tmp_path / f"mismatch-{replay}.pt")
            assert torch.equal(actual, expected), (rows, replay)
            assert not torch.count_nonzero(native_status.cpu()[:, 0])
            guard = torch.full((16,), -99.0, dtype=torch.float16)
            assert torch.equal(storage[:16].cpu(), guard)
            assert torch.equal(storage[-16:].cpu(), guard)
            assert torch.equal(native_output[rows:, :16].cpu(), torch.full((128 - rows, 16), -99.0))


@pytest.mark.parametrize("k", [2048, 4096])
def test_a1_projection_channel_dequant_preserves_live_accumulator(k, tmp_path):
    torch_npu = pytest.importorskip("torch_npu")
    isa = os.environ.get("PTO_ISA_ROOT")
    if not isa:
        pytest.skip("PTO_ISA_ROOT is required")
    from models.qwen3_30b_w8a8.compile_modelslim_a1 import compile_modelslim_program
    from simpler_setup import A1BorrowedLaunch, KernelCompiler

    rows, columns = 20, 256

    @pl.jit.incore
    def projection(
        src: pl.Tensor[[rows, k], pl.INT8],
        weights: pl.Tensor[[1, columns, k], pl.INT8, pl.NZ],
        bias: pl.Tensor[[1, columns], pl.INT32],
        scales: pl.Tensor[[1, columns], pl.INT64],
        output: pl.InOut[pl.Tensor[[rows, columns], pl.FP16]],
        raw_output: pl.InOut[pl.Tensor[[rows, columns], pl.INT32]],
    ):
        activation = pl.tile.load(src, [0, 0], [32, 64], valid_shape=[rows, 64], target_memory=pl.Mem.Mat)
        resident = pl.tile.load(weights, [0, 0, 0], [1, columns, 64], target_memory=pl.Mem.Mat)
        matrix = pl.tile.reshape(resident, [columns, 64])
        right_view = pl.tile.transpose_view(matrix)
        left = pl.tile.extract(activation, 0, 0, [32, 64], target_memory=pl.Mem.Left)
        right = pl.tile.extract(right_view, 0, 0, [64, columns], target_memory=pl.Mem.Right)
        accumulator = pl.tile.matmul(left, right)
        for inner in pl.range(64, k, 64):
            next_activation = pl.tile.load(src, [0, inner], [32, 64], valid_shape=[rows, 64], target_memory=pl.Mem.Mat)
            next_resident = pl.tile.load(weights, [0, 0, inner], [1, columns, 64], target_memory=pl.Mem.Mat)
            next_matrix = pl.tile.reshape(next_resident, [columns, 64])
            next_view = pl.tile.transpose_view(next_matrix)
            next_left = pl.tile.extract(next_activation, 0, 0, [32, 64], target_memory=pl.Mem.Left)
            next_right = pl.tile.extract(next_view, 0, 0, [64, columns], target_memory=pl.Mem.Right)
            accumulator = pl.tile.matmul_acc(accumulator, next_left, next_right)
        for col in pl.range(0, columns, 16):
            values = pl.tile.extract(accumulator, 0, col, [32, 16], target_memory=pl.Mem.Vec)
            channels = pl.tile.load(bias, [0, col], [1, 16])
            packed = pl.tile.load(scales, [0, col], [1, 16])
            expanded = pl.tile.col_expand(values, channels)
            biased = pl.tile.add(values, expanded)
            pl.tile.store(pl.tile.dequantize_channel(biased, packed), [0, col], output)
        for col in pl.range(0, columns, 16):
            original = pl.tile.extract(accumulator, 0, col, [32, 16], target_memory=pl.Mem.Vec)
            pl.tile.store(original, [0, col], raw_output)
        return output, raw_output

    image, order = compile_modelslim_program(projection, tmp_path / "dsl", Path(isa))
    model = Path(__file__).parents[1] / "models/qwen3_30b_w8a8"
    control = tmp_path / "native_projection.cpp"
    control.write_text(
        '#include "a1_nz_projection_batched_wide.h"\n'
        'extern "C" __global__ AICORE void SingleLaunchEntry(__gm__ int64_t* args) {\n'
        f'  BatchedNZProjectionWide<{k}, {columns}, 256, true, true>(args);\n'
        '}\n'
    )
    (tmp_path / "reference").mkdir()
    reference = KernelCompiler("a1").compile_single_launch(
        str(control), isa, build_dir=str(tmp_path / "reference"), extra_include_dirs=[str(model)]
    )
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    source = torch.zeros((rows, k), dtype=torch.int8, device="npu:0")
    native_source = torch.zeros((32, k), dtype=torch.int8, device="npu:0")
    weights = torch.zeros((1, k // 32, columns // 16, 16, 32), dtype=torch.int8, device="npu:0")
    weights = torch_npu.npu_format_cast(weights, 2)
    native_weights = weights.reshape(k // 32 * columns, 32)
    bias = torch.zeros((1, columns), dtype=torch.int32, device="npu:0")
    scales = torch.zeros((1, columns), dtype=torch.int64, device="npu:0")
    native_scales = torch.zeros((1, 2 * columns), dtype=torch.int32, device="npu:0")
    storage = torch.full((rows * columns + 32,), -99.0, dtype=torch.float16, device="npu:0")
    output = storage[16:-16].view(rows, columns)
    raw = torch.empty((rows, columns), dtype=torch.int32, device="npu:0")
    native_output = torch.empty_like(output)
    native_status = torch.full((8, 16), -7, dtype=torch.int32, device="npu:0")
    arguments = {"src": source, "weights": weights, "bias": bias, "scales": scales,
                 "output": output, "raw_output": raw}
    generator = torch.Generator().manual_seed(311011 + k)
    with ExitStack() as owners:
        native_owner = owners.enter_context(A1BorrowedLaunch(reference, device_id=0))
        native_call = native_owner.prepare(native_source, native_weights, bias, native_scales,
                                           native_output, native_status, blocks=8)
        dsl_owner = owners.enter_context(A1BorrowedLaunch(image, device_id=0))
        dsl_call = dsl_owner.prepare(*(arguments[name] for name in order), blocks=1)
        graphs = []
        for call in [native_call, dsl_call]:
            call()
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            owners.callback(graph.reset)
            with torch.npu.graph(graph):
                call()
            graphs.append(graph)
        for replay in range(3):
            x = torch.randint(-128, 128, (rows, k), generator=generator, dtype=torch.int8)
            w = torch.randint(-128, 128, (columns, k), generator=generator, dtype=torch.int8)
            x[0].fill_(127)
            w[0].fill_(127)
            b = torch.randint(-(2**31), 2**31, (1, columns), generator=generator, dtype=torch.int64).int()
            b[0, :4] = torch.tensor([1, 2**24 + 1, 2**31 - 1, -(2**31)], dtype=torch.int32)
            channel = torch.exp2(torch.randint(-28, -12, (1, columns), generator=generator).float())
            channel *= torch.rand((1, columns), generator=generator).add(0.5)
            words = torch.zeros((1, columns, 2), dtype=torch.int32)
            words[:, :, 0].copy_(channel.view(torch.int32))
            source.copy_(x)
            native_source[:rows].copy_(x)
            packed = w.reshape(columns // 16, 16, k // 32, 32).permute(2, 0, 1, 3).unsqueeze(0).contiguous()
            weights.copy_(packed)
            bias.copy_(b)
            scales.copy_(words.view(torch.int64).reshape(1, columns))
            native_scales.copy_(words.reshape(1, 2 * columns))
            storage.fill_(-99.0)
            raw.fill_(-70001)
            for graph in graphs:
                graph.replay()
            assert not torch.count_nonzero(native_status.cpu()[:, 0])
            assert torch.equal(output.cpu().view(torch.int16), native_output.cpu().view(torch.int16)), (k, replay)
            assert torch.equal(raw.cpu(), (x.long() @ w.long().T).int()), ("scratch corrupted live Acc", k, replay)
            guard = torch.full((16,), -99.0, dtype=torch.float16)
            assert torch.equal(storage[:16].cpu(), guard)
            assert torch.equal(storage[-16:].cpu(), guard)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
