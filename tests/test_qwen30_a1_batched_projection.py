# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""Native NZ borrowing and exact B20 INT8/FP16 projection graph checks."""

import json
import os
import statistics
from contextlib import ExitStack
from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize("kind,inner,width", (("qkv", 2048, 5120), ("o", 4096, 2048)))
@pytest.mark.parametrize("wide", (False, True))
def test_batched_nz_projection_graph(kind, inner, width, tmp_path, wide):
    torch_npu = pytest.importorskip("torch_npu")
    isa = os.environ.get("PTO_ISA_ROOT")
    if not isa:
        pytest.skip("PTO_ISA_ROOT is required for the allocated 310P device")
    from models.qwen3_30b_w8a8.linear_batched_modelslim_a1 import (
        compile_a1_batched_modelslim_dequant,
        compile_a1_batched_modelslim_projection,
    )
    from models.qwen3_30b_w8a8.linear_modelslim_a1 import decode_a1_modelslim_scale
    from models.qwen3_30b_w8a8.weights_modelslim_a1 import A1ModelSlimWeights
    from simpler_setup import A1BorrowedLaunch

    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    checkpoint_value = os.environ.get("A1_MODELSLIM_CHECKPOINT")
    if not checkpoint_value:
        pytest.skip("A1_MODELSLIM_CHECKPOINT is required")
    checkpoint = Path(checkpoint_value)
    if not checkpoint.is_dir():
        pytest.fail(f"Qualification checkpoint is missing: {checkpoint}")
    weights = A1ModelSlimWeights(checkpoint).attention(0)
    raw_weight = weights[f"{kind}_weight"]
    weight = torch_npu.npu_format_cast(raw_weight.npu(), 29)
    original_pointer = weight.data_ptr()
    original_bytes = weight.untyped_storage().nbytes()
    x = torch.zeros((32, inner), dtype=torch.int8, device="npu:0")
    acc = torch.zeros((32, width), dtype=torch.int32, device="npu:0")
    bias = weights[f"{kind}_quant_bias"].npu()
    encoded_scale = weights[f"{kind}_deq_scale"].flatten().npu()
    scale = decode_a1_modelslim_scale(weights[f"{kind}_deq_scale"]).npu()
    output = torch.zeros((20, width), dtype=torch.float16, device="npu:0")
    status = torch.full((8, 16), -7, dtype=torch.int32, device="npu:0")
    image = compile_a1_batched_modelslim_projection(kind, Path(isa), wide=wide)
    image_path = os.environ.get("PTO3_B20_TEST_IMAGES")
    epilogue = (torch.load(image_path, map_location="cpu", weights_only=True)["native"]["dequant"]
                if image_path else compile_a1_batched_modelslim_dequant(Path(isa)))
    generator = torch.Generator().manual_seed(310320)
    with ExitStack() as stack:
        owner = stack.enter_context(A1BorrowedLaunch(image))
        with pytest.raises(ValueError, match="ND tensor storage"):
            owner.prepare(x, weight, acc, status, blocks=8)
        with pytest.raises(ValueError, match="full aligned"):
            owner.prepare(x, raw_weight.npu(), acc, status, blocks=8, native_nz_weights=(1,))
        with pytest.raises(ValueError, match="unique argument"):
            owner.prepare(x, weight, acc, status, blocks=8, native_nz_weights=(1, 1))
        call = owner.prepare(x, weight, acc, status, blocks=8, native_nz_weights=(1,))
        epilogue_owner = stack.enter_context(A1BorrowedLaunch(epilogue))
        dequant = epilogue_owner.prepare(acc, bias, scale, output, status, blocks=8)
        call()
        dequant()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        stack.callback(graph.reset)
        with torch.npu.graph(graph):
            call()
            dequant()
        for case in range(2):
            values = torch.randint(-128, 128, (20, inner), dtype=torch.int8, generator=generator)
            x[:20].copy_(values)
            expected_acc = (values.int() @ raw_weight.int().t()).int()
            expected = torch_npu.npu_quant_matmul(
                x[:20], weight.transpose(0, 1), encoded_scale,
                bias=bias.flatten(), output_dtype=torch.float16,
            ).cpu()
            graph.replay()
            actual_acc, actual = acc[:20].cpu(), output.cpu()
            if not torch.equal(actual_acc, expected_acc) or not torch.equal(actual, expected):
                fixture = tmp_path / f"nz-{kind}-{case}.pt"
                torch.save({"x": values, "actual_acc": actual_acc, "expected_acc": expected_acc,
                            "actual": actual, "expected": expected}, fixture)
                pytest.fail(f"{kind}: acc mismatches={torch.count_nonzero(actual_acc != expected_acc)}, "
                            f"fp16 mismatches={torch.count_nonzero(actual != expected)}, fixture={fixture}")
            assert not torch.count_nonzero(status.cpu()[:, 0])
            assert not torch.count_nonzero(acc[20:].cpu())
        assert weight.data_ptr() == original_pointer
        assert weight.untyped_storage().nbytes() == original_bytes


@pytest.mark.parametrize("kind,inner,width", (("qkv", 2048, 5120), ("o", 4096, 2048)))
@pytest.mark.parametrize("packed_scale_words", (False, True))
def test_fused_projection_dequant_graph(
    kind, inner, width, tmp_path, packed_scale_words, *,
    control_source=None, candidate_source=None, control_packed_scale_words=False,
):
    """Require native exactness, changed-input replay and untouched padded rows."""
    torch_npu = pytest.importorskip("torch_npu")
    isa = os.environ.get("PTO_ISA_ROOT")
    if not isa:
        pytest.skip("PTO_ISA_ROOT is required for the allocated 310P device")
    from models.qwen3_30b_w8a8.linear_batched_modelslim_a1 import (
        compile_a1_batched_modelslim_dequant,
        compile_a1_batched_modelslim_projection,
    )
    from models.qwen3_30b_w8a8.linear_modelslim_a1 import decode_a1_modelslim_scale
    from models.qwen3_30b_w8a8.weights_modelslim_a1 import A1ModelSlimWeights
    from simpler_setup import A1BorrowedLaunch, KernelCompiler

    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    checkpoint_value = os.environ.get("A1_MODELSLIM_CHECKPOINT")
    if not checkpoint_value:
        pytest.skip("A1_MODELSLIM_CHECKPOINT is required")
    checkpoint = Path(checkpoint_value)
    weights = A1ModelSlimWeights(checkpoint).attention(0)
    weight = torch_npu.npu_format_cast(weights[f"{kind}_weight"].npu(), 29)
    pointer, storage_bytes = weight.data_ptr(), weight.untyped_storage().nbytes()
    bias = weights[f"{kind}_quant_bias"].npu()
    encoded_scale = weights[f"{kind}_deq_scale"].flatten().npu()
    scale = decode_a1_modelslim_scale(weights[f"{kind}_deq_scale"]).npu()
    x = torch.zeros((32, inner), dtype=torch.int8, device="npu:0")
    acc = torch.zeros((32, width), dtype=torch.int32, device="npu:0")
    control = torch.zeros((20, width), dtype=torch.float16, device="npu:0")
    guarded = torch.full((32, width), -99.0, dtype=torch.float16, device="npu:0")
    fused_control = control_source is not None
    candidate_scale = scale
    if packed_scale_words:
        candidate_scale = encoded_scale.reshape(1, width).view(torch.int32)
        assert candidate_scale.shape == (1, 2 * width)
        assert candidate_scale.data_ptr() == encoded_scale.data_ptr()
    assert not control_packed_scale_words or (fused_control and packed_scale_words)
    control_scale = candidate_scale if control_packed_scale_words else scale
    if fused_control or candidate_source is not None:
        assert control_source is not None and candidate_source is not None
        model = Path(__file__).parents[1] / "models/qwen3_30b_w8a8"
        images = [
            KernelCompiler("a1").compile_single_launch(
                str(source), isa, extra_include_dirs=[str(Path(source).parent), str(model)]
            )
            for source in [control_source, candidate_source]
        ]
    else:
        image_paths = [os.environ.get(name) for name in ("PTO3_B20_CONTROL_IMAGES", "PTO3_B20_TEST_IMAGES")]
        image_sets = [torch.load(path, map_location="cpu", weights_only=True) if path else None for path in image_paths]
        images = [
            image_sets[0]["native"][kind] if image_sets[0] else
            compile_a1_batched_modelslim_projection(kind, Path(isa), wide=True),
            image_sets[0]["native"]["dequant"] if image_sets[0] else
            compile_a1_batched_modelslim_dequant(Path(isa)),
            image_sets[1]["native"][kind] if image_sets[1] else
            compile_a1_batched_modelslim_projection(kind, Path(isa), wide=True, fused_dequant=True,
                                                    packed_scale_words=packed_scale_words),
        ]
    status = [torch.full((8, 16), -7, dtype=torch.int32, device="npu:0") for _ in images]
    generator = torch.Generator().manual_seed(310320)
    with ExitStack() as stack:
        owners = [stack.enter_context(A1BorrowedLaunch(image)) for image in images]
        if fused_control:
            calls = [
                owner.prepare(x, weight, bias, dequant_scale, target, flag, blocks=8,
                              native_nz_weights=(1,))
                for owner, target, flag, dequant_scale in zip(
                    owners, [control, guarded[:20]], status, [control_scale, candidate_scale], strict=True
                )
            ]
            graph_calls = [[calls[0]], [calls[1]]]
        else:
            calls = [owners[0].prepare(x, weight, acc, status[0], blocks=8, native_nz_weights=(1,)),
                     owners[1].prepare(acc, bias, scale, control, status[1], blocks=8),
                     owners[2].prepare(x, weight, bias, candidate_scale, guarded[:20], status[2], blocks=8,
                                       native_nz_weights=(1,))]
            graph_calls = [calls[:2], calls[2:]]
        for call in calls:
            call()
        torch.npu.synchronize()
        graphs = [torch.npu.NPUGraph(), torch.npu.NPUGraph()]
        for graph in graphs:
            stack.callback(graph.reset)
        for graph, sequence in zip(graphs, graph_calls, strict=True):
            with torch.npu.graph(graph):
                for call in sequence:
                    call()
        native_graph = torch.npu.NPUGraph()
        stack.callback(native_graph.reset)
        with torch.npu.graph(native_graph):
            native_output = torch_npu.npu_quant_matmul(
                x[:20], weight.transpose(0, 1), encoded_scale,
                bias=bias.flatten(), output_dtype=torch.float16,
            )
        for case in range(4):
            values = torch.randint(-128, 128, (20, inner), dtype=torch.int8, generator=generator)
            if case == 0:
                values.zero_()
            elif case == 3:
                values[:, ::2] = -128
                values[:, 1::2] = 127
            if case == 2:
                bias.copy_(torch.randint(-(2**24), 2**24, bias.shape, dtype=torch.int32, generator=generator))
            x[:20].copy_(values)
            native_graph.replay()
            expected = native_output.cpu()
            control.fill_(torch.nan)
            guarded[:20].fill_(torch.nan)
            for flag in status:
                flag.fill_(-7)
            for graph in graphs:
                graph.replay()
            actual = guarded[:20].cpu()
            if not torch.equal(actual, expected) or not torch.equal(actual, control.cpu()):
                fixture = tmp_path / f"fused-{kind}-{case}.pt"
                torch.save({"actual": actual, "expected": expected, "control": control.cpu()}, fixture)
                pytest.fail(f"{kind} case {case}: native mismatches={torch.count_nonzero(actual != expected)}, "
                            f"fixture={fixture}")
            assert torch.all(guarded[20:].cpu() == -99)
            for value in status:
                assert not torch.count_nonzero(value.cpu()[:, 0])
        benchmark_graphs = [*graphs, native_graph]
        for graph in benchmark_graphs:
            for _ in range(5):
                graph.replay()
        torch.npu.synchronize()
        timings = [[], [], []]
        for sample in range(5):
            for index in ((0, 1, 2) if sample % 2 == 0 else (2, 1, 0)):
                start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                start.record()
                for _ in range(100):
                    benchmark_graphs[index].replay()
                end.record()
                end.synchronize()
                timings[index].append(start.elapsed_time(end) / 100)
        assert weight.data_ptr() == pointer
        assert weight.untyped_storage().nbytes() == storage_bytes
        report = {"kind": kind, "scope": "standalone graph, not service TPOT", "replays_per_sample": 100,
                  "warmup": 5, "native_exact": True, "fused_control": fused_control,
                  "control_source": str(control_source), "candidate_source": str(candidate_source),
                  "packed_scale_words": packed_scale_words,
                  "control_packed_scale_words": control_packed_scale_words,
                  "milliseconds": timings[:2],
                  "median_ms": [statistics.median(values) for values in timings[:2]],
                  "native_api": "torch_npu.npu_quant_matmul",
                  "native_milliseconds": timings[2], "native_median_ms": statistics.median(timings[2])}
        (tmp_path / f"projection-{kind}-times.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report))


@pytest.mark.parametrize("wide", (False, True))
def test_native_projection_scales_require_fused_dequant(wide):
    from models.qwen3_30b_w8a8.linear_batched_modelslim_a1 import (
        compile_a1_batched_modelslim_projection,
    )

    with pytest.raises(ValueError, match="native scale words require fused dequantization"):
        compile_a1_batched_modelslim_projection(
            "qkv", Path("unused-isa"), wide=wide, packed_scale_words=True,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
