"""Cold-packed VDEQ16 tables preserve full SwiGLU quantized values and scale bits."""

import json
import os
import statistics
from contextlib import ExitStack
from pathlib import Path

import pytest
import torch


def test_swiglu_packed_tables_graph(
    tmp_path, *, control_source=None, candidate_source=None, control_packed=False,
    channel_dequantized=False,
):
    """Vector amax must preserve both FP32 scales and every quantized output bit."""
    import torch_npu
    from models.qwen3_30b_w8a8 import moe_batched_modelslim_a1
    from models.qwen3_30b_w8a8.compile_modelslim_a1 import compile_modelslim_source
    from simpler_setup import A1BorrowedLaunch, KernelCompiler

    isa = os.environ.get("PTO_ISA_ROOT")
    if not isa:
        pytest.skip("PTO_ISA_ROOT is required")
    torch.set_num_threads(2)
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    model = Path(moe_batched_modelslim_a1.__file__).parent

    def compile_arm(source, default, name):
        if source is None:
            return compile_modelslim_source(default, tmp_path / name, Path(isa))
        source = Path(source)
        return KernelCompiler("a1").compile_single_launch(
            str(source), isa, extra_include_dirs=[str(source.parent), str(model)]
        )

    old = compile_arm(control_source, "a1_swiglu_quant_batched_vector_max.cpp", "control")
    new = compile_arm(candidate_source, "a1_swiglu_quant_batched_vector_packed.cpp", "packed")
    images = [old, new]
    generator = torch.Generator().manual_seed(3103275)
    acc = torch.zeros(
        (160, 1536), dtype=torch.float16 if channel_dequantized else torch.int32, device="npu:0"
    )
    activation = torch.rand((20, 16), generator=generator).mul(0.01).add(0.001).npu()
    ids = (torch.arange(160).reshape(20, 8) % 128).int().npu()
    channel = torch.rand((128, 1536), generator=generator).mul(0.0001).add(0.00001).npu()
    packed = torch.zeros((128, 1536, 2), dtype=torch.int32, device="npu:0")
    packed[:, :, 0].copy_(channel.view(torch.int32))
    outputs = [torch.zeros((160, 768), dtype=torch.int8, device="npu:0") for _ in range(2)]
    scales = [torch.zeros((160, 16), dtype=torch.float32, device="npu:0") for _ in range(2)]
    statuses = [torch.full((160, 16), -7, dtype=torch.int32, device="npu:0") for _ in range(2)]
    with ExitStack() as owners:
        graphs, calls = [], []
        for index in range(2):
            owner = owners.enter_context(A1BorrowedLaunch(images[index]))
            table = packed.reshape(128, 3072) if index or control_packed else channel
            call = owner.prepare(
                acc, activation, ids, table, outputs[index], scales[index], statuses[index], blocks=8
            )
            calls.append(call)
            call()
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            owners.callback(graph.reset)
            with torch.npu.graph(graph):
                for _ in range(4):
                    call()
            graphs.append(graph)
        for case in range(4):
            values = torch.randint(-20000000, 20000000, (160, 1536), dtype=torch.int32, generator=generator)
            if channel_dequantized:
                values = torch.randn((160, 1536), generator=generator).mul(16).half()
            if case == 0:
                values.zero_()
            elif case == 1:
                values.zero_()
                for column in (0, 63, 64, 127, 128, 767):
                    values[:, column] = 65504 if channel_dequantized else 20000000
                    values[:, column + 768] = -65504 if channel_dequantized else -20000000
            channel.copy_(torch.rand((128, 1536), generator=generator).mul(0.0001).add(0.00001))
            activation.copy_(torch.rand((20, 16), generator=generator).mul(0.01).add(0.001))
            packed[:, :, 0].copy_(channel.view(torch.int32))
            assert not torch.count_nonzero(packed[:, :, 1].cpu())
            acc.copy_(values)
            ids.copy_(torch.stack([torch.randperm(128, generator=generator)[:8] for _ in range(20)]).int())
            for graph in graphs:
                graph.replay()
            assert torch.equal(outputs[0].cpu(), outputs[1].cpu()), case
            assert torch.equal(scales[0].cpu().view(torch.int32), scales[1].cpu().view(torch.int32)), case
            assert all(not torch.count_nonzero(status.cpu()[:, 0]) for status in statuses)
        for bad_routes in (((0, -1),), ((159, 128),), ((17, -1), (127, 128))):
            invalid_ids = (torch.arange(160).reshape(20, 8) % 128).int()
            for route, expert in bad_routes:
                invalid_ids.flatten()[route] = expert
            ids.copy_(invalid_ids)
            for index, call in enumerate(calls):
                outputs[index].fill_(19)
                scales[index].fill_(-11)
                statuses[index].fill_(-7)
                call()
            assert torch.equal(outputs[0].cpu(), outputs[1].cpu()), bad_routes
            assert torch.equal(scales[0].cpu().view(torch.int32), scales[1].cpu().view(torch.int32))
            assert torch.equal(statuses[0].cpu(), statuses[1].cpu()), bad_routes
            for route, _ in bad_routes:
                assert torch.all(outputs[1][route].cpu() == 19)
                assert torch.all(scales[1][route].cpu() == -11)
                assert statuses[1][route, 0].item() == 2
        ids.copy_((torch.arange(160).reshape(20, 8) % 128).int())
        for graph in graphs:
            for _ in range(5):
                graph.replay()
        samples = [[], []]
        for sample in range(30):
            for index in (sample % 2, 1 - sample % 2):
                begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                begin.record()
                graphs[index].replay()
                end.record()
                end.synchronize()
                samples[index].append(begin.elapsed_time(end) / 4.0)
        (tmp_path / "swiglu-packed-times.json").write_text(
            json.dumps(
                {
                    "captured_repetitions": 4,
                    "alternating_control_candidate": True,
                    "invalid_route_cases_exact": 3,
                    "control_ms": samples[0],
                    "vector_ms": samples[1],
                    "exact": True,
                    "control_median_ms": statistics.median(samples[0]),
                    "candidate_median_ms": statistics.median(samples[1]),
                    "control_source": str(control_source),
                    "candidate_source": str(candidate_source),
                    "control_packed": control_packed,
                    "channel_dequantized_input": channel_dequantized,
                },
                indent=2,
            )
        )


def test_grouped_w13_swiglu_graph(tmp_path, *, swiglu_source=None, w13_source=None,
                                   grouped_control=False, replays_per_sample=1, extra_patterns=None):
    """Moving channel dequantization into W13 preserves quantized SwiGLU bits."""
    isa = os.environ.get("PTO_ISA_ROOT")
    if not isa:
        pytest.skip("PTO_ISA_ROOT is required")
    torch_npu = pytest.importorskip("torch_npu")
    from models.qwen3_30b_w8a8 import moe_batched_modelslim_a1 as moe
    from models.qwen3_30b_w8a8.compile_modelslim_a1 import compile_modelslim_source
    from simpler_setup import A1BorrowedLaunch, KernelCompiler

    torch.set_num_threads(2)
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    model = Path(moe.__file__).parent
    w13_images = [
        KernelCompiler("a1").compile_single_launch(str(model / name), isa)
        for name in (
            "a1_expert_w13_nz_batched_cached_pipeline_wide_n512.cpp",
            "a1_expert_w13_nz_batched_grouped_dequant.cpp",
        )
    ]
    swiglu_images = [
        compile_modelslim_source(name, tmp_path / f"swiglu-{index}", Path(isa))
        for index, name in enumerate(
            ("a1_swiglu_quant_batched_vector_packed.cpp",
             "a1_swiglu_quant_batched_grouped_dequant.cpp")
        )
    ]
    if swiglu_source is not None:
        swiglu_images[1] = KernelCompiler("a1").compile_single_launch(
            str(swiglu_source), isa, extra_include_dirs=[str(model)]
        )
    if grouped_control:
        w13_images[0] = w13_images[1]
        swiglu_images[0] = swiglu_images[1]
    if w13_source is not None:
        w13_images[1] = KernelCompiler("a1").compile_single_launch(
            str(w13_source), isa, extra_include_dirs=[str(model)])
    generator = torch.Generator().manual_seed(31032916)
    x_cpu = torch.randint(-128, 128, (20, 2048), dtype=torch.int8, generator=generator)
    w_cpu = torch.randint(-128, 128, (128, 1536, 2048), dtype=torch.int8, generator=generator)
    x = x_cpu.npu()
    weight = torch_npu.npu_format_cast(w_cpu.npu(), 29)
    ids = (torch.arange(160).reshape(20, 8) % 128).int().npu()
    activation = torch.rand((20, 16), generator=generator).mul(0.01).add(0.001).npu()
    channel = torch.rand((128, 1536), generator=generator).mul(0.0001).add(0.00001).npu()
    packed = torch.zeros((128, 1536, 2), dtype=torch.int32, device="npu:0")
    packed[:, :, 0].copy_(channel.view(torch.int32))
    table = packed.reshape(128, 3072)
    intermediates = [
        torch.empty((160, 1536), dtype=dtype, device="npu:0")
        for dtype in ((torch.float16, torch.float16) if grouped_control else (torch.int32, torch.float16))
    ]
    outputs = [torch.empty((160, 768), dtype=torch.int8, device="npu:0") for _ in range(2)]
    scales = [torch.empty((160, 16), dtype=torch.float32, device="npu:0") for _ in range(2)]
    statuses = [
        [torch.full((160, 16), -7, dtype=torch.int32, device="npu:0") for _ in range(2)]
        for _ in range(2)
    ]
    patterns = {
        "balanced": (torch.arange(160) % 128).reshape(20, 8).int(),
        "twenty_per_expert": torch.arange(8).expand(20, -1).contiguous().int(),
        "all_same": torch.full((20, 8), 127, dtype=torch.int32),
        "modulo_hot": torch.arange(0, 64, 8).expand(20, -1).contiguous().int(),
        "random": torch.stack([torch.randperm(128, generator=generator)[:8] for _ in range(20)]).int(),
        "zero": (torch.arange(160) % 128).reshape(20, 8).int(),
    }
    if extra_patterns:
        assert not (patterns.keys() & extra_patterns.keys()), "Keep all original routing cases"
        patterns.update(extra_patterns)
    report = {"metric": "paired_w13_swiglu_graph_device_ms", "cases": {}, "exact": False,
              "grouped_control": grouped_control, "replays_per_sample": replays_per_sample}
    with ExitStack() as owners:
        graphs, w13_calls = [], []
        for index in range(2):
            w13_owner = owners.enter_context(A1BorrowedLaunch(w13_images[index]))
            w13_args = (x, weight, ids, intermediates[index])
            if index or grouped_control:
                w13_args = (*w13_args, table)
            w13_call = w13_owner.prepare(
                *w13_args, statuses[index][0], blocks=8, native_nz_weights=(1,)
            )
            swiglu_owner = owners.enter_context(A1BorrowedLaunch(swiglu_images[index]))
            swiglu_call = swiglu_owner.prepare(
                intermediates[index], activation, ids, table, outputs[index],
                scales[index], statuses[index][1], blocks=8
            )
            w13_call()
            swiglu_call()
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            owners.callback(graph.reset)
            with torch.npu.graph(graph):
                w13_call()
                swiglu_call()
            graphs.append(graph)
            w13_calls.append(w13_call)
        for name, pattern in patterns.items():
            x_cpu = torch.randint(-128, 128, (20, 2048), dtype=torch.int8, generator=generator)
            if name == "zero":
                x_cpu.zero_()
            x.copy_(x_cpu)
            ids.copy_(pattern)
            activation.copy_(torch.rand((20, 16), generator=generator).mul(0.01).add(0.001))
            channel.copy_(torch.rand((128, 1536), generator=generator).mul(0.0001).add(0.00001))
            packed[:, :, 0].copy_(channel.view(torch.int32))
            for repeat in range(3):
                for index, graph in enumerate(graphs):
                    intermediates[index].fill_(-99)
                    outputs[index].fill_(-89 if index else 91)
                    scales[index].fill_(-7 if index else -9)
                    for status in statuses[index]:
                        status.fill_(-7)
                    graph.replay()
                if grouped_control:
                    assert torch.equal(intermediates[0].cpu().view(torch.int16),
                                       intermediates[1].cpu().view(torch.int16)), (name, repeat, "FP16 W13 bits")
                assert torch.equal(outputs[0].cpu(), outputs[1].cpu()), (name, repeat, "INT8")
                assert torch.equal(
                    scales[0].cpu()[:, 0].view(torch.int32),
                    scales[1].cpu()[:, 0].view(torch.int32)
                ), (name, repeat, "FP32 scale bits")
                assert torch.all(scales[0].cpu()[:, 1:] == -9), (name, "control scale padding")
                assert torch.all(scales[1].cpu()[:, 1:] == -7), (name, "candidate scale padding")
                assert all(not torch.count_nonzero(s.cpu()[:, 0]) for arm in statuses for s in arm)
                if name == "zero":
                    assert not torch.count_nonzero(outputs[1].cpu())
                    assert not torch.count_nonzero(scales[1].cpu()[:, 0])
            if not grouped_control:
                for route in (0, 17, 127, 159):
                    expert = int(pattern.flatten()[route])
                    expected = x_cpu[route // 8].int() @ w_cpu[expert].int().t()
                    assert torch.equal(intermediates[0][route].cpu(), expected), (name, route)
            for _ in range(5):
                for graph in graphs:
                    graph.replay()
            samples = [[], []]
            for sample in range(30):
                for index in (sample % 2, 1 - sample % 2):
                    begin = torch.npu.Event(enable_timing=True)
                    end = torch.npu.Event(enable_timing=True)
                    begin.record()
                    for _ in range(replays_per_sample):
                        graphs[index].replay()
                    end.record()
                    end.synchronize()
                    samples[index].append(begin.elapsed_time(end) / replays_per_sample)
            report["cases"][name] = {
                "control_ms": samples[0], "candidate_ms": samples[1],
                "control_median_ms": statistics.median(samples[0]),
                "candidate_median_ms": statistics.median(samples[1]), "exact": True,
            }
            print("W13_SWIGLU_EXACT", name, report["cases"][name]["control_median_ms"],
                  report["cases"][name]["candidate_median_ms"], flush=True)
        invalid = patterns["balanced"].clone()
        invalid.flatten()[0], invalid.flatten()[159] = -1, 128
        ids.copy_(invalid)
        for index, call in enumerate(w13_calls):
            intermediates[index].fill_(-99)
            statuses[index][0].fill_(-7)
            call()
            codes = statuses[index][0].cpu()[:, 0]
            assert codes[0] == codes[159] == 2
            assert not torch.count_nonzero(codes[1:159])
            assert torch.all(intermediates[index][[0, 159]].cpu() == -99)
    report["exact"] = True
    (tmp_path / "w13-swiglu-grouped-times.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
