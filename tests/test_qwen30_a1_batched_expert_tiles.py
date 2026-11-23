# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""Fresh narrow/wide integer projection equivalence, including independent CPU checks."""

import json
import os
import statistics
from contextlib import ExitStack
from pathlib import Path

import pytest
import torch


def test_batched_expert_tiles(tmp_path):
    isa = os.environ.get("PTO_ISA_ROOT")
    if not isa:
        pytest.skip("PTO_ISA_ROOT is required for the allocated 310P device")
    torch_npu = pytest.importorskip("torch_npu")
    from models.qwen3_30b_w8a8 import moe_batched_modelslim_a1 as moe
    from simpler_setup import A1BorrowedLaunch, KernelCompiler

    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    root = Path(moe.__file__).parent
    ids = (torch.arange(160, device="npu:0", dtype=torch.int32) % 128).reshape(20, 8)
    generator = torch.Generator().manual_seed(310320)
    report = {}
    for kind, inner, width, rows in (("w13", 2048, 1536, 20), ("w2", 768, 2048, 160)):
        sources = (
            Path(__file__).with_name("kernels") / f"a1_expert_{kind}_narrow_control.cpp",
            root / f"a1_expert_{kind}_nz_batched.cpp",
        )
        images = [
            KernelCompiler("a1").compile_single_launch(str(source), isa, extra_include_dirs=[str(root)])
            for source in sources
        ]
        x_cpu = torch.randint(-128, 128, (rows, inner), dtype=torch.int8, generator=generator)
        w_cpu = torch.randint(-128, 128, (128, width, inner), dtype=torch.int8, generator=generator)
        x, w = x_cpu.npu(), torch_npu.npu_format_cast(w_cpu.npu(), 29)
        ys = [torch.zeros((160, width), dtype=torch.int32, device="npu:0") for _ in range(2)]
        statuses = [torch.full((160, 16), -7, dtype=torch.int32, device="npu:0") for _ in range(2)]
        timings = []
        with ExitStack() as stack:
            for binary, y, status in zip(images, ys, statuses, strict=True):
                owner = stack.enter_context(A1BorrowedLaunch(binary))
                call = owner.prepare(x, w, ids, y, status, blocks=8, native_nz_weights=(1,))
                call()
                torch.npu.synchronize()
                graph = torch.npu.NPUGraph()
                stack.callback(graph.reset)
                with torch.npu.graph(graph):
                    call()
                begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                begin.record()
                for _ in range(5):
                    graph.replay()
                end.record()
                end.synchronize()
                timings.append(begin.elapsed_time(end) / 5)
            for route in (0, 17, 127, 159):
                expected = x_cpu[route // 8 if rows == 20 else route].int() @ w_cpu[route % 128].int().t()
                assert torch.equal(ys[1][route].cpu(), expected), (kind, route)
            assert torch.equal(ys[0].cpu(), ys[1].cpu()), kind
            assert all(not torch.count_nonzero(value.cpu()[:, 0]) for value in statuses)
        report[kind] = {"n16_ms": timings[0], "n256_ms": timings[1], "exact": True}
    (tmp_path / "wide-probe-times.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


@pytest.mark.parametrize("cached_inputs", (False, True, "k64", "k128", "pipeline", "pipeline_wide", "pipeline_wide_n512"))
def test_grouped_expert_tiles(tmp_path, cached_inputs, *, source_overrides=None, extra_patterns=None):
    """Compare all INT32 outputs across route distributions and captured replays."""
    isa = os.environ.get("PTO_ISA_ROOT")
    if not isa:
        pytest.skip("PTO_ISA_ROOT is required for the allocated 310P device")
    torch_npu = pytest.importorskip("torch_npu")
    from models.qwen3_30b_w8a8 import moe_batched_modelslim_a1 as moe
    from simpler_setup import A1BorrowedLaunch, KernelCompiler

    torch.set_num_threads(2)
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    root = Path(moe.__file__).parent
    kernels = Path(__file__).with_name("kernels")
    generator = torch.Generator().manual_seed(310326)
    patterns = {
        "balanced": (torch.arange(160) % 128).reshape(20, 8).int(),
        "hot20": ((torch.arange(20)[:, None] + torch.arange(8)[None, :]) % 20).int(),
        "twenty_per_expert": torch.arange(8).expand(20, -1).contiguous().int(),
        "modulo_hot": torch.arange(0, 64, 8).expand(20, -1).contiguous().int(),
        "all_same": torch.full((20, 8), 127, dtype=torch.int32),
        "random": torch.stack([torch.randperm(128, generator=generator)[:8] for _ in range(20)]).int(),
    }
    if extra_patterns:
        for name, pattern in extra_patterns.items():
            assert name not in patterns and tuple(pattern.shape) == (20, 8)
            patterns[name] = pattern.int().cpu()
    report = {"rounds": 20, "warmup": 5, "metric": "graph_replay_device_ms", "projections": {}}
    for kind, inner, width, rows in (("w13", 2048, 1536, 20), ("w2", 768, 2048, 160)):
        suffix = f"_{cached_inputs}" if isinstance(cached_inputs, str) else ""
        sources = ((root / f"a1_expert_{kind}_nz_batched.cpp", root / f"a1_expert_{kind}_nz_batched_cached{suffix}.cpp")
                   if cached_inputs else (kernels / f"a1_expert_{kind}_route_control.cpp",
                                          kernels / f"a1_expert_{kind}_grouped_candidate.cpp"))
        if cached_inputs == "k128":
            # Preserve the rejected service candidate only as a diagnostic control.
            sources = (sources[0], kernels / f"a1_rejected_{kind}_cached_k128.cpp")
        if cached_inputs == "pipeline_wide_n512":
            sources = (root / f"a1_expert_{kind}_nz_batched_cached_pipeline_wide.cpp",
                       root / f"a1_expert_{kind}_nz_batched_cached_pipeline_wide_n512.cpp")
        if cached_inputs == "pipeline_wide":
            sources = (root / f"a1_expert_{kind}_nz_batched_cached_pipeline.cpp",
                       root / f"a1_expert_{kind}_nz_batched_cached_pipeline_wide.cpp")
        if cached_inputs == "pipeline":
            sources = (root / f"a1_expert_{kind}_nz_batched_cached_k64.cpp",
                       root / f"a1_expert_{kind}_nz_batched_cached_pipeline.cpp")
        if source_overrides is not None:
            sources = tuple(Path(source) for source in source_overrides[kind])
            assert len(sources) == 2
        images = [KernelCompiler("a1").compile_single_launch(str(source), isa,
                  extra_include_dirs=[str(kernels), str(root)]) for source in sources]
        x_cpu = torch.randint(-128, 128, (rows, inner), dtype=torch.int8, generator=generator)
        w_cpu = torch.randint(-128, 128, (128, width, inner), dtype=torch.int8, generator=generator)
        x = x_cpu.npu()
        weight = torch_npu.npu_format_cast(w_cpu.npu(), 29)
        ids = patterns["balanced"].npu()
        ys = [torch.empty((160, width), dtype=torch.int32, device="npu:0") for _ in images]
        statuses = [torch.full((160, 16), -7, dtype=torch.int32, device="npu:0") for _ in images]
        distributions = {}
        with ExitStack() as stack:
            graphs = []
            for image, y, status in zip(images, ys, statuses, strict=True):
                owner = stack.enter_context(A1BorrowedLaunch(image))
                call = owner.prepare(x, weight, ids, y, status, blocks=8, native_nz_weights=(1,))
                call()
                torch.npu.synchronize()
                graph = torch.npu.NPUGraph()
                stack.callback(graph.reset)
                with torch.npu.graph(graph):
                    call()
                graphs.append(graph)
            for name, pattern in patterns.items():
                if cached_inputs in ("pipeline", "pipeline_wide", "pipeline_wide_n512"):
                    x_cpu = torch.randint(-128, 128, (rows, inner), dtype=torch.int8, generator=generator)
                    x.copy_(x_cpu)
                ids.copy_(pattern)
                for graph, y, status in zip(graphs, ys, statuses, strict=True):
                    y.fill_(-123456789)
                    status.fill_(-7)
                    graph.replay()
                actual = [y.cpu() for y in ys]
                assert torch.equal(*actual), (kind, name, int(torch.count_nonzero(actual[0] != actual[1])))
                assert all(not torch.count_nonzero(status.cpu()[:, 0]) for status in statuses)
                for route in (0, 17, 127, 159):
                    expert = int(pattern.flatten()[route])
                    expected = x_cpu[route // 8 if rows == 20 else route].int() @ w_cpu[expert].int().t()
                    assert torch.equal(actual[1][route], expected), (kind, name, route)
                samples = []
                for graph in graphs:
                    for _ in range(5):
                        graph.replay()
                    torch.npu.synchronize()
                    times = []
                    for _ in range(20):
                        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                        start.record()
                        graph.replay()
                        end.record()
                        end.synchronize()
                        times.append(start.elapsed_time(end))
                    samples.append(times)
                distributions[name] = {"control_ms": samples[0], "candidate_ms": samples[1],
                    "control_median_ms": statistics.median(samples[0]),
                    "candidate_median_ms": statistics.median(samples[1]), "exact": True,
                    "control": sources[0].stem if source_overrides else ("grouped" if cached_inputs else "route"),
                    "candidate": sources[1].stem if source_overrides else (f"cached_inputs{suffix}" if cached_inputs else "grouped"),
                    "sources": [str(source) for source in sources]}
                assert torch.equal(ys[0].cpu(), ys[1].cpu()), (kind, name, "repeated replay")
                print(json.dumps({"kind": kind, "pattern": name, **distributions[name]}), flush=True)
            invalid = patterns["balanced"].clone()
            invalid.flatten()[0], invalid.flatten()[159] = -1, 128
            ids.copy_(invalid)
            for graph, y, status in zip(graphs, ys, statuses, strict=True):
                y.fill_(-123456789)
                status.fill_(-7)
                graph.replay()
                result, codes = y.cpu(), status.cpu()[:, 0]
                assert codes[0] == codes[159] == 2
                assert not torch.count_nonzero(codes[1:159])
                assert torch.all(result[[0, 159]] == -123456789)
        report["projections"][kind] = distributions
    (tmp_path / "grouped-probe-times.json").write_text(json.dumps(report, indent=2))




def test_swiglu_vector_maximum_graph(tmp_path):
    """Vector amax must preserve both FP32 scales and every quantized output bit."""
    import torch_npu
    from models.qwen3_30b_w8a8.compile_modelslim_a1 import compile_modelslim_source
    from simpler_setup import A1BorrowedLaunch

    isa = os.environ.get("PTO_ISA_ROOT")
    if not isa:
        pytest.skip("PTO_ISA_ROOT is required")
    torch.set_num_threads(2)
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    images = [compile_modelslim_source(f"a1_swiglu_quant_batched{suffix}.cpp", tmp_path / str(index), Path(isa))
              for index, suffix in enumerate(("", "_vector_max"))]
    generator = torch.Generator().manual_seed(3103275)
    acc = torch.zeros((160, 1536), dtype=torch.int32, device="npu:0")
    activation = torch.rand((20, 16), generator=generator).mul(0.01).add(0.001).npu()
    ids = (torch.arange(160).reshape(20, 8) % 128).int().npu()
    channel = torch.rand((128, 1536), generator=generator).mul(0.0001).add(0.00001).npu()
    outputs = [torch.zeros((160, 768), dtype=torch.int8, device="npu:0") for _ in range(2)]
    scales = [torch.zeros((160, 16), dtype=torch.float32, device="npu:0") for _ in range(2)]
    statuses = [torch.full((160, 16), -7, dtype=torch.int32, device="npu:0") for _ in range(2)]
    with ExitStack() as owners:
        graphs = []
        for index in range(2):
            owner = owners.enter_context(A1BorrowedLaunch(images[index]))
            call = owner.prepare(acc, activation, ids, channel, outputs[index], scales[index], statuses[index], blocks=8)
            call()
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            owners.callback(graph.reset)
            with torch.npu.graph(graph):
                call()
            graphs.append(graph)
        for case in range(4):
            values = torch.randint(-20000000, 20000000, (160, 1536), dtype=torch.int32, generator=generator)
            if case == 0:
                values.zero_()
            elif case == 1:
                values.zero_()
                for column in (0, 63, 64, 127, 128, 767):
                    values[:, column] = 20000000
                    values[:, column + 768] = -20000000
            acc.copy_(values)
            ids.copy_(torch.stack([torch.randperm(128, generator=generator)[:8] for _ in range(20)]).int())
            for graph in graphs:
                graph.replay()
            assert torch.equal(outputs[0].cpu(), outputs[1].cpu()), case
            assert torch.equal(scales[0].cpu(), scales[1].cpu()), case
            assert all(not torch.count_nonzero(status.cpu()[:, 0]) for status in statuses)
        samples = []
        for graph in graphs:
            for _ in range(5):
                graph.replay()
            times = []
            for _ in range(30):
                begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                begin.record()
                graph.replay()
                end.record()
                end.synchronize()
                times.append(begin.elapsed_time(end))
            samples.append(times)
        (tmp_path / "swiglu-vector-max-times.json").write_text(json.dumps({
            "control_ms": samples[0], "vector_ms": samples[1], "exact": True,
            "control_median_ms": statistics.median(samples[0]),
            "candidate_median_ms": statistics.median(samples[1])}, indent=2))






def test_router_m16_batching_graph(tmp_path, *, control_source=None, candidate_source=None):
    """Batch rows without altering K16 summation or accepting a logit ULP."""
    import torch_npu
    from models.qwen3_30b_w8a8 import moe_batched_modelslim_a1 as moe
    from simpler_setup import A1BorrowedLaunch, KernelCompiler

    isa = os.environ.get("PTO_ISA_ROOT")
    if not isa:
        pytest.skip("PTO_ISA_ROOT is required")
    torch.set_num_threads(2)
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    root = Path(moe.__file__).parent
    sources = (Path(control_source) if control_source else root / "a1_modelslim_router_batched.cpp",
               Path(candidate_source) if candidate_source else root / "a1_modelslim_router_batched_m16.cpp")
    images = [KernelCompiler("a1").compile_single_launch(str(source), isa,
              extra_include_dirs=[str(root)]) for source in sources]
    generator = torch.Generator().manual_seed(3103271)
    x = torch.zeros((20, 2048), dtype=torch.float16, device="npu:0")
    weight = torch.zeros((2048, 128), dtype=torch.float16, device="npu:0")
    outputs = [torch.zeros((20, 128), dtype=torch.float16, device="npu:0") for _ in range(2)]
    statuses = [torch.full((8, 16), -7, dtype=torch.int32, device="npu:0") for _ in range(2)]
    with ExitStack() as owners:
        graphs = []
        layout_diagnostics = []
        for index in range(2):
            owner = owners.enter_context(A1BorrowedLaunch(images[index]))
            call = owner.prepare(x, weight, outputs[index], statuses[index], blocks=8)
            call()
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            owners.callback(graph.reset)
            with torch.npu.graph(graph):
                call()
            graphs.append(graph)
        for case in range(5):
            values = torch.randn((20, 2048), generator=generator).half()
            weights = (torch.randn((2048, 128), generator=generator) / 45.25).half()
            if case == 0:
                values.zero_()
            elif case == 1:
                values = (torch.randint(-16, 17, (20, 2048), generator=generator).float() / 16).half()
                weights = (torch.randint(-16, 17, (2048, 128), generator=generator).float() / 128).half()
            x.copy_(values)
            weight.copy_(weights)
            # AscendUnquantizedLinearMethod uses maybe_trans_nz for this FP16 gate.
            # Keep ND as a diagnostic rather than silently treating it as the service path.
            native_nd_weight = weight.t().contiguous()
            native_nz_weight = torch_npu.npu_format_cast(native_nd_weight, 29)
            assert torch_npu.get_npu_format(native_nz_weight) == 29
            reference = torch.nn.functional.linear(x, native_nz_weight).cpu()
            reference_nd = torch.nn.functional.linear(x, native_nd_weight).cpu()
            for graph, output, status in zip(graphs, outputs, statuses, strict=True):
                output.fill_(float("nan"))
                status.fill_(-7)
                graph.replay()
            actual = [value.cpu() for value in outputs]
            if not torch.equal(actual[0], actual[1]) or not torch.equal(actual[1], reference):
                torch.save({"x": values, "weight": weights, "control": actual[0], "batched": actual[1],
                            "native_nz": reference, "diagnostic_native_nd": reference_nd},
                           tmp_path / f"router-m16-mismatch-{case}.pt")
            layout_diagnostics.append({"case": case, "native_weight_format": 29,
                                       "nd_vs_nz_mismatches": int((reference_nd != reference).sum()),
                                       "m16_vs_old_mismatches": int((actual[0] != actual[1]).sum()),
                                       "m16_vs_native_nz_mismatches": int((actual[1] != reference).sum())})
            assert torch.equal(actual[0], actual[1]), case
            assert torch.equal(actual[1], reference), case
            assert all(not torch.count_nonzero(value.cpu()[:, 0]) for value in statuses)
        times = []
        for graph in graphs:
            for _ in range(5):
                graph.replay()
            samples = []
            for _ in range(30):
                begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                begin.record()
                graph.replay()
                end.record()
                end.synchronize()
                samples.append(begin.elapsed_time(end))
            times.append(samples)
        (tmp_path / "router-native-layout-comparison.json").write_text(json.dumps(layout_diagnostics, indent=2))
        (tmp_path / "router-m16-times.json").write_text(json.dumps({
            "control_ms": times[0], "batched_ms": times[1], "exact": True,
            "sources": [str(source) for source in sources], "rounds": 30, "warmup": 5,
            "control_median_ms": statistics.median(times[0]),
            "candidate_median_ms": statistics.median(times[1])}, indent=2))
        assert torch.equal(outputs[0].cpu(), outputs[1].cpu()), "repeated replay"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
