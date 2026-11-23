# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""B20 ModelSlim routing and complete 160-route MoE against native 310P primitives."""

import json
import os
import statistics
from contextlib import ExitStack
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="module")
def resources(tmp_path_factory):
    torch_npu = pytest.importorskip("torch_npu")
    isa = os.environ.get("PTO_ISA_ROOT")
    if not isa:
        pytest.skip("PTO_ISA_ROOT is required for the allocated 310P device")
    from models.qwen3_30b_w8a8.moe_batched_modelslim_a1 import compile_a1_batched_moe_images
    from models.qwen3_30b_w8a8.weights_modelslim_a1 import A1ModelSlimWeights

    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.npu.config.allow_internal_format = True
    checkpoint = os.environ.get("A1_MODELSLIM_CHECKPOINT")
    if not checkpoint:
        pytest.skip("A1_MODELSLIM_CHECKPOINT is required")
    loader = A1ModelSlimWeights(Path(checkpoint))
    image_path = os.environ.get("PTO3_B20_TEST_IMAGES")
    images = (torch.load(image_path, map_location="cpu", weights_only=True)["moe"] if image_path else
              compile_a1_batched_moe_images(tmp_path_factory.mktemp("b20_moe_images"), Path(isa)))
    return loader, images


def _check_exact(actual, expected, tmp_path):
    for name, value in actual.items():
        reference = expected[name]
        if not torch.equal(value, reference):
            fixture = tmp_path / f"moe-b20-{name}.pt"
            torch.save({"actual": actual, "expected": expected}, fixture)
            pytest.fail(f"{name}: mismatches={torch.count_nonzero(value != reference)}, "
                        f"max_abs={float((value.float()-reference.float()).abs().max())}, fixture={fixture}")


def _native_routes(logits):
    import torch_npu

    selected, ids, _ = torch_npu.npu_moe_gating_top_k_softmax(logits, k=8)
    weights = (selected / selected.sum(-1, keepdim=True)).half()
    return weights, ids.int()


def test_b20_modelslim_router_graph(resources, tmp_path):
    import torch_npu
    from simpler_setup import A1BorrowedLaunch

    loader, images = resources
    raw_weight = loader.attention(0)["mlp.gate.weight"]
    weight = raw_weight.t().contiguous().npu()
    native_weight = torch_npu.npu_format_cast(raw_weight.npu(), 29)
    generator = torch.Generator().manual_seed(310320)
    hidden = torch.randn((20, 2048), generator=generator).half().npu()
    logits = torch.zeros((20, 128), dtype=torch.float16).npu()
    ids = torch.zeros((20, 8), dtype=torch.int32).npu()
    weights = torch.zeros((20, 8), dtype=torch.float32).npu()
    router_status = torch.full((8, 16), -7, dtype=torch.int32).npu()
    topk_status = torch.full((20, 16), -7, dtype=torch.int32).npu()
    with ExitStack() as stack:
        router = stack.enter_context(A1BorrowedLaunch(images["router"]))
        topk = stack.enter_context(A1BorrowedLaunch(images["topk"]))
        route = router.prepare(hidden, weight, logits, router_status, blocks=8)
        select = topk.prepare(logits, ids, weights, topk_status, blocks=8)
        route()
        select()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        stack.callback(graph.reset)
        with torch.npu.graph(graph):
            route()
            select()
        for _ in range(2):
            hidden.copy_(torch.randn((20, 2048), generator=generator).half())
            native_logits = torch.nn.functional.linear(hidden, native_weight)
            native_weights, native_ids = _native_routes(native_logits)
            graph.replay()
            _check_exact({"logits": logits.cpu(), "ids": ids.cpu(), "weights": weights.cpu()},
                         {"logits": native_logits.cpu(), "ids": native_ids.cpu(),
                          "weights": native_weights.float().cpu()}, tmp_path)
            assert not torch.count_nonzero(router_status.cpu()[:, 0])
            assert not torch.count_nonzero(topk_status.cpu()[:, 0])


def test_b20_modelslim_topk_graph(tmp_path, *, sources=None, baseline_source=None):
    """Require native routing equality across changed logits and repeated graphs."""
    torch_npu = pytest.importorskip("torch_npu")
    isa = os.environ.get("PTO_ISA_ROOT")
    if not isa:
        pytest.skip("PTO_ISA_ROOT is required for the allocated 310P device")
    from simpler_setup import A1BorrowedLaunch, KernelCompiler

    select_experts = pytest.importorskip("vllm_ascend._310p.fused_moe.experts_selector").select_experts

    torch.set_num_threads(2)
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    root = Path(__file__).parents[1] / "models/qwen3_30b_w8a8"
    sources = tuple(Path(source) for source in sources) if sources else (root / "a1_modelslim_topk_batched.cpp",)
    if baseline_source:
        sources = (Path(baseline_source), *sources)
    images = [KernelCompiler("a1").compile_single_launch(str(source), isa,
              extra_include_dirs=[str(root)]) for source in sources]
    logits = torch.zeros((20, 128), dtype=torch.float16, device="npu:0")
    indices = [torch.empty((20, 8), dtype=torch.int32, device="npu:0") for _ in images]
    weights = [torch.empty((20, 8), dtype=torch.float32, device="npu:0") for _ in images]
    statuses = [torch.empty((20, 16), dtype=torch.int32, device="npu:0") for _ in images]
    generator = torch.Generator().manual_seed(310330)
    random_logits = torch.randn((20, 128), generator=generator).half() * 4
    cutoff_tie = torch.arange(128).half().expand(20, -1).contiguous() / 128
    cutoff_tie[:, :9] = 10
    normalization_ties = torch.arange(128).half().expand(20, -1).contiguous() * 6e-8
    normalization_ties[:, -1] = 20
    positive_ties = torch.arange(128).half().expand(20, -1).contiguous() * 1e-6
    positive_ties[:, -1] = 1
    small_positive_ties = positive_ties.clone()
    small_positive_ties[:, -1] = 0.01
    nonzero_tail_ties = positive_ties.clone()
    nonzero_tail_ties[:, -1] = 10
    upper_half_ties = torch.full((20, 128), -10.0, dtype=torch.float16)
    upper_half_ties[:, 64:] = 10
    cross_half_ties = torch.full((20, 128), -10.0, dtype=torch.float16)
    cross_half_ties[:, 60:69] = 10
    block_boundary_ties = torch.full((20, 128), -10.0, dtype=torch.float16)
    block_boundary_ties[:, [15, 16, 31, 32, 63, 64, 95, 96, 127]] = 10
    cases = {
        "random": random_logits,
        "zero_ties": torch.zeros((20, 128), dtype=torch.float16),
        "cutoff_ties": cutoff_tie,
        "reversed": random_logits.flip(-1),
        "underflow": ((torch.arange(128) - 64) * 200).half().expand(20, -1).contiguous(),
        "normalization_ties": normalization_ties,
        "near_zero_logits": torch.arange(128).half().expand(20, -1).contiguous() * 6e-8,
        "positive_fp16_ties": positive_ties,
        "small_positive_fp16_ties": small_positive_ties,
        "nonzero_tail_fp16_ties": nonzero_tail_ties,
        "upper_half_ties": upper_half_ties,
        "cross_half_ties": cross_half_ties,
        "block_boundary_ties": block_boundary_ties,
    }
    report = {"cases": [], "scope": "standalone_graph_event_ms_not_vllm", "warmup": 5, "rounds": 20}
    with ExitStack() as stack:
        graphs = []
        for image, ids, values, status in zip(images, indices, weights, statuses, strict=True):
            owner = stack.enter_context(A1BorrowedLaunch(image))
            call = owner.prepare(logits, ids, values, status, blocks=8)
            call()
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            stack.callback(graph.reset)
            with torch.npu.graph(graph):
                call()
            graphs.append(graph)
        def native_routes():
            return select_experts(hidden_states=logits, router_logits=logits, top_k=8,
                                  use_grouped_topk=False, renormalize=True)

        native_routes()
        native_graph = torch.npu.NPUGraph()
        stack.callback(native_graph.reset)
        with torch.npu.graph(native_graph):
            native_weights, native_ids = native_routes()
        for name, values in cases.items():
            logits.copy_(values)
            native_graph.replay()
            expected = {"ids": native_ids.cpu(), "weights": native_weights.float().cpu()}
            case_report = {"name": name, "variants": []}
            for index, graph in enumerate(graphs):
                indices[index].fill_(-7)
                weights[index].fill_(float("nan"))
                statuses[index].fill_(-7)
                graph.replay()
                actual = {"ids": indices[index].cpu(), "weights": weights[index].cpu()}
                mismatches = {key: int(torch.count_nonzero(actual[key] != value)) for key, value in expected.items()}
                case_report["variants"].append({"source": str(sources[index]), "mismatches": mismatches})
                if not (baseline_source and index == 0):
                    if any(mismatches.values()):
                        torch.save({"case": name, "logits": values, "actual": actual, "expected": expected},
                                   tmp_path / f"topk-{name}-mismatch.pt")
                    _check_exact(actual, expected, tmp_path)
                elif name not in {"underflow", "normalization_ties", "near_zero_logits",
                                  "positive_fp16_ties", "small_positive_fp16_ties", "nonzero_tail_fp16_ties"}:
                    _check_exact(actual, expected, tmp_path)
                assert not torch.count_nonzero(statuses[index].cpu()[:, 0]), (name, index)
            report["cases"].append(case_report)
            (tmp_path / "topk-precision-progress.json").write_text(json.dumps(report, indent=2))
        if baseline_source:
            negative = next(case for case in report["cases"] if case["name"] == "underflow")
            assert negative["variants"][0]["mismatches"]["ids"] > 0, "Missing negative control"
        logits.copy_(random_logits)
        timed_graphs = [*graphs, native_graph]
        for graph in timed_graphs:
            for _ in range(5):
                graph.replay()
        torch.npu.synchronize()
        samples = [[] for _ in timed_graphs]
        for repeat in range(20):
            order = range(len(timed_graphs)) if repeat % 2 == 0 else reversed(range(len(timed_graphs)))
            for index in order:
                begin = torch.npu.Event(enable_timing=True)
                end = torch.npu.Event(enable_timing=True)
                begin.record()
                for _ in range(10):
                    timed_graphs[index].replay()
                end.record()
                end.synchronize()
                samples[index].append(begin.elapsed_time(end) / 10)
        expected = {"ids": native_ids.cpu(), "weights": native_weights.float().cpu()}
        for ids, values in zip(indices, weights, strict=True):
            _check_exact({"ids": ids.cpu(), "weights": values.cpu()}, expected, tmp_path)
        report["variants"] = [
            {"source": str(source), "samples_ms": times, "median_ms": statistics.median(times)}
            for source, times in zip(sources, samples[:-1], strict=True)
        ]
        report["native_median_ms"] = statistics.median(samples[-1])
        report["timed_case"] = "random"
        report["native_reference"] = "vllm_ascend._310p.fused_moe.experts_selector.select_experts"
        report["baseline_known_failure_recorded"] = bool(baseline_source)
    (tmp_path / "topk-native-graph.json").write_text(json.dumps(report, indent=2))


def _native_gmm(x, weight, scale, groups):
    import torch_npu

    return torch_npu.npu_quant_grouped_matmul_dequant(
        x=x, quantized_weight=weight, weight_scale=scale, group_list=groups, quant_mode="pertoken"
    )


def test_b20_modelslim_moe_graph(resources, tmp_path):
    import torch_npu
    import torchair

    from models.qwen3_30b_w8a8.moe_batched_modelslim_a1 import A1BatchedModelSlimMoe

    loader, images = resources
    attention, experts = loader.attention(0), loader.experts(0)
    raw_router = attention["mlp.gate.weight"]
    native_router = torch_npu.npu_format_cast(raw_router.npu(), 29)
    router = raw_router.t().contiguous().npu()
    w13 = torch_npu.npu_format_cast(experts.w13.npu(), 29)
    w2 = torch_npu.npu_format_cast(experts.w2.npu(), 29)
    s13, s2 = experts.w13_scale.npu(), experts.w2_scale.npu()
    # Independent native graph preparation has its own qualification-only ND
    # inputs. The production PTO model borrows vLLM's existing NZ allocation.
    native_w13, native_w2 = experts.w13.npu(), experts.w2.npu()
    generator = torch.Generator().manual_seed(310320)
    hidden = torch.randn((20, 2048), generator=generator).half().npu()
    output = torch.zeros_like(hidden)
    config = torchair.CompilerConfig()
    config.mode = "reduce-overhead"
    config.debug.run_eagerly = True
    config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
    native_gmm = torch.compile(_native_gmm, backend=torchair.get_npu_backend(compiler_config=config), fullgraph=True)
    model = A1BatchedModelSlimMoe(images, hidden, router, w13, s13, w2, s2, output)
    graph = torch.npu.NPUGraph()
    try:
        model.enqueue()
        torch.npu.synchronize()
        with torch.npu.graph(graph):
            model.enqueue()
        for _ in range(2):
            hidden.copy_(torch.randn((20, 2048), generator=generator).half())
            native_logits = torch.nn.functional.linear(hidden, native_router)
            route_weights, ids = _native_routes(native_logits)
            order = torch.argsort(ids.flatten(), stable=True)
            groups = (ids.flatten()[:, None] == torch.arange(128, device="npu:0")[None, :]).sum(0).long().cumsum(0)
            routed = hidden.repeat_interleave(8, dim=0)[order].contiguous()
            gate_up = native_gmm(routed, native_w13, s13, groups)
            activated = torch_npu.npu_swiglu(gate_up)
            expert_y = native_gmm(activated, native_w2, s2, groups)
            inverse = torch.empty(160, dtype=torch.int32, device="npu:0")
            inverse.scatter_(0, order, torch.arange(160, dtype=torch.int32, device="npu:0"))
            native_output = torch_npu.npu_moe_token_unpermute(
                permuted_tokens=expert_y, sorted_indices=inverse, probs=route_weights
            )
            expected = {"logits": native_logits.cpu(), "ids": ids.cpu(), "weights": route_weights.float().cpu(),
                        "expert_y": expert_y[inverse.long()].cpu(), "output": native_output.cpu()}
            native_quant, native_scale = torch_npu.npu_dynamic_quant(hidden)
            native_quant13, native_scale13 = torch_npu.npu_dynamic_quant(activated)
            graph.replay()
            actual = {name: model.buffers[name].cpu() for name in ("logits", "ids", "weights", "expert_y")}
            actual["output"] = output.cpu()
            if not torch.equal(actual["expert_y"], expected["expert_y"]):
                from simpler_setup import A1BorrowedLaunch, KernelCompiler
                from models.qwen3_30b_w8a8 import moe_batched_modelslim_a1

                source = Path(moe_batched_modelslim_a1.__file__).with_name("a1_expert_dequant_batched.cpp")
                image = KernelCompiler("a1").compile_single_launch(str(source), os.environ["PTO_ISA_ROOT"])
                gate_debug = torch.empty((160, 1536), dtype=torch.float16, device="npu:0")
                debug_status = torch.full((160, 16), -7, dtype=torch.int32, device="npu:0")
                with A1BorrowedLaunch(image) as debug:
                    debug.prepare(model.buffers["acc13"], model.buffers["scale"], model.buffers["ids"], s13,
                                  gate_debug, debug_status, blocks=8)()
                    torch.npu.synchronize()
                native_gate = gate_up[inverse.long()].cpu()
                native_activated = activated[inverse.long()].cpu()
                pto_gate = gate_debug.cpu()
                print(f"B20_STAGE gate_up mismatches={torch.count_nonzero(pto_gate != native_gate)}, "
                      f"max_abs={float((pto_gate.float()-native_gate.float()).abs().max())}", flush=True)
                identity = torch.eye(768, dtype=torch.int8).unsqueeze(0).expand(128, -1, -1).contiguous().npu()
                identity_scale = torch.ones((128, 768), dtype=torch.float32, device="npu:0")
                probed = native_gmm(activated, identity, identity_scale, groups)[inverse.long()].cpu()
                frozen = {name: tensor.cpu() for name, tensor in model.buffers.items()}
                inferred_quant = (probed.float() / frozen["scale13"][:, :1]).round().to(torch.int8)
                print(f"B20_STAGE inferred_gmm_quant13 mismatches={torch.count_nonzero(inferred_quant != frozen['quant13'])}", flush=True)
                for selected_route in (0, 1, 16, 40, 159):
                    expert = int(frozen["ids"].flatten()[selected_route])
                    integer = frozen["quant13"][selected_route].int() @ experts.w2[expert].int().t()
                    print(f"B20_STAGE int32_w2 route={selected_route} mismatches="
                          f"{torch.count_nonzero(integer != frozen['acc2'][selected_route])}", flush=True)
                fixture = tmp_path / "moe-stage-diagnosis.pt"
                torch.save({"actual": frozen, "inferred_quant13": inferred_quant, "identity_probe": probed,
                            "hidden": hidden.cpu(), "gate_up": gate_up[inverse.long()].cpu(),
                            "activated": activated[inverse.long()].cpu(),
                            "native_quant": native_quant.cpu(), "native_scale": native_scale.cpu(),
                            "native_quant13": native_quant13[inverse.long()].cpu(),
                            "native_scale13": native_scale13[inverse.long()].cpu(),
                            "expected": expected}, fixture)
                torch.save({"pto_gate": pto_gate, "native_gate": native_gate,
                            "native_activated": native_activated}, tmp_path / "gate-boundary.pt")
                for name, reference in (("quant", native_quant.cpu()), ("quant13", native_quant13[inverse.long()].cpu())):
                    value = model.buffers[name].cpu()
                    print(f"B20_STAGE {name} mismatches={torch.count_nonzero(value != reference)}", flush=True)
                print(f"B20_STAGE fixture={fixture}", flush=True)
            _check_exact(actual, expected, tmp_path)
            assert all(not torch.count_nonzero(status.cpu()[:, 0]) for status in model.status.values())
    finally:
        graph.reset()
        model.close()


@pytest.mark.parametrize("scale_rows", (20, 160))
def test_b20_retained_swiglu_graph(tmp_path, scale_rows):
    """Require bit-exact INT8/scales under changed inputs and graph replay."""
    import json
    import statistics

    pytest.importorskip("torch_npu")
    from simpler_setup import A1BorrowedLaunch

    paths = [os.environ.get(name) for name in ("PTO3_B20_CONTROL_IMAGES", "PTO3_B20_TEST_IMAGES")]
    if not all(paths):
        pytest.skip("Both control and candidate image paths are required")
    torch.npu.set_device(0)
    images = [torch.load(path, map_location="cpu", weights_only=True)["moe"]["swiglu"] for path in paths]
    generator = torch.Generator().manual_seed(310326)
    acc = torch.zeros((160, 1536), dtype=torch.int32, device="npu:0")
    scales = torch.ones((scale_rows, 16), dtype=torch.float32, device="npu:0")
    ids = torch.zeros((20, 8), dtype=torch.int32, device="npu:0")
    channel = torch.ones((128, 1536), dtype=torch.float32, device="npu:0")
    outputs = [torch.empty((160, 768), dtype=torch.int8, device="npu:0") for _ in images]
    out_scales = [torch.full((160, 16), -7, dtype=torch.float32, device="npu:0") for _ in images]
    statuses = [torch.full((160, 16), -7, dtype=torch.int32, device="npu:0") for _ in images]
    reports = []
    with ExitStack() as stack:
        graphs = []
        for image, output, scale, status in zip(images, outputs, out_scales, statuses, strict=True):
            owner = stack.enter_context(A1BorrowedLaunch(image))
            call = owner.prepare(acc, scales, ids, channel, output, scale, status, blocks=8)
            call()
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            stack.callback(graph.reset)
            with torch.npu.graph(graph):
                call()
            graphs.append(graph)
        for case in range(4):
            integer = torch.randint(-100000, 100001, (160, 1536), generator=generator, dtype=torch.int32)
            if case == 0:
                integer.zero_()
            elif case == 1:
                integer[:4].zero_()
            acc.copy_(integer)
            scales.copy_(torch.rand((scale_rows, 16), generator=generator) * 0.1)
            channel.copy_(torch.rand((128, 1536), generator=generator) * 0.001)
            ids.copy_(torch.randint(128, (20, 8), generator=generator, dtype=torch.int32))
            for graph, output, scale, status in zip(graphs, outputs, out_scales, statuses, strict=True):
                output.fill_(-113)
                scale.fill_(-7)
                status.fill_(-7)
                graph.replay()
            assert torch.equal(outputs[0].cpu(), outputs[1].cpu()), (case, "INT8")
            assert torch.equal(out_scales[0].cpu(), out_scales[1].cpu()), (case, "FP32 scale")
            assert all(not torch.count_nonzero(value.cpu()[:, 0]) for value in statuses)
            assert all(torch.all(value.cpu()[:, 1:] == -7) for value in out_scales)
            if case == 0:
                assert not torch.count_nonzero(outputs[1].cpu())
                assert not torch.count_nonzero(out_scales[1].cpu()[:, 0])
            reports.append({"case": case, "exact": True})
        for graph in graphs:
            for _ in range(5):
                graph.replay()
        torch.npu.synchronize()
        samples = [[], []]
        for repeat in range(5):
            for index in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                begin.record()
                for _ in range(10):
                    graphs[index].replay()
                end.record()
                end.synchronize()
                samples[index].append(begin.elapsed_time(end) / 10)
    report = {"scale_rows": scale_rows, "checks": reports, "scope": "standalone_graph_event_ms_not_vllm",
              "control_ms": samples[0], "candidate_ms": samples[1],
              "control_median_ms": statistics.median(samples[0]),
              "candidate_median_ms": statistics.median(samples[1])}
    (tmp_path / "retained-swiglu.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
