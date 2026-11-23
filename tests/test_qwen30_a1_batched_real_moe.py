"""Replay an actual vLLM MoE boundary without changing the native reference."""

import os
from pathlib import Path

import pytest
import torch


def test_actual_native_b20_moe(tmp_path):
    root = os.environ.get("PTO3_B20_REAL_FIXTURE")
    if not root:
        pytest.skip("PTO3_B20_REAL_FIXTURE must name the isolated vLLM trace directory")
    import torch_npu
    import torchair
    from test_qwen30_a1_batched_moe import _native_gmm, _native_routes

    from models.qwen3_30b_w8a8.moe_batched_modelslim_a1 import A1BatchedModelSlimMoe

    root = Path(root)
    state = torch.load(root / "layer-16-resident.pt", map_location="cpu", weights_only=True)
    snapshot = torch.load(Path(os.environ.get("PTO3_B20_REAL_SNAPSHOT", str(root / "step-1.pt"))),
                          map_location="cpu", weights_only=True)
    weights = state["weights"]
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    hidden = snapshot["stages"][16]["post_norm"][0].npu()
    raw_router = weights["mlp.gate.weight"]
    router = raw_router.t().contiguous().npu()
    native_router = torch_npu.npu_format_cast(raw_router.npu(), 29)
    w13, w2 = [torch_npu.npu_format_cast(weights[f"mlp.experts.{name}_weight"].npu(), 29)
               for name in ("w13", "w2")]
    s13, s2 = [weights[f"mlp.experts.{name}_weight_scale"].npu() for name in ("w13", "w2")]
    image_path = os.environ.get("PTO3_B20_TEST_IMAGES")
    if image_path:
        images = torch.load(Path(image_path), map_location="cpu", weights_only=True)["moe"]
    else:
        from models.qwen3_30b_w8a8.moe_batched_modelslim_a1 import compile_a1_batched_moe_images

        images = compile_a1_batched_moe_images(tmp_path / "images", Path(os.environ["PTO_ISA_ROOT"]))
    output = torch.zeros_like(hidden)
    config = torchair.CompilerConfig()
    config.mode = "reduce-overhead"
    config.debug.run_eagerly = True
    config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
    gmm = torch.compile(_native_gmm, backend=torchair.get_npu_backend(compiler_config=config), fullgraph=True)
    model = A1BatchedModelSlimMoe(images, hidden, router, w13, s13, w2, s2, output)
    graph = torch.npu.NPUGraph()
    try:
        model.enqueue()
        torch.npu.synchronize()
        with torch.npu.graph(graph):
            model.enqueue()
        logits = torch.nn.functional.linear(hidden, native_router)
        probabilities, ids = _native_routes(logits)
        if "topk" in snapshot:
            native_logits, native_selected, native_ids = snapshot["topk"][16]
            print("ACTUAL_ROUTER", native_logits.dtype,
                  int(torch.count_nonzero(native_logits != logits.cpu())),
                  float((native_logits.float()-logits.cpu().float()).abs().max()))
            print("ACTUAL_ROUTER_IDS", int(torch.count_nonzero(native_ids.int() != ids.cpu())))
            print("ACTUAL_ROUTER_WEIGHT", int(torch.count_nonzero(
                native_selected / native_selected.sum(-1,keepdim=True) != probabilities.cpu())))
        order = torch.argsort(ids.flatten(), stable=True)
        groups = (ids.flatten()[:, None] == torch.arange(128, device="npu:0")[None, :]).sum(0).long().cumsum(0)
        expanded = hidden.repeat_interleave(8, dim=0)[order].contiguous()
        up_gate = gmm(expanded, w13, s13, groups)
        expert_output = gmm(torch_npu.npu_swiglu(up_gate), w2, s2, groups)
        inverse = torch.empty(160, dtype=torch.int32, device="npu:0")
        inverse.scatter_(0, order, torch.arange(160, dtype=torch.int32, device="npu:0"))
        expected = torch_npu.npu_moe_token_unpermute(permuted_tokens=expert_output,
                                                    sorted_indices=inverse, probs=probabilities)
        fused_hidden, fused_indices, fused_groups, _ = torch_npu.npu_moe_init_routing_v2(
            hidden, ids, active_num=160, expert_num=128, drop_pad_mode=0,
            active_expert_range=[0,128], quant_mode=-1, row_idx_type=0)
        fused_up_gate = gmm(fused_hidden, w13, s13, fused_groups.long())
        fused_expert = gmm(torch_npu.npu_swiglu(fused_up_gate), w2, s2, fused_groups.long())
        fused_output = torch_npu.npu_moe_token_unpermute(permuted_tokens=fused_expert,
                                                       sorted_indices=fused_indices, probs=probabilities)
        print("FUSED_DISPATCH_NATIVE_MISMATCHES", int(torch.count_nonzero(fused_output.cpu() != snapshot["layers"][16][0])))
        print("FUSED_DISPATCH_VS_STABLE", int(torch.count_nonzero(fused_output != expected)))
        graph.replay()
        actual = {name: model.buffers[name].cpu() for name in model.buffers}
        expected_stages = {"logits": logits.cpu(), "ids": ids.cpu(), "weights": probabilities.float().cpu(),
                           "output": expected.cpu(), "expert_y": expert_output[inverse.long()].cpu(),
                           "up_gate": up_gate[inverse.long()].cpu()}
        actual["output"] = output.cpu()
        torch.save({"actual": actual, "expected": expected_stages,
                    "full_vllm_hidden": snapshot["layers"][16][0]}, tmp_path / "actual-vllm-moe.pt")
        print("FULL_NATIVE_GMM_MISMATCHES", int(torch.count_nonzero(expected.cpu() != snapshot["layers"][16][0])))
        for name in ("logits", "ids", "weights", "expert_y", "output"):
            a, b = actual[name], expected_stages[name]
            print(name, int(torch.count_nonzero(a != b)), float((a.float()-b.float()).abs().max()))
        for name in ("logits", "ids", "weights", "expert_y", "output"):
            assert torch.equal(actual[name], expected_stages[name]), (name, tmp_path / "actual-vllm-moe.pt")
        assert torch.equal(expected.cpu(), snapshot["layers"][16][0]), "Independent GMM must match actual native vLLM"
    finally:
        graph.reset()
        model.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
