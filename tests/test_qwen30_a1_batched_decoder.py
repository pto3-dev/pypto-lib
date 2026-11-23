# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""One complete B20 decoder, including native PA cache ownership and 160-route MoE."""

import os
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="module")
def resources():
    torch_npu = pytest.importorskip("torch_npu")
    isa = os.environ.get("PTO_ISA_ROOT")
    if not isa:
        pytest.skip("PTO_ISA_ROOT is required for the allocated 310P device")
    from models.qwen3_30b_w8a8.decode_batched_modelslim_a1 import compile_a1_batched_decoder_images
    from models.qwen3_30b_w8a8.linear_modelslim_a1 import decode_a1_modelslim_scale
    from models.qwen3_30b_w8a8.weights_modelslim_a1 import A1ModelSlimWeights

    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.npu.config.allow_internal_format = True
    checkpoint = os.environ.get("A1_MODELSLIM_CHECKPOINT")
    if not checkpoint:
        pytest.skip("A1_MODELSLIM_CHECKPOINT is required")
    loader = A1ModelSlimWeights(Path(checkpoint))
    raw, experts = loader.attention(0), loader.experts(0)
    native = {name: value.npu() for name, value in raw.items()}
    weights = {}
    for kind, inner in (("qkv", 2048), ("o", 4096)):
        weights[f"{kind}_weight"] = torch_npu.npu_format_cast(native[f"{kind}_weight"], 29)
        weights[f"{kind}_scale"] = decode_a1_modelslim_scale(raw[f"{kind}_deq_scale"]).npu()
        weights[f"{kind}_bias"] = native[f"{kind}_quant_bias"]
        weights[f"{kind}_reciprocal"] = (1 / native[f"{kind}_input_scale"]).repeat(1, inner)
        weights[f"{kind}_offset"] = native[f"{kind}_input_offset"].half().repeat(1, inner)
    for name, original in (("input_norm", "input_layernorm"), ("post_norm", "post_attention_layernorm")):
        for field in ("weight", "bias"):
            weights[f"{name}_{field}"] = native[f"{original}.{field}"]
    weights.update({"q_norm": native["self_attn.q_norm.weight"], "k_norm": native["self_attn.k_norm.weight"],
                    "router": native["mlp.gate.weight"].t().contiguous(),
                    "w13": torch_npu.npu_format_cast(experts.w13.npu(), 29),
                    "w2": torch_npu.npu_format_cast(experts.w2.npu(), 29),
                    "w13_scale": experts.w13_scale.npu(), "w2_scale": experts.w2_scale.npu()})
    phase = torch.arange(4096).float()[:, None] / (1000000.0 ** (torch.arange(0, 128, 2).float() / 128))[None, :]
    weights["rope"] = torch.cat((phase.cos(), phase.sin()), -1).half().npu()
    directory = Path(__file__).parents[1] / "build_output/a1-b20-modelslim"
    images = compile_a1_batched_decoder_images(directory, Path(isa))
    torch.save(images, directory / "images.pt")
    return native, weights, experts, images


@pytest.mark.parametrize("first_layer", (True, False))
def test_b20_complete_decoder_graph(resources, first_layer, tmp_path):
    import torch_npu
    import torchair
    from test_qwen30_a1_batched_moe import _native_gmm, _native_routes

    from models.qwen3_30b_w8a8.decode_batched_modelslim_a1 import A1BatchedModelSlimDecoderLayer
    from models.qwen3_30b_w8a8.kv_cache_batched_a1 import A1BatchedMetadata

    native, weights, experts, images = resources
    generator = torch.Generator().manual_seed(310320)
    hidden = torch.zeros((20, 2048), dtype=torch.float16, device="npu:0")
    residual, output, residual_out = (torch.zeros_like(hidden) for _ in range(3))
    metadata = A1BatchedMetadata.allocate("npu:0")
    cache = tuple(torch_npu.npu_format_cast(torch.zeros((640, 32, 128, 16), dtype=torch.float16).npu(), 2)
                  for _ in range(2))
    layer = A1BatchedModelSlimDecoderLayer(images, hidden, residual, metadata, cache, weights, output,
                                          residual_out, first_layer=first_layer)
    config = torchair.CompilerConfig()
    config.mode = "reduce-overhead"
    config.debug.run_eagerly = True
    config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
    gmm = torch.compile(_native_gmm, backend=torchair.get_npu_backend(compiler_config=config), fullgraph=True)
    native_w13, native_w2 = experts.w13.npu(), experts.w2.npu()
    native_router = torch_npu.npu_format_cast(native["mlp.gate.weight"], 29)
    graph = torch.npu.NPUGraph()

    def norm(x, skip, prefix, fused):
        if fused:
            value, _, skip_out = torch_npu.npu_add_rms_norm(x, skip, native[f"{prefix}.weight"], 1e-6)
        else:
            value, _ = torch_npu.npu_rms_norm(x, native[f"{prefix}.weight"], 1e-6)
            skip_out = x.clone()
        return value + native[f"{prefix}.bias"], skip_out

    def project(x, kind):
        quant = torch_npu.npu_quantize(x, 1/native[f"{kind}_input_scale"],
                                       native[f"{kind}_input_offset"].half(), torch.qint8, -1, False)
        return torch_npu.npu_quant_matmul(quant, weights[f"{kind}_weight"].t(), native[f"{kind}_deq_scale"].flatten(),
                                          bias=native[f"{kind}_quant_bias"].flatten(), output_dtype=torch.float16)

    def pack(dense):
        return dense.reshape(640, 128, 32, 16).permute(0, 2, 1, 3).contiguous()

    try:
        layer.enqueue()
        torch.npu.synchronize()
        with torch.npu.graph(graph):
            layer.enqueue()
        for case in range(2):
            hidden.copy_(torch.randn((20, 2048), generator=generator).half())
            residual.copy_(torch.randn((20, 2048), generator=generator).half())
            lengths = torch.linspace(2048 + case, 2447 + case, 20).int()
            table = torch.randperm(640, generator=generator).reshape(20, 32).int()
            slots = table[torch.arange(20), (lengths-1)//128] * 128 + (lengths-1)%128
            metadata.update((lengths-1).npu(), slots.npu(), lengths.npu(), table.npu())
            dense_k = (torch.randn((640*128, 512), generator=generator) / 2).half()
            dense_v = (torch.randn((640*128, 512), generator=generator) / 2).half()
            cache[0].copy_(pack(dense_k))
            cache[1].copy_(pack(dense_v))
            normalized, skip = norm(hidden, residual, "input_layernorm", not first_layer)
            qkv = project(normalized, "qkv")
            selected = weights["rope"][(lengths-1).long().npu()]
            cosine = selected[:, :64].repeat(1, 2).reshape(20, 1, 1, 128)
            sine = selected[:, 64:].repeat(1, 2).reshape(20, 1, 1, 128)
            rotated = {}
            for kind, start, heads in (("q", 0, 32), ("k", 4096, 4)):
                x = qkv[:, start:start+heads*128].contiguous().reshape(20, 1, heads, 128)
                normalized_heads, _ = torch_npu.npu_rms_norm(x, native[f"self_attn.{kind}_norm.weight"], 1e-6)
                rotated[kind], _ = torch_npu.npu_apply_rotary_pos_emb(normalized_heads.clone(), normalized_heads.clone(),
                                                                 cosine, sine, rotary_mode="half")
            dense_k[slots.long()] = rotated["k"].cpu().reshape(20, 512)
            dense_v[slots.long()] = qkv[:, 4608:].cpu()
            reference_cache = tuple(torch_npu.npu_format_cast(pack(dense).npu(), 2) for dense in (dense_k, dense_v))
            attention = torch.zeros((20, 32, 128), dtype=torch.float16, device="npu:0")
            torch_npu._npu_paged_attention(query=rotated["q"].reshape(20, 32, 128), key_cache=reference_cache[0],
                                          value_cache=reference_cache[1], num_kv_heads=4, num_heads=32,
                                          scale_value=128**-0.5, block_table=table.npu(), context_lens=lengths.npu(), out=attention)
            projected = project(attention.reshape(20, 4096), "o")
            post, expected_residual = norm(projected, skip, "post_attention_layernorm", True)
            logits = torch.nn.functional.linear(post, native_router)
            probabilities, ids = _native_routes(logits)
            order = torch.argsort(ids.flatten(), stable=True)
            groups = (ids.flatten()[:, None] == torch.arange(128, device="npu:0")[None, :]).sum(0).long().cumsum(0)
            routed = post.repeat_interleave(8, dim=0)[order].contiguous()
            up_gate = gmm(routed, native_w13, weights["w13_scale"], groups)
            expert_out = gmm(torch_npu.npu_swiglu(up_gate), native_w2, weights["w2_scale"], groups)
            inverse = torch.empty(160, dtype=torch.int32, device="npu:0")
            inverse.scatter_(0, order, torch.arange(160, dtype=torch.int32, device="npu:0"))
            expected = torch_npu.npu_moe_token_unpermute(permuted_tokens=expert_out, sorted_indices=inverse, probs=probabilities)
            frozen = {"norm": normalized.cpu(), "qkv": qkv.cpu(), "attention": attention.cpu().reshape(20, 4096),
                      "o": projected.cpu(), "post_norm": post.cpu(), "residual": expected_residual.cpu(), "output": expected.cpu()}
            graph.replay()
            actual = {name: layer.buffers[name].cpu() for name in ("norm", "qkv", "attention", "o", "post_norm")}
            actual.update({"residual": residual_out.cpu(), "output": output.cpu()})
            for name, value in actual.items():
                if not torch.equal(value, frozen[name]):
                    fixture = tmp_path / f"decoder-{first_layer}-{case}-{name}.pt"
                    torch.save({"actual": actual, "expected": frozen}, fixture)
                    pytest.fail(f"First unequal stage {name}: {torch.count_nonzero(value != frozen[name])} mismatches; {fixture}")
            assert torch.equal(cache[0].cpu(), pack(dense_k))
            assert torch.equal(cache[1].cpu(), pack(dense_v))
            assert all(not torch.count_nonzero(status.cpu()[:, 0]) for status in layer.status.values())
    finally:
        graph.reset()
        layer.close()
