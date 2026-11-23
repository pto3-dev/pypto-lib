# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""Twenty independent device positions and exact native head-Norm/RoPE replay."""

import os
from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize("batched_heads", (False, True))
def test_b20_modelslim_head_norm_rope_graph(tmp_path, batched_heads):
    torch_npu = pytest.importorskip("torch_npu")
    isa = os.environ.get("PTO_ISA_ROOT")
    if not isa:
        pytest.skip("PTO_ISA_ROOT is required for the allocated 310P device")
    from models.qwen3_30b_w8a8.qk_batched_modelslim_a1 import (
        A1BatchedHeadNormRope,
        compile_a1_batched_qk_images,
    )
    from models.qwen3_30b_w8a8.weights_modelslim_a1 import A1ModelSlimWeights

    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    checkpoint = os.environ.get("A1_MODELSLIM_CHECKPOINT")
    if not checkpoint:
        pytest.skip("A1_MODELSLIM_CHECKPOINT is required")
    weights = A1ModelSlimWeights(Path(checkpoint)).attention(0)
    generator = torch.Generator().manual_seed(310320)
    qkv = torch.zeros((20, 5120), dtype=torch.float16, device="npu:0")
    positions = torch.zeros((20, 8), dtype=torch.int32, device="npu:0")
    frequency = 1 / (1000000.0 ** (torch.arange(0, 128, 2).float() / 128))
    phase = torch.arange(4096).float()[:, None] * frequency[None, :]
    table = torch.cat((phase.cos(), phase.sin()), dim=-1).half().npu()
    images = compile_a1_batched_qk_images(tmp_path, Path(isa), batched_heads=batched_heads)
    gamma = {kind: weights[f"self_attn.{kind}_norm.weight"].npu() for kind in ("q", "k")}
    owner = A1BatchedHeadNormRope(images, qkv, positions, table, gamma["q"], gamma["k"])
    assert len(owner._calls) == (4 if batched_heads else 42)
    graph = torch.npu.NPUGraph()
    try:
        owner.enqueue()
        torch.npu.synchronize()
        with torch.npu.graph(graph):
            owner.enqueue()
        for case in range(2):
            qkv.copy_(torch.randn((20, 5120), generator=generator).half())
            selected = torch.arange(20) * 19 + 2048 + case
            positions[:, 0].copy_(selected)
            selected_cache = table[selected.to("npu:0")]
            cosine = selected_cache[:, :64].repeat(1, 2).reshape(20, 1, 1, 128)
            sine = selected_cache[:, 64:].repeat(1, 2).reshape(20, 1, 1, 128)
            expected = {}
            for kind, start, heads in (("q", 0, 32), ("k", 4096, 4)):
                source = qkv[:, start:start + heads * 128].contiguous().reshape(20, heads, 128)
                norm, _ = torch_npu.npu_rms_norm(source, gamma[kind], 1e-6)
                rotated, _ = torch_npu.npu_apply_rotary_pos_emb(
                    norm.clone().reshape(20, 1, heads, 128), norm.clone().reshape(20, 1, heads, 128),
                    cosine, sine, rotary_mode="half",
                )
                expected[kind] = rotated.reshape(20, heads, 128).cpu()
                expected[f"{kind}_norm"] = norm.cpu()
            graph.replay()
            b = owner.buffers
            assert torch.equal(b["value"].cpu(), qkv[:, 4608:].cpu())
            assert torch.equal(b["rope"].cpu(), selected_cache.cpu())
            for kind, heads, padded in (("q", 32, 32), ("k", 4, 8)):
                for field, reference in (("norm", expected[f"{kind}_norm"]), ("rotated", expected[kind])):
                    actual = b[f"{kind}_{field}"].reshape(20, padded, 128)[:, :heads].cpu()
                    if not torch.equal(actual, reference):
                        fixture = tmp_path / f"qk-b20-{kind}-{field}-{case}.pt"
                        torch.save({"actual": actual, "expected": reference, "qkv": qkv.cpu()}, fixture)
                        pytest.fail(f"{kind}/{field}: mismatches={torch.count_nonzero(actual != reference)}, fixture={fixture}")
            assert torch.equal(b["query"][:, :8].cpu().reshape(20, 32, 128), expected["q"])
            assert not torch.count_nonzero(b["query"][:, 8:].cpu())
            assert torch.equal(b["key"].cpu().reshape(20, 4, 128), expected["k"])
            assert all(not torch.count_nonzero(status.cpu()[:, 0]) for status in owner.status.values())
    finally:
        graph.reset()
        owner.close()
