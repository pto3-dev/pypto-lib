# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""Exact ModelSlim loading contracts against the requested real checkpoint."""

import json
import os
from pathlib import Path

import pytest
import torch
from models.qwen3_30b_w8a8.linear_modelslim_a1 import decode_a1_modelslim_scale
from models.qwen3_30b_w8a8.weights_modelslim_a1 import A1ModelSlimWeights
from safetensors import safe_open


def test_fp16_storage_conversion_guard():
    values = torch.tensor([1.25, -2.0, 0.0], dtype=torch.bfloat16)
    actual = A1ModelSlimWeights._floating(values, (3,), "probe")
    assert actual.dtype == torch.float16
    assert torch.equal(actual, values.half())
    with pytest.raises(ValueError, match="finite FP16"):
        A1ModelSlimWeights._floating(torch.tensor([1e10]), (1,), "overflow")
    with pytest.raises(ValueError, match="floating checkpoint storage"):
        A1ModelSlimWeights._floating(torch.ones(1, dtype=torch.int32), (1,), "integer")


def test_modelslim_scale_decodes_bits_and_rejects_invalid_encoding():
    encoded = torch.tensor([[0x3f800000, 0x35800000]], dtype=torch.int64)
    expected = torch.tensor([[1.0, 2**-20]], dtype=torch.float32)
    assert torch.equal(decode_a1_modelslim_scale(encoded), expected)
    with pytest.raises(ValueError, match="upper 32 bits"):
        decode_a1_modelslim_scale(torch.tensor([0x13f800000], dtype=torch.int64))
    for value in (0, 0x7fc00000, 0xbf800000):
        with pytest.raises(ValueError, match="finite positive"):
            decode_a1_modelslim_scale(torch.tensor([value], dtype=torch.int64))


@pytest.mark.skipif(os.getenv("PYPTO_A1_MODELSLIM_TEST") != "1", reason="requires real ModelSlim weights")
def test_real_modelslim_attention_expert_loading():
    workspace = Path(__file__).resolve().parents[3]
    checkpoint = workspace / "models/Qwen3-30B-A3B-w8a8-QuaRot-310"
    verified = json.loads((workspace / "reports/quarot-310-download-verified.json").read_text())
    assert verified["revision"] == "3392bab98441ecee223325d92ecc09fc51d35d2a"
    loader = A1ModelSlimWeights(checkpoint)

    def original(name):
        with safe_open(checkpoint / loader.index[name], framework="pt", device="cpu") as shard:
            return shard.get_tensor(name)

    for layer in (0, 47):
        attention = loader.attention(layer)
        prefix = f"model.layers.{layer}."
        offset = 0
        for kind, width in (("q", 4096), ("k", 512), ("v", 512)):
            source = f"{prefix}self_attn.{kind}_proj"
            assert torch.equal(attention["qkv_weight"][offset:offset + width], original(f"{source}.weight"))
            for field in ("quant_bias", "deq_scale"):
                actual = attention[f"qkv_{field}"][0, offset:offset + width]
                assert torch.equal(actual, original(f"{source}.{field}"))
            offset += width
        for norm in ("input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm"):
            for field in ("weight", "bias"):
                name = f"{norm}.{field}"
                assert torch.equal(attention[name], original(prefix + name).half())
    experts = loader.experts(0)
    for expert in (0, 63, 127):
        prefix = f"model.layers.0.mlp.experts.{expert}"
        for kind, start in (("gate_proj", 0), ("up_proj", 768)):
            assert torch.equal(experts.w13[expert, start:start + 768], original(f"{prefix}.{kind}.weight"))
            expected = original(f"{prefix}.{kind}.weight_scale").float().flatten()
            assert torch.equal(experts.w13_scale[expert, start:start + 768], expected)
        assert torch.equal(experts.w2[expert], original(f"{prefix}.down_proj.weight"))
        expected_scale = original(f"{prefix}.down_proj.weight_scale").float().flatten()
        assert torch.equal(experts.w2_scale[expert], expected_scale)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
