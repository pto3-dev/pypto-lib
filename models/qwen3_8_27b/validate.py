# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compile and validate PTO text operators on one NPU, retaining DFX artifacts."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from golden import TensorSpec, run

from . import reference
from .attention import make_attention, make_rope
from .dense import make_gemm, make_norm, make_pointwise
from .kv_cache import make_append
from .layout import make_embedding, make_gdn_unpack, make_query_gate_unpack
from .linear_attention import make_conv, make_gates, make_qk_norm, make_recurrent


def random(shape, dtype=torch.bfloat16):
    return torch.randn(shape, dtype=dtype)


def cases():
    torch.manual_seed(183)
    starts = torch.tensor([0, 2, 5, 6], dtype=torch.int32)
    indices = torch.tensor([2, 0, -1], dtype=torch.int32)
    initial = torch.tensor([1, 0, 1], dtype=torch.int32)

    def conv_golden(d):
        d["out"][:] = reference.causal_conv(
            d["x"], d["weight"].T, d["state"], d["starts"], d["indices"], d["initial"]
        )

    conv = {
        "x": random([6, 10240]),
        "weight": random([4, 10240]),
        "state": random([4, 3, 10240]),
        "starts": starts,
        "indices": indices,
        "initial": initial,
        "out": torch.empty(6, 10240, dtype=torch.bfloat16),
    }
    yield "conv", make_conv(10240), conv, conv_golden, 0.02, 0.03

    def gates_golden(d):
        d["alpha"][:], d["beta"][:] = reference.gating(d["a"], d["b"], d["a_log"], d["dt_bias"])

    gates = {
        "a": random([6, 48]),
        "b": random([6, 48]),
        "a_log": random([1, 48], torch.float32),
        "dt_bias": random([1, 48], torch.float32),
        "alpha": torch.empty(6, 48),
        "beta": torch.empty(6, 48),
    }
    yield "gates", make_gates(48), gates, gates_golden, 1e-4, 1e-5

    def qk_golden(d):
        q, k = d["q"].float(), d["k"].float()
        d["q_out"][:] = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) / 128**0.5
        d["k_out"][:] = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)

    qk = {
        "q": random([6, 16, 128]),
        "k": random([6, 16, 128]),
        "q_out": torch.empty(6, 16, 128),
        "k_out": torch.empty(6, 16, 128),
    }
    yield "qk_norm", make_qk_norm(16), qk, qk_golden, 1e-4, 1e-5

    def recurrent_golden(d):
        # The recurrent kernel receives FP32 Q/K after normalization.
        out = torch.zeros_like(d["out"])
        for req, slot in enumerate(d["indices"].tolist()):
            if slot < 0:
                continue
            s = d["state"][slot].clone() if d["initial"][req] else torch.zeros_like(d["state"][slot])
            for t in range(int(d["starts"][req]), int(d["starts"][req + 1])):
                qt = d["q"][t].repeat_interleave(3, dim=0)
                kt = d["k"][t].repeat_interleave(3, dim=0)
                s *= d["alpha"][t, :, None, None]
                error = d["v"][t].float() - (s * kt[:, None, :]).sum(-1)
                s += (error * d["beta"][t, :, None])[:, :, None] * kt[:, None, :]
                out[t] = (s * qt[:, None, :]).sum(-1).to(out.dtype)
            d["state"][slot] = s
        d["out"][:] = out

    q = F.normalize(random([6, 16, 128], torch.float32), dim=-1) / 128**0.5
    k = F.normalize(random([6, 16, 128], torch.float32), dim=-1)
    recurrent = {
        "q": q,
        "k": k,
        "v": random([6, 48, 128]),
        "alpha": torch.rand(6, 48),
        "beta": torch.rand(6, 48),
        "state": random([4, 48, 128, 128], torch.float32),
        "starts": starts,
        "indices": indices,
        "initial": initial,
        "out": torch.empty(6, 48, 128, dtype=torch.bfloat16),
    }
    yield "recurrent", make_recurrent(16, 48), recurrent, recurrent_golden, 0.02, 0.02

    def ascend_recurrent_golden(d):
        original = d["state"]
        d["state"] = original.transpose(-1, -2).contiguous()
        recurrent_golden(d)
        original.copy_(d["state"].transpose(-1, -2))
        d["state"] = original

    ascend_recurrent = {name: value.clone() for name, value in recurrent.items()}
    ascend_recurrent["state"] = recurrent["state"].transpose(-1, -2).contiguous()
    yield (
        "recurrent_ascend",
        make_recurrent(16, 48, "KV"),
        ascend_recurrent,
        ascend_recurrent_golden,
        0.02,
        0.02,
    )

    def kv_golden(d):
        for t, slot in enumerate(d["slots"].tolist()):
            if slot >= 0:
                d["key_cache"][slot] = d["k"][t]
                d["value_cache"][slot] = d["v"][t]

    kv = {
        "k": random([4, 1024]),
        "v": random([4, 1024]),
        "key_cache": random([896, 1024]),
        "value_cache": random([896, 1024]),
        "slots": torch.tensor([383, 128, -1, 769], dtype=torch.int64),
    }
    yield "kv_append", make_append(4), kv, kv_golden, 0, 0

    def attention_golden(d):
        cache = torch.stack((d["key_cache"], d["value_cache"])).view(2, 7, 128, 4, 256)
        d["out"][:] = reference.paged_attention(d["q"], cache, d["table"], d["seq_lens"], d["starts"])

    attn = {
        "q": random([3, 24, 256]),
        "key_cache": random([896, 4, 256]),
        "value_cache": random([896, 4, 256]),
        "table": torch.tensor([[5, 1, 6], [3, 0, 2]], dtype=torch.int32),
        "seq_lens": torch.tensor([257, 13], dtype=torch.int32),
        "starts": torch.tensor([0, 2, 3], dtype=torch.int32),
        "out": torch.empty(3, 24, 256, dtype=torch.bfloat16),
    }
    yield "attention", make_attention(24, 4), attn, attention_golden, 0.03, 0.02

    def native_block_attention_golden(d):
        cache = torch.stack((d["key_cache"], d["value_cache"])).view(2, 4, 1536, 4, 256)
        d["out"][:] = reference.paged_attention(d["q"], cache, d["table"], d["seq_lens"], d["starts"])

    native_block = {
        "q": random([3, 24, 256]),
        "key_cache": random([6144, 4, 256]),
        "value_cache": random([6144, 4, 256]),
        "table": torch.tensor([[3, 0], [2, 1]], dtype=torch.int32),
        "seq_lens": torch.tensor([1601, 1537], dtype=torch.int32),
        "starts": torch.tensor([0, 2, 3], dtype=torch.int32),
        "out": torch.empty(3, 24, 256, dtype=torch.bfloat16),
    }
    yield (
        "attention_native_block",
        make_attention(24, 4, 1536),
        native_block,
        native_block_attention_golden,
        0.03,
        0.02,
    )

    for physical_block in (128, 1536):
        lengths = torch.tensor([1, 2, 5, 13, 127, 128, 129, 257, 1537, 1601], dtype=torch.int32)
        query_starts = torch.tensor([0, 1, 3, 8, 9, 10, 11, 12, 13, 14, 15], dtype=torch.int32)
        blocks = 16 if physical_block == 128 else 4
        columns = (1601 + physical_block - 1) // physical_block
        keys = random([blocks * physical_block, 4, 256]) + 1
        values = random([blocks * physical_block, 4, 256])
        table = torch.stack([torch.randperm(blocks)[:columns] for _ in range(10)]).int()
        short = {
            "q": random([15, 24, 256]) + 1,
            "key_cache": keys,
            "value_cache": values,
            "table": table,
            "seq_lens": lengths,
            "starts": query_starts,
            "out": torch.empty(15, 24, 256, dtype=torch.bfloat16),
        }

        def short_golden(d, size=physical_block, count=blocks):
            cache = torch.stack((d["key_cache"], d["value_cache"])).view(2, count, size, 4, 256)
            d["out"][:] = reference.paged_attention(d["q"], cache, d["table"], d["seq_lens"], d["starts"])

        yield (
            f"attention_short_contexts_{physical_block}",
            make_attention(24, 4, physical_block),
            short,
            short_golden,
            0.03,
            0.02,
        )

    def norm_golden(d):
        x = d["x"].float()
        d["out"][:] = (
            x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * (1 + d["weight"].float())
        ).to(d["out"].dtype)

    norm = {
        "x": random([3, 5120]),
        "weight": random([1, 5120]),
        "z": random([3, 5120]),
        "out": torch.empty(3, 5120, dtype=torch.bfloat16),
    }
    yield "norm", make_norm(5120), norm, norm_golden, 0.015, 0.015

    def native_round_golden(d):
        x = d["x"].float()
        normalized = (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)).bfloat16().float()
        gamma = (1 + d["weight"].float()).bfloat16().float()
        d["out"][:] = (normalized * gamma).bfloat16()

    native_round = {name: value.clone() for name, value in norm.items()}
    yield (
        "native_round_norm",
        make_norm(5120, native_round=True),
        native_round,
        native_round_golden,
        0.003,
        0.001,
    )

    def gated_norm_golden(d):
        x = d["x"].float()
        normalized = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
        d["out"][:] = (normalized * d["weight"].float() * F.silu(d["z"].float())).to(d["out"].dtype)

    gated_norm = {
        "x": random([3 * 48, 128]),
        "weight": random([1, 128]),
        "z": random([3 * 48, 128]),
        "out": torch.empty(3 * 48, 128, dtype=torch.bfloat16),
    }
    yield "gated_norm", make_norm(128, 0.0, True), gated_norm, gated_norm_golden, 0.015, 0.02

    def gemm_golden(d):
        d["out"][:] = (d["x"].float() @ d["weight"].float().T).to(d["out"].dtype)

    gemm = {
        "x": random([3, 5120]),
        "weight": random([1536, 5120]),
        "out": torch.empty(3, 1536, dtype=torch.bfloat16),
    }
    yield "gemm", make_gemm(5120, 1536), gemm, gemm_golden, 0.025, 0.1

    def pointwise_golden(d):
        d["out"][:] = (F.silu(d["x"].float()) * d["y"].float()).to(d["out"].dtype)

    pointwise = {
        "x": random([3, 17408]),
        "y": random([3, 17408]),
        "out": torch.empty(3, 17408, dtype=torch.bfloat16),
    }
    yield "pointwise", make_pointwise(17408, "silu_mul"), pointwise, pointwise_golden, 0.015, 0.02

    def rope_golden(d):
        x = d["x"].float()
        c = d["cos"][d["positions"].long()].float()[:, None]
        s = d["sin"][d["positions"].long()].float()[:, None]
        result = x.clone()
        result[..., :32] = x[..., :32] * c - x[..., 32:64] * s
        result[..., 32:64] = x[..., 32:64] * c + x[..., :32] * s
        d["out"][:] = result.to(d["out"].dtype)

    angles = torch.randn(64, 32)
    rope = {
        "x": random([3, 24, 256]),
        "cos": angles.cos().bfloat16(),
        "sin": angles.sin().bfloat16(),
        "positions": torch.tensor([2, 19, 37], dtype=torch.int64),
        "out": torch.empty(3, 24, 256, dtype=torch.bfloat16),
    }
    yield "rope", make_rope(24), rope, rope_golden, 0.015, 0.015

    def gdn_split_golden(d):
        d["q"][:] = d["x"][:, :2048].reshape(3, 16, 128)
        d["k"][:] = d["x"][:, 2048:4096].reshape(3, 16, 128)
        d["v"][:] = d["x"][:, 4096:].reshape(3, 48, 128)

    gdn_split = {
        "x": random([3, 10240]),
        "q": torch.empty(3, 16, 128, dtype=torch.bfloat16),
        "k": torch.empty(3, 16, 128, dtype=torch.bfloat16),
        "v": torch.empty(3, 48, 128, dtype=torch.bfloat16),
    }
    yield "gdn_split", make_gdn_unpack(16, 48), gdn_split, gdn_split_golden, 0, 0

    def query_split_golden(d):
        packed = d["x"].view(3, 24, 512)
        d["q"][:] = packed[..., :256]
        d["gate"][:] = packed[..., 256:].reshape(3, 6144)

    query_split = {
        "x": random([3, 12288]),
        "q": torch.empty(3, 24, 256, dtype=torch.bfloat16),
        "gate": torch.empty(3, 6144, dtype=torch.bfloat16),
    }
    yield "query_split", make_query_gate_unpack(24), query_split, query_split_golden, 0, 0

    def embedding_golden(d):
        d["out"][:] = F.embedding(d["ids"], d["weight"])

    embed = {
        "ids": torch.tensor([256, 0, 183], dtype=torch.int64),
        "weight": random([257, 5120]),
        "out": torch.empty(3, 5120, dtype=torch.bfloat16),
    }
    yield "embedding", make_embedding(257), embed, embedding_golden, 0, 0

    gate_projection = {
        "x": random([3, 5120]),
        "weight": random([48, 5120]),
        "out": torch.empty(3, 48, dtype=torch.bfloat16),
    }
    yield "gemm_gates", make_gemm(5120, 48), gate_projection, gemm_golden, 0.025, 0.1

    def add_golden(d):
        d["out"][:] = d["x"] + d["y"]

    add = {"x": random([3, 5120]), "y": random([3, 5120]), "out": torch.empty(3, 5120, dtype=torch.bfloat16)}
    yield "residual_add", make_pointwise(5120, "add"), add, add_golden, 0, 0

    def sigmoid_golden(d):
        d["out"][:] = d["x"].sigmoid() * d["y"]

    sigmoid = {
        "x": random([3, 6144]),
        "y": random([3, 6144]),
        "out": torch.empty(3, 6144, dtype=torch.bfloat16),
    }
    yield "attention_gate", make_pointwise(6144, "sigmoid_mul"), sigmoid, sigmoid_golden, 0.015, 0.015


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--case", default="all")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--swimlane", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--golden-root", type=Path)
    args = parser.parse_args()
    manifest = args.output / "summary.json"
    summary = [] if args.case == "all" or not manifest.exists() else json.loads(manifest.read_text())
    selected = set(args.case.split(","))
    for name, fn, data, golden, rtol, atol in cases():
        if "all" not in selected and name not in selected:
            continue
        specs = [
            TensorSpec(key, list(value.shape), value.dtype, init_value=lambda saved=value: saved.clone())
            for key, value in data.items()
        ]
        config = dict(
            platform="a2a3",
            device_id=args.device,
            save_kernels=True,
            save_kernels_dir=str(args.output / name),
            enable_chip_swimlane=3 if args.swimlane else 0,
        )
        golden_data = None if args.golden_root is None else args.golden_root / name / "data"
        result = run(
            fn=fn,
            specs=specs,
            golden_fn=golden,
            config=config,
            save_data=True,
            golden_data=golden_data,
            rtol=rtol,
            atol=atol,
            compile_only=args.compile_only,
        )
        summary = [record for record in summary if record["case"] != name]
        summary.append(
            {
                "case": name,
                "passed": result.passed,
                "error": str(result.error),
                "device_validated": not args.compile_only,
                "device": args.device,
                "shapes": {key: list(value.shape) for key, value in data.items()},
            }
        )
        args.output.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(summary, indent=2))
        if not result.passed:
            raise SystemExit(f"Failed {name}: {result.error}")


if __name__ == "__main__":
    main()
