# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under
# the terms and conditions of CANN Open Software License Agreement Version 2.0.
# Please refer to the License for details. You may not use this file except in
# compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the
# License.
# ------------------------------------------------------------------------------
"""Golden validation for the vLLM QKV-to-fused-paged-attention adapter."""

import math
import os
import sys
from pathlib import Path

import pytest
import torch
import torch_npu  # noqa: F401
from golden import TensorSpec, run

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "models" / "qwen3_14b"))
from paged_attention_vllm import (  # noqa: E402
    BLOCK_SIZE,
    HEAD_DIM,
    KV_SIZE,
    NUM_KV_HEADS,
    NUM_Q_HEADS,
    Q_PER_KV,
    Q_SIZE,
    SCALE,
    qwen3_vllm_fused_attention,
)

BATCH = 8
CAPACITY = 384
MAX_BLOCKS = CAPACITY // BLOCK_SIZE
NUM_BLOCKS = 32
QKV_SIZE = Q_SIZE + 2 * KV_SIZE
EPS = 1.0e-6


def _rope_tables() -> tuple[torch.Tensor, torch.Tensor]:
    half = HEAD_DIM // 2
    positions = torch.arange(CAPACITY, dtype=torch.float32).view(-1, 1)
    frequencies = torch.pow(
        torch.tensor(10000.0, dtype=torch.float32),
        -torch.arange(half, dtype=torch.float32) / half,
    ).view(1, -1)
    angles = positions * frequencies
    return (
        torch.cat((angles.cos(), angles.cos()), dim=1).contiguous(),
        torch.cat((angles.sin(), angles.sin()), dim=1).contiguous(),
    )


SEQ_LENS = torch.tensor([2, 5, 65, 127, 129, 191, 257, 383], dtype=torch.int32)
BLOCK_TABLE = torch.tensor(
    [
        [31, 6, 19],
        [2, 27, 10],
        [29, 4, 21],
        [8, 25, 12],
        [23, 14, 17],
        [16, 1, 18],
        [7, 30, 11],
        [28, 5, 20],
    ],
    dtype=torch.int32,
)
SLOT_MAPPING = torch.tensor(
    [
        int(BLOCK_TABLE[b, (int(length) - 1) // BLOCK_SIZE]) * BLOCK_SIZE + (int(length) - 1) % BLOCK_SIZE
        for b, length in enumerate(SEQ_LENS)
    ],
    dtype=torch.int32,
)
ROPE_COS, ROPE_SIN = _rope_tables()


def _golden(values: dict[str, torch.Tensor]) -> None:
    qkv = values["qkv"].float()
    q, k, v = qkv.split([Q_SIZE, KV_SIZE, KV_SIZE], dim=-1)
    q = q.view(BATCH, NUM_Q_HEADS, HEAD_DIM)
    k = k.view(BATCH, NUM_KV_HEADS, HEAD_DIM)
    v = v.view(BATCH, NUM_KV_HEADS, HEAD_DIM)
    q = q * torch.rsqrt(q.square().mean(dim=-1, keepdim=True) + EPS)
    k = k * torch.rsqrt(k.square().mean(dim=-1, keepdim=True) + EPS)
    q = q * values["q_norm_w"].float().view(1, 1, HEAD_DIM)
    k = k * values["k_norm_w"].float().view(1, 1, HEAD_DIM)
    positions = values["seq_lens"].long() - 1
    cos = values["rope_cos"][positions].float().view(BATCH, 1, HEAD_DIM)
    sin = values["rope_sin"][positions].float().view(BATCH, 1, HEAD_DIM)

    def rotate(rows: torch.Tensor) -> torch.Tensor:
        lo, hi = rows[..., : HEAD_DIM // 2], rows[..., HEAD_DIM // 2 :]
        return torch.cat(
            (
                lo * cos[..., : HEAD_DIM // 2] - hi * sin[..., : HEAD_DIM // 2],
                hi * cos[..., HEAD_DIM // 2 :] + lo * sin[..., HEAD_DIM // 2 :],
            ),
            dim=-1,
        )

    q = rotate(q).to(torch.bfloat16)
    k = rotate(k).to(torch.bfloat16)
    v = v.to(torch.bfloat16)
    key_cache = values["key_cache"].view(NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
    value_cache = values["value_cache"].view(NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
    for batch_idx, slot in enumerate(values["slot_mapping"].long()):
        page, offset = divmod(int(slot), BLOCK_SIZE)
        key_cache[page, offset] = k[batch_idx]
        value_cache[page, offset] = v[batch_idx]

    output = values["out"].view(BATCH, NUM_Q_HEADS, HEAD_DIM)
    table = values["block_table"].view(BATCH, MAX_BLOCKS).long()
    for batch_idx, seq_len_value in enumerate(values["seq_lens"]):
        seq_len = int(seq_len_value)
        pages = table[batch_idx, : math.ceil(seq_len / BLOCK_SIZE)]
        for kv_head in range(NUM_KV_HEADS):
            key = key_cache[pages, :, kv_head].reshape(-1, HEAD_DIM)[:seq_len].float()
            value = value_cache[pages, :, kv_head].reshape(-1, HEAD_DIM)[:seq_len].float()
            q_start = kv_head * Q_PER_KV
            scores = q[batch_idx, q_start : q_start + Q_PER_KV].float() @ key.t()
            probabilities = torch.softmax(scores * SCALE, dim=-1)
            output[batch_idx, q_start : q_start + Q_PER_KV] = (probabilities @ value).to(torch.bfloat16)


@pytest.mark.skipif(not torch.npu.is_available(), reason="Ascend NPU unavailable")
def test_qwen3_14b_vllm_fused_attention_golden() -> None:
    torch.manual_seed(20260923)
    specs = [
        TensorSpec(
            "qkv",
            [BATCH, QKV_SIZE],
            torch.bfloat16,
            init_value=lambda: torch.randn(BATCH, QKV_SIZE, dtype=torch.bfloat16) * 0.1,
        ),
        TensorSpec(
            "key_cache",
            [NUM_BLOCKS * BLOCK_SIZE, KV_SIZE],
            torch.bfloat16,
            init_value=lambda: torch.randn(NUM_BLOCKS * BLOCK_SIZE, KV_SIZE, dtype=torch.bfloat16) * 0.1,
        ),
        TensorSpec(
            "value_cache",
            [NUM_BLOCKS * BLOCK_SIZE, KV_SIZE],
            torch.bfloat16,
            init_value=lambda: torch.randn(NUM_BLOCKS * BLOCK_SIZE, KV_SIZE, dtype=torch.bfloat16) * 0.1,
        ),
        TensorSpec(
            "block_table", [BATCH * MAX_BLOCKS], torch.int32, init_value=lambda: BLOCK_TABLE.flatten()
        ),
        TensorSpec("seq_lens", [BATCH], torch.int32, init_value=lambda: SEQ_LENS),
        TensorSpec("slot_mapping", [BATCH], torch.int32, init_value=lambda: SLOT_MAPPING),
        TensorSpec("rope_cos", [CAPACITY, HEAD_DIM], torch.float32, init_value=lambda: ROPE_COS),
        TensorSpec("rope_sin", [CAPACITY, HEAD_DIM], torch.float32, init_value=lambda: ROPE_SIN),
        TensorSpec(
            "q_norm_w", [1, HEAD_DIM], torch.float32, init_value=lambda: torch.rand(1, HEAD_DIM) * 0.4 + 0.8
        ),
        TensorSpec(
            "k_norm_w", [1, HEAD_DIM], torch.float32, init_value=lambda: torch.rand(1, HEAD_DIM) * 0.4 + 0.8
        ),
        TensorSpec("out", [BATCH, Q_SIZE], torch.bfloat16),
    ]
    result = run(
        fn=qwen3_vllm_fused_attention,
        specs=specs,
        golden_fn=_golden,
        config={"platform": "a2a3", "device_id": int(os.getenv("PYPTO_TEST_DEVICE", "0"))},
        rtol=0.02,
        atol=0.02,
    )
    assert result.passed, result.error
