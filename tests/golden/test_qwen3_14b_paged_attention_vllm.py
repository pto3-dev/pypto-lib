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
"""Golden validation for the Qwen3-14B one-page vLLM attention bridge."""

import os
import sys
from pathlib import Path

import pytest
import torch
import torch_npu  # noqa: F401
from golden import TensorSpec, run

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "models" / "qwen3_14b"))
from paged_attention_vllm import (  # noqa: E402
    BATCH_PAD,
    BLOCK_SIZE,
    HEAD_DIM,
    KV_SIZE,
    NUM_KV_HEADS,
    NUM_Q_HEADS,
    Q_PER_KV,
    Q_SIZE,
    SCALE,
    qwen3_paged_attention_one_page,
)


def _golden(values: dict[str, torch.Tensor]) -> None:
    q = values["q"].float().view(BATCH_PAD, NUM_Q_HEADS, HEAD_DIM)
    key = values["key_blocks"].float().view(BATCH_PAD, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
    value = values["value_blocks"].float().view(BATCH_PAD, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
    output = torch.empty_like(q)
    for batch_idx, seq_len in enumerate(values["seq_lens"].tolist()):
        for kv_head in range(NUM_KV_HEADS):
            q_start = kv_head * Q_PER_KV
            query = q[batch_idx, q_start : q_start + Q_PER_KV]
            key_head = key[batch_idx, :seq_len, kv_head]
            value_head = value[batch_idx, :seq_len, kv_head]
            scores = torch.matmul(query, key_head.transpose(0, 1)) * SCALE
            exp_scores = torch.exp(scores - scores.max(dim=-1, keepdim=True).values)
            probabilities = exp_scores / exp_scores.sum(dim=-1, keepdim=True)
            context = torch.matmul(probabilities, value_head)
            output[batch_idx, q_start : q_start + Q_PER_KV] = context
    values["out"][:] = output.to(torch.bfloat16).view(BATCH_PAD, Q_SIZE)


@pytest.mark.skipif(not torch.npu.is_available(), reason="Ascend NPU unavailable")
def test_qwen3_14b_paged_attention_vllm_golden() -> None:
    torch.manual_seed(0)
    seq_lens = torch.tensor(
        [1, 2, 7, 16, 31, 32, 47, 64, 65, 79, 96, 111, 120, 126, 127, 128],
        dtype=torch.int32,
    )
    specs = [
        TensorSpec(
            "q",
            [BATCH_PAD, Q_SIZE],
            torch.bfloat16,
            init_value=lambda: torch.randn(BATCH_PAD, Q_SIZE, dtype=torch.bfloat16) * 0.1,
        ),
        TensorSpec(
            "key_blocks",
            [BATCH_PAD, BLOCK_SIZE, KV_SIZE],
            torch.bfloat16,
            init_value=lambda: torch.randn(BATCH_PAD, BLOCK_SIZE, KV_SIZE, dtype=torch.bfloat16) * 0.1,
        ),
        TensorSpec(
            "value_blocks",
            [BATCH_PAD, BLOCK_SIZE, KV_SIZE],
            torch.bfloat16,
            init_value=lambda: torch.randn(BATCH_PAD, BLOCK_SIZE, KV_SIZE, dtype=torch.bfloat16) * 0.1,
        ),
        TensorSpec(
            "seq_lens",
            [BATCH_PAD],
            torch.int32,
            init_value=lambda: seq_lens,
        ),
        TensorSpec("out", [BATCH_PAD, Q_SIZE], torch.bfloat16),
    ]
    result = run(
        fn=qwen3_paged_attention_one_page,
        specs=specs,
        golden_fn=_golden,
        config={
            "platform": "a2a3",
            "device_id": int(os.getenv("PYPTO_TEST_DEVICE", "0")),
        },
        rtol=0.02,
        atol=0.02,
    )
    assert result.passed, result.error
