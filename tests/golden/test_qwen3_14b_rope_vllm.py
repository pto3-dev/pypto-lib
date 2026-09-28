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
# -----------------------------------------------------------------------------
"""Golden validation for Qwen3-14B RoPE."""

import os
import sys
from pathlib import Path

import pytest
import torch
import torch_npu  # noqa: F401
from golden import TensorSpec, run

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "models" / "qwen3_14b"))
from rope_vllm import (  # noqa: E402
    BATCH_PAD,
    HALF_DIM,
    HEAD_DIM,
    KV_SIZE,
    NUM_KV_HEADS,
    NUM_Q_HEADS,
    Q_SIZE,
    qwen3_rope_decode,
)


def _rope(
    value: torch.Tensor,
    heads: int,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    shaped = value.float().view(BATCH_PAD, heads, HEAD_DIM)
    lo, hi = shaped[..., :HALF_DIM], shaped[..., HALF_DIM:]
    cos = cos.float().unsqueeze(1)
    sin = sin.float().unsqueeze(1)
    return (
        torch.cat((lo * cos - hi * sin, hi * cos + lo * sin), dim=-1)
        .to(torch.bfloat16)
        .reshape(BATCH_PAD, heads * HEAD_DIM)
    )


def _golden(values: dict[str, torch.Tensor]) -> None:
    values["q_out"][:] = _rope(values["q"], NUM_Q_HEADS, values["rope_cos"], values["rope_sin"])
    values["k_out"][:] = _rope(values["k"], NUM_KV_HEADS, values["rope_cos"], values["rope_sin"])


@pytest.mark.skipif(not torch.npu.is_available(), reason="Ascend NPU unavailable")
def test_qwen3_14b_rope_vllm_golden() -> None:
    torch.manual_seed(0)
    specs = [
        TensorSpec("q", [BATCH_PAD, Q_SIZE], torch.bfloat16, init_value=torch.randn),
        TensorSpec("k", [BATCH_PAD, KV_SIZE], torch.bfloat16, init_value=torch.randn),
        TensorSpec(
            "rope_cos",
            [BATCH_PAD, HALF_DIM],
            torch.bfloat16,
            init_value=lambda: torch.rand(BATCH_PAD, HALF_DIM, dtype=torch.bfloat16),
        ),
        TensorSpec(
            "rope_sin",
            [BATCH_PAD, HALF_DIM],
            torch.bfloat16,
            init_value=lambda: torch.rand(BATCH_PAD, HALF_DIM, dtype=torch.bfloat16),
        ),
        TensorSpec("q_out", [BATCH_PAD, Q_SIZE], torch.bfloat16),
        TensorSpec("k_out", [BATCH_PAD, KV_SIZE], torch.bfloat16),
    ]
    result = run(
        fn=qwen3_rope_decode,
        specs=specs,
        golden_fn=_golden,
        config={"platform": "a2a3", "device_id": int(os.getenv("PYPTO_TEST_DEVICE", "0"))},
        rtol=0.01,
        atol=0.01,
    )
    assert result.passed, result.error
