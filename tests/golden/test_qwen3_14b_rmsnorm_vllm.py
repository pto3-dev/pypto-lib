# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under
# the terms and conditions of CANN Open Software License Agreement Version 2.0.
# Please refer to the License for details. You may not use this file except in
# compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------
"""Golden checks for vLLM-compatible Qwen3 Decode RMSNorm entries."""

import os
import sys
from pathlib import Path

import pytest
import torch

from golden import TensorSpec, run

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "models" / "qwen3_14b"))
from rmsnorm_vllm import (  # noqa: E402
    BATCH_PAD,
    HIDDEN,
    qwen3_add_rmsnorm_decode,
    qwen3_rmsnorm_decode,
)


def _plain_golden(values: dict[str, torch.Tensor]) -> None:
    x = values["x"].float()
    variance = x.square().mean(dim=-1, keepdim=True)
    values["out"][:] = (x * torch.rsqrt(variance + 1e-6) * values["weight"].float()).to(torch.bfloat16)


def _fused_golden(values: dict[str, torch.Tensor]) -> None:
    summed = (values["x"].float() + values["residual"].float()).to(torch.bfloat16)
    values["residual_out"][:] = summed
    summed_fp32 = summed.float()
    variance = summed_fp32.square().mean(dim=-1, keepdim=True)
    values["out"][:] = (summed_fp32 * torch.rsqrt(variance + 1e-6) * values["weight"].float()).to(
        torch.bfloat16
    )


@pytest.mark.parametrize("with_residual", [False, True])
def test_qwen3_14b_rmsnorm_vllm_golden(with_residual: bool) -> None:
    torch.manual_seed(20260924)
    specs = [
        TensorSpec("x", [BATCH_PAD, HIDDEN], torch.bfloat16, init_value=torch.randn),
    ]
    if with_residual:
        specs.append(TensorSpec("residual", [BATCH_PAD, HIDDEN], torch.bfloat16, init_value=torch.randn))
    specs.append(
        TensorSpec(
            "weight",
            [1, HIDDEN],
            torch.bfloat16,
            init_value=lambda: (1 + 0.1 * torch.randn(1, HIDDEN, dtype=torch.float32)).to(torch.bfloat16),
        )
    )
    if with_residual:
        specs.extend(
            [
                TensorSpec("out", [BATCH_PAD, HIDDEN], torch.bfloat16),
                TensorSpec("residual_out", [BATCH_PAD, HIDDEN], torch.bfloat16),
            ]
        )
    else:
        specs.append(TensorSpec("out", [BATCH_PAD, HIDDEN], torch.bfloat16))
    result = run(
        fn=qwen3_add_rmsnorm_decode if with_residual else qwen3_rmsnorm_decode,
        specs=specs,
        golden_fn=_fused_golden if with_residual else _plain_golden,
        config={
            "platform": os.getenv("PYPTO_TEST_PLATFORM", "a2a3sim"),
            "device_id": int(os.getenv("PYPTO_TEST_DEVICE", "0")),
        },
        rtol=0.01,
        atol=0.01,
    )
    assert result.passed, result.error
