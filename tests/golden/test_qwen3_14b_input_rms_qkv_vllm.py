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
"""Golden checks for the fused input RMSNorm and QKV Decode callables."""

import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401
from golden import TensorSpec, run

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "models" / "qwen3_14b"))
from input_rms_qkv_vllm import qwen3_add_input_rms_qkv_decode, qwen3_input_rms_qkv_decode  # noqa: E402
from qkv_vllm import BATCH_PAD, HIDDEN, QKV_SIZE  # noqa: E402


def _golden(values: dict[str, torch.Tensor]) -> None:
    x = values["x"]
    if "residual" in values:
        x = (x.float() + values["residual"].float()).to(torch.bfloat16)
        values["residual_out"][:] = x
    x_fp32 = x.float()
    variance = x_fp32.square().mean(dim=-1, keepdim=True)
    normed = (x_fp32 * torch.rsqrt(variance + 1e-6) * values["norm_weight"].float()).to(torch.bfloat16)
    values["qkv_out"][:] = F.linear(normed, values["qkv_weight"])


@pytest.mark.skipif(not torch.npu.is_available(), reason="Ascend NPU unavailable")
@pytest.mark.parametrize("with_residual", [False, True])
def test_qwen3_14b_input_rms_qkv_vllm_golden(with_residual: bool) -> None:
    torch.manual_seed(20260929)
    specs = [TensorSpec("x", [BATCH_PAD, HIDDEN], torch.bfloat16, init_value=torch.randn)]
    if with_residual:
        specs.append(TensorSpec("residual", [BATCH_PAD, HIDDEN], torch.bfloat16, init_value=torch.randn))
    specs.extend(
        [
            TensorSpec(
                "norm_weight",
                [1, HIDDEN],
                torch.bfloat16,
                init_value=lambda: (1 + 0.1 * torch.randn(1, HIDDEN)).to(torch.bfloat16),
            ),
            TensorSpec(
                "qkv_weight",
                [QKV_SIZE, HIDDEN],
                torch.bfloat16,
                init_value=lambda: torch.randn(QKV_SIZE, HIDDEN, dtype=torch.bfloat16).mul_(0.01),
            ),
            TensorSpec("qkv_out", [BATCH_PAD, QKV_SIZE], torch.bfloat16),
        ]
    )
    if with_residual:
        specs.append(TensorSpec("residual_out", [BATCH_PAD, HIDDEN], torch.bfloat16))
    result = run(
        fn=qwen3_add_input_rms_qkv_decode if with_residual else qwen3_input_rms_qkv_decode,
        specs=specs,
        golden_fn=_golden,
        config={"platform": "a2a3", "device_id": int(os.getenv("PYPTO_TEST_DEVICE", "0"))},
        rtol=0.02,
        atol=0.02,
    )
    assert result.passed, result.error
