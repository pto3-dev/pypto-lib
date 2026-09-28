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
# --------------------------------------------------------------------------------
"""Golden validation for the vLLM-layout Qwen3-14B MLP kernel."""

import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401
from golden import TensorSpec, run

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "models" / "qwen3_14b"))
from mlp_vllm import BATCH_PAD, GATE_UP, HIDDEN, INTERMEDIATE, qwen3_mlp_decode  # noqa: E402


def _random_weight(shape: tuple[int, ...]) -> torch.Tensor:
    return torch.randn(shape, dtype=torch.bfloat16).mul_(0.01)


def _golden(values: dict[str, torch.Tensor]) -> None:
    gate_up = F.linear(values["x"], values["gate_up_weight"])
    gate, up = gate_up.chunk(2, dim=-1)
    values["out"][:] = F.linear(F.silu(gate) * up, values["down_weight"])


@pytest.mark.skipif(not torch.npu.is_available(), reason="Ascend NPU unavailable")
def test_qwen3_14b_mlp_vllm_golden() -> None:
    torch.manual_seed(0)
    specs = [
        TensorSpec("x", [BATCH_PAD, HIDDEN], torch.bfloat16, init_value=torch.randn),
        TensorSpec(
            "gate_up_weight",
            [GATE_UP, HIDDEN],
            torch.bfloat16,
            init_value=lambda: _random_weight((GATE_UP, HIDDEN)),
        ),
        TensorSpec(
            "down_weight",
            [HIDDEN, INTERMEDIATE],
            torch.bfloat16,
            init_value=lambda: _random_weight((HIDDEN, INTERMEDIATE)),
        ),
        TensorSpec("out", [BATCH_PAD, HIDDEN], torch.bfloat16),
    ]
    result = run(
        fn=qwen3_mlp_decode,
        specs=specs,
        golden_fn=_golden,
        config={"platform": "a2a3", "device_id": int(os.getenv("PYPTO_TEST_DEVICE", "0"))},
        rtol=0.02,
        atol=0.02,
    )
    assert result.passed, result.error
