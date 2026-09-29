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
"""Qwen3-14B post-attention RMSNorm and MLP in one vLLM Decode callable."""

import pypto.language as pl

from mlp_vllm import BATCH_PAD, GATE_UP, HIDDEN, INTERMEDIATE, qwen3_mlp_inline
from rmsnorm_vllm import qwen3_add_rmsnorm_inline


@pl.jit(auto_scope=False)
def qwen3_post_rms_mlp_decode(
    x: pl.Tensor[[BATCH_PAD, HIDDEN], pl.BF16],
    residual: pl.Tensor[[BATCH_PAD, HIDDEN], pl.BF16],
    norm_weight: pl.Tensor[[1, HIDDEN], pl.BF16],
    gate_up_weight: pl.Tensor[[GATE_UP, HIDDEN], pl.BF16],
    down_weight: pl.Tensor[[HIDDEN, INTERMEDIATE], pl.BF16],
    out: pl.Out[pl.Tensor[[BATCH_PAD, HIDDEN], pl.BF16]],
    residual_out: pl.Out[pl.Tensor[[BATCH_PAD, HIDDEN], pl.BF16]],
):
    normed = pl.create_tensor([BATCH_PAD, HIDDEN], dtype=pl.BF16)
    normed, residual_out = qwen3_add_rmsnorm_inline(x, residual, norm_weight, normed, residual_out)
    out = qwen3_mlp_inline(normed, gate_up_weight, down_weight, out)
    return out, residual_out
