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
"""Qwen3-14B Decode input RMSNorm and QKV projection in one callable."""

import pypto.language as pl

from qkv_vllm import QKV_SIZE, qwen3_qkv_inline
from rmsnorm_vllm import BATCH_PAD, HIDDEN, qwen3_add_rmsnorm_inline, qwen3_rmsnorm_inline


@pl.jit(auto_scope=False)
def qwen3_input_rms_qkv_decode(
    x: pl.Tensor[[BATCH_PAD, HIDDEN], pl.BF16],
    norm_weight: pl.Tensor[[1, HIDDEN], pl.BF16],
    qkv_weight: pl.Tensor[[QKV_SIZE, HIDDEN], pl.BF16],
    qkv_out: pl.Out[pl.Tensor[[BATCH_PAD, QKV_SIZE], pl.BF16]],
):
    normed = pl.create_tensor([BATCH_PAD, HIDDEN], dtype=pl.BF16)
    normed = qwen3_rmsnorm_inline(x, norm_weight, normed)
    qkv_out = qwen3_qkv_inline(normed, qkv_weight, qkv_out)
    return qkv_out


@pl.jit(auto_scope=False)
def qwen3_add_input_rms_qkv_decode(
    x: pl.Tensor[[BATCH_PAD, HIDDEN], pl.BF16],
    residual: pl.Tensor[[BATCH_PAD, HIDDEN], pl.BF16],
    norm_weight: pl.Tensor[[1, HIDDEN], pl.BF16],
    qkv_weight: pl.Tensor[[QKV_SIZE, HIDDEN], pl.BF16],
    qkv_out: pl.Out[pl.Tensor[[BATCH_PAD, QKV_SIZE], pl.BF16]],
    residual_out: pl.Out[pl.Tensor[[BATCH_PAD, HIDDEN], pl.BF16]],
):
    normed = pl.create_tensor([BATCH_PAD, HIDDEN], dtype=pl.BF16)
    normed, residual_out = qwen3_add_rmsnorm_inline(x, residual, norm_weight, normed, residual_out)
    qkv_out = qwen3_qkv_inline(normed, qkv_weight, qkv_out)
    return qkv_out, residual_out
