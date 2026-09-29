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
# ---------------------------------------------------------------------------------
"""Qwen3-14B packed QKV projection with vLLM's BF16 weight layout.

The output replaces only qkv_proj. Q/K normalization, RoPE, paged attention,
KV writes, and output projection remain in native vLLM.
"""

import pypto.language as pl


BATCH_PAD = 16
HIDDEN = 5120
Q_SIZE = 5120
KV_SIZE = 1024
QKV_SIZE = Q_SIZE + 2 * KV_SIZE
K_TILE = 256
N_TILE = 256


@pl.jit.inline(auto_scope=False)
def qwen3_qkv_inline(
    x: pl.Tensor[[BATCH_PAD, HIDDEN], pl.BF16],
    qkv_weight: pl.Tensor[[QKV_SIZE, HIDDEN], pl.BF16],
    out: pl.Tensor[[BATCH_PAD, QKV_SIZE], pl.BF16],
):
    """Project hidden states into packed Q, K, and V rows."""
    for n0 in pl.parallel(0, QKV_SIZE, N_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="qkv_proj"):
            acc = pl.create_tensor([BATCH_PAD, N_TILE], dtype=pl.FP32)
            for k0 in pl.pipeline(0, HIDDEN, K_TILE, stage=2):
                x_tile = x[:, k0 : k0 + K_TILE]
                w_tile = qkv_weight[n0 : n0 + N_TILE, k0 : k0 + K_TILE]
                acc = pl.matmul_acc(acc, x_tile, w_tile, b_trans=True, init_cond=(k0 == 0))
            out = pl.assemble(out, pl.cast(acc, pl.BF16, mode="rint"), [0, n0])
    return out


@pl.jit(auto_scope=False)
def qwen3_qkv_decode(
    x: pl.Tensor[[BATCH_PAD, HIDDEN], pl.BF16],
    qkv_weight: pl.Tensor[[QKV_SIZE, HIDDEN], pl.BF16],
    out: pl.Out[pl.Tensor[[BATCH_PAD, QKV_SIZE], pl.BF16]],
):
    out = qwen3_qkv_inline(x, qkv_weight, out)
    return out
