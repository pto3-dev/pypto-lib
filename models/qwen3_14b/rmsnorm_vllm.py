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
"""Qwen3-14B Decode RMSNorm entries with vLLM's BF16 tensor boundaries."""

import pypto.language as pl


BATCH_PAD = 16
HIDDEN = 5120
K_TILE = 256
EPS = 1e-6


@pl.jit.inline(auto_scope=False)
def qwen3_rmsnorm_inline(
    x: pl.Tensor[[BATCH_PAD, HIDDEN], pl.BF16],
    weight: pl.Tensor[[1, HIDDEN], pl.BF16],
    out: pl.Tensor[[BATCH_PAD, HIDDEN], pl.BF16],
):
    inv_rms_states = pl.create_tensor([1, BATCH_PAD], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="input_rms_reduce"):
        square_sum = pl.full([1, BATCH_PAD], dtype=pl.FP32, value=0.0)
        for k0 in pl.range(0, HIDDEN, K_TILE):
            x_tile = pl.cast(pl.slice(x, [BATCH_PAD, K_TILE], [0, k0]), pl.FP32)
            row_squares = pl.reshape(pl.row_sum(pl.mul(x_tile, x_tile)), [1, BATCH_PAD])
            square_sum = pl.add(square_sum, row_squares)
        inv_rms = pl.rsqrt(pl.add(pl.mul(square_sum, 1.0 / HIDDEN), EPS))
        inv_rms_states = pl.assemble(inv_rms_states, inv_rms, [0, 0])
    for k0 in pl.parallel(0, HIDDEN, K_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="input_rms_apply"):
            x_tile = pl.cast(pl.slice(x, [BATCH_PAD, K_TILE], [0, k0]), pl.FP32)
            inv_rms_column = pl.reshape(pl.slice(inv_rms_states, [1, BATCH_PAD], [0, 0]), [BATCH_PAD, 1])
            weight_tile = pl.cast(pl.slice(weight, [1, K_TILE], [0, k0]), pl.FP32)
            normalized = pl.row_expand_mul(x_tile, inv_rms_column)
            out_tile = pl.cast(pl.col_expand_mul(normalized, weight_tile), pl.BF16, mode="rint")
            out = pl.assemble(out, out_tile, [0, k0])
    return out


@pl.jit(auto_scope=False)
def qwen3_rmsnorm_decode(
    x: pl.Tensor[[BATCH_PAD, HIDDEN], pl.BF16],
    weight: pl.Tensor[[1, HIDDEN], pl.BF16],
    out: pl.Out[pl.Tensor[[BATCH_PAD, HIDDEN], pl.BF16]],
):
    out = qwen3_rmsnorm_inline(x, weight, out)
    return out


@pl.jit.inline(auto_scope=False)
def qwen3_add_rmsnorm_inline(
    x: pl.Tensor[[BATCH_PAD, HIDDEN], pl.BF16],
    residual: pl.Tensor[[BATCH_PAD, HIDDEN], pl.BF16],
    weight: pl.Tensor[[1, HIDDEN], pl.BF16],
    out: pl.Tensor[[BATCH_PAD, HIDDEN], pl.BF16],
    residual_out: pl.Tensor[[BATCH_PAD, HIDDEN], pl.BF16],
):
    inv_rms_states = pl.create_tensor([1, BATCH_PAD], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="add_rms_reduce"):
        square_sum = pl.full([1, BATCH_PAD], dtype=pl.FP32, value=0.0)
        for k0 in pl.range(0, HIDDEN, K_TILE):
            x_tile = pl.cast(pl.slice(x, [BATCH_PAD, K_TILE], [0, k0]), pl.FP32)
            residual_tile = pl.cast(pl.slice(residual, [BATCH_PAD, K_TILE], [0, k0]), pl.FP32)
            summed = pl.cast(pl.add(x_tile, residual_tile), pl.BF16, mode="rint")
            summed_fp32 = pl.cast(summed, pl.FP32)
            row_squares = pl.reshape(pl.row_sum(pl.mul(summed_fp32, summed_fp32)), [1, BATCH_PAD])
            square_sum = pl.add(square_sum, row_squares)
        inv_rms = pl.rsqrt(pl.add(pl.mul(square_sum, 1.0 / HIDDEN), EPS))
        inv_rms_states = pl.assemble(inv_rms_states, inv_rms, [0, 0])
    for k0 in pl.parallel(0, HIDDEN, K_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="add_rms_apply"):
            x_tile = pl.cast(pl.slice(x, [BATCH_PAD, K_TILE], [0, k0]), pl.FP32)
            residual_tile = pl.cast(pl.slice(residual, [BATCH_PAD, K_TILE], [0, k0]), pl.FP32)
            summed = pl.cast(pl.add(x_tile, residual_tile), pl.BF16, mode="rint")
            inv_rms_column = pl.reshape(pl.slice(inv_rms_states, [1, BATCH_PAD], [0, 0]), [BATCH_PAD, 1])
            weight_tile = pl.cast(pl.slice(weight, [1, K_TILE], [0, k0]), pl.FP32)
            normalized = pl.row_expand_mul(pl.cast(summed, pl.FP32), inv_rms_column)
            out_tile = pl.cast(pl.col_expand_mul(normalized, weight_tile), pl.BF16, mode="rint")
            out = pl.assemble(out, out_tile, [0, k0])
            residual_out = pl.assemble(residual_out, summed, [0, k0])
    return out, residual_out


@pl.jit(auto_scope=False)
def qwen3_add_rmsnorm_decode(
    x: pl.Tensor[[BATCH_PAD, HIDDEN], pl.BF16],
    residual: pl.Tensor[[BATCH_PAD, HIDDEN], pl.BF16],
    weight: pl.Tensor[[1, HIDDEN], pl.BF16],
    out: pl.Out[pl.Tensor[[BATCH_PAD, HIDDEN], pl.BF16]],
    residual_out: pl.Out[pl.Tensor[[BATCH_PAD, HIDDEN], pl.BF16]],
):
    out, residual_out = qwen3_add_rmsnorm_inline(x, residual, weight, out, residual_out)
    return out, residual_out
