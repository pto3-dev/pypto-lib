# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Select local heads from replicated small projection outputs."""

import pypto.language as pl

TOKENS = pl.dynamic("tokens")


def make_selected_gates(heads, input_heads, head_offset, inline=False, tokens=TOKENS):
    tile_heads = (heads + 15) // 16 * 16
    decorate = pl.jit.inline if inline else pl.jit

    @decorate
    def gates(
        a: pl.Tensor[[tokens, input_heads], pl.BF16],
        b: pl.Tensor[[tokens, input_heads], pl.BF16],
        a_log: pl.Tensor[[1, heads], pl.FP32],
        dt_bias: pl.Tensor[[1, heads], pl.FP32],
        alpha: pl.Out[pl.Tensor[[tokens, heads], pl.FP32]],
        beta: pl.Out[pl.Tensor[[tokens, heads], pl.FP32]],
    ):
        tokens = pl.tensor.dim(a, 0)
        for t in pl.spmd(tokens, name_hint="gdn_gates"):
            at = pl.cast(
                pl.slice(
                    a, [1, tile_heads], [t, head_offset], valid_shape=[1, heads], pad_value=0, clamp=True
                ),
                pl.FP32,
            )
            bt = pl.cast(
                pl.slice(
                    b, [1, tile_heads], [t, head_offset], valid_shape=[1, heads], pad_value=0, clamp=True
                ),
                pl.FP32,
            )
            bias = pl.slice(dt_bias, [1, tile_heads], [0, 0], valid_shape=[1, heads], pad_value=0, clamp=True)
            log_decay = pl.slice(
                a_log, [1, tile_heads], [0, 0], valid_shape=[1, heads], pad_value=0, clamp=True
            )
            x = pl.add(at, bias)
            softplus = pl.add(pl.maximum(x, 0.0), pl.log(pl.add(pl.exp(pl.neg(pl.abs(x))), 1.0)))
            decay = pl.neg(pl.mul(pl.exp(log_decay), softplus))
            alpha_t = pl.tensor.set_validshape(pl.exp(decay), 1, heads)
            alpha = pl.assemble(alpha, alpha_t, [t, 0])
            beta_t = pl.recip(pl.add(pl.exp(pl.neg(bt)), 1.0))
            beta_valid = pl.tensor.set_validshape(beta_t, 1, heads)
            beta = pl.assemble(beta, beta_valid, [t, 0])
        return alpha, beta

    return gates
