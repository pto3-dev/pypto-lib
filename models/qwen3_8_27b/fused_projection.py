# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Consume locally packed gate/up projections without changing TP head ownership."""

import pypto.language as pl

ROWS = pl.dynamic("rows")


def make_packed_silu(dim, inline=False, tile=512):
    decorate = pl.jit.inline if inline else pl.jit

    @decorate
    def activate(
        packed: pl.Tensor[[ROWS, dim * 2], pl.BF16],
        out: pl.Out[pl.Tensor[[ROWS, dim], pl.BF16]],
    ):
        rows = pl.tensor.dim(packed, 0)
        for job in pl.spmd(rows * ((dim + tile - 1) // tile), name_hint="packed_mlp_activation"):
            row = job // ((dim + tile - 1) // tile)
            c0 = job % ((dim + tile - 1) // tile) * tile
            valid = pl.min(tile, dim - c0)
            gate = pl.cast(
                pl.slice(packed, [1, tile], [row, c0], valid_shape=[1, valid], clamp=True), pl.FP32
            )
            up = pl.cast(
                pl.slice(packed, [1, tile], [row, dim + c0], valid_shape=[1, valid], clamp=True), pl.FP32
            )
            sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate)), 1.0))
            result = pl.mul(pl.mul(sigmoid, gate), up)
            out = pl.assemble(out, pl.cast(result, pl.BF16, mode="rint"), [row, c0])
        return out

    return activate
