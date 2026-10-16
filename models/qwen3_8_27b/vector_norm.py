# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Same128-element RMS partial sums, one GM load and scalar ordered fold."""

import pypto.language as pl

ROWS = pl.dynamic("rows")


def make_vector_norm(native_round=False, inline=False):
    decorate = pl.jit.inline if inline else pl.jit

    @decorate
    def norm(
        x: pl.Tensor[[ROWS, 5120], pl.BF16],
        weight: pl.Tensor[[1, 5120], pl.BF16],
        z: pl.Tensor[[ROWS, 5120], pl.BF16],
        out: pl.Out[pl.Tensor[[ROWS, 5120], pl.BF16]],
    ):
        rows = pl.tensor.dim(x, 0)
        for row in pl.spmd(rows, name_hint="rmsnorm_vectorized_partials"):
            whole = pl.cast(pl.slice(x, [1, 5120], [row, 0]), pl.FP32)
            chunks = pl.reshape(whole, [40, 128])
            partials = pl.row_sum(pl.mul(chunks, chunks))
            zero = pl.full([1, 128], dtype=pl.FP32, value=0.0)
            total = zero
            for group in pl.range(0, 40):
                total = pl.add(total, pl.read(partials, [group, 0]))
            variance = pl.add(pl.mul(total, 1.0 / 5120), 1e-6)
            inv = pl.rsqrt(variance)
            inv = pl.mul(inv, pl.add(pl.mul(pl.mul(variance, pl.mul(inv, inv)), -0.5), 1.5))
            scalar_inv = pl.read(inv, [0, 0])
            gamma = pl.add(pl.cast(pl.slice(weight, [1, 5120], [0, 0]), pl.FP32), 1.0)
            normalized = pl.mul(whole, scalar_inv)
            if native_round:
                normalized = pl.cast(pl.cast(normalized, pl.BF16, mode="rint"), pl.FP32)
                gamma = pl.cast(pl.cast(gamma, pl.BF16, mode="rint"), pl.FP32)
            out = pl.assemble(out, pl.cast(pl.mul(normalized, gamma), pl.BF16, mode="rint"), [row, 0])
        return out

    return norm
