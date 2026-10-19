# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Native TP2 Q/K RMSNorm + partial RoPE rounding boundaries in one PTO task."""

import pypto.language as pl

TOKENS = pl.dynamic("tokens")
POSITIONS = pl.dynamic("positions")


def make_norm_rope(heads):
    @pl.jit.inline
    def fused(
        x: pl.Tensor[[TOKENS, heads, 256], pl.BF16],
        weight: pl.Tensor[[1, 256], pl.BF16],
        cos: pl.Tensor[[POSITIONS, 32], pl.BF16],
        sin: pl.Tensor[[POSITIONS, 32], pl.BF16],
        positions: pl.Tensor[[TOKENS], pl.INT64],
        out: pl.Out[pl.Tensor[[TOKENS, heads, 256], pl.BF16]],
    ):
        tokens = pl.tensor.dim(x, 0)
        xf = pl.reshape(x, [tokens * heads, 256])
        of = pl.reshape(out, [tokens * heads, 256])
        for job in pl.spmd((tokens * heads + 7) // 8, name_hint="native_qk_norm_rope"):
            row = job * 8
            valid = pl.min(8, tokens * heads - row)
            value = pl.cast(pl.slice(xf, [8, 256], [row, 0], [valid, 256], pad_value=0, clamp=True), pl.FP32)
            variance = pl.add(pl.mul(pl.row_sum(pl.mul(value, value)), 1.0 / 256), 1e-6)
            inverse = pl.rsqrt(variance)
            inverse = pl.mul(inverse, pl.add(pl.mul(pl.mul(variance, pl.mul(inverse, inverse)), -0.5), 1.5))
            # Native 1+gamma is evaluated in BF16; normalized Q/K stay FP32
            # until the final RoPE store, including the unrotated 192 elements.
            gamma = pl.cast(
                pl.cast(
                    pl.add(pl.cast(pl.slice(weight, [1, 256], [0, 0]), pl.FP32), 1.0), pl.BF16, mode="rint"
                ),
                pl.FP32,
            )
            normalized = pl.col_expand_mul(pl.row_expand_mul(value, inverse), gamma)
            position = pl.read(positions, [0])
            ct = pl.cast(pl.slice(cos, [1, 32], [position, 0]), pl.FP32)
            st = pl.cast(pl.slice(sin, [1, 32], [position, 0]), pl.FP32)
            first = pl.slice(normalized, [8, 32], [0, 0])
            second = pl.slice(normalized, [8, 32], [0, 32])
            rotated_first = pl.sub(pl.col_expand_mul(first, ct), pl.col_expand_mul(second, st))
            rotated_second = pl.add(pl.col_expand_mul(second, ct), pl.col_expand_mul(first, st))
            of = pl.assemble(
                of, pl.tensor.set_validshape(pl.cast(normalized, pl.BF16, mode="rint"), valid, 256), [row, 0]
            )
            of = pl.assemble(
                of,
                pl.tensor.set_validshape(pl.cast(rotated_first, pl.BF16, mode="rint"), valid, 32),
                [row, 0],
            )
            of = pl.assemble(
                of,
                pl.tensor.set_validshape(pl.cast(rotated_second, pl.BF16, mode="rint"), valid, 32),
                [row, 32],
            )
        return out

    return fused


def make_packed_norm_rope(heads):
    @pl.jit.inline
    def fused(
        x: pl.Tensor[[TOKENS, heads * 512], pl.BF16],
        weight: pl.Tensor[[1, 256], pl.BF16],
        cos: pl.Tensor[[POSITIONS, 32], pl.BF16],
        sin: pl.Tensor[[POSITIONS, 32], pl.BF16],
        positions: pl.Tensor[[TOKENS], pl.INT64],
        out: pl.Out[pl.Tensor[[TOKENS, heads, 256], pl.BF16]],
        gate: pl.Out[pl.Tensor[[TOKENS, heads * 256], pl.BF16]],
    ):
        tokens = pl.tensor.dim(x, 0)
        xf = pl.reshape(x, [tokens * heads, 512])
        gf = pl.reshape(gate, [tokens * heads, 256])
        of = pl.reshape(out, [tokens * heads, 256])
        for job in pl.spmd((tokens * heads + 7) // 8, name_hint="packed_query_norm_rope"):
            row = job * 8
            valid = pl.min(8, tokens * heads - row)
            value = pl.cast(pl.slice(xf, [8, 256], [row, 0], [valid, 256], pad_value=0, clamp=True), pl.FP32)
            gate_value = pl.slice(xf, [8, 256], [row, 256], [valid, 256], pad_value=0, clamp=True)
            gf = pl.assemble(gf, gate_value, [row, 0])
            variance = pl.add(pl.mul(pl.row_sum(pl.mul(value, value)), 1.0 / 256), 1e-6)
            inverse = pl.rsqrt(variance)
            inverse = pl.mul(inverse, pl.add(pl.mul(pl.mul(variance, pl.mul(inverse, inverse)), -0.5), 1.5))
            # Native 1+gamma is evaluated in BF16; normalized Q/K stay FP32
            # until the final RoPE store, including the unrotated 192 elements.
            gamma = pl.cast(
                pl.cast(
                    pl.add(pl.cast(pl.slice(weight, [1, 256], [0, 0]), pl.FP32), 1.0), pl.BF16, mode="rint"
                ),
                pl.FP32,
            )
            normalized = pl.col_expand_mul(pl.row_expand_mul(value, inverse), gamma)
            position = pl.read(positions, [0])
            ct = pl.cast(pl.slice(cos, [1, 32], [position, 0]), pl.FP32)
            st = pl.cast(pl.slice(sin, [1, 32], [position, 0]), pl.FP32)
            first = pl.slice(normalized, [8, 32], [0, 0])
            second = pl.slice(normalized, [8, 32], [0, 32])
            rotated_first = pl.sub(pl.col_expand_mul(first, ct), pl.col_expand_mul(second, st))
            rotated_second = pl.add(pl.col_expand_mul(second, ct), pl.col_expand_mul(first, st))
            of = pl.assemble(
                of, pl.tensor.set_validshape(pl.cast(normalized, pl.BF16, mode="rint"), valid, 256), [row, 0]
            )
            of = pl.assemble(
                of,
                pl.tensor.set_validshape(pl.cast(rotated_first, pl.BF16, mode="rint"), valid, 32),
                [row, 0],
            )
            of = pl.assemble(
                of,
                pl.tensor.set_validshape(pl.cast(rotated_second, pl.BF16, mode="rint"), valid, 32),
                [row, 32],
            )
        return out, gate

    return fused


def make_norm_rope_append(heads=2):
    if heads != 2:
        raise ValueError("This decode specialization requires two KV heads.")
    cache_rows = pl.dynamic("cache_rows")

    @pl.jit.inline
    def fused(
        x: pl.Tensor[[1, 2, 256], pl.BF16],
        weight: pl.Tensor[[1, 256], pl.BF16],
        cos: pl.Tensor[[POSITIONS, 32], pl.BF16],
        sin: pl.Tensor[[POSITIONS, 32], pl.BF16],
        positions: pl.Tensor[[1], pl.INT64],
        v: pl.Tensor[[1, 512], pl.BF16],
        key_cache: pl.InOut[pl.Tensor[[cache_rows, 512], pl.BF16]],
        value_cache: pl.InOut[pl.Tensor[[cache_rows, 512], pl.BF16]],
        slots: pl.Tensor[[1], pl.INT64],
    ):
        xf = pl.reshape(x, [2, 256])
        kf = pl.reshape(key_cache, [pl.tensor.dim(key_cache, 0) * 2, 256])
        for job in pl.spmd(1, name_hint="key_rope_append"):
            value = pl.cast(pl.slice(xf, [8, 256], [0, 0], [2, 256], pad_value=0, clamp=True), pl.FP32)
            variance = pl.add(pl.mul(pl.row_sum(pl.mul(value, value)), 1.0 / 256), 1e-6)
            inverse = pl.rsqrt(variance)
            inverse = pl.mul(inverse, pl.add(pl.mul(pl.mul(variance, pl.mul(inverse, inverse)), -0.5), 1.5))
            gamma = pl.cast(
                pl.cast(
                    pl.add(pl.cast(pl.slice(weight, [1, 256], [0, 0]), pl.FP32), 1.0), pl.BF16, mode="rint"
                ),
                pl.FP32,
            )
            normalized = pl.col_expand_mul(pl.row_expand_mul(value, inverse), gamma)
            position = pl.read(positions, [0])
            ct = pl.cast(pl.slice(cos, [1, 32], [position, 0]), pl.FP32)
            st = pl.cast(pl.slice(sin, [1, 32], [position, 0]), pl.FP32)
            first = pl.slice(normalized, [8, 32], [0, 0])
            second = pl.slice(normalized, [8, 32], [0, 32])
            rotated_first = pl.sub(pl.col_expand_mul(first, ct), pl.col_expand_mul(second, st))
            rotated_second = pl.add(pl.col_expand_mul(second, ct), pl.col_expand_mul(first, st))
            slot = pl.read(slots, [0])
            if slot >= 0:
                kf = pl.assemble(
                    kf,
                    pl.tensor.set_validshape(pl.cast(normalized, pl.BF16, mode="rint"), 2, 256),
                    [slot * 2, 0],
                )
                kf = pl.assemble(
                    kf,
                    pl.tensor.set_validshape(pl.cast(rotated_first, pl.BF16, mode="rint"), 2, 32),
                    [slot * 2, 0],
                )
                kf = pl.assemble(
                    kf,
                    pl.tensor.set_validshape(pl.cast(rotated_second, pl.BF16, mode="rint"), 2, 32),
                    [slot * 2, 32],
                )
                vt = pl.slice(v, [1, 512], [0, 0])
                value_cache = pl.assemble(value_cache, vt, [slot, 0])
        return key_cache, value_cache

    return fused
