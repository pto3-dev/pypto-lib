# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""norm_pointwise operators. Explicit PyPTO functions; no source generator at build time.

Migrated from the validated L1K2560 graph; tuning is audited separately.
"""

import pypto.language as pl


@pl.function(type=pl.FunctionType.InCore)
def norm_plain_kernel(
    x: pl.Tensor[[24, 5120], pl.BF16],
    weight: pl.Tensor[[1, 5120], pl.BF16],
    z: pl.Tensor[[24, 5120], pl.BF16],
    out: pl.Tensor[[24, 5120], pl.BF16],
) -> pl.Tensor[[24, 5120], pl.BF16]:
    pl.func_attr({"static_block_mapping": True})
    rows = 24
    row = pl.tile.get_block_idx()
    whole = pl.cast(pl.slice(x, [1, 5120], [row, 0]), pl.FP32)
    chunks = pl.reshape(whole, [40, 128])
    partials = pl.row_sum(pl.mul(chunks, chunks))
    zero = pl.full([1, 8], dtype=pl.FP32, value=0.0)
    total = zero
    for group in pl.range(0, 40):
        total = pl.add(total, pl.read(partials, [group, 0]))
    variance = pl.add(pl.mul(total, 1.0 / 5120), 1e-06)
    inv = pl.rsqrt(variance)
    inv = pl.mul(inv, pl.add(pl.mul(pl.mul(variance, pl.mul(inv, inv)), -0.5), 1.5))
    scalar_inv = pl.read(inv, [0, 0])
    gamma = pl.add(pl.cast(pl.slice(weight, [1, 5120], [0, 0]), pl.FP32), 1.0)
    normalized = pl.mul(whole, scalar_inv)
    out = pl.assemble(out, pl.cast(pl.mul(normalized, gamma), pl.BF16, mode="rint"), [row, 0])
    return out


@pl.function(type=pl.FunctionType.Inline)
def norm_plain(
    x: pl.Tensor[[24, 5120], pl.BF16],
    weight: pl.Tensor[[1, 5120], pl.BF16],
    z: pl.Tensor[[24, 5120], pl.BF16],
    out: pl.Tensor[[24, 5120], pl.BF16],
) -> pl.Tensor[[24, 5120], pl.BF16]:
    rows = 24
    with pl.spmd(rows, allow_early_resolve=True):
        out = norm_plain_kernel(x, weight, z, out)
    return out


@pl.function(type=pl.FunctionType.InCore)
def norm_hidden_kernel(
    x: pl.Tensor[[24, 5120], pl.BF16],
    weight: pl.Tensor[[1, 5120], pl.BF16],
    z: pl.Tensor[[24, 5120], pl.BF16],
    out: pl.Tensor[[24, 5120], pl.BF16],
) -> pl.Tensor[[24, 5120], pl.BF16]:
    pl.func_attr({"static_block_mapping": True})
    rows = 24
    row = pl.tile.get_block_idx()
    whole = pl.cast(pl.slice(x, [1, 5120], [row, 0]), pl.FP32)
    chunks = pl.reshape(whole, [40, 128])
    partials = pl.row_sum(pl.mul(chunks, chunks))
    zero = pl.full([1, 8], dtype=pl.FP32, value=0.0)
    total = zero
    for group in pl.range(0, 40):
        total = pl.add(total, pl.read(partials, [group, 0]))
    variance = pl.add(pl.mul(total, 1.0 / 5120), 1e-06)
    inv = pl.rsqrt(variance)
    inv = pl.mul(inv, pl.add(pl.mul(pl.mul(variance, pl.mul(inv, inv)), -0.5), 1.5))
    scalar_inv = pl.read(inv, [0, 0])
    gamma = pl.add(pl.cast(pl.slice(weight, [1, 5120], [0, 0]), pl.FP32), 1.0)
    normalized = pl.mul(whole, scalar_inv)
    normalized = pl.cast(pl.cast(normalized, pl.BF16, mode="rint"), pl.FP32)
    gamma = pl.cast(pl.cast(gamma, pl.BF16, mode="rint"), pl.FP32)
    out = pl.assemble(out, pl.cast(pl.mul(normalized, gamma), pl.BF16, mode="rint"), [row, 0])
    return out


@pl.function(type=pl.FunctionType.Inline)
def norm_hidden(
    x: pl.Tensor[[24, 5120], pl.BF16],
    weight: pl.Tensor[[1, 5120], pl.BF16],
    z: pl.Tensor[[24, 5120], pl.BF16],
    out: pl.Tensor[[24, 5120], pl.BF16],
) -> pl.Tensor[[24, 5120], pl.BF16]:
    rows = 24
    with pl.spmd(rows, allow_early_resolve=True):
        out = norm_hidden_kernel(x, weight, z, out)
    return out


@pl.function(type=pl.FunctionType.InCore)
def norm_gdn_kernel(
    x: pl.Tensor[[288, 128], pl.BF16],
    weight: pl.Tensor[[1, 128], pl.BF16],
    z: pl.Tensor[[288, 128], pl.BF16],
    out: pl.Tensor[[288, 128], pl.BF16],
) -> pl.Tensor[[288, 128], pl.BF16]:
    pl.func_attr({"static_block_mapping": True})
    rows = 288
    job = pl.tile.get_block_idx()
    r0 = job * 8
    valid = pl.min(8, rows - r0)
    xr = pl.slice(x, [8, 128], [r0, 0], [valid, 128], clamp=True)
    zr = pl.slice(z, [8, 128], [r0, 0], [valid, 128], clamp=True)
    wr = pl.slice(weight, [1, 128], [0, 0])
    xt = pl.cast(xr, pl.FP32)
    zt = pl.cast(zr, pl.FP32)
    wt = pl.add(pl.cast(wr, pl.FP32), 0.0)
    sq = pl.row_sum(pl.mul(xt, xt))
    variance = pl.add(pl.mul(sq, 1.0 / 128), 1e-06)
    inv = pl.rsqrt(variance)
    correction = pl.add(pl.mul(pl.mul(variance, pl.mul(inv, inv)), -0.5), 1.5)
    inv = pl.mul(inv, correction)
    for c0 in pl.range(0, 128, 128):
        norm_x = xt
        normalized = pl.row_expand_mul(norm_x, inv)
        result = pl.col_expand_mul(normalized, wt)
        silu = pl.mul(zt, pl.recip(pl.add(pl.exp(pl.neg(zt)), 1.0)))
        result = pl.mul(result, silu)
        out = pl.assemble(out, pl.cast(result, pl.BF16, mode="rint"), [r0, c0])
    return out


@pl.function(type=pl.FunctionType.Inline)
def norm_gdn(
    x: pl.Tensor[[288, 128], pl.BF16],
    weight: pl.Tensor[[1, 128], pl.BF16],
    z: pl.Tensor[[288, 128], pl.BF16],
    out: pl.Tensor[[288, 128], pl.BF16],
) -> pl.Tensor[[288, 128], pl.BF16]:
    rows = 288
    with pl.spmd((rows + 7) // 8, allow_early_resolve=True):
        out = norm_gdn_kernel(x, weight, z, out)
    return out


@pl.function(type=pl.FunctionType.InCore)
def add_kernel(
    x: pl.Tensor[[24, 5120], pl.BF16], y: pl.Tensor[[24, 5120], pl.BF16], out: pl.Tensor[[24, 5120], pl.BF16]
) -> pl.Tensor[[24, 5120], pl.BF16]:
    pl.func_attr({"static_block_mapping": True})
    rows = 24
    job = pl.tile.get_block_idx()
    row = job // ((5120 + 1280 - 1) // 1280)
    c0 = job % ((5120 + 1280 - 1) // 1280) * 1280
    valid = pl.min(1280, 5120 - c0)
    xt = pl.cast(pl.slice(x, [1, 1280], [row, c0], [1, valid], clamp=True), pl.FP32)
    yt = pl.cast(pl.slice(y, [1, 1280], [row, c0], [1, valid], clamp=True), pl.FP32)
    if 0 == 0:
        result = pl.add(xt, yt)
    else:
        sigmoid = pl.recip(pl.add(pl.exp(pl.neg(xt)), 1.0))
        if 0 == 1:
            sigmoid = pl.mul(sigmoid, xt)
        else:
            sigmoid = pl.cast(pl.cast(sigmoid, pl.BF16, mode="rint"), pl.FP32)
        result = pl.mul(sigmoid, yt)
    out = pl.assemble(out, pl.cast(result, pl.BF16, mode="rint"), [row, c0])
    return out


@pl.function(type=pl.FunctionType.Inline)
def add(
    x: pl.Tensor[[24, 5120], pl.BF16], y: pl.Tensor[[24, 5120], pl.BF16], out: pl.Tensor[[24, 5120], pl.BF16]
) -> pl.Tensor[[24, 5120], pl.BF16]:
    rows = 24
    with pl.spmd(rows * ((5120 + 1280 - 1) // 1280), allow_early_resolve=True):
        out = add_kernel(x, y, out)
    return out


@pl.function(type=pl.FunctionType.InCore)
def activate_kernel(
    x: pl.Tensor[[24, 4352], pl.BF16], y: pl.Tensor[[24, 4352], pl.BF16], out: pl.Tensor[[24, 4352], pl.BF16]
) -> pl.Tensor[[24, 4352], pl.BF16]:
    pl.func_attr({"static_block_mapping": True})
    rows = 24
    job = pl.tile.get_block_idx()
    row = job // ((4352 + 512 - 1) // 512)
    c0 = job % ((4352 + 512 - 1) // 512) * 512
    valid = pl.min(512, 4352 - c0)
    xt = pl.cast(pl.slice(x, [1, 512], [row, c0], [1, valid], clamp=True), pl.FP32)
    yt = pl.cast(pl.slice(y, [1, 512], [row, c0], [1, valid], clamp=True), pl.FP32)
    if 1 == 0:
        result = pl.add(xt, yt)
    else:
        sigmoid = pl.recip(pl.add(pl.exp(pl.neg(xt)), 1.0))
        if 1 == 1:
            sigmoid = pl.mul(sigmoid, xt)
        else:
            sigmoid = pl.cast(pl.cast(sigmoid, pl.BF16, mode="rint"), pl.FP32)
        result = pl.mul(sigmoid, yt)
    out = pl.assemble(out, pl.cast(result, pl.BF16, mode="rint"), [row, c0])
    return out


@pl.function(type=pl.FunctionType.Inline)
def activate(
    x: pl.Tensor[[24, 4352], pl.BF16], y: pl.Tensor[[24, 4352], pl.BF16], out: pl.Tensor[[24, 4352], pl.BF16]
) -> pl.Tensor[[24, 4352], pl.BF16]:
    rows = 24
    with pl.spmd(rows * ((4352 + 512 - 1) // 512), allow_early_resolve=True):
        out = activate_kernel(x, y, out)
    return out


@pl.function(type=pl.FunctionType.InCore)
def gate_attention_kernel(
    x: pl.Tensor[[24, 1536], pl.BF16], y: pl.Tensor[[24, 1536], pl.BF16], out: pl.Tensor[[24, 1536], pl.BF16]
) -> pl.Tensor[[24, 1536], pl.BF16]:
    pl.func_attr({"static_block_mapping": True})
    rows = 24
    job = pl.tile.get_block_idx()
    row = job // ((1536 + 512 - 1) // 512)
    c0 = job % ((1536 + 512 - 1) // 512) * 512
    valid = pl.min(512, 1536 - c0)
    xt = pl.cast(pl.slice(x, [1, 512], [row, c0], [1, valid], clamp=True), pl.FP32)
    yt = pl.cast(pl.slice(y, [1, 512], [row, c0], [1, valid], clamp=True), pl.FP32)
    if 2 == 0:
        result = pl.add(xt, yt)
    else:
        sigmoid = pl.recip(pl.add(pl.exp(pl.neg(xt)), 1.0))
        if 2 == 1:
            sigmoid = pl.mul(sigmoid, xt)
        else:
            sigmoid = pl.cast(pl.cast(sigmoid, pl.BF16, mode="rint"), pl.FP32)
        result = pl.mul(sigmoid, yt)
    out = pl.assemble(out, pl.cast(result, pl.BF16, mode="rint"), [row, c0])
    return out


@pl.function(type=pl.FunctionType.Inline)
def gate_attention(
    x: pl.Tensor[[24, 1536], pl.BF16], y: pl.Tensor[[24, 1536], pl.BF16], out: pl.Tensor[[24, 1536], pl.BF16]
) -> pl.Tensor[[24, 1536], pl.BF16]:
    rows = 24
    with pl.spmd(rows * ((1536 + 512 - 1) // 512), allow_early_resolve=True):
        out = gate_attention_kernel(x, y, out)
    return out
