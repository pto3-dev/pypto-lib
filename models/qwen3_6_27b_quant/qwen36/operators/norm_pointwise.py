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
def norm_plain_kernel(x: pl.Tensor[[24, 5120], pl.BF16], weight: pl.Tensor[[1, 5120], pl.BF16], z: pl.Tensor[[24, 5120], pl.BF16], out: pl.Tensor[[24, 5120], pl.BF16]) -> pl.Tensor[[24, 5120], pl.BF16]:
    pl.func_attr({'static_block_mapping': True})
    rows = 24
    row = pl.tile.get_block_idx()
    whole = pl.cast(pl.slice(x, [1, 5120], [row, 0]), pl.FP32)
    # CANN9.1 GemmaRmsNorm key0, measured in437: ReduceSumHalfInterval.
    # Fold the tail into the largest power-of-two span, then halve to64.
    # Eight physical reduction rows keep the statistic32-byte addressable;
    # only row0 represents this logical model row.
    squares = pl.mul(whole, whole)
    first = pl.add(pl.slice(squares, [1, 1024], [0, 0]), pl.slice(squares, [1, 1024], [0, 4096]))
    fold4096 = pl.concat(first, pl.slice(squares, [1, 3072], [0, 1024]))
    fold2048 = pl.add(pl.slice(fold4096, [1, 2048], [0, 0]), pl.slice(fold4096, [1, 2048], [0, 2048]))
    fold1024 = pl.add(pl.slice(fold2048, [1, 1024], [0, 0]), pl.slice(fold2048, [1, 1024], [0, 1024]))
    fold512 = pl.add(pl.slice(fold1024, [1, 512], [0, 0]), pl.slice(fold1024, [1, 512], [0, 512]))
    fold256 = pl.add(pl.slice(fold512, [1, 256], [0, 0]), pl.slice(fold512, [1, 256], [0, 256]))
    fold128 = pl.add(pl.slice(fold256, [1, 128], [0, 0]), pl.slice(fold256, [1, 128], [0, 128]))
    fold64 = pl.add(pl.slice(fold128, [1, 64], [0, 0]), pl.slice(fold128, [1, 64], [0, 64]))
    padded = pl.reshape(pl.concat(fold64, pl.full([1, 448], dtype=pl.FP32, value=0.0)), [8, 64])
    total = pl.row_sum(padded)
    # CANN9.1 key0 executes scalar FP32 MADD, SQRT, DIV (trace447/453).
    # Typed scalar reads preserve FP32 operands without double literals.
    total_scalar = pl.read(total, [0, 0])
    average_tile = pl.full([1, 8], dtype=pl.FP32, value=0.0001953125)
    epsilon_tile = pl.full([1, 8], dtype=pl.FP32, value=1e-06)
    average = pl.read(average_tile, [0, 0])
    epsilon = pl.read(epsilon_tile, [0, 0])
    variance_scalar = total_scalar * average + epsilon
    variance = pl.mul(pl.full([1, 8], dtype=pl.FP32, value=1.0), variance_scalar)
    # Compensate the vector sqrt before reciprocal to match native scalar SQRT.
    # Existing DSL only; actual F32 intermediates were calibrated in455.
    root0 = pl.sqrt(variance)
    splitter = pl.mul(root0, 4097.0)
    high = pl.sub(splitter, pl.sub(splitter, root0))
    low = pl.sub(root0, high)
    residual = pl.sub(pl.sub(pl.sub(variance, pl.mul(high, high)), pl.mul(pl.mul(high, low), 2.0)), pl.mul(low, low))
    root = pl.add(root0, pl.div(residual, pl.mul(root0, 2.0)))
    # Use native scalar FP32 DIV; vector reciprocal can differ by1ULP.
    root_scalar = pl.read(root, [0, 0])
    one_tile = pl.full([1, 8], dtype=pl.FP32, value=1.0)
    one_scalar = pl.read(one_tile, [0, 0])
    scalar_inv = one_scalar / root_scalar
    gamma = pl.add(pl.cast(pl.slice(weight, [1, 5120], [0, 0]), pl.FP32), 1.0)
    normalized = pl.mul(whole, scalar_inv)
    out = pl.assemble(out, pl.cast(pl.mul(normalized, gamma), pl.BF16, mode='rint'), [row, 0])
    return out

@pl.function(type=pl.FunctionType.Inline)
def norm_plain(x: pl.Tensor[[24, 5120], pl.BF16], weight: pl.Tensor[[1, 5120], pl.BF16], z: pl.Tensor[[24, 5120], pl.BF16], out: pl.Tensor[[24, 5120], pl.BF16]) -> pl.Tensor[[24, 5120], pl.BF16]:
    rows = 24
    with pl.spmd(rows, allow_early_resolve=True):
        out = norm_plain_kernel(x, weight, z, out)
    return out

@pl.function(type=pl.FunctionType.InCore)
def norm_hidden_kernel(x: pl.Tensor[[24, 5120], pl.BF16], weight: pl.Tensor[[1, 5120], pl.BF16], z: pl.Tensor[[24, 5120], pl.BF16], out: pl.Tensor[[24, 5120], pl.BF16]) -> pl.Tensor[[24, 5120], pl.BF16]:
    pl.func_attr({'static_block_mapping': True})
    rows = 24
    row = pl.tile.get_block_idx()
    whole = pl.tile.cast(pl.tile.load(x, [row, 0], [1, 5120], target_memory=pl.Mem.Vec), pl.FP32)
    # Native AddRMSNorm: scale each square, accumulate80 contiguous64-lane
    # vectors in order, then one WholeReduceSum-equivalent TROWSUM.
    terms = pl.tile.muls(pl.tile.mul(whole, whole), 1.0 / 5120)
    lanes = pl.tile.full([1, 64], dtype=pl.FP32, value=0.0)
    for repeat in pl.range(80):
        values = pl.tile.slice(terms, [1, 64], [0, repeat * 64])
        lanes = pl.tile.add(values, lanes)
    temporary = pl.tile.create([1, 64], pl.FP32, target_memory=pl.Mem.Vec)
    total = pl.tile.row_sum(lanes, temporary, physical_rows=8)
    variance = pl.tile.adds(total, 1e-06)
    inv = pl.tile.recip(pl.tile.sqrt(variance))
    scalar_inv = pl.read(inv, [0, 0])
    gamma = pl.add(pl.cast(pl.tile.load(weight, [0, 0], [1, 5120], target_memory=pl.Mem.Vec), pl.FP32), 1.0)
    normalized = pl.mul(whole, scalar_inv)
    normalized = pl.cast(pl.cast(normalized, pl.BF16, mode='rint'), pl.FP32)
    gamma = pl.cast(pl.cast(gamma, pl.BF16, mode='rint'), pl.FP32)
    out = pl.tile.store(pl.cast(pl.mul(normalized, gamma), pl.BF16, mode='rint'), [row, 0], out)
    return out

@pl.function(type=pl.FunctionType.Inline)
def norm_hidden(x: pl.Tensor[[24, 5120], pl.BF16], weight: pl.Tensor[[1, 5120], pl.BF16], z: pl.Tensor[[24, 5120], pl.BF16], out: pl.Tensor[[24, 5120], pl.BF16]) -> pl.Tensor[[24, 5120], pl.BF16]:
    rows = 24
    with pl.spmd(rows, allow_early_resolve=True):
        out = norm_hidden_kernel(x, weight, z, out)
    return out

@pl.function(type=pl.FunctionType.InCore)
def norm_gdn_kernel(x: pl.Tensor[[288, 128], pl.BF16], weight: pl.Tensor[[1, 128], pl.BF16], z: pl.Tensor[[288, 128], pl.BF16], out: pl.Tensor[[288, 128], pl.BF16]) -> pl.Tensor[[288, 128], pl.BF16]:
    pl.func_attr({'static_block_mapping': True})
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
    # Native gated RMSNorm uses 1 / sqrt; the rsqrt/Newton form changes
    # FP32 rounding before the final BF16 cast. Real-input evidence: 33b.
    inv = pl.recip(pl.sqrt(variance))
    for c0 in pl.range(0, 128, 128):
        norm_x = xt
        normalized = pl.row_expand_mul(norm_x, inv)
        result = pl.col_expand_mul(normalized, wt)
        # Real layer0/layer1 native calibration121 selects direct SiLU Div;
        # reciprocal multiplication changes a BF16 rounding boundary.
        silu = pl.div(zt, pl.add(pl.exp(pl.neg(zt)), 1.0))
        result = pl.mul(result, silu)
        out = pl.assemble(out, pl.cast(result, pl.BF16, mode='rint'), [r0, c0])
    return out

@pl.function(type=pl.FunctionType.Inline)
def norm_gdn(x: pl.Tensor[[288, 128], pl.BF16], weight: pl.Tensor[[1, 128], pl.BF16], z: pl.Tensor[[288, 128], pl.BF16], out: pl.Tensor[[288, 128], pl.BF16]) -> pl.Tensor[[288, 128], pl.BF16]:
    rows = 288
    with pl.spmd((rows + 7) // 8, allow_early_resolve=True):
        out = norm_gdn_kernel(x, weight, z, out)
    return out

@pl.function(type=pl.FunctionType.InCore)
def add_kernel(x: pl.Tensor[[24, 5120], pl.BF16], y: pl.Tensor[[24, 5120], pl.BF16], out: pl.Tensor[[24, 5120], pl.BF16]) -> pl.Tensor[[24, 5120], pl.BF16]:
    pl.func_attr({'static_block_mapping': True})
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
            sigmoid = pl.cast(pl.cast(sigmoid, pl.BF16, mode='rint'), pl.FP32)
        result = pl.mul(sigmoid, yt)
    out = pl.assemble(out, pl.cast(result, pl.BF16, mode='rint'), [row, c0])
    return out

@pl.function(type=pl.FunctionType.Inline)
def add(x: pl.Tensor[[24, 5120], pl.BF16], y: pl.Tensor[[24, 5120], pl.BF16], out: pl.Tensor[[24, 5120], pl.BF16]) -> pl.Tensor[[24, 5120], pl.BF16]:
    rows = 24
    with pl.spmd(rows * ((5120 + 1280 - 1) // 1280), allow_early_resolve=True):
        out = add_kernel(x, y, out)
    return out

@pl.function(type=pl.FunctionType.InCore)
def activate_kernel(x: pl.Tensor[[24, 4352], pl.BF16], y: pl.Tensor[[24, 4352], pl.BF16], out: pl.Tensor[[24, 4352], pl.BF16]) -> pl.Tensor[[24, 4352], pl.BF16]:
    pl.func_attr({'static_block_mapping': True})
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
            sigmoid = pl.cast(pl.cast(sigmoid, pl.BF16, mode='rint'), pl.FP32)
        result = pl.mul(sigmoid, yt)
    out = pl.assemble(out, pl.cast(result, pl.BF16, mode='rint'), [row, c0])
    return out

@pl.function(type=pl.FunctionType.Inline)
def activate(x: pl.Tensor[[24, 4352], pl.BF16], y: pl.Tensor[[24, 4352], pl.BF16], out: pl.Tensor[[24, 4352], pl.BF16]) -> pl.Tensor[[24, 4352], pl.BF16]:
    rows = 24
    with pl.spmd(rows * ((4352 + 512 - 1) // 512), allow_early_resolve=True):
        out = activate_kernel(x, y, out)
    return out

@pl.function(type=pl.FunctionType.InCore)
def gate_attention_kernel(x: pl.Tensor[[24, 1536], pl.BF16], y: pl.Tensor[[24, 1536], pl.BF16], out: pl.Tensor[[24, 1536], pl.BF16]) -> pl.Tensor[[24, 1536], pl.BF16]:
    pl.func_attr({'static_block_mapping': True})
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
            sigmoid = pl.cast(pl.cast(sigmoid, pl.BF16, mode='rint'), pl.FP32)
        result = pl.mul(sigmoid, yt)
    out = pl.assemble(out, pl.cast(result, pl.BF16, mode='rint'), [row, c0])
    return out

@pl.function(type=pl.FunctionType.Inline)
def gate_attention(x: pl.Tensor[[24, 1536], pl.BF16], y: pl.Tensor[[24, 1536], pl.BF16], out: pl.Tensor[[24, 1536], pl.BF16]) -> pl.Tensor[[24, 1536], pl.BF16]:
    rows = 24
    with pl.spmd(rows * ((1536 + 512 - 1) // 512), allow_early_resolve=True):
        out = gate_attention_kernel(x, y, out)
    return out
