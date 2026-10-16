# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""PTO varlen causal attention over vLLM-owned BSND cache pages."""

import pypto.language as pl

TOKENS = pl.dynamic("tokens")
CACHE_ROWS = pl.dynamic("cache_rows")
REQUESTS = pl.dynamic("requests")
BOUNDARIES = pl.dynamic("boundaries")
TABLE_COLS = pl.dynamic("table_cols")
POSITIONS = pl.dynamic("positions")


def make_attention(query_heads, kv_heads, block_size=128):
    if block_size % 128:
        raise ValueError("Native block_size must be a multiple of the 128-token compute tile.")
    group = query_heads // kv_heads

    @pl.jit
    def attention(
        q: pl.Tensor[[TOKENS, query_heads, 256], pl.BF16],
        key_cache: pl.Tensor[[CACHE_ROWS, kv_heads, 256], pl.BF16],
        value_cache: pl.Tensor[[CACHE_ROWS, kv_heads, 256], pl.BF16],
        table: pl.Tensor[[REQUESTS, TABLE_COLS], pl.INT32],
        seq_lens: pl.Tensor[[REQUESTS], pl.INT32],
        starts: pl.Tensor[[BOUNDARIES], pl.INT32],
        out: pl.Out[pl.Tensor[[TOKENS, query_heads, 256], pl.BF16]],
    ):
        qf = pl.reshape(q, [pl.tensor.dim(q, 0) * query_heads, 256])
        of = pl.reshape(out, [pl.tensor.dim(q, 0) * query_heads, 256])
        kf = pl.reshape(key_cache, [pl.tensor.dim(key_cache, 0), kv_heads * 256])
        vf = pl.reshape(value_cache, [pl.tensor.dim(value_cache, 0), kv_heads * 256])
        for job in pl.spmd(pl.tensor.dim(table, 0) * kv_heads, name_hint="paged_attention"):
            request = job // kv_heads
            head = job % kv_heads
            begin = pl.read(starts, [request])
            end = pl.read(starts, [request + 1])
            length = pl.read(seq_lens, [request])
            for t in pl.range(begin, end):
                visible = length - (end - t - 1)
                qrow = t * query_heads + head * group
                qt = pl.slice(qf, [16, 256], [qrow, 0], [group, 256], pad_value=0, clamp=True)
                qt = pl.tensor.set_validshape(qt, 16, 256)
                zero_tile = pl.full([16, 128], dtype=pl.FP32, value=0.0)
                zero_column = pl.row_sum(zero_tile)
                maximum = pl.add(zero_column, -1e30)
                total = zero_column
                acc = pl.full([16, 256], dtype=pl.FP32, value=0.0)
                for tile in pl.range((visible + 127) // 128):
                    logical = tile * 128 // block_size
                    physical = pl.read(table, [request, logical])
                    valid = pl.min(128, visible - tile * 128)
                    offset = physical * block_size + tile * 128 % block_size
                    kt = pl.slice(kf, [128, 256], [offset, head * 256], [valid, 256], pad_value=0)
                    vt = pl.slice(vf, [128, 256], [offset, head * 256], [valid, 256], pad_value=0)
                    kt = pl.tensor.set_validshape(kt, 128, 256)
                    score = pl.matmul(qt, kt, b_trans=True, out_dtype=pl.FP32)
                    score = pl.mul(score, 0.0625)
                    score = pl.tensor.set_validshape(score, group, valid)
                    score = pl.fillpad(score, pad_value=pl.PadValue.min)
                    score = pl.tensor.set_validshape(score, 16, 128)
                    local_max = pl.row_max(score)
                    next_max = pl.maximum(maximum, local_max)
                    rescale = pl.exp(pl.sub(maximum, next_max))
                    probability = pl.exp(pl.row_expand_sub(score, next_max))
                    total = pl.add(pl.mul(total, rescale), pl.row_sum(probability))
                    pv = pl.matmul(pl.cast(probability, pl.BF16, mode="rint"), vt, out_dtype=pl.FP32)
                    acc = pl.add(pl.row_expand_mul(acc, rescale), pv)
                    maximum = next_max
                result = pl.row_expand_div(acc, total)
                result = pl.tensor.set_validshape(pl.cast(result, pl.BF16, mode="rint"), group, 256)
                of = pl.assemble(of, result, [qrow, 0])
        return out

    return attention


def make_rope(heads):
    @pl.jit
    def rope(
        x: pl.Tensor[[TOKENS, heads, 256], pl.BF16],
        cos: pl.Tensor[[POSITIONS, 32], pl.BF16],
        sin: pl.Tensor[[POSITIONS, 32], pl.BF16],
        positions: pl.Tensor[[TOKENS], pl.INT64],
        out: pl.Out[pl.Tensor[[TOKENS, heads, 256], pl.BF16]],
    ):
        xf = pl.reshape(x, [pl.tensor.dim(x, 0) * heads, 256])
        of = pl.reshape(out, [pl.tensor.dim(x, 0) * heads, 256])
        for job in pl.spmd(pl.tensor.dim(x, 0) * heads, name_hint="partial_rope"):
            position = pl.read(positions, [job // heads])
            ct = pl.cast(pl.slice(cos, [1, 32], [position, 0]), pl.FP32)
            st = pl.cast(pl.slice(sin, [1, 32], [position, 0]), pl.FP32)
            left = pl.cast(pl.slice(xf, [1, 32], [job, 0]), pl.FP32)
            right = pl.cast(pl.slice(xf, [1, 32], [job, 32]), pl.FP32)
            ol = pl.sub(pl.mul(left, ct), pl.mul(right, st))
            ore = pl.add(pl.mul(right, ct), pl.mul(left, st))
            of = pl.assemble(of, pl.cast(ol, pl.BF16, mode="rint"), [job, 0])
            of = pl.assemble(of, pl.cast(ore, pl.BF16, mode="rint"), [job, 32])
            tail = pl.slice(xf, [1, 192], [job, 64])
            of = pl.assemble(of, tail, [job, 64])
        return out

    return rope
