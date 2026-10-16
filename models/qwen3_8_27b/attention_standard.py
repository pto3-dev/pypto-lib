# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Paged attention with a normalized BF16 softmax rounding boundary.

Separate score, softmax, and PV tasks also parallelize independent query tokens.
"""

import pypto.language as pl

TOKENS = pl.dynamic("tokens")
ROWS = pl.dynamic("attention_rows")
CACHE_ROWS = pl.dynamic("cache_rows")
TABLE_COLS = pl.dynamic("table_cols")
BOUNDARIES = pl.dynamic("boundaries")
REQUESTS = pl.dynamic("requests")


def make_standard_attention(query_heads, kv_heads, max_seq, cache_block_size=1536):
    group = query_heads // kv_heads

    @pl.jit.inline
    def scores(
        q: pl.Tensor[[TOKENS, query_heads, 256], pl.BF16],
        cache: pl.Tensor[[CACHE_ROWS, kv_heads, 256], pl.BF16],
        table: pl.Tensor[[REQUESTS, TABLE_COLS], pl.INT32],
        lengths: pl.Tensor[[REQUESTS], pl.INT32],
        out: pl.Out[pl.Tensor[[ROWS, max_seq], pl.FP32]],
    ):
        tokens = pl.tensor.dim(q, 0)
        qf = pl.reshape(q, [tokens * query_heads, 256])
        cf = pl.reshape(cache, [pl.tensor.dim(cache, 0), kv_heads * 256])
        for job in pl.spmd(tokens * kv_heads, name_hint="attention_scores"):
            token = job // kv_heads
            head = job % kv_heads
            visible = pl.read(lengths, [0]) - tokens + token + 1
            qrow = token * query_heads + head * group
            qt = pl.slice(qf, [16, 256], [qrow, 0], [group, 256], pad_value=0, clamp=True)
            qt_full = pl.tensor.set_validshape(qt, 16, 256)
            for offset in pl.range(0, max_seq, 128):
                physical = (
                    pl.read(table, [0, offset // cache_block_size]) * cache_block_size
                    + offset % cache_block_size
                )
                valid = pl.max(0, pl.min(128, visible - offset))
                kt = pl.slice(cf, [128, 256], [physical, head * 256], [valid, 256], pad_value=0)
                kt_full = pl.tensor.set_validshape(kt, 128, 256)
                product = pl.matmul(qt_full, kt_full, b_trans=True, out_dtype=pl.FP32)
                scaled = pl.mul(product, 0.0625)
                masked = pl.tensor.set_validshape(scaled, group, valid)
                padded = pl.fillpad(masked, pad_value=pl.PadValue.min)
                stored = pl.tensor.set_validshape(padded, group, 128)
                out = pl.assemble(out, stored, [qrow, offset])
        return out

    @pl.jit.inline
    def softmax(x: pl.Tensor[[ROWS, max_seq], pl.FP32], out: pl.Out[pl.Tensor[[ROWS, max_seq], pl.BF16]]):
        rows = pl.tensor.dim(x, 0)
        for job in pl.spmd((rows + 7) // 8, name_hint="attention_softmax"):
            row = job * 8
            valid = pl.min(8, rows - row)
            score = pl.slice(x, [8, max_seq], [row, 0], [valid, max_seq], pad_value=0, clamp=True)
            maximum = pl.row_max(score)
            probability = pl.exp(pl.row_expand_sub(score, maximum))
            denominator = pl.row_sum(probability)
            normalized = pl.row_expand_div(probability, denominator)
            narrowed = pl.cast(normalized, pl.BF16, mode="rint")
            stored = pl.tensor.set_validshape(narrowed, valid, max_seq)
            out = pl.assemble(out, stored, [row, 0])
        return out

    @pl.jit.inline
    def values(
        probability: pl.Tensor[[ROWS, max_seq], pl.BF16],
        cache: pl.Tensor[[CACHE_ROWS, kv_heads, 256], pl.BF16],
        table: pl.Tensor[[REQUESTS, TABLE_COLS], pl.INT32],
        lengths: pl.Tensor[[REQUESTS], pl.INT32],
        out: pl.Out[pl.Tensor[[TOKENS, query_heads, 256], pl.BF16]],
    ):
        tokens = pl.tensor.dim(out, 0)
        cf = pl.reshape(cache, [pl.tensor.dim(cache, 0), kv_heads * 256])
        of = pl.reshape(out, [tokens * query_heads, 256])
        for job in pl.spmd(tokens * kv_heads, name_hint="attention_values"):
            token = job // kv_heads
            head = job % kv_heads
            visible = pl.read(lengths, [0]) - tokens + token + 1
            qrow = token * query_heads + head * group
            acc = pl.create_tensor([16, 256], pl.FP32)
            for offset in pl.range(0, max_seq, 128):
                physical = (
                    pl.read(table, [0, offset // cache_block_size]) * cache_block_size
                    + offset % cache_block_size
                )
                valid = pl.max(0, pl.min(128, visible - offset))
                pt = pl.slice(probability, [16, 128], [qrow, offset], [group, 128], pad_value=0, clamp=True)
                vt = pl.slice(cf, [128, 256], [physical, head * 256], [valid, 256], pad_value=0)
                pt_full = pl.tensor.set_validshape(pt, 16, 128)
                vt_full = pl.tensor.set_validshape(vt, 128, 256)
                acc = pl.matmul_acc(acc, pt_full, vt_full, init_cond=(offset == 0))
            narrowed = pl.cast(acc, pl.BF16, mode="rint")
            stored = pl.tensor.set_validshape(narrowed, group, 256)
            of = pl.assemble(of, stored, [qrow, 0])
        return out

    return scores, softmax, values
