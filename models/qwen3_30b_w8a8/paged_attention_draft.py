# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -------------------------------------------------------------------------------------------------------
"""Rank-local four-page Decode attention with device-side lengths and native BSND cache views."""

from typing import Any
import pypto.language as pl

from .linear import HEAD_DIM, LOCAL_Q_HEADS, LOCAL_KV_HEADS

CACHE_TOKENS = pl.dynamic("QWEN30_ATTENTION_CACHE_TOKENS")
TABLE_COLS = pl.dynamic("QWEN30_ATTENTION_TABLE_COLS")
PAGE = 128
STACK = 512
HEAD_TILE = 16
GROUP = LOCAL_Q_HEADS // LOCAL_KV_HEADS
SCALE = HEAD_DIM**-0.5


def make_softmax_pv(rows: int, *, early: bool = False) -> Any:
    """Keep sequential softmax and BF16 probabilities inside one mixed task."""
    if rows != 1:
        raise ValueError(f"Fused softmax/PV currently requires rows=1, got {rows}")
    tasks = rows * LOCAL_KV_HEADS

    @pl.jit.inline
    def softmax_pv(
        scores: pl.Tensor[[tasks * HEAD_TILE, STACK], pl.FP32],
        values: pl.Tensor[[tasks * STACK, HEAD_DIM], pl.BF16],
        lengths: pl.Tensor[[rows, 1], pl.INT32],
        out: pl.Tensor[[rows * LOCAL_Q_HEADS, HEAD_DIM], pl.BF16],
    ) -> pl.Tensor:
        for task in pl.spmd(tasks, name_hint="attention_softmax_pv", allow_early_resolve=early):
            token = task // LOCAL_KV_HEADS
            length = pl.cast(pl.read(lengths, [token, 0]), pl.INDEX)
            base = task * HEAD_TILE
            running_max = pl.row_max(pl.full([HEAD_TILE, 8], dtype=pl.FP32, value=-3.4028234663852886e38))
            raw_score = pl.mul(scores[base : base + HEAD_TILE, :], SCALE)
            masked = pl.set_validshape(
                pl.fillpad(pl.set_validshape(raw_score, HEAD_TILE, length), pad_value=pl.PadValue.min),
                HEAD_TILE, STACK,
            )
            running_max = pl.maximum(running_max, pl.row_max(masked))
            exponential = pl.exp(pl.row_expand_sub(masked, running_max))
            sum64 = pl.row_sum(exponential[:, :64])
            for section in pl.unroll(64, STACK, 64):
                sum64 = pl.add(sum64, pl.row_sum(exponential[:, section : section + 64]))
            probabilities = pl.cast(exponential, pl.BF16, mode="rint")
            acc = pl.matmul(probabilities, values[task * STACK : (task + 1) * STACK, :], out_dtype=pl.FP32)
            normalized = pl.cast(
                pl.row_expand_div(pl.add(acc, 0.0), pl.add(sum64, 0.0)), pl.BF16, mode="rint"
            )
            output_row = token * LOCAL_Q_HEADS + task % LOCAL_KV_HEADS * GROUP
            out[output_row : output_row + GROUP, :] = normalized[:GROUP, :]
        return out

    return softmax_pv


def make_softmax_pv_quant(rows: int) -> Any:
    """Fuse PV output quantization with one scale across every local query head."""
    if rows != 1:
        raise ValueError(f"Fused softmax/PV currently requires rows=1, got {rows}")
    tasks = rows * LOCAL_KV_HEADS

    @pl.jit.inline
    def softmax_pv_quant(
        scores: pl.Tensor[[tasks * HEAD_TILE, STACK], pl.FP32],
        values: pl.Tensor[[tasks * STACK, HEAD_DIM], pl.BF16],
        lengths: pl.Tensor[[rows, 1], pl.INT32],
        quant: pl.Tensor[[rows, LOCAL_Q_HEADS * HEAD_DIM], pl.INT8],
        scale: pl.Tensor[[rows, 1], pl.FP32],
    ) -> tuple[pl.Tensor, pl.Tensor]:
        for _block in pl.spmd(1, name_hint="attention_softmax_pv_quant"):
            context = pl.full([LOCAL_Q_HEADS, HEAD_DIM], dtype=pl.BF16, value=0.0)
            for task in pl.range(tasks):
                token = task // LOCAL_KV_HEADS
                length = pl.cast(pl.read(lengths, [token, 0]), pl.INDEX)
                base = task * HEAD_TILE
                running_max = pl.row_max(pl.full([HEAD_TILE, 8], dtype=pl.FP32, value=-3.4028234663852886e38))
                raw_score = pl.mul(scores[base : base + HEAD_TILE, :], SCALE)
                masked = pl.set_validshape(
                    pl.fillpad(pl.set_validshape(raw_score, HEAD_TILE, length), pad_value=pl.PadValue.min),
                    HEAD_TILE, STACK,
                )
                running_max = pl.maximum(running_max, pl.row_max(masked))
                exponential = pl.exp(pl.row_expand_sub(masked, running_max))
                sum64 = pl.row_sum(exponential[:, :64])
                for section in pl.unroll(64, STACK, 64):
                    sum64 = pl.add(sum64, pl.row_sum(exponential[:, section : section + 64]))
                probabilities = pl.cast(exponential, pl.BF16, mode="rint")
                acc = pl.matmul(probabilities, values[task * STACK : (task + 1) * STACK, :], out_dtype=pl.FP32)
                normalized = pl.cast(
                    pl.row_expand_div(pl.add(acc, 0.0), pl.add(sum64, 0.0)), pl.BF16, mode="rint"
                )
                output_row = token * LOCAL_Q_HEADS + task % LOCAL_KV_HEADS * GROUP
                context = pl.assemble(context, normalized[:GROUP, :], [output_row, 0])
            row = pl.cast(pl.reshape(context, [1, LOCAL_Q_HEADS * HEAD_DIM]), pl.FP32)
            amax = pl.reshape(pl.row_max(pl.reshape(pl.abs(row), [8, LOCAL_Q_HEADS * HEAD_DIM // 8])), [1, 8])
            max_value = pl.read(amax, [0, 0])
            for lane in pl.unroll(1, 8):
                candidate = pl.read(amax, [0, lane])
                if candidate > max_value:
                    max_value = candidate
            if max_value > 0.0:
                pl.write(amax, [0, 0], max_value)
                inverse_scale = pl.div(pl.full([1, 8], dtype=pl.FP32, value=127.0), amax)
                inverse_value = pl.read(inverse_scale, [0, 0])
                pl.write(scale, [0, 0], pl.div(1.0, inverse_value))
                rounded = pl.cast(pl.mul(row, inverse_value), pl.INT32, mode="rint")
                quant[:, :] = pl.cast(pl.cast(rounded, pl.FP16), pl.INT8, mode="trunc")
            else:
                quant[:, :] = pl.cast(row, pl.INT8, mode="rint")
                pl.write(scale, [0, 0], 0.0)
        return quant, scale

    return softmax_pv_quant


def make_paged_attention_quant(rows: int, *, page_qk_impl: Any = None) -> Any:
    """B1 candidate: page/QK then one mixed softmax/PV/global-quant task."""
    if rows != 1:
        raise ValueError(f"Fused Attention quantization requires rows=1, got {rows}")
    tasks = rows * LOCAL_KV_HEADS
    page_qk = make_page_qk(rows, pages_per_block=4) if page_qk_impl is None else page_qk_impl
    softmax_pv_quant = make_softmax_pv_quant(rows)

    @pl.jit.inline
    def attention_quant(
        query: pl.Tensor[[rows * LOCAL_Q_HEADS, HEAD_DIM], pl.BF16],
        key_cache: pl.Tensor[[CACHE_TOKENS, LOCAL_KV_HEADS * HEAD_DIM], pl.BF16],
        value_cache: pl.Tensor[[CACHE_TOKENS, LOCAL_KV_HEADS * HEAD_DIM], pl.BF16],
        block_table: pl.Tensor[[rows, TABLE_COLS], pl.INT32],
        lengths: pl.Tensor[[rows, 1], pl.INT32],
        quant: pl.Tensor[[rows, LOCAL_Q_HEADS * HEAD_DIM], pl.INT8],
        scale: pl.Tensor[[rows, 1], pl.FP32],
    ) -> tuple[pl.Tensor, pl.Tensor]:
        key_cache.bind_dynamic(0, CACHE_TOKENS)
        value_cache.bind_dynamic(0, CACHE_TOKENS)
        block_table.bind_dynamic(1, TABLE_COLS)
        values = pl.create_tensor([tasks * STACK, HEAD_DIM], dtype=pl.BF16)
        scores = pl.create_tensor([tasks * HEAD_TILE, STACK], dtype=pl.FP32)
        scores_after, values_after = page_qk(
            query, key_cache, value_cache, block_table, lengths, scores, values
        )
        quant_after, scale_after = softmax_pv_quant(scores_after, values_after, lengths, quant, scale)
        return quant_after, scale_after

    return attention_quant


def make_qk_softmax_pv(rows: int) -> Any:
    """Keep sequential softmax and BF16 probabilities inside one paged mixed task."""
    if rows != 1:
        raise ValueError(f"Fused QK/softmax/PV currently requires rows=1, got {rows}")
    tasks = rows * LOCAL_KV_HEADS

    @pl.jit.inline
    def qk_softmax_pv(
        query: pl.Tensor[[rows * LOCAL_Q_HEADS, HEAD_DIM], pl.BF16],
        key_cache: pl.Tensor[[CACHE_TOKENS, LOCAL_KV_HEADS * HEAD_DIM], pl.BF16],
        value_cache: pl.Tensor[[CACHE_TOKENS, LOCAL_KV_HEADS * HEAD_DIM], pl.BF16],
        block_table: pl.Tensor[[rows, TABLE_COLS], pl.INT32],
        lengths: pl.Tensor[[rows, 1], pl.INT32],
        out: pl.Tensor[[rows * LOCAL_Q_HEADS, HEAD_DIM], pl.BF16],
    ) -> pl.Tensor:
        for task in pl.spmd(tasks, name_hint="attention_qk_softmax_pv"):
            token = task // LOCAL_KV_HEADS
            length = pl.cast(pl.read(lengths, [token, 0]), pl.INDEX)
            kv_head = task % LOCAL_KV_HEADS
            q_base = token * LOCAL_Q_HEADS + kv_head * GROUP
            q_valid = pl.slice(query, [HEAD_TILE, HEAD_DIM], [q_base, 0], valid_shape=[GROUP, HEAD_DIM])
            q_full = pl.set_validshape(pl.fillpad(q_valid, pad_value=pl.PadValue.zero), HEAD_TILE, HEAD_DIM)
            local_scores = pl.full([HEAD_TILE, STACK], dtype=pl.FP32, value=0.0)
            for page in pl.range(STACK // PAGE):
                col = page * PAGE
                if col < length:
                    physical = pl.cast(pl.read(block_table, [token, page]), pl.INDEX)
                    valid = pl.min(PAGE, length - col)
                    key_valid = pl.slice(
                        key_cache, [PAGE, HEAD_DIM], [physical * PAGE, kv_head * HEAD_DIM],
                        valid_shape=[valid, HEAD_DIM],
                    )
                    key_full = pl.set_validshape(
                        pl.fillpad(key_valid, pad_value=pl.PadValue.zero), PAGE, HEAD_DIM
                    )
                    score = pl.matmul(q_full, key_full, b_trans=True, out_dtype=pl.FP32)
                    local_scores = pl.assemble(local_scores, score, [0, col])
            running_max = pl.row_max(pl.full([HEAD_TILE, 8], dtype=pl.FP32, value=-3.4028234663852886e38))
            raw_score = pl.mul(local_scores, SCALE)
            masked = pl.set_validshape(
                pl.fillpad(pl.set_validshape(raw_score, HEAD_TILE, pl.max(0, pl.min(STACK, length))), pad_value=pl.PadValue.min),
                HEAD_TILE, STACK,
            )
            running_max = pl.maximum(running_max, pl.row_max(masked))
            exponential = pl.exp(pl.row_expand_sub(masked, running_max))
            sum64 = pl.row_sum(exponential[:, :64])
            for section in pl.unroll(64, STACK, 64):
                sum64 = pl.add(sum64, pl.row_sum(exponential[:, section : section + 64]))
            probabilities = pl.cast(exponential, pl.BF16, mode="rint")
            acc = pl.create_tensor([HEAD_TILE, HEAD_DIM], dtype=pl.FP32)
            for page in pl.range(STACK // PAGE):
                col = page * PAGE
                value_full = pl.full([PAGE, HEAD_DIM], dtype=pl.BF16, value=0.0)
                if col < length:
                    physical = pl.cast(pl.read(block_table, [token, page]), pl.INDEX)
                    valid = pl.min(PAGE, length - col)
                    value_valid = pl.slice(
                        value_cache, [PAGE, HEAD_DIM], [physical * PAGE, kv_head * HEAD_DIM],
                        valid_shape=[valid, HEAD_DIM],
                    )
                    value_full = pl.set_validshape(
                        pl.fillpad(value_valid, pad_value=pl.PadValue.zero), PAGE, HEAD_DIM
                    )
                acc = pl.matmul_acc(
                    acc, probabilities[:, col : col + PAGE], value_full, init_cond=(page == 0)
                )
            normalized = pl.cast(
                pl.row_expand_div(pl.add(acc, 0.0), pl.add(sum64, 0.0)), pl.BF16, mode="rint"
            )
            output_row = token * LOCAL_Q_HEADS + task % LOCAL_KV_HEADS * GROUP
            out[output_row : output_row + GROUP, :] = normalized[:GROUP, :]
        return out

    return qk_softmax_pv


def make_page_qk(rows: int, *, pages_per_block: int = 4) -> Any:
    """Read K pages into the QK tile; retain only V's contiguous GM staging."""
    if type(pages_per_block) is not int or pages_per_block not in (1, 4):
        raise ValueError(f"Expected page QK block to own 1 or 4 pages, got {pages_per_block}")
    tasks = rows * LOCAL_KV_HEADS
    page_groups = STACK // PAGE // pages_per_block

    @pl.jit.inline
    def page_qk(
        query: pl.Tensor[[rows * LOCAL_Q_HEADS, HEAD_DIM], pl.BF16],
        key_cache: pl.Tensor[[CACHE_TOKENS, LOCAL_KV_HEADS * HEAD_DIM], pl.BF16],
        value_cache: pl.Tensor[[CACHE_TOKENS, LOCAL_KV_HEADS * HEAD_DIM], pl.BF16],
        block_table: pl.Tensor[[rows, TABLE_COLS], pl.INT32],
        lengths: pl.Tensor[[rows, 1], pl.INT32],
        scores: pl.Out[pl.Tensor[[tasks * HEAD_TILE, STACK], pl.FP32]],
        values: pl.Out[pl.Tensor[[tasks * STACK, HEAD_DIM], pl.BF16]],
    ) -> tuple[pl.Tensor, pl.Tensor]:
        for block in pl.spmd(tasks * page_groups, name_hint="attention_page_qk", allow_early_resolve=True):
            task = block // page_groups
            page_start = (block % page_groups) * pages_per_block
            token = task // LOCAL_KV_HEADS
            kv_head = task % LOCAL_KV_HEADS
            length = pl.cast(pl.read(lengths, [token, 0]), pl.INDEX)
            q_base = token * LOCAL_Q_HEADS + kv_head * GROUP
            q_valid = pl.slice(query, [HEAD_TILE, HEAD_DIM], [q_base, 0], valid_shape=[GROUP, HEAD_DIM])
            q_full = pl.set_validshape(pl.fillpad(q_valid, pad_value=pl.PadValue.zero), HEAD_TILE, HEAD_DIM)
            for page in pl.range(page_start, page_start + pages_per_block):
                row = task * STACK + page * PAGE
                col = page * PAGE
                if col < length:
                    physical = pl.cast(pl.read(block_table, [token, page]), pl.INDEX)
                    valid = pl.min(PAGE, length - col)
                    key_valid = pl.slice(
                        key_cache, [PAGE, HEAD_DIM], [physical * PAGE, kv_head * HEAD_DIM],
                        valid_shape=[valid, HEAD_DIM],
                    )
                    key_full = pl.set_validshape(
                        pl.fillpad(key_valid, pad_value=pl.PadValue.zero), PAGE, HEAD_DIM
                    )
                    score = pl.matmul(q_full, key_full, b_trans=True, out_dtype=pl.FP32)
                    scores[task * HEAD_TILE : (task + 1) * HEAD_TILE, col : col + PAGE] = score
                    value_valid = pl.slice(
                        value_cache, [PAGE, HEAD_DIM], [physical * PAGE, kv_head * HEAD_DIM],
                        valid_shape=[valid, HEAD_DIM],
                    )
                    values[row : row + PAGE, :] = pl.set_validshape(
                        pl.fillpad(value_valid, pad_value=pl.PadValue.zero), PAGE, HEAD_DIM
                    )
                else:
                    scores[task * HEAD_TILE : (task + 1) * HEAD_TILE, col : col + PAGE] = pl.full(
                        [HEAD_TILE, PAGE], dtype=pl.FP32, value=0.0
                    )
                    values[row : row + PAGE, :] = pl.full([PAGE, HEAD_DIM], dtype=pl.BF16, value=0.0)
        return scores, values

    return page_qk


def make_paged_attention(
    rows: int, chunk: int = 256, *, fold_sum: bool = False, reciprocal_normalize: bool = False,
    sequential_sum: bool = False, inline: bool = False, pv_cols: int = 32,
    fuse_pv_normalize: bool = False,
    fuse_page_qk: bool = False,
    page_qk_pages_per_block: int = 4,
    fuse_softmax_pv: bool = False,
    fuse_qk_softmax_pv: bool = False,
    page_qk_impl: Any = None,
) -> Any:
    """Specialize a <=512-context graph bucket with two local KV heads."""
    if not 1 <= rows <= 16:
        raise ValueError(f"Unsupported attention bucket: {rows}")
    if chunk not in (128, 256, 512):
        raise ValueError(f"Unsupported attention chunk: {chunk}")
    if fold_sum and sequential_sum:
        raise ValueError("Choose either folded or sequential softmax reduction")
    if pv_cols not in (32, 64, 128):
        raise ValueError(f"Expected PV column tile in (32,64,128), got {pv_cols}")
    if fuse_pv_normalize and (chunk != STACK or pv_cols != HEAD_DIM or reciprocal_normalize):
        raise ValueError("Fused PV normalization requires chunk=512, pv_cols=128 and division")
    if fuse_page_qk and (rows != 1 or chunk != STACK):
        raise ValueError("Fused page QK currently requires rows=1 and chunk=512")
    if fuse_softmax_pv and (rows != 1 or chunk != STACK or not sequential_sum or not fuse_pv_normalize):
        raise ValueError("Fused softmax/PV requires rows=1, chunk=512, sequential_sum and fused PV normalization")
    if fuse_qk_softmax_pv and (
        rows != 1 or chunk != STACK or not sequential_sum or not fuse_pv_normalize
    ):
        raise ValueError("Fused QK/softmax/PV requires rows=1, chunk=512, sequential_sum and fused PV normalization")
    tasks = rows * LOCAL_KV_HEADS
    page_qk = make_page_qk(rows, pages_per_block=page_qk_pages_per_block) if page_qk_impl is None else page_qk_impl
    softmax_pv = make_softmax_pv(1, early=rows == 1)
    qk_softmax_pv = make_qk_softmax_pv(1)

    decorator = pl.jit.inline if inline else pl.jit

    @decorator
    def attention(
        query: pl.Tensor[[rows * LOCAL_Q_HEADS, HEAD_DIM], pl.BF16],
        key_cache: pl.Tensor[[CACHE_TOKENS, LOCAL_KV_HEADS * HEAD_DIM], pl.BF16],
        value_cache: pl.Tensor[[CACHE_TOKENS, LOCAL_KV_HEADS * HEAD_DIM], pl.BF16],
        block_table: pl.Tensor[[rows, TABLE_COLS], pl.INT32],
        lengths: pl.Tensor[[rows, 1], pl.INT32],
        out: pl.Out[pl.Tensor[[rows * LOCAL_Q_HEADS, HEAD_DIM], pl.BF16]],
    ) -> pl.Tensor:
        key_cache.bind_dynamic(0, CACHE_TOKENS)
        value_cache.bind_dynamic(0, CACHE_TOKENS)
        block_table.bind_dynamic(1, TABLE_COLS)
        keys = pl.create_tensor([tasks * STACK, HEAD_DIM], dtype=pl.BF16)
        values = pl.create_tensor([tasks * STACK, HEAD_DIM], dtype=pl.BF16)
        scores = pl.create_tensor([tasks * HEAD_TILE, STACK], dtype=pl.FP32)
        probabilities = pl.create_tensor([tasks * HEAD_TILE, STACK], dtype=pl.BF16)
        chunks = STACK // chunk
        sums = pl.create_tensor([tasks * chunks * HEAD_TILE, 1], dtype=pl.FP32)
        maxima = pl.create_tensor([tasks * chunks * HEAD_TILE, 1], dtype=pl.FP32)
        numerators = pl.create_tensor([tasks * chunks * HEAD_TILE, HEAD_DIM], dtype=pl.FP32)
        if fuse_qk_softmax_pv:
            out = qk_softmax_pv(query, key_cache, value_cache, block_table, lengths, out)
        else:
            if fuse_page_qk:
                _scores_after, _values_after = page_qk(
                    query, key_cache, value_cache, block_table, lengths, scores, values
                )
            else:
                for task in pl.spmd(tasks, name_hint="attention_stage_pages"):
                    token = task // LOCAL_KV_HEADS
                    kv_head = task % LOCAL_KV_HEADS
                    length = pl.cast(pl.read(lengths, [token, 0]), pl.INDEX)
                    for page in pl.range(STACK // PAGE):
                        row = task * STACK + page * PAGE
                        if page * PAGE < length:
                            physical = pl.cast(pl.read(block_table, [token, page]), pl.INDEX)
                            valid = pl.min(PAGE, length - page * PAGE)
                            key_valid = pl.slice(
                                key_cache,
                                [PAGE, HEAD_DIM],
                                [physical * PAGE, kv_head * HEAD_DIM],
                                valid_shape=[valid, HEAD_DIM],
                            )
                            key_full = pl.set_validshape(
                                pl.fillpad(key_valid, pad_value=pl.PadValue.zero), PAGE, HEAD_DIM
                            )
                            value_valid = pl.slice(
                                value_cache,
                                [PAGE, HEAD_DIM],
                                [physical * PAGE, kv_head * HEAD_DIM],
                                valid_shape=[valid, HEAD_DIM],
                            )
                            value_full = pl.set_validshape(
                                pl.fillpad(value_valid, pad_value=pl.PadValue.zero), PAGE, HEAD_DIM
                            )
                            keys[row : row + PAGE, :] = key_full
                            values[row : row + PAGE, :] = value_full
                        else:
                            keys[row : row + PAGE, :] = pl.full([PAGE, HEAD_DIM], dtype=pl.BF16, value=0.0)
                            values[row : row + PAGE, :] = pl.full([PAGE, HEAD_DIM], dtype=pl.BF16, value=0.0)
                for task in pl.spmd(tasks, name_hint="attention_qk"):
                    token = task // LOCAL_KV_HEADS
                    kv_head = task % LOCAL_KV_HEADS
                    q_base = token * LOCAL_Q_HEADS + kv_head * GROUP
                    q_valid = pl.slice(query, [HEAD_TILE, HEAD_DIM], [q_base, 0], valid_shape=[GROUP, HEAD_DIM])
                    q_full = pl.set_validshape(pl.fillpad(q_valid, pad_value=pl.PadValue.zero), HEAD_TILE, HEAD_DIM)
                    score = pl.matmul(
                        q_full, keys[task * STACK : (task + 1) * STACK, :], b_trans=True, out_dtype=pl.FP32
                    )
                    scores[task * HEAD_TILE : (task + 1) * HEAD_TILE, :] = score
            if fuse_softmax_pv:
                out = softmax_pv(scores, values, lengths, out)
            else:
                for task in pl.spmd(tasks, name_hint="attention_softmax_online"):
                    token = task // LOCAL_KV_HEADS
                    length = pl.cast(pl.read(lengths, [token, 0]), pl.INDEX)
                    base = task * HEAD_TILE
                    running_max = pl.row_max(pl.full([HEAD_TILE, 8], dtype=pl.FP32, value=-3.4028234663852886e38))
                    for part in pl.unroll(STACK // chunk):
                        col = part * chunk
                        row = (task * (STACK // chunk) + part) * HEAD_TILE
                        raw_score = pl.mul(scores[base : base + HEAD_TILE, col : col + chunk], SCALE)
                        valid_cols = pl.max(0, pl.min(chunk, length - col))
                        masked = pl.set_validshape(
                            pl.fillpad(
                                pl.set_validshape(raw_score, HEAD_TILE, valid_cols), pad_value=pl.PadValue.min
                            ),
                            HEAD_TILE,
                            chunk,
                        )
                        running_max = pl.maximum(running_max, pl.row_max(masked))
                        exponential = pl.exp(pl.row_expand_sub(masked, running_max))
                        if sequential_sum:
                            sum64 = pl.row_sum(exponential[:, :64])
                            for section in pl.unroll(64, chunk, 64):
                                sum64 = pl.add(sum64, pl.row_sum(exponential[:, section:section + 64]))
                            sums[row : row + HEAD_TILE, :] = sum64
                        elif fold_sum:
                            half = pl.add(exponential[:, :chunk // 2], exponential[:, chunk // 2:chunk])
                            quarter = pl.add(half[:, :chunk // 4], half[:, chunk // 4:chunk // 2])
                            eighth = pl.add(quarter[:, :chunk // 8], quarter[:, chunk // 8:chunk // 4])
                            if chunk == 512:
                                sums[row : row + HEAD_TILE, :] = pl.row_sum(eighth)
                            elif chunk == 256:
                                sums[row : row + HEAD_TILE, :] = pl.row_sum(quarter)
                            else:
                                sums[row : row + HEAD_TILE, :] = pl.row_sum(half)
                        else:
                            sums[row : row + HEAD_TILE, :] = pl.row_sum(exponential)
                        if not fuse_pv_normalize:
                            maxima[row : row + HEAD_TILE, :] = running_max
                        probabilities[base : base + HEAD_TILE, col : col + chunk] = pl.cast(
                            exponential, pl.BF16, mode="rint"
                        )
                for task in pl.spmd(tasks, name_hint="attention_pv_online"):
                    base = task * HEAD_TILE
                    for part in pl.unroll(STACK // chunk):
                        col_k = part * chunk
                        row = (task * (STACK // chunk) + part) * HEAD_TILE
                        for col in pl.unroll(0, HEAD_DIM, pv_cols):
                            acc = pl.matmul(
                                probabilities[base : base + HEAD_TILE, col_k : col_k + chunk],
                                values[task * STACK + col_k : task * STACK + col_k + chunk, col : col + pv_cols],
                                out_dtype=pl.FP32,
                            )
                            if fuse_pv_normalize:
                                output_row = task // LOCAL_KV_HEADS * LOCAL_Q_HEADS + task % LOCAL_KV_HEADS * GROUP
                                fused_num = pl.add(acc, 0.0)
                                fused_denom = pl.add(sums[row : row + HEAD_TILE, :], 0.0)
                                normalized = pl.cast(
                                    pl.row_expand_div(fused_num, fused_denom), pl.BF16, mode="rint"
                                )
                                out[output_row : output_row + GROUP, :] = normalized[:GROUP, :]
                            else:
                                numerators[row : row + HEAD_TILE, col : col + pv_cols] = acc
                if not fuse_pv_normalize:
                    for task in pl.spmd(tasks, name_hint="attention_normalize"):
                        token = task // LOCAL_KV_HEADS
                        kv_head = task % LOCAL_KV_HEADS
                        output_row = token * LOCAL_Q_HEADS + kv_head * GROUP
                        previous_max = pl.row_max(pl.full([GROUP, 8], dtype=pl.FP32, value=-3.4028234663852886e38))
                        num = pl.full([GROUP, HEAD_DIM], dtype=pl.FP32, value=0.0)
                        denom = pl.row_sum(pl.full([GROUP, 8], dtype=pl.FP32, value=0.0))
                        for part in pl.unroll(STACK // chunk):
                            base = (task * (STACK // chunk) + part) * HEAD_TILE
                            current_max = maxima[base : base + GROUP, :]
                            correction = pl.exp(pl.sub(previous_max, current_max))
                            num = pl.add(pl.row_expand_mul(num, correction), numerators[base : base + GROUP, :])
                            denom = pl.add(pl.mul(denom, correction), sums[base : base + GROUP, :])
                            previous_max = current_max
                        if reciprocal_normalize:
                            out[output_row : output_row + GROUP, :] = pl.cast(
                                pl.row_expand_mul(num, pl.recip(denom)), pl.BF16, mode="rint"
                            )
                        else:
                            out[output_row : output_row + GROUP, :] = pl.cast(
                                pl.row_expand_div(num, denom), pl.BF16, mode="rint"
                            )
        return out

    return attention
