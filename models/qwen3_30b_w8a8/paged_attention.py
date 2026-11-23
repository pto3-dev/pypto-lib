# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -------------------------------------------------------------------------------------------------------
"""Decode attention with device-side lengths and fixed context-capacity buckets.

Native CANN 8.5.1 probes use 128-token chunks below a batch maximum of 256,
and 256-token chunks at/above it. This bounded rule still requires graph
boundary and heterogeneous-length qualification; it is not a general ATB ABI.
"""

from typing import Any

import pypto.language as pl

from .linear import HEAD_DIM, LOCAL_Q_HEADS, LOCAL_KV_HEADS
from .paged_attention_draft import CACHE_TOKENS, TABLE_COLS, PAGE, HEAD_TILE, GROUP, SCALE


def make_paged_attention(
    rows: int,
    fold_sum: bool = True,
    *,
    inline: bool = False,
    pv_cols: int = 128,
    max_context: int = 512,
    direct_pages: bool | None = None,
    active_chunks: bool | None = None,
) -> Any:
    if not 1 <= rows <= 16:
        raise ValueError(f"Unsupported attention bucket: {rows}")
    if pv_cols not in (32, 64, 128):
        raise ValueError(f"Unsupported PV column tile: {pv_cols}; expected 32, 64, or 128")
    if max_context not in (512, 1024, 2048, 4096):
        raise ValueError(f"Unsupported context capacity: {max_context}; expected 512, 1024, 2048, or 4096")
    if direct_pages is None:
        direct_pages = rows == 1 and max_context > 512
    if active_chunks is None:
        active_chunks = rows == 1 and max_context > 512
    tasks = rows * LOCAL_KV_HEADS

    decorator = pl.jit.inline if inline else pl.jit

    @decorator
    def adaptive_attention(
        query: pl.Tensor[[rows * LOCAL_Q_HEADS, HEAD_DIM], pl.BF16],
        key_cache: pl.Tensor[[CACHE_TOKENS, LOCAL_KV_HEADS * HEAD_DIM], pl.BF16],
        value_cache: pl.Tensor[[CACHE_TOKENS, LOCAL_KV_HEADS * HEAD_DIM], pl.BF16],
        block_table: pl.Tensor[[rows, TABLE_COLS], pl.INT32],
        lengths: pl.Tensor[[rows, 1], pl.INT32],
        out: pl.Out[pl.Tensor[[rows * LOCAL_Q_HEADS, HEAD_DIM], pl.BF16]],
    ) -> pl.Tensor:
        if not inline:
            key_cache.bind_dynamic(0, CACHE_TOKENS)
            value_cache.bind_dynamic(0, CACHE_TOKENS)
            block_table.bind_dynamic(1, TABLE_COLS)
        keys = pl.create_tensor([tasks * max_context, HEAD_DIM], dtype=pl.BF16)
        values = pl.create_tensor([tasks * max_context, HEAD_DIM], dtype=pl.BF16)
        scores = pl.create_tensor([tasks * HEAD_TILE, max_context], dtype=pl.FP32)
        probabilities = pl.create_tensor([tasks * HEAD_TILE, max_context], dtype=pl.BF16)
        sums = pl.create_tensor([tasks * (max_context // 128) * HEAD_TILE, 1], dtype=pl.FP32)
        maxima = pl.create_tensor([tasks * (max_context // 128) * HEAD_TILE, 1], dtype=pl.FP32)
        numerators = pl.create_tensor([tasks * (max_context // 128) * HEAD_TILE, HEAD_DIM], dtype=pl.FP32)
        for task in pl.spmd(tasks, name_hint="attention_stage_pages"):
            token = task // LOCAL_KV_HEADS
            kv_head = task % LOCAL_KV_HEADS
            length = pl.cast(pl.read(lengths, [token, 0]), pl.INDEX)
            for page in pl.range(max_context // PAGE):
                row = task * max_context + page * PAGE
                if page * PAGE < length:
                    physical = pl.cast(pl.read(block_table, [token, page]), pl.INDEX)
                    valid = pl.min(PAGE, length - page * PAGE)
                    if not direct_pages:
                        key_valid = pl.slice(
                            key_cache,
                            [PAGE, HEAD_DIM],
                            [physical * PAGE, kv_head * HEAD_DIM],
                            valid_shape=[valid, HEAD_DIM],
                        )
                        key_full = pl.set_validshape(
                            pl.fillpad(key_valid, pad_value=pl.PadValue.zero), PAGE, HEAD_DIM
                        )
                        keys[row : row + PAGE, :] = key_full
                    value_valid = pl.slice(
                        value_cache,
                        [PAGE, HEAD_DIM],
                        [physical * PAGE, kv_head * HEAD_DIM],
                        valid_shape=[valid, HEAD_DIM],
                    )
                    value_full = pl.set_validshape(
                        pl.fillpad(value_valid, pad_value=pl.PadValue.zero), PAGE, HEAD_DIM
                    )
                    values[row : row + PAGE, :] = value_full
                else:
                    if not direct_pages:
                        keys[row : row + PAGE, :] = pl.full([PAGE, HEAD_DIM], dtype=pl.BF16, value=0.0)
                    values[row : row + PAGE, :] = pl.full([PAGE, HEAD_DIM], dtype=pl.BF16, value=0.0)
        if direct_pages:
            for page_task in pl.spmd(tasks * (max_context // PAGE), name_hint="attention_qk_direct_pages"):
                task = page_task // (max_context // PAGE)
                page = page_task % (max_context // PAGE)
                token = task // LOCAL_KV_HEADS
                kv_head = task % LOCAL_KV_HEADS
                length = pl.cast(pl.read(lengths, [token, 0]), pl.INDEX)
                if page * PAGE < length:
                    q_base = token * LOCAL_Q_HEADS + kv_head * GROUP
                    q_valid = pl.slice(
                        query, [HEAD_TILE, HEAD_DIM], [q_base, 0], valid_shape=[GROUP, HEAD_DIM]
                    )
                    q_full = pl.set_validshape(
                        pl.fillpad(q_valid, pad_value=pl.PadValue.zero), HEAD_TILE, HEAD_DIM
                    )
                    physical = pl.cast(pl.read(block_table, [token, page]), pl.INDEX)
                    direct_key_valid = pl.slice(
                        key_cache,
                        [PAGE, HEAD_DIM],
                        [physical * PAGE, kv_head * HEAD_DIM],
                        valid_shape=[pl.min(PAGE, length - page * PAGE), HEAD_DIM],
                    )
                    key_full = pl.set_validshape(
                        pl.fillpad(direct_key_valid, pad_value=pl.PadValue.zero), PAGE, HEAD_DIM
                    )
                    page_score = pl.matmul(q_full, key_full, b_trans=True, out_dtype=pl.FP32)
                    scores[task * HEAD_TILE : (task + 1) * HEAD_TILE, page * PAGE : (page + 1) * PAGE] = (
                        page_score
                    )
                else:
                    scores[task * HEAD_TILE : (task + 1) * HEAD_TILE, page * PAGE : (page + 1) * PAGE] = (
                        pl.full([HEAD_TILE, PAGE], dtype=pl.FP32, value=0.0)
                    )
        else:
            for task in pl.spmd(tasks, name_hint="attention_qk"):
                token = task // LOCAL_KV_HEADS
                kv_head = task % LOCAL_KV_HEADS
                q_base = token * LOCAL_Q_HEADS + kv_head * GROUP
                q_valid = pl.slice(query, [HEAD_TILE, HEAD_DIM], [q_base, 0], valid_shape=[GROUP, HEAD_DIM])
                q_full = pl.set_validshape(
                    pl.fillpad(q_valid, pad_value=pl.PadValue.zero), HEAD_TILE, HEAD_DIM
                )
                for col in pl.unroll(0, max_context, 512):
                    score = pl.matmul(
                        q_full,
                        keys[task * max_context + col : task * max_context + col + 512, :],
                        b_trans=True,
                        out_dtype=pl.FP32,
                    )
                    scores[task * HEAD_TILE : (task + 1) * HEAD_TILE, col : col + 512] = score
        for task in pl.spmd(tasks, name_hint="attention_softmax_adaptive"):
            token = task // LOCAL_KV_HEADS
            length = pl.cast(pl.read(lengths, [token, 0]), pl.INDEX)
            max_length = length
            for other in pl.range(rows):
                max_length = pl.max(max_length, pl.cast(pl.read(lengths, [other, 0]), pl.INDEX))
            base = task * HEAD_TILE
            running_max = pl.row_max(pl.full([HEAD_TILE, 8], dtype=pl.FP32, value=-3.4028234663852886e38))
            part_count = max_context // 128
            if active_chunks:
                if max_length < 256:
                    part_count = pl.min(max_context // 128, (max_length + 127) // 128)
                else:
                    part_count = pl.min(max_context // 256, (max_length + 255) // 256)
            for part in pl.range(part_count):
                meta_row = (task * (max_context // 128) + part) * HEAD_TILE
                if max_length < 256:
                    col128 = part * 128
                    raw128 = pl.mul(scores[base : base + HEAD_TILE, col128 : col128 + 128], SCALE)
                    valid128 = pl.max(0, pl.min(128, length - col128))
                    masked128 = pl.set_validshape(
                        pl.fillpad(pl.set_validshape(raw128, HEAD_TILE, valid128), pad_value=pl.PadValue.min),
                        HEAD_TILE,
                        128,
                    )
                    running_max = pl.maximum(running_max, pl.row_max(masked128))
                    exp128 = pl.exp(pl.row_expand_sub(masked128, running_max))
                    if fold_sum:
                        folded128 = pl.add(exp128[:, :64], exp128[:, 64:])
                        sums[meta_row : meta_row + HEAD_TILE, :] = pl.row_sum(folded128)
                    else:
                        sums[meta_row : meta_row + HEAD_TILE, :] = pl.row_sum(exp128)
                    maxima[meta_row : meta_row + HEAD_TILE, :] = running_max
                    probabilities[base : base + HEAD_TILE, col128 : col128 + 128] = pl.cast(
                        exp128, pl.BF16, mode="rint"
                    )
                elif part < max_context // 256:
                    col256 = part * 256
                    raw256 = pl.mul(scores[base : base + HEAD_TILE, col256 : col256 + 256], SCALE)
                    valid256 = pl.max(0, pl.min(256, length - col256))
                    masked256 = pl.set_validshape(
                        pl.fillpad(pl.set_validshape(raw256, HEAD_TILE, valid256), pad_value=pl.PadValue.min),
                        HEAD_TILE,
                        256,
                    )
                    running_max = pl.maximum(running_max, pl.row_max(masked256))
                    exp256 = pl.exp(pl.row_expand_sub(masked256, running_max))
                    if fold_sum:
                        folded256 = pl.add(exp256[:, :128], exp256[:, 128:])
                        folded64 = pl.add(folded256[:, :64], folded256[:, 64:])
                        sums[meta_row : meta_row + HEAD_TILE, :] = pl.row_sum(folded64)
                    else:
                        sums[meta_row : meta_row + HEAD_TILE, :] = pl.row_sum(exp256)
                    maxima[meta_row : meta_row + HEAD_TILE, :] = running_max
                    probabilities[base : base + HEAD_TILE, col256 : col256 + 256] = pl.cast(
                        exp256, pl.BF16, mode="rint"
                    )
        for task in pl.spmd(tasks, name_hint="attention_pv_adaptive"):
            max_length = pl.cast(pl.read(lengths, [0, 0]), pl.INDEX)
            for other in pl.range(rows):
                max_length = pl.max(max_length, pl.cast(pl.read(lengths, [other, 0]), pl.INDEX))
            base = task * HEAD_TILE
            part_count = max_context // 128
            if active_chunks:
                if max_length < 256:
                    part_count = pl.min(max_context // 128, (max_length + 127) // 128)
                else:
                    part_count = pl.min(max_context // 256, (max_length + 255) // 256)
            for part in pl.range(part_count):
                meta_row = (task * (max_context // 128) + part) * HEAD_TILE
                for col in pl.unroll(0, HEAD_DIM, pv_cols):
                    if max_length < 256:
                        k128 = part * 128
                        acc128 = pl.matmul(
                            probabilities[base : base + HEAD_TILE, k128 : k128 + 128],
                            values[
                                task * max_context + k128 : task * max_context + k128 + 128,
                                col : col + pv_cols,
                            ],
                            out_dtype=pl.FP32,
                        )
                        numerators[meta_row : meta_row + HEAD_TILE, col : col + pv_cols] = acc128
                    elif part < max_context // 256:
                        k256 = part * 256
                        acc256 = pl.matmul(
                            probabilities[base : base + HEAD_TILE, k256 : k256 + 256],
                            values[
                                task * max_context + k256 : task * max_context + k256 + 256,
                                col : col + pv_cols,
                            ],
                            out_dtype=pl.FP32,
                        )
                        numerators[meta_row : meta_row + HEAD_TILE, col : col + pv_cols] = acc256
        for task in pl.spmd(tasks, name_hint="attention_normalize_adaptive"):
            max_length = pl.cast(pl.read(lengths, [0, 0]), pl.INDEX)
            for other in pl.range(rows):
                max_length = pl.max(max_length, pl.cast(pl.read(lengths, [other, 0]), pl.INDEX))
            token = task // LOCAL_KV_HEADS
            kv_head = task % LOCAL_KV_HEADS
            output_row = token * LOCAL_Q_HEADS + kv_head * GROUP
            previous_max = pl.row_max(pl.full([GROUP, 8], dtype=pl.FP32, value=-3.4028234663852886e38))
            num = pl.full([GROUP, HEAD_DIM], dtype=pl.FP32, value=0.0)
            denom = pl.row_sum(pl.full([GROUP, 8], dtype=pl.FP32, value=0.0))
            part_count = max_context // 128
            if active_chunks:
                if max_length < 256:
                    part_count = pl.min(max_context // 128, (max_length + 127) // 128)
                else:
                    part_count = pl.min(max_context // 256, (max_length + 255) // 256)
            for part in pl.range(part_count):
                if max_length < 256 or part < max_context // 256:
                    meta_row = (task * (max_context // 128) + part) * HEAD_TILE
                    current_max = maxima[meta_row : meta_row + GROUP, :]
                    correction = pl.exp(pl.sub(previous_max, current_max))
                    num = pl.add(
                        pl.row_expand_mul(num, correction), numerators[meta_row : meta_row + GROUP, :]
                    )
                    denom = pl.add(pl.mul(denom, correction), sums[meta_row : meta_row + GROUP, :])
                    previous_max = current_max
            length = pl.cast(pl.read(lengths, [token, 0]), pl.INDEX)
            if length > 0:
                out[output_row : output_row + GROUP, :] = pl.cast(
                    pl.row_expand_div(num, denom), pl.BF16, mode="rint"
                )
            else:
                out[output_row : output_row + GROUP, :] = pl.full([GROUP, HEAD_DIM], dtype=pl.BF16, value=0.0)
        return out

    return adaptive_attention
