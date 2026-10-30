# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""attention operators. Explicit PyPTO functions; no source generator at build time.

Migrated from the validated L1K2560 graph; tuning is audited separately.
"""

import pypto.language as pl


@pl.function(type=pl.FunctionType.InCore)
def split_query_kernel(
    x: pl.Tensor[[24, 3072], pl.BF16],
    q: pl.Tensor[[24, 6, 256], pl.BF16],
    gate: pl.Tensor[[24, 1536], pl.BF16],
    qf: pl.Tensor[[24, 1536], pl.BF16],
) -> tuple[pl.Tensor[[24, 6, 256], pl.BF16], pl.Tensor[[24, 1536], pl.BF16]]:
    pl.func_attr({"static_block_mapping": True})
    job = pl.tile.get_block_idx()
    token = job // 6
    head = job % 6
    query = pl.slice(x, [1, 256], [token, head * 512])
    gating = pl.slice(x, [1, 256], [token, head * 512 + 256])
    qf = pl.assemble(qf, query, [token, head * 256])
    gate = pl.assemble(gate, gating, [token, head * 256])
    return (q, gate)


@pl.function(type=pl.FunctionType.Inline)
def split_query(
    x: pl.Tensor[[24, 3072], pl.BF16],
    q: pl.Tensor[[24, 6, 256], pl.BF16],
    gate: pl.Tensor[[24, 1536], pl.BF16],
) -> tuple[pl.Tensor[[24, 6, 256], pl.BF16], pl.Tensor[[24, 1536], pl.BF16]]:
    qf = pl.reshape(q, [24, 6 * 256])
    with pl.spmd(24 * 6, allow_early_resolve=True):
        q, gate = split_query_kernel(x, q, gate, qf)
    return (q, gate)


@pl.function(type=pl.FunctionType.InCore)
def norm_rope_q_kernel(
    x: pl.Tensor[[24, 3072], pl.BF16],
    weight: pl.Tensor[[1, 256], pl.BF16],
    cos: pl.Tensor[[43008, 32], pl.BF16],
    sin: pl.Tensor[[43008, 32], pl.BF16],
    positions: pl.Tensor[[24], pl.INT64],
    out: pl.Tensor[[24, 6, 256], pl.BF16],
    gate: pl.Tensor[[24, 1536], pl.BF16],
    xf: pl.Tensor[[144, 512], pl.BF16],
    gf: pl.Tensor[[144, 256], pl.BF16],
    of: pl.Tensor[[144, 256], pl.BF16],
) -> tuple[pl.Tensor[[24, 6, 256], pl.BF16], pl.Tensor[[24, 1536], pl.BF16]]:
    pl.func_attr({"static_block_mapping": True})
    tokens = 24
    job = pl.tile.get_block_idx()
    token = job // ((6 + 7) // 8)
    head = job % ((6 + 7) // 8) * 8
    row = token * 6 + head
    valid = pl.min(8, 6 - head)
    value = pl.cast(pl.slice(xf, [8, 256], [row, 0], [valid, 256], pad_value=0, clamp=True), pl.FP32)
    gate_value = pl.slice(xf, [8, 256], [row, 256], [valid, 256], pad_value=0, clamp=True)
    gf = pl.assemble(gf, gate_value, [row, 0])
    variance = pl.add(pl.mul(pl.row_sum(pl.mul(value, value)), 1.0 / 256), 1e-06)
    inverse = pl.rsqrt(variance)
    inverse = pl.mul(inverse, pl.add(pl.mul(pl.mul(variance, pl.mul(inverse, inverse)), -0.5), 1.5))
    gamma = pl.cast(
        pl.cast(pl.add(pl.cast(pl.slice(weight, [1, 256], [0, 0]), pl.FP32), 1.0), pl.BF16, mode="rint"),
        pl.FP32,
    )
    normalized = pl.col_expand_mul(pl.row_expand_mul(value, inverse), gamma)
    position = pl.read(positions, [token])
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
        of, pl.tensor.set_validshape(pl.cast(rotated_first, pl.BF16, mode="rint"), valid, 32), [row, 0]
    )
    of = pl.assemble(
        of, pl.tensor.set_validshape(pl.cast(rotated_second, pl.BF16, mode="rint"), valid, 32), [row, 32]
    )
    return (out, gate)


@pl.function(type=pl.FunctionType.Inline)
def norm_rope_q(
    x: pl.Tensor[[24, 3072], pl.BF16],
    weight: pl.Tensor[[1, 256], pl.BF16],
    cos: pl.Tensor[[43008, 32], pl.BF16],
    sin: pl.Tensor[[43008, 32], pl.BF16],
    positions: pl.Tensor[[24], pl.INT64],
    out: pl.Tensor[[24, 6, 256], pl.BF16],
    gate: pl.Tensor[[24, 1536], pl.BF16],
) -> tuple[pl.Tensor[[24, 6, 256], pl.BF16], pl.Tensor[[24, 1536], pl.BF16]]:
    tokens = 24
    xf = pl.reshape(x, [tokens * 6, 512])
    gf = pl.reshape(gate, [tokens * 6, 256])
    of = pl.reshape(out, [tokens * 6, 256])
    with pl.spmd(tokens * ((6 + 7) // 8), allow_early_resolve=True):
        out, gate = norm_rope_q_kernel(x, weight, cos, sin, positions, out, gate, xf, gf, of)
    return (out, gate)


@pl.function(type=pl.FunctionType.InCore)
def norm_rope_k_append_kernel(
    x: pl.Tensor[[24, 1, 256], pl.BF16],
    weight: pl.Tensor[[1, 256], pl.BF16],
    cos: pl.Tensor[[43008, 32], pl.BF16],
    sin: pl.Tensor[[43008, 32], pl.BF16],
    positions: pl.Tensor[[24], pl.INT64],
    v: pl.Tensor[[24, 256], pl.BF16],
    key_cache: pl.Tensor[[294912, 256], pl.BF16],
    value_cache: pl.Tensor[[294912, 256], pl.BF16],
    slots: pl.Tensor[[24], pl.INT64],
    xf: pl.Tensor[[24, 256], pl.BF16],
    of: pl.Tensor[[294912, 256], pl.BF16],
) -> tuple[pl.Tensor[[294912, 256], pl.BF16], pl.Tensor[[294912, 256], pl.BF16]]:
    pl.func_attr({"static_block_mapping": True})
    tokens = 24
    job = pl.tile.get_block_idx()
    token = job // ((1 + 7) // 8)
    head = job % ((1 + 7) // 8) * 8
    row = token * 1 + head
    valid = pl.min(8, 1 - head)
    value = pl.cast(pl.slice(xf, [8, 256], [row, 0], [valid, 256], pad_value=0, clamp=True), pl.FP32)
    variance = pl.add(pl.mul(pl.row_sum(pl.mul(value, value)), 1.0 / 256), 1e-06)
    inverse = pl.rsqrt(variance)
    inverse = pl.mul(inverse, pl.add(pl.mul(pl.mul(variance, pl.mul(inverse, inverse)), -0.5), 1.5))
    gamma = pl.cast(
        pl.cast(pl.add(pl.cast(pl.slice(weight, [1, 256], [0, 0]), pl.FP32), 1.0), pl.BF16, mode="rint"),
        pl.FP32,
    )
    normalized = pl.col_expand_mul(pl.row_expand_mul(value, inverse), gamma)
    position = pl.read(positions, [token])
    ct = pl.cast(pl.slice(cos, [1, 32], [position, 0]), pl.FP32)
    st = pl.cast(pl.slice(sin, [1, 32], [position, 0]), pl.FP32)
    first = pl.slice(normalized, [8, 32], [0, 0])
    second = pl.slice(normalized, [8, 32], [0, 32])
    rotated_first = pl.sub(pl.col_expand_mul(first, ct), pl.col_expand_mul(second, st))
    rotated_second = pl.add(pl.col_expand_mul(second, ct), pl.col_expand_mul(first, st))
    slot = pl.read(slots, [token])
    if slot >= 0:
        of = pl.assemble(
            of,
            pl.tensor.set_validshape(pl.cast(normalized, pl.BF16, mode="rint"), valid, 256),
            [slot * 1 + head, 0],
        )
        of = pl.assemble(
            of,
            pl.tensor.set_validshape(pl.cast(rotated_first, pl.BF16, mode="rint"), valid, 32),
            [slot * 1 + head, 0],
        )
        of = pl.assemble(
            of,
            pl.tensor.set_validshape(pl.cast(rotated_second, pl.BF16, mode="rint"), valid, 32),
            [slot * 1 + head, 32],
        )
        vt = pl.slice(v, [1, 1 * 256], [token, 0])
        value_cache = pl.assemble(value_cache, vt, [slot, 0])
    return (key_cache, value_cache)


@pl.function(type=pl.FunctionType.Inline)
def norm_rope_k_append(
    x: pl.Tensor[[24, 1, 256], pl.BF16],
    weight: pl.Tensor[[1, 256], pl.BF16],
    cos: pl.Tensor[[43008, 32], pl.BF16],
    sin: pl.Tensor[[43008, 32], pl.BF16],
    positions: pl.Tensor[[24], pl.INT64],
    v: pl.Tensor[[24, 256], pl.BF16],
    key_cache: pl.Tensor[[294912, 256], pl.BF16],
    value_cache: pl.Tensor[[294912, 256], pl.BF16],
    slots: pl.Tensor[[24], pl.INT64],
) -> tuple[pl.Tensor[[294912, 256], pl.BF16], pl.Tensor[[294912, 256], pl.BF16]]:
    tokens = 24
    xf = pl.reshape(x, [tokens * 1, 256])
    of = pl.reshape(key_cache, [294912 * 1, 256])
    with pl.spmd(tokens * ((1 + 7) // 8), allow_early_resolve=True):
        key_cache, value_cache = norm_rope_k_append_kernel(
            x, weight, cos, sin, positions, v, key_cache, value_cache, slots, xf, of
        )
    return (key_cache, value_cache)


@pl.function(type=pl.FunctionType.Inline)
def reduce_gate(
    partials: pl.Tensor[[4032, 256], pl.FP32],
    gate: pl.Tensor[[24, 1536], pl.BF16],
    out: pl.Tensor[[24, 1536], pl.BF16],
) -> pl.Tensor[[24, 1536], pl.BF16]:
    tokens = 24
    rows = tokens * 6
    gf = pl.reshape(gate, [rows, 256])
    of = pl.reshape(out, [rows, 256])
    available = pl.system.available_aiv_count()
    width = pl.min(available, rows)
    for core in pl.spmd(width, name_hint="reduce_values_gate", allow_early_resolve=True):
        for row in pl.range(core, rows, pl.tile.get_block_num()):
            acc = pl.full([1, 256], dtype=pl.FP32, value=0.0)
            for chunk in pl.range(28):
                value = pl.slice(partials, [1, 256], [chunk * rows + row, 0])
                acc = pl.add(acc, value)
            rounded = pl.cast(pl.cast(acc, pl.BF16, mode="rint"), pl.FP32)
            g = pl.cast(pl.slice(gf, [1, 256], [row, 0]), pl.FP32)
            sigmoid = pl.recip(pl.add(pl.exp(pl.neg(g)), 1.0))
            sigmoid = pl.cast(pl.cast(sigmoid, pl.BF16, mode="rint"), pl.FP32)
            result = pl.cast(pl.mul(sigmoid, rounded), pl.BF16, mode="rint")
            of = pl.assemble(of, result, [row, 0])
    return out


@pl.function(type=pl.FunctionType.Inline)
def compute_scores(
    q: pl.Tensor[[24, 6, 256], pl.BF16],
    cache: pl.Tensor[[294912, 1, 256], pl.BF16],
    table: pl.Tensor[[24, 336], pl.INT32],
    lengths: pl.Tensor[[24], pl.INT32],
    out: pl.Tensor[[144, 43008], pl.FP32],
) -> pl.Tensor[[144, 43008], pl.FP32]:
    tokens = 24
    qf = pl.reshape(q, [tokens * 6, 256])
    cf = pl.reshape(cache, [294912, 1 * 256])
    jobs = tokens // 6 * 1 * 28
    available = pl.system.available_cluster_count()
    width = pl.min(available, jobs)
    for core in pl.spmd(width, name_hint="s2_scores", allow_early_resolve=True):
        token = core % (tokens // 6)
        local_core = core // (tokens // 6)
        token_width = (pl.tile.get_block_num() + tokens // 6 - 1 - token) // (tokens // 6)
        visible = pl.read(lengths, [token * 6 + 5])
        for head in pl.range(1):
            qrow = token * 6 * 6 + head * 6 * 6
            qt = pl.slice(qf, [48, 256], [qrow, 0], [6 * 6, 256], pad_value=0, clamp=True)
            qt = pl.tensor.set_validshape(qt, 48, 256)
            for chunk in pl.range(local_core, 28, token_width):
                for local_offset in pl.pipeline(0, 1536, 128, stage=2):
                    offset = chunk * 1536 + local_offset
                    physical = pl.read(table, [token * 6, offset // 128]) * 128 + offset % 128
                    valid = pl.max(0, pl.min(128, visible - offset))
                    kt = pl.slice(cf, [128, 256], [physical, head * 256], [valid, 256], pad_value=0)
                    kt = pl.tensor.set_validshape(kt, 128, 256)
                    product = pl.matmul(qt, kt, b_trans=True, out_dtype=pl.FP32)
                    out = pl.assemble(out, pl.tensor.set_validshape(product, 6 * 6, 128), [qrow, offset])
    return out


@pl.function(type=pl.FunctionType.Inline)
def softmax_scores(
    x: pl.Tensor[[144, 43008], pl.FP32],
    lengths: pl.Tensor[[24], pl.INT32],
    columns: pl.Tensor[[1, 1024], pl.FP32],
    out: pl.Tensor[[144, 43008], pl.BF16],
) -> pl.Tensor[[144, 43008], pl.BF16]:
    rows = 144
    available = pl.system.available_aiv_count()
    jobs = rows
    width = pl.min(available, jobs)
    for core in pl.spmd(width, name_hint="tiled_softmax", allow_early_resolve=True):
        for job in pl.range(core, jobs, pl.tile.get_block_num()):
            row = job
            visible = pl.read(lengths, [row // 6])
            column_tile = pl.tensor.set_validshape(pl.slice(columns, [1, 1024], [0, 0]), 1, 1024)
            valid = 1
            maximum_vector = pl.full([1, 1024], dtype=pl.FP32, value=-3.4028234663852886e38)
            for offset in pl.range(0, 43008, 1024):
                raw = pl.slice(x, [1, 1024], [row, offset], [valid, 1024], pad_value=0, clamp=True)
                raw = pl.tensor.set_validshape(raw, 1, 1024)
                scaled = pl.mul(raw, 0.0625)
                delta_i32 = pl.cast(offset - visible, pl.INT32)
                delta_fp32 = pl.cast(delta_i32, pl.FP32)
                distance = pl.add(column_tile, delta_fp32)
                mask = pl.minimum(pl.maximum(pl.mul(distance, -1.0), 0.0), 1.0)
                inverse_mask = pl.add(pl.mul(mask, -1.0), 1.0)
                score = pl.add(pl.mul(scaled, mask), pl.mul(inverse_mask, -3.4028234663852886e38))
                maximum_vector = pl.maximum(maximum_vector, score)
            maximum_padded = pl.reshape(
                pl.concat(maximum_vector, pl.full([1, 7168], dtype=pl.FP32, value=0.0)), [8, 1024]
            )
            maximum = pl.row_max(maximum_padded)
            ones = pl.full([1, 1024], dtype=pl.FP32, value=1.0)
            max_broadcast = pl.mul(ones, pl.read(maximum, [0, 0]))
            denominator_vector = pl.full([1, 1024], dtype=pl.FP32, value=0.0)
            for offset in pl.range(0, 43008, 1024):
                raw = pl.slice(x, [1, 1024], [row, offset], [valid, 1024], pad_value=0, clamp=True)
                raw = pl.tensor.set_validshape(raw, 1, 1024)
                scaled = pl.mul(raw, 0.0625)
                delta_i32 = pl.cast(offset - visible, pl.INT32)
                delta_fp32 = pl.cast(delta_i32, pl.FP32)
                distance = pl.add(column_tile, delta_fp32)
                mask = pl.minimum(pl.maximum(pl.mul(distance, -1.0), 0.0), 1.0)
                inverse_mask = pl.add(pl.mul(mask, -1.0), 1.0)
                score = pl.add(pl.mul(scaled, mask), pl.mul(inverse_mask, -3.4028234663852886e38))
                exponent = pl.exp(pl.sub(score, max_broadcast))
                denominator_vector = pl.add(denominator_vector, exponent)
            denominator_padded = pl.reshape(
                pl.concat(denominator_vector, pl.full([1, 7168], dtype=pl.FP32, value=0.0)), [8, 1024]
            )
            denominator = pl.row_sum(denominator_padded)
            denominator_broadcast = pl.mul(ones, pl.read(denominator, [0, 0]))
            for offset in pl.range(0, 43008, 1024):
                raw = pl.slice(x, [1, 1024], [row, offset], [valid, 1024], pad_value=0, clamp=True)
                raw = pl.tensor.set_validshape(raw, 1, 1024)
                scaled = pl.mul(raw, 0.0625)
                delta_i32 = pl.cast(offset - visible, pl.INT32)
                delta_fp32 = pl.cast(delta_i32, pl.FP32)
                distance = pl.add(column_tile, delta_fp32)
                mask = pl.minimum(pl.maximum(pl.mul(distance, -1.0), 0.0), 1.0)
                inverse_mask = pl.add(pl.mul(mask, -1.0), 1.0)
                score = pl.add(pl.mul(scaled, mask), pl.mul(inverse_mask, -3.4028234663852886e38))
                exponent = pl.exp(pl.sub(score, max_broadcast))
                probability = pl.div(exponent, denominator_broadcast)
                out = pl.assemble(
                    out,
                    pl.tensor.set_validshape(pl.cast(probability, pl.BF16, mode="rint"), valid, 1024),
                    [row, offset],
                )
    return out


@pl.function(type=pl.FunctionType.Inline)
def compute_partial_values(
    probability: pl.Tensor[[144, 43008], pl.BF16],
    cache: pl.Tensor[[294912, 1, 256], pl.BF16],
    table: pl.Tensor[[24, 336], pl.INT32],
    lengths: pl.Tensor[[24], pl.INT32],
    out: pl.Tensor[[4032, 256], pl.FP32],
) -> pl.Tensor[[4032, 256], pl.FP32]:
    tokens = 24
    cf = pl.reshape(cache, [294912, 1 * 256])
    jobs = tokens // 6 * 1 * 28
    available = pl.system.available_cluster_count()
    width = pl.min(available, jobs)
    for core in pl.spmd(width, name_hint="s2_partial_values", allow_early_resolve=True):
        token = core % (tokens // 6)
        local_core = core // (tokens // 6)
        token_width = (pl.tile.get_block_num() + tokens // 6 - 1 - token) // (tokens // 6)
        visible = pl.read(lengths, [token * 6 + 5])
        for job in pl.range(local_core, 1 * 28, token_width):
            head = job // 28
            chunk = job % 28
            qrow = token * 6 * 6 + head * 6 * 6
            acc = pl.create_tensor([48, 256], pl.FP32)
            for local_offset in pl.pipeline(0, 1536, 128, stage=2):
                offset = chunk * 1536 + local_offset
                physical = pl.read(table, [token * 6, offset // 128]) * 128 + offset % 128
                valid = pl.max(0, pl.min(128, visible - offset))
                pt = pl.slice(probability, [48, 128], [qrow, offset], [6 * 6, 128], pad_value=0, clamp=True)
                vt = pl.slice(cf, [128, 256], [physical, head * 256], [valid, 256], pad_value=0)
                pt = pl.tensor.set_validshape(pt, 48, 128)
                vt = pl.tensor.set_validshape(vt, 128, 256)
                acc = pl.matmul_acc(acc, pt, vt, init_cond=local_offset == 0)
            out = pl.assemble(out, pl.tensor.set_validshape(acc, 6 * 6, 256), [chunk * tokens * 6 + qrow, 0])
    return out


@pl.function(type=pl.FunctionType.Inline)
def reduce_values(
    partials: pl.Tensor[[4032, 256], pl.FP32], out: pl.Tensor[[144, 256], pl.BF16]
) -> pl.Tensor[[144, 256], pl.BF16]:
    rows = 144
    available = pl.system.available_aiv_count()
    width = pl.min(available, rows)
    for core in pl.spmd(width, name_hint="s2_reduce_values", allow_early_resolve=True):
        for row in pl.range(core, rows, pl.tile.get_block_num()):
            acc = pl.full([1, 256], dtype=pl.FP32, value=0.0)
            for chunk in pl.range(28):
                value = pl.slice(partials, [1, 256], [chunk * rows + row, 0])
                acc = pl.add(acc, value)
            out = pl.assemble(out, pl.cast(acc, pl.BF16, mode="rint"), [row, 0])
    return out
