# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Readable experimental FD mathematics, normal PyPTO compilation only.

`partition_ids[request, block]` groups512-column stacks into ordered independent
FD partitions. It is a shape/planning input, never a value-dependent correction.
All physical core limits are queried. This module is not a serving default.
"""
import pypto.language as pl
from qwen36.sequence_config import CACHE_ROWS, TABLE_COLS, SEQUENCE

BLOCK = 512
BLOCKS = (SEQUENCE + BLOCK - 1) // BLOCK
STAT_WIDTH = BLOCKS * 3
PARTIAL_ROWS = BLOCKS * 144
REDUCE_STAGES = BLOCKS.bit_length()


@pl.function(type=pl.FunctionType.InCore)
def fd_softmax_kernel(scores: pl.Tensor[[144, SEQUENCE], pl.FP32],
                      lengths: pl.Tensor[[24], pl.INT32],
                      columns: pl.Tensor[[1, 1024], pl.FP32],
                      partition_ids: pl.Tensor[[4, BLOCKS], pl.INT32],
                      probability: pl.Out[pl.Tensor[[144, SEQUENCE], pl.BF16]],
                      statistics: pl.Out[pl.Tensor[[144, STAT_WIDTH], pl.FP32]]):
    for row in pl.range(pl.tile.get_block_idx(), 144, pl.tile.get_block_num()):
        request = row // 36
        visible = pl.read(lengths, [row // 6])
        request_visible = pl.read(lengths, [request * 6])
        for query in pl.range(1, 6):
            request_visible = pl.max(request_visible, pl.read(lengths, [request * 6 + query]))
        count = (request_visible + BLOCK - 1) // BLOCK
        running_max = pl.tile.full([1, 8], pl.FP32, -3.4028234663852886e38)
        running_den = pl.tile.full([1, 8], pl.FP32, 0.0)
        for block in pl.range(BLOCKS):
            if block < count:
                part = pl.cast(pl.read(partition_ids, [request, block]), pl.INT64)
                reset = block == 0
                if block > 0:
                    reset = part != pl.read(partition_ids, [request, block - 1])
                if reset:
                    running_max = pl.tile.full([1, 8], pl.FP32, -3.4028234663852886e38)
                    running_den = pl.tile.full([1, 8], pl.FP32, 0.0)
                raw = pl.load(scores, [row, block * BLOCK], [1, BLOCK])
                scaled = pl.mul(raw, 0.0625)
                column = pl.load(columns, [0, 0], [1, BLOCK])
                distance = pl.add(column, pl.cast(pl.cast(block * BLOCK - visible, pl.INT32), pl.FP32))
                valid = pl.tile.cmps(distance, 0.0, cmp_type=2)
                select_scratch = pl.create_tile([1, BLOCK], pl.FP32, target_memory=pl.Mem.Vec)
                masked = pl.tile.sels(valid, scaled, select_scratch, -3.4028234663852886e38)
                max_scratch = pl.create_tile([1, BLOCK], pl.FP32, target_memory=pl.Mem.Vec)
                local_max = pl.tile.row_max(masked, max_scratch)
                broadcast_max = pl.tile.expands(pl.create_tile([1, 8], pl.FP32, target_memory=pl.Mem.Vec), pl.tile.read(local_max, [0, 0]))
                updated_max = pl.maximum(running_max, broadcast_max)
                alpha = pl.exp(pl.sub(running_max, updated_max))
                exponent = pl.tile.row_expand_expdif(masked, updated_max, packed=True)
                # A completely masked query/partition must contribute zero.
                exponent = pl.tile.sels(valid, exponent, select_scratch, 0.0)
                blocks8 = pl.tile.reshape(exponent, [64, 8])
                scratch8 = pl.create_tile([64, 8], pl.FP32, target_memory=pl.Mem.Vec)
                sums8 = pl.tile.row_sum(blocks8, scratch8)
                blocks64 = pl.tile.reshape(sums8, [8, 8])
                scratch64 = pl.create_tile([8, 8], pl.FP32, target_memory=pl.Mem.Vec)
                sums64 = pl.tile.row_sum(blocks64, scratch64)
                sum_row = pl.tile.reshape(sums64, [1, 8])
                sum_scratch = pl.create_tile([1, 8], pl.FP32, target_memory=pl.Mem.Vec)
                summed = pl.tile.row_sum(sum_row, sum_scratch)
                local_den = pl.tile.read(summed, [0, 0])
                actual_columns = pl.min(BLOCK, request_visible - block * BLOCK)
                # CANN CalcLocalRowSum:512/256 use their block trees;
                # other tails accumulate consecutive64-element groups.
                if actual_columns != 512:
                    if actual_columns != 256:
                        local_den = pl.tile.read(sums64, [0, 0])
                        for group in pl.range(1, 8):
                            if group * 64 < actual_columns:
                                local_den = local_den + pl.tile.read(sums64, [group, 0])
                local_den_tile = pl.tile.expands(pl.create_tile([1, 8], pl.FP32, target_memory=pl.Mem.Vec), local_den)
                running_den = pl.add(pl.mul(running_den, alpha), local_den_tile)
                running_max = updated_max
                pl.store(pl.cast(exponent, pl.BF16, mode='rint'), [row, block * BLOCK], probability)
                pl.store(pl.tile.set_validshape(alpha, 1, 1), [row, block], statistics)
                pl.store(pl.tile.set_validshape(running_den, 1, 1), [row, BLOCKS + block], statistics)
                pl.store(pl.tile.set_validshape(running_max, 1, 1), [row, 2 * BLOCKS + block], statistics)
            else:
                pl.store(pl.tile.full([1, BLOCK], pl.BF16, 0.0), [row, block * BLOCK], probability)
    return probability, statistics


@pl.function(type=pl.FunctionType.Inline)
def fd_softmax(scores: pl.Tensor[[144, SEQUENCE], pl.FP32], lengths: pl.Tensor[[24], pl.INT32],
               columns: pl.Tensor[[1, 1024], pl.FP32], partition_ids: pl.Tensor[[4, BLOCKS], pl.INT32],
               probability: pl.Out[pl.Tensor[[144, SEQUENCE], pl.BF16]],
               statistics: pl.Out[pl.Tensor[[144, STAT_WIDTH], pl.FP32]]):
    width = pl.min(pl.system.available_aiv_count(), 144)
    with pl.spmd(width):
        probability, statistics = fd_softmax_kernel(scores, lengths, columns, partition_ids, probability, statistics)
    return probability, statistics


@pl.function(type=pl.FunctionType.Inline)
def fd_partial_values(probability: pl.Tensor[[144, SEQUENCE], pl.BF16],
                      values: pl.Tensor[[CACHE_ROWS, 1, 256], pl.BF16],
                      table: pl.Tensor[[24, TABLE_COLS], pl.INT32], lengths: pl.Tensor[[24], pl.INT32],
                      partial: pl.Out[pl.Tensor[[PARTIAL_ROWS, 256], pl.FP32]]):
    flat = pl.reshape(values, [CACHE_ROWS, 256])
    width = pl.min(pl.system.available_cluster_count(), 4 * BLOCKS)
    for core in pl.spmd(width):
        pl.set_cache_policy(flat, pl.CachePolicy.BYPASS)
        request = core % 4
        local = core // 4
        stride = (pl.tile.get_block_num() + 3 - request) // 4
        request_visible = pl.read(lengths, [request * 6])
        for query in pl.range(1, 6):
            request_visible = pl.max(request_visible, pl.read(lengths, [request * 6 + query]))
        count = (request_visible + BLOCK - 1) // BLOCK
        for block in pl.range(local, count, stride):
            acc = pl.create_tensor([48, 256], pl.FP32)
            for offset in pl.pipeline(0, pl.min(BLOCK, request_visible - block * BLOCK), 128, stage=2):
                logical = block * BLOCK + offset
                physical = pl.cast(pl.read(table, [request * 6, logical // 128]), pl.INT64) * 128
                valid = pl.min(128, request_visible - logical)
                pt = pl.slice(probability, [48, 128], [request * 36, logical], [36, valid], pad_value=0, clamp=True)
                vt = pl.slice(flat, [128, 256], [physical, 0], [valid, 256], pad_value=0)
                acc = pl.matmul_acc(acc, pt, vt, init_cond=offset == 0)
            partial = pl.assemble(partial, pl.tensor.set_validshape(acc, 36, 256), [block * 144 + request * 36, 0])
    return partial


@pl.function(type=pl.FunctionType.InCore)
def fd_combine_kernel(partial: pl.Tensor[[PARTIAL_ROWS, 256], pl.FP32],
                      statistics: pl.Tensor[[144, STAT_WIDTH], pl.FP32],
                      partition_ids: pl.Tensor[[4, BLOCKS], pl.INT32],
                      lengths: pl.Tensor[[24], pl.INT32], gate: pl.Tensor[[24, 1536], pl.BF16],
                      normalized_parts: pl.Out[pl.Tensor[[PARTIAL_ROWS, 256], pl.FP32]],
                      part_lse: pl.Out[pl.Tensor[[PARTIAL_ROWS, 8], pl.FP32]],
                      raw: pl.Out[pl.Tensor[[24, 1536], pl.BF16]],
                      out: pl.Out[pl.Tensor[[24, 1536], pl.BF16]],
                      gf: pl.Tensor[[144, 256], pl.BF16],
                      rf: pl.Tensor[[144, 256], pl.BF16],
                      of: pl.Tensor[[144, 256], pl.BF16]):
    # GM alias views are created by the Inline caller, not loaded as whole
    # Vec tiles by an InCore reshape. Only the accessed row is loaded/stored.
    for row in pl.range(pl.tile.get_block_idx(), 144, pl.tile.get_block_num()):
        request = row // 36
        request_visible = pl.read(lengths, [request * 6])
        for query in pl.range(1, 6):
            request_visible = pl.max(request_visible, pl.read(lengths, [request * 6 + query]))
        count = (request_visible + BLOCK - 1) // BLOCK
        visible = pl.read(lengths, [row // 6])
        # The planner sets inactive/padded query lengths to0. Never read a
        # negative partition index or stale partials for these logical rows.
        if count > 0 and visible > 0:
            parts = pl.cast(pl.read(partition_ids, [request, count - 1]), pl.INT64) + 1
            lse_row = pl.full([BLOCKS, 8], dtype=pl.FP32, value=-3.4028234663852886e38)
            acc = pl.full([1, 256], dtype=pl.FP32, value=0.0)
            for block in pl.range(count):
                part = pl.cast(pl.read(partition_ids, [request, block]), pl.INT64)
                first = block == 0
                if block > 0:
                    first = part != pl.read(partition_ids, [request, block - 1])
                pv = pl.slice(partial, [1, 256], [block * 144 + row, 0])
                if first:
                    acc = pv
                else:
                    alpha = pl.read(statistics, [row, block])
                    acc = pl.add(pl.mul(acc, alpha), pv)
                last = block + 1 == count
                if block + 1 < count:
                    last = part != pl.read(partition_ids, [request, block + 1])
                if last:
                    den = pl.read(statistics, [row, BLOCKS + block])
                    maximum = pl.read(statistics, [row, 2 * BLOCKS + block])
                    if den > 0.0:
                        divisor = pl.mul(pl.full([1, 256], dtype=pl.FP32, value=1.0), den)
                        normal = pl.div(acc, divisor)
                        den_tile = pl.mul(pl.full([1, 8], dtype=pl.FP32, value=1.0), den)
                        lse = pl.add(pl.log(den_tile), maximum)
                    else:
                        normal = pl.full([1, 256], dtype=pl.FP32, value=0.0)
                        lse = pl.full([1, 8], dtype=pl.FP32, value=-3.4028234663852886e38)
                    normalized_parts = pl.assemble(normalized_parts, normal, [part * 144 + row, 0])
                    lse_row = pl.assemble(lse_row, lse, [part, 0])
            part_lse = pl.assemble(part_lse, lse_row, [row * BLOCKS, 0])
            max_lse = pl.read(pl.col_max(lse_row), [0, 0])
            lse_exp = pl.exp(pl.sub(lse_row, max_lse))
            # Native CombineScale reduces across the partition dimension (RA).
            # CANN9.1 BinaryReduceByFirstAxis: fold the non-power-of-two
            # tail, then add corresponding rows of the two halves at each level.
            # The logical capacity is fixed; active rows depend only on FD shape.
            aligned = ((parts + 7) // 8) * 8
            split = 1
            for stage in pl.range(REDUCE_STAGES):
                if split * 2 <= aligned:
                    split = split * 2
            remain = aligned - split
            tree = pl.full([BLOCKS, 8], dtype=pl.FP32, value=0.0)
            for index in pl.range(split):
                term = pl.slice(lse_exp, [1, 8], [index, 0])
                if index < remain:
                    term = pl.add(term, pl.slice(lse_exp, [1, 8], [split + index, 0]))
                tree = pl.assemble(tree, term, [index, 0])
            for stage in pl.range(REDUCE_STAGES):
                if split > 1:
                    half = split // 2
                    for index in pl.range(half):
                        left = pl.slice(tree, [1, 8], [index, 0])
                        right = pl.slice(tree, [1, 8], [half + index, 0])
                        tree = pl.assemble(tree, pl.add(left, right), [index, 0])
                    split = half
            sum_exp = pl.slice(tree, [1, 8], [0, 0])
            global_lse = pl.add(pl.log(sum_exp), max_lse)
            total = pl.full([1, 256], dtype=pl.FP32, value=0.0)
            for part in pl.range(parts):
                local_lse = pl.read(lse_row, [part, 0])
                weight = pl.exp(pl.sub(pl.mul(pl.full([1, 8], dtype=pl.FP32, value=1.0), local_lse), global_lse))
                scalar_weight = pl.read(weight, [0, 0])
                normal = pl.slice(normalized_parts, [1, 256], [part * 144 + row, 0])
                total = pl.add(total, pl.mul(normal, scalar_weight))
            rounded = pl.cast(total, pl.BF16, mode='rint')
            rf = pl.assemble(rf, rounded, [row, 0])
            g = pl.cast(pl.slice(gf, [1, 256], [row, 0]), pl.FP32)
            sigmoid = pl.cast(pl.recip(pl.add(pl.exp(pl.neg(g)), 1.0)), pl.BF16, mode='rint')
            of = pl.assemble(of, pl.cast(pl.mul(pl.cast(rounded, pl.FP32), pl.cast(sigmoid, pl.FP32)), pl.BF16, mode='rint'), [row, 0])
        else:
            zero = pl.full([1, 256], dtype=pl.BF16, value=0.0)
            rf = pl.assemble(rf, zero, [row, 0])
            of = pl.assemble(of, zero, [row, 0])
    return normalized_parts, part_lse, raw, out


@pl.function(type=pl.FunctionType.Inline)
def fd_combine(partial: pl.Tensor[[PARTIAL_ROWS, 256], pl.FP32],
               statistics: pl.Tensor[[144, STAT_WIDTH], pl.FP32],
               partition_ids: pl.Tensor[[4, BLOCKS], pl.INT32],
               lengths: pl.Tensor[[24], pl.INT32], gate: pl.Tensor[[24, 1536], pl.BF16],
               normalized_parts: pl.Out[pl.Tensor[[PARTIAL_ROWS, 256], pl.FP32]],
               part_lse: pl.Out[pl.Tensor[[PARTIAL_ROWS, 8], pl.FP32]],
               raw: pl.Out[pl.Tensor[[24, 1536], pl.BF16]], out: pl.Out[pl.Tensor[[24, 1536], pl.BF16]]):
    gf = pl.reshape(gate, [144, 256])
    rf = pl.reshape(raw, [144, 256])
    of = pl.reshape(out, [144, 256])
    width = pl.min(pl.system.available_aiv_count(), 144)
    with pl.spmd(width):
        normalized_parts, part_lse, raw, out = fd_combine_kernel(partial, statistics, partition_ids, lengths, gate, normalized_parts, part_lse, raw, out, gf, rf, of)
    return normalized_parts, part_lse, raw, out
