# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Same-order row-cached softmax, expressed entirely with existing tile DSL.

The reduction order matches the production 1024-column vector accumulator.
Input is loaded once, retained in UB, and output conversion/store is pipelined.
This is a migration acceptance kernel, not a claim of measured performance.
"""

import pypto.language as pl
from pypto import backend, ir


def make_row_cached_softmax(rows=144, sequence=43008, heads=6):
    if sequence % 2048:
        raise ValueError("sequence must contain complete pairs of 1024-column tiles")
    capacity = backend.get_backend_instance(backend.BackendType.Ascend910B).get_mem_size(ir.MemorySpace.Vec)
    if sequence * 4 + 16384 > capacity:
        raise ValueError("row cache plus working buffers exceeds backend Vec capacity")

    cache_slots = sequence // 1024
    stat_columns = (
        backend.get_backend_instance(backend.BackendType.Ascend910B)
        .get_handler()
        .get_vec_tile_inner_alignment_bytes()
        // 4
    )

    @pl.jit.incore
    def softmax_kernel(
        output: pl.Out[pl.Tensor[[rows, sequence], pl.BF16]],
        lengths: pl.Tensor[[(rows + heads - 1) // heads], pl.INT32],
        columns: pl.Tensor[[1, 1024], pl.FP32],
        scores: pl.Tensor[[rows, sequence], pl.FP32],
    ):
        core = pl.tile.get_block_idx()
        for row in pl.range(core, rows, pl.tile.get_block_num()):
            visible = pl.read(lengths, [row // heads])
            active = sequence // 1024
            if visible > 1:
                active = pl.min((visible + 1023) // 1024, sequence // 1024)
            maximum = pl.tile.full([1, 1024], pl.FP32, -3.4028234663852886e38)
            bulk = active - 1
            if visible <= 1:
                bulk = 0
            # Declared cache views cross loops; drain this core before slot reuse.
            pl.system.bar_all()
            for prefetch_index in pl.pipeline(bulk, stage=2):
                bulk_offset = prefetch_index * 1024
                bulk_raw: pl.Tile[
                    [1, 1024], pl.FP32, pl.MemRef("row_cache", slots=cache_slots)[prefetch_index], pl.Mem.Vec
                ] = pl.load(scores, [row, bulk_offset], [1, 1024])
            # Make the cross-loop MTE2 -> V handoff explicit.
            pl.system.bar_all()
            for compute_index in pl.range(bulk):
                compute_raw: pl.Tile[
                    [1, 1024], pl.FP32, pl.MemRef("row_cache", slots=cache_slots)[compute_index], pl.Mem.Vec
                ] = pl.create_tile([1, 1024], pl.FP32, target_memory=pl.Mem.Vec)
                bulk_scaled: pl.Tile[
                    [1, 1024], pl.FP32, pl.MemRef("row_cache", slots=cache_slots)[compute_index], pl.Mem.Vec
                ] = pl.mul(compute_raw, 0.0625)
                maximum = pl.maximum(maximum, bulk_scaled)
            for load_index in pl.range(bulk, active):
                offset = load_index * 1024
                raw: pl.Tile[
                    [1, 1024], pl.FP32, pl.MemRef("row_cache", slots=cache_slots)[load_index], pl.Mem.Vec
                ] = pl.load(scores, [row, offset], [1, 1024])
                scaled: pl.Tile[
                    [1, 1024], pl.FP32, pl.MemRef("row_cache", slots=cache_slots)[load_index], pl.Mem.Vec
                ] = pl.mul(raw, 0.0625)
                if offset + 1024 >= visible:
                    column = pl.load(columns, [0, 0], [1, 1024])
                    delta = pl.cast(pl.cast(offset - visible, pl.INT32), pl.FP32)
                    distance = pl.add(column, delta)
                    mask = pl.minimum(pl.maximum(pl.mul(distance, -1.0), 0.0), 1.0)
                    masked: pl.Tile[
                        [1, 1024], pl.FP32, pl.MemRef("row_cache", slots=cache_slots)[load_index], pl.Mem.Vec
                    ] = pl.mul(scaled, mask)
                    penalty = pl.mul(pl.add(pl.mul(mask, -1.0), 1.0), -3.4028234663852886e38)
                    score: pl.Tile[
                        [1, 1024], pl.FP32, pl.MemRef("row_cache", slots=cache_slots)[load_index], pl.Mem.Vec
                    ] = pl.add(masked, penalty)
                else:
                    score = scaled
                maximum = pl.maximum(maximum, score)
            scratch = pl.create_tile([1, 1024], pl.FP32, target_memory=pl.Mem.Vec)
            max_stat = pl.row_max(maximum, scratch)
            max_value = pl.tile.read(max_stat, [0, 0])
            max_broadcast = pl.tile.expands(
                pl.create_tile([1, stat_columns], pl.FP32, target_memory=pl.Mem.Vec), max_value
            )
            denominator = pl.tile.full([1, 1024], pl.FP32, 0.0)
            for exp_index in pl.range(active):
                offset = exp_index * 1024
                cached_score: pl.Tile[
                    [1, 1024], pl.FP32, pl.MemRef("row_cache", slots=cache_slots)[exp_index], pl.Mem.Vec
                ] = pl.create_tile([1, 1024], pl.FP32, target_memory=pl.Mem.Vec)
                exponent: pl.Tile[
                    [1, 1024], pl.FP32, pl.MemRef("row_cache", slots=cache_slots)[exp_index], pl.Mem.Vec
                ] = pl.tile.row_expand_expdif(cached_score, max_broadcast, packed=True)
                denominator = pl.add(denominator, exponent)
            den_stat = pl.row_sum(denominator, scratch)
            den_value = pl.tile.read(den_stat, [0, 0])
            den_broadcast = pl.tile.expands(
                pl.create_tile([1, stat_columns], pl.FP32, target_memory=pl.Mem.Vec), den_value
            )
            for pair_index in pl.range(cache_slots // 2):
                # Output slot 0: wait only when reusing this slot.
                store_index = pair_index * 2 + 0
                offset = store_index * 1024
                if store_index < active:
                    cached_exponent: pl.Tile[
                        [1, 1024], pl.FP32, pl.MemRef("row_cache", slots=cache_slots)[store_index], pl.Mem.Vec
                    ] = pl.create_tile([1, 1024], pl.FP32, target_memory=pl.Mem.Vec)
                    normalized: pl.Tile[
                        [1, 1024], pl.FP32, pl.MemRef("row_cache", slots=cache_slots)[store_index], pl.Mem.Vec
                    ] = pl.tile.row_expand_div(cached_exponent, den_broadcast, packed=True)
                    result: pl.Tile[[1, 1024], pl.BF16, pl.MemRef("output", slots=2)[0], pl.Mem.Vec] = (
                        pl.cast(normalized, pl.BF16, mode="rint")
                    )
                else:
                    result: pl.Tile[[1, 1024], pl.BF16, pl.MemRef("output", slots=2)[0], pl.Mem.Vec] = (
                        pl.tile.full([1, 1024], pl.BF16, 0.0)
                    )
                pl.store(result, [row, offset], output)
                # Output slot 1: wait only when reusing this slot.
                store_index_odd = pair_index * 2 + 1
                offset_odd = store_index_odd * 1024
                if store_index_odd < active:
                    cached_exponent_odd: pl.Tile[
                        [1, 1024],
                        pl.FP32,
                        pl.MemRef("row_cache", slots=cache_slots)[store_index_odd],
                        pl.Mem.Vec,
                    ] = pl.create_tile([1, 1024], pl.FP32, target_memory=pl.Mem.Vec)
                    normalized_odd: pl.Tile[
                        [1, 1024],
                        pl.FP32,
                        pl.MemRef("row_cache", slots=cache_slots)[store_index_odd],
                        pl.Mem.Vec,
                    ] = pl.tile.row_expand_div(cached_exponent_odd, den_broadcast, packed=True)
                    result_odd: pl.Tile[[1, 1024], pl.BF16, pl.MemRef("output", slots=2)[1], pl.Mem.Vec] = (
                        pl.cast(normalized_odd, pl.BF16, mode="rint")
                    )
                else:
                    result_odd: pl.Tile[[1, 1024], pl.BF16, pl.MemRef("output", slots=2)[1], pl.Mem.Vec] = (
                        pl.tile.full([1, 1024], pl.BF16, 0.0)
                    )
                pl.store(result_odd, [row, offset_odd], output)
        return output

    @pl.jit
    def row_cached_softmax(
        output: pl.Out[pl.Tensor[[rows, sequence], pl.BF16]],
        lengths: pl.Tensor[[(rows + heads - 1) // heads], pl.INT32],
        columns: pl.Tensor[[1, 1024], pl.FP32],
        scores: pl.Tensor[[rows, sequence], pl.FP32],
    ):
        available = pl.system.available_aiv_count()
        width = pl.min(available, rows)
        with pl.spmd(width, allow_early_resolve=True):
            softmax_kernel(output, lengths, columns, scores)
        return output

    return row_cached_softmax
