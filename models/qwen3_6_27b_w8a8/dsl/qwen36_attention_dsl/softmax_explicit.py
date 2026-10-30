# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DSL softmax with complete, explicit core-local pipeline synchronization.

PTOAS skips automatic synchronization in functions containing explicit flags.
Every vector dependency, scalar read and asynchronous store is ordered here.
A complete FP32 row stays in declared UB slots; two output slots overlap stores.
"""

import pypto.language as pl
from pypto import backend, ir


def make_explicit_row_cached_softmax(rows=144, sequence=43008, heads=6):
    device = backend.get_backend_instance(backend.BackendType.Ascend910B)
    capacity = device.get_mem_size(ir.MemorySpace.Vec)
    if sequence % 2048 or sequence * 4 + 16384 > capacity:
        raise ValueError("row cache and working set do not fit this backend")
    slots = sequence // 1024
    stat_columns = device.get_handler().get_vec_tile_inner_alignment_bytes() // 4

    @pl.jit.incore
    def softmax_kernel(
        output: pl.Out[pl.Tensor[[rows, sequence], pl.BF16]],
        lengths: pl.Tensor[[(rows + heads - 1) // heads], pl.INT32],
        columns: pl.Tensor[[1, 1024], pl.FP32],
        scores: pl.Tensor[[rows, sequence], pl.FP32],
    ):
        core = pl.tile.get_block_idx()
        # Prime the empty output slots. Each store returns its own token.
        pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
        for row in pl.range(core, rows, pl.tile.get_block_num()):
            visible = pl.read(lengths, [row // heads])
            active = slots
            if visible > 1:
                active = pl.min((visible + 1023) // 1024, slots)
            maximum = pl.tile.full([1, 1024], pl.FP32, -3.4028234663852886e38)
            pl.system.bar_v()
            first_raw: pl.Tile[[1, 1024], pl.FP32, pl.MemRef("row_cache", slots=slots)[0], pl.Mem.Vec] = (
                pl.load(scores, [row, 0], [1, 1024])
            )
            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            for index in pl.range(active):
                next_index = index + 1
                if next_index < active:
                    next_raw: pl.Tile[
                        [1, 1024], pl.FP32, pl.MemRef("row_cache", slots=slots)[next_index], pl.Mem.Vec
                    ] = pl.load(scores, [row, next_index * 1024], [1, 1024])
                    if index % 2 == 0:
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=1)
                    else:
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                if index % 2 == 0:
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                else:
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=1)
                raw: pl.Tile[[1, 1024], pl.FP32, pl.MemRef("row_cache", slots=slots)[index], pl.Mem.Vec] = (
                    pl.create_tile([1, 1024], pl.FP32, target_memory=pl.Mem.Vec)
                )
                scaled: pl.Tile[
                    [1, 1024], pl.FP32, pl.MemRef("row_cache", slots=slots)[index], pl.Mem.Vec
                ] = pl.mul(raw, 0.0625)
                pl.system.bar_v()
                if index * 1024 + 1024 >= visible:
                    # Scratch may contain earlier vector work; order its next DMA.
                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=2)
                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=2)
                    column = pl.load(columns, [0, 0], [1, 1024])
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=2)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=2)
                    delta = pl.cast(pl.cast(index * 1024 - visible, pl.INT32), pl.FP32)
                    distance = pl.add(column, delta)
                    pl.system.bar_v()
                    negated = pl.mul(distance, -1.0)
                    pl.system.bar_v()
                    lower = pl.maximum(negated, 0.0)
                    pl.system.bar_v()
                    mask = pl.minimum(lower, 1.0)
                    pl.system.bar_v()
                    masked: pl.Tile[
                        [1, 1024], pl.FP32, pl.MemRef("row_cache", slots=slots)[index], pl.Mem.Vec
                    ] = pl.mul(scaled, mask)
                    pl.system.bar_v()
                    negative_mask = pl.mul(mask, -1.0)
                    pl.system.bar_v()
                    inverted = pl.add(negative_mask, 1.0)
                    pl.system.bar_v()
                    penalty = pl.mul(inverted, -3.4028234663852886e38)
                    pl.system.bar_v()
                    score: pl.Tile[
                        [1, 1024], pl.FP32, pl.MemRef("row_cache", slots=slots)[index], pl.Mem.Vec
                    ] = pl.add(masked, penalty)
                    pl.system.bar_v()
                else:
                    score = scaled
                maximum = pl.maximum(maximum, score)
                pl.system.bar_v()
            scratch = pl.create_tile([1, 1024], pl.FP32, target_memory=pl.Mem.Vec)
            max_stat = pl.row_max(maximum, scratch)
            pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.S, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.S, event_id=0)
            max_value = pl.tile.read(max_stat, [0, 0])
            max_block = pl.tile.expands(
                pl.create_tile([1, stat_columns], pl.FP32, target_memory=pl.Mem.Vec), max_value
            )
            denominator = pl.tile.full([1, 1024], pl.FP32, 0.0)
            pl.system.bar_v()
            for exp_index in pl.range(active):
                cached: pl.Tile[
                    [1, 1024], pl.FP32, pl.MemRef("row_cache", slots=slots)[exp_index], pl.Mem.Vec
                ] = pl.create_tile([1, 1024], pl.FP32, target_memory=pl.Mem.Vec)
                exponent: pl.Tile[
                    [1, 1024], pl.FP32, pl.MemRef("row_cache", slots=slots)[exp_index], pl.Mem.Vec
                ] = pl.tile.row_expand_expdif(cached, max_block, packed=True)
                pl.system.bar_v()
                denominator = pl.add(denominator, exponent)
                pl.system.bar_v()
            den_stat = pl.row_sum(denominator, scratch)
            pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.S, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.S, event_id=0)
            den_value = pl.tile.read(den_stat, [0, 0])
            den_block = pl.tile.expands(
                pl.create_tile([1, stat_columns], pl.FP32, target_memory=pl.Mem.Vec), den_value
            )
            pl.system.bar_v()
            for pair in pl.range(slots // 2):
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=0)
                out_index = pair * 2
                if out_index < active:
                    source: pl.Tile[
                        [1, 1024], pl.FP32, pl.MemRef("row_cache", slots=slots)[out_index], pl.Mem.Vec
                    ] = pl.create_tile([1, 1024], pl.FP32, target_memory=pl.Mem.Vec)
                    divided: pl.Tile[
                        [1, 1024], pl.FP32, pl.MemRef("row_cache", slots=slots)[out_index], pl.Mem.Vec
                    ] = pl.tile.row_expand_div(source, den_block, packed=True)
                    pl.system.bar_v()
                    result: pl.Tile[[1, 1024], pl.BF16, pl.MemRef("output", slots=2)[0], pl.Mem.Vec] = (
                        pl.cast(divided, pl.BF16, mode="rint")
                    )
                else:
                    result: pl.Tile[[1, 1024], pl.BF16, pl.MemRef("output", slots=2)[0], pl.Mem.Vec] = (
                        pl.tile.full([1, 1024], pl.BF16, 0.0)
                    )
                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
                pl.store(result, [row, out_index * 1024], output)
                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
                out_index_odd = pair * 2 + 1
                if out_index_odd < active:
                    source_odd: pl.Tile[
                        [1, 1024], pl.FP32, pl.MemRef("row_cache", slots=slots)[out_index_odd], pl.Mem.Vec
                    ] = pl.create_tile([1, 1024], pl.FP32, target_memory=pl.Mem.Vec)
                    divided_odd: pl.Tile[
                        [1, 1024], pl.FP32, pl.MemRef("row_cache", slots=slots)[out_index_odd], pl.Mem.Vec
                    ] = pl.tile.row_expand_div(source_odd, den_block, packed=True)
                    pl.system.bar_v()
                    result_odd: pl.Tile[[1, 1024], pl.BF16, pl.MemRef("output", slots=2)[1], pl.Mem.Vec] = (
                        pl.cast(divided_odd, pl.BF16, mode="rint")
                    )
                else:
                    result_odd: pl.Tile[[1, 1024], pl.BF16, pl.MemRef("output", slots=2)[1], pl.Mem.Vec] = (
                        pl.tile.full([1, 1024], pl.BF16, 0.0)
                    )
                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.store(result_odd, [row, out_index_odd * 1024], output)
                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
            # Cache slots cannot be overwritten by the next row while V uses them.
            pl.system.bar_all()
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=1)
        return output

    @pl.jit
    def row_cached_softmax(
        output: pl.Out[pl.Tensor[[rows, sequence], pl.BF16]],
        lengths: pl.Tensor[[(rows + heads - 1) // heads], pl.INT32],
        columns: pl.Tensor[[1, 1024], pl.FP32],
        scores: pl.Tensor[[rows, sequence], pl.FP32],
    ):
        width = pl.min(pl.system.available_aiv_count(), rows)
        with pl.spmd(width, allow_early_resolve=True):
            softmax_kernel(output, lengths, columns, scores)
        return output

    return row_cached_softmax
