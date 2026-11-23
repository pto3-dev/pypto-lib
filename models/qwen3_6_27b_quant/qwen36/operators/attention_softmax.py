# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""45056-column bounded-row-cache softmax, pure PyPTO DSL.

Resident cache size is derived from the selected backend's actual Vec capacity.
Nonresident tail tiles are reloaded, never narrowed into BF16 before reduction.
The 1024-column FP32 maximum/sum order is retained. Emit unnormalized BF16
exponents plus an FP32 denominator, so PV is normalized before final BF16
rounding rather than rounding normalized probabilities before the matrix.
"""
import pypto.language as pl
from pypto import backend,ir
from ..sequence_config import SEQUENCE,SOFTMAX_SLOTS
_handler=backend.get_backend_instance(backend.BackendType.Ascend910B)
VEC_CAPACITY=_handler.get_mem_size(ir.MemorySpace.Vec)
STAT_COLUMNS=_handler.get_handler().get_vec_tile_inner_alignment_bytes()//4
# A tuning reserve for live vectors/statistics/output; not a hardware capacity.
SCRATCH_RESERVE=9*1024*4
RESIDENT_SLOTS=min(SOFTMAX_SLOTS,(VEC_CAPACITY-SCRATCH_RESERVE)//4096)
if not 1<=RESIDENT_SLOTS<=SOFTMAX_SLOTS:raise ValueError('Backend Vec capacity cannot hold a row-cache tile and scratch')
ROW_CACHE=pl.MemRef("bounded_row",slots=RESIDENT_SLOTS)

@pl.function(type=pl.FunctionType.InCore)
def softmax_cached_kernel(
    output:pl.Out[pl.Tensor[[144,SEQUENCE],pl.BF16]],
    lengths:pl.Tensor[[24],pl.INT32],
    columns:pl.Tensor[[1,1024],pl.FP32],
    scores:pl.Tensor[[144,SEQUENCE],pl.FP32],
    denominators:pl.Out[pl.Tensor[[144,1],pl.FP32]],
):
    for row in pl.range(pl.tile.get_block_idx(),144,pl.tile.get_block_num()):
        # A2/A3 TSELS requires a same-dtype row scratch in this backend ABI.
        select_scratch=pl.create_tile([1,1024],pl.FP32,target_memory=pl.Mem.Vec)
        visible=pl.read(lengths,[row//6])
        active=SOFTMAX_SLOTS
        if visible>1:
            active=pl.min((visible+1023)//1024,SOFTMAX_SLOTS)
        cache_count=pl.min(active,RESIDENT_SLOTS)
        maximum=pl.tile.full([1,1024],pl.FP32,-3.4028234663852886e38)
        bulk=pl.min(active-1,RESIDENT_SLOTS)
        if visible<=1:bulk=0
        pl.system.bar_all()
        for prefetch_index in pl.pipeline(bulk,stage=2):
            prefetched_raw:pl.Tile[[1,1024],pl.FP32,ROW_CACHE[prefetch_index],pl.Mem.Vec]=pl.load(scores,[row,prefetch_index*1024],[1,1024])
        pl.system.bar_all()
        for compute_index in pl.range(bulk):
            compute_raw:pl.Tile[[1,1024],pl.FP32,ROW_CACHE[compute_index],pl.Mem.Vec]=pl.create_tile([1,1024],pl.FP32,target_memory=pl.Mem.Vec)
            compute_score:pl.Tile[[1,1024],pl.FP32,ROW_CACHE[compute_index],pl.Mem.Vec]=pl.mul(compute_raw,0.0625)
            maximum=pl.maximum(maximum,compute_score)
        # At most one resident tile needs the causal mask.
        for cached_tail in pl.range(bulk,cache_count):
            offset=cached_tail*1024
            cached_tail_raw:pl.Tile[[1,1024],pl.FP32,ROW_CACHE[cached_tail],pl.Mem.Vec]=pl.load(scores,[row,offset],[1,1024])
            cached_tail_scaled:pl.Tile[[1,1024],pl.FP32,ROW_CACHE[cached_tail],pl.Mem.Vec]=pl.mul(cached_tail_raw,0.0625)
            column=pl.load(columns,[0,0],[1,1024])
            distance=pl.add(column,pl.cast(pl.cast(offset-visible,pl.INT32),pl.FP32))
            # A mask multiplication preserves NaN/Inf in inactive QK lanes.
            # Value selection must happen before max/exp/sum reductions.
            cached_tail_score:pl.Tile[[1,1024],pl.FP32,ROW_CACHE[cached_tail],pl.Mem.Vec]=pl.tile.sels(pl.tile.cmps(distance,0.0,cmp_type=2),cached_tail_scaled,select_scratch,-3.4028234663852886e38)
            maximum=pl.maximum(maximum,cached_tail_score)
        for tail_index in pl.range(cache_count,active):
            offset=tail_index*1024
            raw=pl.load(scores,[row,offset],[1,1024])
            score=pl.mul(raw,0.0625)
            if offset+1024>=visible:
                column=pl.load(columns,[0,0],[1,1024])
                distance=pl.add(column,pl.cast(pl.cast(offset-visible,pl.INT32),pl.FP32))
                score=pl.tile.sels(pl.tile.cmps(distance,0.0,cmp_type=2),score,select_scratch,-3.4028234663852886e38)
            maximum=pl.maximum(maximum,score)
        scratch=pl.create_tile([1,1024],pl.FP32,target_memory=pl.Mem.Vec)
        max_stat=pl.row_max(maximum,scratch)
        max_value=pl.tile.read(max_stat,[0,0])
        max_broadcast=pl.tile.expands(pl.create_tile([1,STAT_COLUMNS],pl.FP32,target_memory=pl.Mem.Vec),max_value)
        denominator=pl.tile.full([1,1024],pl.FP32,0.0)
        for exp_index in pl.range(cache_count):
            cached:pl.Tile[[1,1024],pl.FP32,ROW_CACHE[exp_index],pl.Mem.Vec]=pl.create_tile([1,1024],pl.FP32,target_memory=pl.Mem.Vec)
            cached_exponent:pl.Tile[[1,1024],pl.FP32,ROW_CACHE[exp_index],pl.Mem.Vec]=pl.tile.row_expand_expdif(cached,max_broadcast,packed=True)
            denominator=pl.add(denominator,cached_exponent)
        # Reload only the nonresident tail, preserving the exact lane-wise sum order.
        for tail_index in pl.range(cache_count,active):
            offset=tail_index*1024
            score=pl.mul(pl.load(scores,[row,offset],[1,1024]),0.0625)
            if offset+1024>=visible:
                column=pl.load(columns,[0,0],[1,1024])
                distance=pl.add(column,pl.cast(pl.cast(offset-visible,pl.INT32),pl.FP32))
                score=pl.tile.sels(pl.tile.cmps(distance,0.0,cmp_type=2),score,select_scratch,-3.4028234663852886e38)
            exponent=pl.tile.row_expand_expdif(score,max_broadcast,packed=True)
            denominator=pl.add(denominator,exponent)
        # FAInfer RowsumTAILTILE: reduce consecutive32-byte blocks, then
        # each64-element group, and accumulate those groups in source order.
        # This preserves a materially different FP32 tree from TROWSUM1024.
        sum_blocks=pl.tile.reshape(denominator,[128,8])
        sum_block_scratch=pl.tile.create([128,8],pl.FP32,target_memory=pl.Mem.Vec)
        block_sums=pl.tile.row_sum(sum_blocks,sum_block_scratch)
        sum_vectors=pl.tile.reshape(block_sums,[16,8])
        sum_vector_scratch=pl.tile.create([16,8],pl.FP32,target_memory=pl.Mem.Vec)
        vector_sums=pl.tile.row_sum(sum_vectors,sum_vector_scratch)
        native_sum=pl.tile.read(vector_sums,[0,0])
        for sum_group in pl.range(1,16):
            native_sum=native_sum+pl.tile.read(vector_sums,[sum_group,0])
        den_block=pl.tile.expands(pl.create_tile([1,STAT_COLUMNS],pl.FP32,target_memory=pl.Mem.Vec),native_sum)
        den_stat=pl.tile.set_validshape(den_block,1,1)
        den_value=pl.tile.read(den_stat,[0,0])
        # The row sum remains FP32 until the PV consumer. This scalar GM
        # publication is explicit in the normal compiler dependency graph.
        pl.store(pl.tile.set_validshape(den_stat,1,1),[row,0],denominators)
        # One output slot: explicit conservative lifetime until codegen audit.
        for output_index in pl.range(SOFTMAX_SLOTS):
            output_offset=output_index*1024
            if output_index<active:
                if output_index<cache_count:
                    output_cached:pl.Tile[[1,1024],pl.FP32,ROW_CACHE[output_index],pl.Mem.Vec]=pl.create_tile([1,1024],pl.FP32,target_memory=pl.Mem.Vec)
                    output_result_cached=pl.cast(output_cached,pl.BF16,mode='rint')
                    pl.store(output_result_cached,[row,output_offset],output)
                else:
                    output_score=pl.mul(pl.load(scores,[row,output_offset],[1,1024]),0.0625)
                    if output_offset+1024>=visible:
                        output_column=pl.load(columns,[0,0],[1,1024])
                        output_distance=pl.add(output_column,pl.cast(pl.cast(output_offset-visible,pl.INT32),pl.FP32))
                        output_score=pl.tile.sels(pl.tile.cmps(output_distance,0.0,cmp_type=2),output_score,select_scratch,-3.4028234663852886e38)
                    output_exponent=pl.tile.row_expand_expdif(output_score,max_broadcast,packed=True)
                    output_result_tail=pl.cast(output_exponent,pl.BF16,mode='rint')
                    pl.store(output_result_tail,[row,output_offset],output)
            else:
                output_zero=pl.tile.full([1,1024],pl.BF16,0.0)
                pl.store(output_zero,[row,output_offset],output)
    return output,denominators

@pl.function(type=pl.FunctionType.Inline)
def softmax_scores(x:pl.Tensor[[144,SEQUENCE],pl.FP32],lengths:pl.Tensor[[24],pl.INT32],columns:pl.Tensor[[1,1024],pl.FP32],out:pl.Tensor[[144,SEQUENCE],pl.BF16],denominators:pl.Tensor[[144,1],pl.FP32])->tuple[pl.Tensor[[144,SEQUENCE],pl.BF16],pl.Tensor[[144,1],pl.FP32]]:
    available=pl.system.available_aiv_count()
    width=pl.min(available,144)
    with pl.spmd(width,allow_early_resolve=True):
        out,denominators=softmax_cached_kernel(out,lengths,columns,x,denominators)
    return out,denominators
