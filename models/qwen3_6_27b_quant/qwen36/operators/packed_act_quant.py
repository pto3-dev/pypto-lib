# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Donor-style complete MLP window, normal DSL only.

Packed gate/up is a pure AIC producer. One AIV owns one complete row for
BF16 dequant/SwiGLU and dynamic quant, avoiding intermediate activation GM.
Weight/activation scaling order and all BF16 rounding points are preserved.
"""
import pypto.language as pl

@pl.function(type=pl.FunctionType.InCore)
def packed_gateup_kernel(x:pl.Tensor[[24,5120],pl.INT8],weight:pl.Tensor[[16,5120,8704],pl.INT8,pl.NZ],ints:pl.Tensor[[24,8704],pl.INT32],weight_group:pl.Scalar[pl.INT64])->pl.Tensor[[24,8704],pl.INT32]:
    pl.func_attr({'static_block_mapping':False})
    core=pl.tile.get_block_idx()
    grid=pl.tile.get_block_num()
    activation=pl.tile.load(x,[0,0],[32,5120],valid_shape=[24,5120],target_memory=pl.Mem.Mat)
    for job in pl.range(core,(8704+223)//224,grid):
        n0=job*224
        if job<38:
            full_acc=pl.tile.create([32,224],pl.INT32,target_memory=pl.Mem.Acc,compact=True)
            for full_k1 in pl.pipeline(0,5120,512,stage=3):
                full_w=pl.tile.reshape(pl.tile.load(weight,[weight_group,full_k1,n0],[1,512,224],target_memory=pl.Mem.Mat,cache=pl.CachePolicy.BYPASS),[512,224])
                for full_k0 in pl.pipeline(0,512,128,stage=2):
                    full_left=pl.tile.extract(activation,0,full_k1+full_k0,[32,128],target_memory=pl.Mem.Left)
                    full_right:pl.Tile[[128,224],pl.INT8,pl.MemRef('packed_right',slots=2)[(full_k0//128)%2],pl.Mem.Right]=pl.tile.extract(full_w,full_k0,0,[128,224],target_memory=pl.Mem.Right)
                    full_acc=pl.tile.matmul_acc(full_acc,full_left,full_right,init_cond=full_k1==0 and full_k0==0)
            ints=pl.tile.store(pl.tile.set_validshape(full_acc,24,224),[0,n0],ints)
        else:
            tail_acc=pl.tile.create([32,192],pl.INT32,target_memory=pl.Mem.Acc,compact=True)
            for tail_k1 in pl.pipeline(0,5120,512,stage=3):
                tail_w=pl.tile.reshape(pl.tile.load(weight,[weight_group,tail_k1,n0],[1,512,192],target_memory=pl.Mem.Mat,cache=pl.CachePolicy.BYPASS),[512,192])
                for tail_k0 in pl.pipeline(0,512,128,stage=2):
                    tail_left=pl.tile.extract(activation,0,tail_k1+tail_k0,[32,128],target_memory=pl.Mem.Left)
                    tail_right:pl.Tile[[128,192],pl.INT8,pl.MemRef('packed_right',slots=2)[(tail_k0//128)%2],pl.Mem.Right]=pl.tile.extract(tail_w,tail_k0,0,[128,192],target_memory=pl.Mem.Right)
                    tail_acc=pl.tile.matmul_acc(tail_acc,tail_left,tail_right,init_cond=tail_k1==0 and tail_k0==0)
            ints=pl.tile.store(pl.tile.set_validshape(tail_acc,24,192),[0,n0],ints)
    return ints

@pl.function(type=pl.FunctionType.InCore)
def swiglu_quant_kernel(ints:pl.Tensor[[24,8704],pl.INT32],weight_scale:pl.Tensor[[16,8704],pl.FP32],activation_scale:pl.Tensor[[24,1],pl.FP32],q:pl.Out[pl.Tensor[[24,4352],pl.INT8]],scales:pl.Out[pl.Tensor[[24,1],pl.FP32]],weight_group:pl.Scalar[pl.INT64])->tuple[pl.Tensor[[24,4352],pl.INT8],pl.Tensor[[24,1],pl.FP32]]:
    pl.func_attr({'static_block_mapping':True})
    lane=pl.tile.get_block_idx()
    width=pl.tile.get_block_num()
    # Full column follows the existing DN-compatible DSL load path. Its24
    # scale values occupy96valid bytes; each lane reads its own scalar.
    xs_column=pl.tile.load(activation_scale,[0,0],[32,1],valid_shape=[24,1],target_memory=pl.Mem.Vec)
    for row in pl.range(lane,24,width):
        xs=pl.tile.read(xs_column,[row,0])
        gate_int=pl.tile.load(ints,[row,0],[1,4352],target_memory=pl.Mem.Vec)
        up_int=pl.tile.load(ints,[row,4352],[1,4352],target_memory=pl.Mem.Vec)
        gs=pl.tile.load(weight_scale,[weight_group,0],[1,4352],target_memory=pl.Mem.Vec)
        us=pl.tile.load(weight_scale,[weight_group,4352],[1,4352],target_memory=pl.Mem.Vec)
        gate_fp=pl.tile.cast(gate_int,pl.FP32)
        up_fp=pl.tile.cast(up_int,pl.FP32)
        gate_scale=pl.tile.muls(gs,xs)
        up_scale=pl.tile.muls(us,xs)
        gate_bf=pl.tile.cast(pl.tile.mul(gate_fp,gate_scale),pl.BF16,mode='rint')
        up_bf=pl.tile.cast(pl.tile.mul(up_fp,up_scale),pl.BF16,mode='rint')
        gate=pl.tile.cast(gate_bf,pl.FP32)
        up=pl.tile.cast(up_bf,pl.FP32)
        # CANN9.1 BF16 SwiGLU uses direct F32 DIV before multiplying up.
        # Reciprocal followed by multiply changes a BF16 midpoint (real row5).
        silu=pl.tile.div(gate,pl.tile.adds(pl.tile.exp(pl.tile.neg(gate)),1.0))
        act_bf=pl.tile.cast(pl.tile.mul(silu,up),pl.BF16,mode='rint')
        value=pl.tile.cast(act_bf,pl.FP32)
        chunks=pl.tile.reshape(pl.tile.abs(value),[8,544])
        tmp=pl.tile.create([8,544],pl.FP32)
        maximum=pl.tile.row_max(chunks,tmp)
        magnitude=pl.tile.read(maximum,[0,0])
        for part in pl.range(1,8):
            partial=pl.tile.read(maximum,[part,0])
            if partial>magnitude:
                magnitude=partial
        if magnitude==0.0:
            normalized=pl.tile.muls(value,0.0)
        else:
            denominator=pl.tile.muls(pl.tile.full([1,8],pl.FP32,1.0),magnitude)
            numerator=pl.tile.full([1,8],pl.FP32,127.0)
            ratio_tile=pl.tile.div(numerator,denominator)
            ratio=pl.tile.read(ratio_tile,[0,0])
            normalized=pl.tile.muls(value,ratio)
        rounded=pl.tile.cast(normalized,pl.INT32,mode='rint')
        half=pl.tile.cast(rounded,pl.FP16,mode='rint')
        quantized=pl.tile.cast(half,pl.INT8,mode='trunc')
        q=pl.tile.store(quantized,[row,0],q)
        scale_tile=pl.tile.muls(pl.tile.full([1,8],pl.FP32,1.0),magnitude*(1.0/127.0))
        scales=pl.tile.store(pl.tile.set_validshape(scale_tile,1,1),[row,0],scales)
    return q,scales

from .quantized import project_down_kernel

@pl.function(type=pl.FunctionType.Inline)
def packed_mlp_from_quant(q:pl.Tensor[[24,5120],pl.INT8],gate:pl.Tensor[[16,5120,8704],pl.INT8,pl.NZ],out:pl.Tensor[[24,5120],pl.FP32],group:pl.Scalar[pl.INT64],gate_scale:pl.Tensor[[16,8704],pl.FP32],xscale:pl.Tensor[[24,1],pl.FP32],down:pl.Tensor[[16,4352,5120],pl.INT8,pl.NZ],down_scale:pl.Tensor[[16,5120],pl.FP32])->pl.Tensor[[24,5120],pl.FP32]:
    ints=pl.create_tensor([24,8704],pl.INT32)
    aq=pl.create_tensor([24,4352],pl.INT8)
    scale=pl.create_tensor([24,1],pl.FP32)
    cc=pl.system.available_cluster_count()
    av=pl.system.available_aiv_count()
    with pl.spmd(pl.min(cc,39),allow_early_resolve=True):
        ints=packed_gateup_kernel(q,gate,ints,group)
    with pl.spmd(pl.min(av,24),allow_early_resolve=True):
        aq,scale=swiglu_quant_kernel(ints,gate_scale,xscale,aq,scale,group)
    with pl.spmd(pl.min(cc,40),allow_early_resolve=True):
        out=project_down_kernel(aq,down,out,group,down_scale,scale)
    return out
