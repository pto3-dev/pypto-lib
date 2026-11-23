# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""INT32 down publication; dequantization is owned by the row communication consumer.

All K/N tiling and integer accumulation follow the packed reference. BF16 rounding
moves intact to the consumer before peer publication; no rank sum order changes.
"""
import pypto.language as pl
from qwen36.tuning import DOWN_N,DOWN_L1_K,DOWN_L0_K,DOWN_JOBS
from .packed_act_quant import packed_gateup_kernel,swiglu_quant_kernel

@pl.function(type=pl.FunctionType.InCore)
def project_down_i32_kernel(x: pl.Tensor[[24, 4352], pl.INT8], weight: pl.Tensor[[16, 4352, 5120], pl.INT8, pl.NZ], out: pl.Tensor[[24, 5120], pl.INT32], weight_group: pl.Scalar[pl.INT64]) -> pl.Tensor[[24, 5120], pl.INT32]:
    pl.func_attr({'static_block_mapping': False})
    core = pl.tile.get_block_idx()
    grid = pl.tile.get_block_num()
    activation = pl.tile.load(x, [0, 0], [32, 4352], valid_shape=[24, 4352], target_memory=pl.Mem.Mat)
    for job in pl.range(core, DOWN_JOBS, grid):
        n0 = job * DOWN_N
        value_acc = pl.tile.create([32, DOWN_N], pl.INT32, target_memory=pl.Mem.Acc, compact=True)
        for k0 in pl.pipeline(0, 4352, DOWN_L1_K, stage=2):
            wt = pl.tile.reshape(pl.tile.load(weight, [weight_group, k0, n0], [1, DOWN_L1_K, DOWN_N], target_memory=pl.Mem.Mat, cache=pl.CachePolicy.BYPASS), [DOWN_L1_K, DOWN_N])
            left = pl.tile.extract(activation, 0, k0, [32, DOWN_L0_K], target_memory=pl.Mem.Left)
            right = pl.tile.extract(wt, 0, 0, [DOWN_L0_K, DOWN_N], target_memory=pl.Mem.Right)
            value_acc = pl.tile.matmul_acc(value_acc, left, right, init_cond=k0 == 0)
        out = pl.tile.store(pl.tile.set_validshape(value_acc, 24, DOWN_N), [0, n0], out)
    return out



@pl.function(type=pl.FunctionType.Inline)
def packed_mlp_i32_from_quant(q:pl.Tensor[[24,5120],pl.INT8],gate:pl.Tensor[[16,5120,8704],pl.INT8,pl.NZ],out:pl.Tensor[[24,5120],pl.INT32],group:pl.Scalar[pl.INT64],gate_scale:pl.Tensor[[16,8704],pl.FP32],xscale:pl.Tensor[[24,1],pl.FP32],down:pl.Tensor[[16,4352,5120],pl.INT8,pl.NZ],down_activation_scale:pl.Tensor[[24,1],pl.FP32])->tuple[pl.Tensor[[24,5120],pl.INT32],pl.Tensor[[24,1],pl.FP32]]:
    ints=pl.create_tensor([24,8704],pl.INT32)
    aq=pl.create_tensor([24,4352],pl.INT8)
    cc=pl.system.available_cluster_count()
    av=pl.system.available_aiv_count()
    with pl.spmd(pl.min(cc,39),allow_early_resolve=True):
        ints=packed_gateup_kernel(q,gate,ints,group)
    with pl.spmd(pl.min(av,24),allow_early_resolve=True):
        aq,down_activation_scale=swiglu_quant_kernel(ints,gate_scale,xscale,aq,down_activation_scale,group)
    with pl.spmd(pl.min(cc,DOWN_JOBS),allow_early_resolve=True):
        out=project_down_i32_kernel(aq,down,out,group)
    return out,down_activation_scale
