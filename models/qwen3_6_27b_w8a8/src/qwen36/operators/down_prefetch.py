# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Experimental down: true GM/L1 K2176 and L1/L0 K128 pipelines.

Default model never imports this unless the explicit compile variant is chosen.
Integer accumulation, BF16 rounding and FP32 partial output are unchanged.
"""

import pypto.language as pl
from .quantized import quant_4352

OUTER_K = 2176
INNER_K = 128
N_TILE = 64
JOBS = 5120 // N_TILE


@pl.function(type=pl.FunctionType.InCore)
def project_down_kernel(
    x: pl.Tensor[[24, 4352], pl.INT8],
    weight: pl.Tensor[[16, 4352, 5120], pl.INT8, pl.NZ],
    out: pl.Tensor[[24, 5120], pl.FP32],
    weight_group: pl.Scalar[pl.INT64],
    weight_scale: pl.Tensor[[16, 5120], pl.FP32],
    activation_scale: pl.Tensor[[24, 1], pl.FP32],
) -> pl.Tensor[[24, 5120], pl.FP32]:
    pl.func_attr({"static_block_mapping": True})
    core = pl.tile.get_block_idx()
    grid = pl.tile.get_block_num()
    activation = pl.tile.load(x, [0, 0], [32, 4352], valid_shape=[24, 4352], target_memory=pl.Mem.Mat)
    xs = pl.tile.load(activation_scale, [0, 0], [32, 1], valid_shape=[24, 1], target_memory=pl.Mem.Vec)
    for job in pl.range(core, JOBS, grid):
        n0 = job * N_TILE
        acc = pl.tile.create([32, N_TILE], pl.INT32, target_memory=pl.Mem.Acc, compact=True)
        for k1 in pl.pipeline(0, 4352, OUTER_K, stage=2):
            block = pl.tile.reshape(
                pl.tile.load(
                    weight,
                    [weight_group, k1, n0],
                    [1, OUTER_K, N_TILE],
                    target_memory=pl.Mem.Mat,
                    cache=pl.CachePolicy.BYPASS,
                ),
                [OUTER_K, N_TILE],
            )
            for k0 in pl.pipeline(0, OUTER_K, INNER_K, stage=2):
                left = pl.tile.extract(activation, 0, k1 + k0, [32, INNER_K], target_memory=pl.Mem.Left)
                right = pl.tile.extract(block, k0, 0, [INNER_K, N_TILE], target_memory=pl.Mem.Right)
                acc = pl.tile.matmul_acc(acc, left, right, init_cond=k1 == 0 and k0 == 0)
        ws = pl.tile.load(weight_scale, [weight_group, n0], [1, N_TILE], target_memory=pl.Mem.Vec)
        fp = pl.tile.cast(pl.tile.set_validshape(acc, 24, N_TILE), pl.FP32)
        ones = pl.tile.set_validshape(pl.tile.full([32, N_TILE], pl.FP32, 1.0), 24, N_TILE)
        combined_scale = pl.tile.row_expand_mul(pl.tile.col_expand_mul(ones, ws), xs)
        dequant = pl.tile.mul(fp, combined_scale)
        value = pl.tile.cast(dequant, pl.BF16, mode="rint")
        output_value = pl.tile.cast(value, pl.FP32)
        out = pl.tile.store(output_value, [0, n0], out)
    return out


@pl.function(type=pl.FunctionType.Inline)
def project_down(
    x: pl.Tensor[[24, 4352], pl.BF16],
    weight: pl.Tensor[[16, 4352, 5120], pl.INT8, pl.NZ],
    out: pl.Tensor[[24, 5120], pl.FP32],
    weight_group: pl.Scalar[pl.INT64],
    weight_scale: pl.Tensor[[16, 5120], pl.FP32],
) -> pl.Tensor[[24, 5120], pl.FP32]:
    q = pl.create_tensor([24, 4352], pl.INT8)
    xs = pl.create_tensor([24, 1], pl.FP32)
    available_aiv = pl.system.available_aiv_count()
    vector_width = pl.min(available_aiv, 24)
    with pl.spmd(vector_width, allow_early_resolve=True):
        q, xs = quant_4352(x, q, xs)
    available_cube = pl.system.available_cluster_count()
    cube_width = pl.min(available_cube, JOBS)
    with pl.spmd(cube_width, allow_early_resolve=True):
        out = project_down_kernel(q, weight, out, weight_group, weight_scale, xs)
    return out
