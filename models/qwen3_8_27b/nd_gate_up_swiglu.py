# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Full-row ND loads with gate compute overlapping the up-weight load."""

import pypto.language as pl


def make_gemm(
    input_dim,
    output_dim,
    inline=False,
    n_tile=16,
    k_tile=5120,
    cube_bf16_store=True,
    stage=1,
    m_tile=16,
    launch_cap=100000,
    inner_k=640,
    inner_stage=2,
    weight_cache_bypass=False,
):
    intermediate = output_dim // 2
    weight_cache_policy = pl.CachePolicy.BYPASS if weight_cache_bypass else pl.CachePolicy.DEFAULT
    if input_dim % inner_k:
        raise ValueError("Require exact L0 chunks.")
    decorate = pl.jit.inline if inline else pl.jit

    @decorate
    def gemm(
        x: pl.Tensor[[1, input_dim], pl.BF16],
        weight: pl.Tensor[[output_dim, input_dim], pl.BF16],
        out: pl.Out[pl.Tensor[[1, intermediate], pl.BF16]],
    ):
        for core in pl.spmd(pl.system.available_cluster_count(), name_hint="gate_up_swiglu"):
            grid = pl.tile.get_block_num()
            jobs = intermediate // n_tile
            count = jobs // grid
            remainder = jobs % grid
            first = core * count + pl.min(core, remainder)
            last = (core + 1) * count + pl.min(core + 1, remainder)
            xt = pl.tile.load(
                x, [0, 0], [16, input_dim], valid_shape=[1, input_dim], target_memory=pl.Mem.Mat
            )
            for job in pl.range(first, last):
                n0 = job * n_tile
                gate_acc = pl.tile.create([16, n_tile], pl.FP32, target_memory=pl.Mem.Acc)
                up_acc = pl.tile.create([16, n_tile], pl.FP32, target_memory=pl.Mem.Acc)
                wg = pl.tile.load(
                    weight, [n0, 0], [n_tile, input_dim], target_memory=pl.Mem.Mat, cache=weight_cache_policy
                )
                wu = pl.tile.load(
                    weight,
                    [intermediate + n0, 0],
                    [n_tile, input_dim],
                    target_memory=pl.Mem.Mat,
                    cache=weight_cache_policy,
                )
                wgt = pl.tile.transpose_view(wg)
                wut = pl.tile.transpose_view(wu)
                for kk in pl.pipeline(0, input_dim, inner_k, stage=inner_stage):
                    ag = pl.tile.extract(xt, 0, kk, [16, inner_k], target_memory=pl.Mem.Left)
                    bg = pl.tile.extract(wgt, kk, 0, [inner_k, n_tile], target_memory=pl.Mem.Right)
                    gate_acc = pl.tile.matmul_acc(gate_acc, ag, bg, init_cond=(kk == 0))
                for kk in pl.pipeline(0, input_dim, inner_k, stage=inner_stage):
                    au = pl.tile.extract(xt, 0, kk, [16, inner_k], target_memory=pl.Mem.Left)
                    bu = pl.tile.extract(wut, kk, 0, [inner_k, n_tile], target_memory=pl.Mem.Right)
                    up_acc = pl.tile.matmul_acc(up_acc, au, bu, init_cond=(kk == 0))
                gate = pl.tile.cast(pl.tile.cast(gate_acc, pl.BF16, mode="rint"), pl.FP32)
                up = pl.tile.cast(pl.tile.cast(up_acc, pl.BF16, mode="rint"), pl.FP32)
                sigmoid = pl.tile.recip(pl.tile.adds(pl.tile.exp(pl.tile.neg(gate)), 1.0))
                activated = pl.tile.cast(pl.tile.mul(pl.tile.mul(sigmoid, gate), up), pl.BF16, mode="rint")
                out = pl.tile.store(pl.tile.set_validshape(activated, 1, n_tile), [0, n0], out)
        return out

    return gemm
