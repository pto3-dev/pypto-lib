# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Decode GEMM: load the activation once per core and stream weight K tiles."""

import pypto.language as pl


def make_gemm(
    input_dim,
    output_dim,
    inline=False,
    out_fp32=False,
    n_tile=64,
    k_tile=1024,
    inner_k=256,
    outer_stage=2,
    weight_cache_bypass=False,
):
    if input_dim % k_tile or k_tile % inner_k or output_dim % n_tile:
        raise ValueError("Static-A projection requires exact aligned tiles.")
    policy = pl.CachePolicy.BYPASS if weight_cache_bypass else pl.CachePolicy.DEFAULT
    decorate = pl.jit.inline if inline else pl.jit

    @decorate
    def gemm(
        x: pl.Tensor[[1, input_dim], pl.BF16],
        weight: pl.Tensor[[output_dim, input_dim], pl.BF16],
        out: pl.Out[pl.Tensor[[1, output_dim], pl.FP32 if out_fp32 else pl.BF16]],
    ):
        for core in pl.spmd(pl.system.available_cluster_count(), name_hint="static_a_projection"):
            grid = pl.tile.get_block_num()
            jobs = output_dim // n_tile
            count = jobs // grid
            remainder = jobs % grid
            first = core * count + pl.min(core, remainder)
            last = (core + 1) * count + pl.min(core + 1, remainder)
            activation = pl.tile.load(
                x, [0, 0], [16, input_dim], valid_shape=[1, input_dim], target_memory=pl.Mem.Mat
            )
            for job in pl.range(first, last):
                n0 = job * n_tile
                acc = pl.tile.create([16, n_tile], pl.FP32, target_memory=pl.Mem.Acc)
                for k0 in pl.pipeline(0, input_dim, k_tile, stage=outer_stage):
                    weight_tile = pl.tile.load(
                        weight, [n0, k0], [n_tile, k_tile], target_memory=pl.Mem.Mat, cache=policy
                    )
                    transposed = pl.tile.transpose_view(weight_tile)
                    for kk in pl.pipeline(0, k_tile, inner_k, stage=2):
                        left = pl.tile.extract(
                            activation, 0, k0 + kk, [16, inner_k], target_memory=pl.Mem.Left
                        )
                        right = pl.tile.extract(
                            transposed, kk, 0, [inner_k, n_tile], target_memory=pl.Mem.Right
                        )
                        acc = pl.tile.matmul_acc(acc, left, right, init_cond=(k0 == 0 and kk == 0))
                out = pl.tile.store(pl.tile.set_validshape(acc, 1, n_tile), [0, n0], out)
        return out

    return gemm
