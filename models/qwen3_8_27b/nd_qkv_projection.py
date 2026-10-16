# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import pypto.language as pl

ROWS = pl.dynamic("rows")


def make_nd_qkv_gemm(
    input_dim,
    first_dim,
    second_dim,
    third_dim,
    inline=False,
    n_tile=128,
    k_tile=256,
    pipeline_stage=2,
    weight_cache_bypass=False,
):
    if first_dim % n_tile or second_dim % n_tile or input_dim % k_tile:
        raise ValueError("Split projection requires complete tiles in both outputs.")
    weight_cache_policy = pl.CachePolicy.BYPASS if weight_cache_bypass else pl.CachePolicy.DEFAULT
    output_dim = first_dim + second_dim + third_dim
    decorate = pl.jit.inline if inline else pl.jit

    @decorate
    def gemm(
        x: pl.Tensor[[ROWS, input_dim], pl.BF16],
        weight: pl.Tensor[[output_dim, input_dim], pl.BF16],
        mixed: pl.Out[pl.Tensor[[ROWS, first_dim], pl.BF16]],
        z: pl.Out[pl.Tensor[[ROWS, second_dim], pl.BF16]],
        v: pl.Out[pl.Tensor[[ROWS, third_dim], pl.BF16]],
    ):
        rows = pl.tensor.dim(x, 0)
        for core in pl.spmd(pl.system.available_cluster_count(), name_hint="joint_attention_projection"):
            pl.set_cache_policy(weight, weight_cache_policy)
            for job in pl.range(core, ((rows + 15) // 16) * (output_dim // n_tile), pl.tile.get_block_num()):
                m0 = job // (output_dim // n_tile) * 16
                column = job % (output_dim // n_tile)
                n0 = column * n_tile
                vm = pl.min(16, rows - m0)
                acc = pl.create_tensor([16, n_tile], pl.FP32)
                for kb in pl.pipeline(0, input_dim // k_tile, stage=pipeline_stage):
                    k0 = kb * k_tile
                    xt = pl.slice(x, [16, k_tile], [m0, k0], [vm, k_tile], pad_value=0, clamp=True)
                    wt = pl.slice(weight, [n_tile, k_tile], [n0, k0])
                    acc = pl.matmul_acc(acc, xt, wt, b_trans=True, init_cond=(kb == 0))
                valid = pl.tensor.set_validshape(acc, vm, n_tile)
                if n0 < first_dim:
                    mixed = pl.assemble(mixed, valid, [m0, n0])
                elif n0 < first_dim + second_dim:
                    z = pl.assemble(z, valid, [m0, n0 - first_dim])
                else:
                    v = pl.assemble(v, valid, [m0, n0 - first_dim - second_dim])
        return mixed, z, v

    return gemm
