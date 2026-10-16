# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""PTO BF16 projections, RMS normalization, and pointwise activations."""

import pypto.language as pl

ROWS = pl.dynamic("rows")


def make_gemm(
    input_dim,
    output_dim,
    inline=False,
    out_fp32=False,
    n_tile=128,
    k_tile=256,
    cube_bf16_store=False,
    stage=2,
    m_tile=16,
    launch_cap=100000,
    weight_cache_bypass=False,
):
    weight_cache_policy = pl.CachePolicy.BYPASS if weight_cache_bypass else pl.CachePolicy.DEFAULT
    output_dtype = pl.FP32 if out_fp32 else pl.BF16
    decorate = pl.jit.inline if inline else pl.jit

    @decorate
    def gemm(
        x: pl.Tensor[[ROWS, input_dim], pl.BF16],
        weight: pl.Tensor[[output_dim, input_dim], pl.BF16],
        out: pl.Out[pl.Tensor[[ROWS, output_dim], output_dtype]],
    ):
        rows = pl.tensor.dim(x, 0)
        # Query launch geometry on the owning worker; use that launch width in-core.
        for core in pl.spmd(pl.system.available_cluster_count(), name_hint="projection"):
            pl.set_cache_policy(weight, weight_cache_policy)
            grid = pl.tile.get_block_num()
            jobs = ((rows + (m_tile - 1)) // m_tile) * ((output_dim + (n_tile - 1)) // n_tile)
            per_core = jobs // grid
            remainder = jobs % grid
            first = core * per_core + pl.min(core, remainder)
            last = (core + 1) * per_core + pl.min(core + 1, remainder)
            for job in pl.range(first, last):
                m0 = job // ((output_dim + (n_tile - 1)) // n_tile) * m_tile
                n0 = job % ((output_dim + (n_tile - 1)) // n_tile) * n_tile
                vm = pl.min(m_tile, rows - m0)
                vn = pl.min(n_tile, output_dim - n0)
                acc = pl.create_tensor([m_tile, n_tile], pl.FP32)
                for kb in pl.pipeline(0, (input_dim + (k_tile - 1)) // k_tile, stage=stage):
                    k0 = kb * k_tile
                    vk = pl.min(k_tile, input_dim - k0)
                    xt = pl.slice(
                        x, [m_tile, k_tile], [m0, k0], valid_shape=[vm, vk], pad_value=0, clamp=True
                    )
                    wt = pl.slice(
                        weight, [n_tile, k_tile], [n0, k0], valid_shape=[vn, vk], pad_value=0, clamp=True
                    )
                    acc = pl.matmul_acc(acc, xt, wt, b_trans=True, init_cond=(kb == 0))
                if out_fp32:
                    fp32_result = pl.tensor.set_validshape(acc, vm, vn)
                    out = pl.assemble(out, fp32_result, [m0, n0])
                elif cube_bf16_store:
                    # FIXPIPE can narrow an FP32 accumulator directly into
                    # BF16 GM with round-to-nearest-even, keeping the epilogue
                    # on Cube and avoiding a separate vector worker per tile.
                    narrowed_valid = pl.tensor.set_validshape(acc, vm, vn)
                    out = pl.assemble(out, narrowed_valid, [m0, n0])
                else:
                    bf16_result = pl.cast(acc, pl.BF16, mode="rint")
                    bf16_valid = pl.tensor.set_validshape(bf16_result, vm, vn)
                    out = pl.assemble(out, bf16_valid, [m0, n0])
        return out

    return gemm
