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


def make_gemm(
    input_dim,
    output_dim,
    inline=False,
    out_fp32=False,
    n_tile=16,
    k_tile=512,
    cube_bf16_store=False,
    stage=1,
    m_tile=16,
    launch_cap=100000,
):
    dtype = pl.FP32 if out_fp32 else pl.BF16
    decorate = pl.jit.inline if inline else pl.jit

    @decorate
    def gemm(
        x: pl.Tensor[[ROWS, input_dim], pl.BF16],
        weight: pl.Tensor[[output_dim, input_dim], pl.BF16],
        out: pl.Out[pl.Tensor[[ROWS, output_dim], dtype]],
    ):
        rows = pl.tensor.dim(x, 0)
        available = pl.system.available_aiv_count()
        width = pl.min(available, pl.min(launch_cap, rows * (output_dim // n_tile)))
        for core in pl.spmd(width, name_hint="vector_gemv"):
            for job in pl.range(core, rows * (output_dim // n_tile), pl.tile.get_block_num()):
                r = job // (output_dim // n_tile)
                n0 = job % (output_dim // n_tile) * n_tile
                seed = pl.full([n_tile, 128], dtype=pl.FP32, value=0.0)
                acc = pl.row_sum(seed)
                for kb in pl.pipeline(0, input_dim // k_tile, stage=stage):
                    xt = pl.cast(pl.slice(x, [1, k_tile], [r, kb * k_tile]), pl.FP32)
                    wt = pl.cast(pl.slice(weight, [n_tile, k_tile], [n0, kb * k_tile]), pl.FP32)
                    prod = pl.col_expand_mul(wt, xt)
                    acc = pl.add(acc, pl.row_sum(prod))
                value = pl.reshape(acc, [1, n_tile])
                if out_fp32:
                    out = pl.assemble(out, value, [r, n0])
                else:
                    out = pl.assemble(out, pl.cast(value, pl.BF16, mode="rint"), [r, n0])
        return out

    return gemm
