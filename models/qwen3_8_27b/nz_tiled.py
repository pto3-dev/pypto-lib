# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tile-local NZ weights that satisfy the pinned PTOAS layout inference."""

import pypto.language as pl

ROWS = pl.dynamic("rows")


def pack_nz_tiles(weight, row_tile=256, k_tile=256):
    n, k = weight.shape
    if n % row_tile or k % k_tile:
        raise ValueError("NZ tiles require complete blocks.")
    blocked = weight.reshape(n // row_tile, row_tile // 16, 16, k // k_tile, k_tile // 16, 16)
    return (
        blocked.permute(0, 3, 4, 1, 2, 5)
        .contiguous()
        .reshape((n // row_tile) * (k // k_tile), row_tile, k_tile)
    )


def make_nz_tiled_gemm(k, n, nt=128, kt=256, out_fp32=False, inline=False):
    rt = 256
    if n % rt or k % kt or rt % nt:
        raise ValueError("NZ projection needs 256-row storage tiles.")
    dtype = pl.FP32 if out_fp32 else pl.BF16
    decorate = pl.jit.inline if inline else pl.jit

    @decorate
    def gemm(
        x: pl.Tensor[[ROWS, k], pl.BF16],
        weight: pl.Tensor[[(n // rt) * (k // kt), rt, kt], pl.BF16, pl.NZ],
        out: pl.Out[pl.Tensor[[ROWS, n], dtype]],
    ):
        rows = pl.tensor.dim(x, 0)
        for core in pl.spmd(pl.system.available_cluster_count(), name_hint="nz_tiled_projection"):
            for job in pl.range(core, ((rows + 15) // 16) * (n // nt), pl.tile.get_block_num()):
                m0 = job // (n // nt) * 16
                column = job % (n // nt)
                vm = pl.min(16, rows - m0)
                for half in pl.range(0, rt // nt):
                    if half == column % (rt // nt):
                        acc = pl.create_tensor([16, nt], pl.FP32)
                        for kb in pl.pipeline(0, k // kt, stage=2):
                            k0 = kb * kt
                            xt = pl.slice(x, [16, kt], [m0, k0], [vm, kt], pad_value=0, clamp=True)
                            location = (column // (rt // nt)) * (k // kt) + kb
                            packed = pl.slice(weight, [1, nt, kt], [location, half * nt, 0])
                            wt = pl.reshape(packed, [nt, kt])
                            acc = pl.matmul_acc(acc, xt, wt, b_trans=True, init_cond=(kb == 0))
                        out = pl.assemble(out, pl.tensor.set_validshape(acc, vm, nt), [m0, column * nt])
        return out

    return gemm
