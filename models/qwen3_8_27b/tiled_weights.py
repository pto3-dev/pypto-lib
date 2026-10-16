# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""ND weight tiles packed once on CPU and kept resident on the NPU."""

import pypto.language as pl

ROWS = pl.dynamic("rows")


def pack_tiles(weight, n_tile=128, k_tile=256, transpose=False):
    n, k = weight.shape
    if n % n_tile or k % k_tile:
        raise ValueError("Resident weight packing requires complete tiles.")
    order = (0, 2, 3, 1) if transpose else (0, 2, 1, 3)
    return weight.reshape(n // n_tile, n_tile, k // k_tile, k_tile).permute(*order).contiguous()


def make_tiled_gemm(
    input_dim,
    output_dim,
    inline=False,
    out_fp32=False,
    n_tile=128,
    k_tile=256,
    transpose=False,
    pipeline_stage=2,
):
    if input_dim % k_tile or output_dim % n_tile:
        raise ValueError("Tiled GEMM requires complete N/K tiles.")
    if pipeline_stage not in (1, 2, 3, 4):
        raise ValueError("Projection pipeline stage must be in [1, 4].")
    output_dtype = pl.FP32 if out_fp32 else pl.BF16
    tile_rows = k_tile if transpose else n_tile
    tile_cols = n_tile if transpose else k_tile
    b_transpose = not transpose
    decorate = pl.jit.inline if inline else pl.jit

    @decorate
    def gemm(
        x: pl.Tensor[[ROWS, input_dim], pl.BF16],
        weight: pl.Tensor[[output_dim // n_tile, input_dim // k_tile, tile_rows, tile_cols], pl.BF16],
        out: pl.Out[pl.Tensor[[ROWS, output_dim], output_dtype]],
    ):
        rows = pl.tensor.dim(x, 0)
        for core in pl.spmd(pl.system.available_cluster_count(), name_hint="tiled_projection"):
            for job in pl.range(core, ((rows + 15) // 16) * (output_dim // n_tile), pl.tile.get_block_num()):
                m0 = job // (output_dim // n_tile) * 16
                column = job % (output_dim // n_tile)
                n0 = column * n_tile
                vm = pl.min(16, rows - m0)
                acc = pl.create_tensor([16, n_tile], pl.FP32)
                for kb in pl.pipeline(0, input_dim // k_tile, stage=pipeline_stage):
                    k0 = kb * k_tile
                    xt = pl.slice(x, [16, k_tile], [m0, k0], [vm, k_tile], pad_value=0, clamp=True)
                    packed = pl.slice(weight, [1, 1, tile_rows, tile_cols], [column, kb, 0, 0])
                    wt = pl.reshape(packed, [tile_rows, tile_cols])
                    acc = pl.matmul_acc(acc, xt, wt, b_trans=b_transpose, init_cond=(kb == 0))
                valid = pl.tensor.set_validshape(acc, vm, n_tile)
                out = pl.assemble(out, valid, [m0, n0])
        return out

    return gemm


def make_split_tiled_gemm(
    input_dim, first_dim, second_dim, inline=False, n_tile=128, k_tile=256, pipeline_stage=2
):
    if first_dim % n_tile or second_dim % n_tile or input_dim % k_tile:
        raise ValueError("Split projection requires complete tiles in both outputs.")
    output_dim = first_dim + second_dim
    decorate = pl.jit.inline if inline else pl.jit

    @decorate
    def gemm(
        x: pl.Tensor[[ROWS, input_dim], pl.BF16],
        weight: pl.Tensor[[output_dim // n_tile, input_dim // k_tile, n_tile, k_tile], pl.BF16],
        mixed: pl.Out[pl.Tensor[[ROWS, first_dim], pl.BF16]],
        z: pl.Out[pl.Tensor[[ROWS, second_dim], pl.BF16]],
    ):
        rows = pl.tensor.dim(x, 0)
        for core in pl.spmd(pl.system.available_cluster_count(), name_hint="split_tiled_projection"):
            for job in pl.range(core, ((rows + 15) // 16) * (output_dim // n_tile), pl.tile.get_block_num()):
                m0 = job // (output_dim // n_tile) * 16
                column = job % (output_dim // n_tile)
                n0 = column * n_tile
                vm = pl.min(16, rows - m0)
                acc = pl.create_tensor([16, n_tile], pl.FP32)
                for kb in pl.pipeline(0, input_dim // k_tile, stage=pipeline_stage):
                    k0 = kb * k_tile
                    xt = pl.slice(x, [16, k_tile], [m0, k0], [vm, k_tile], pad_value=0, clamp=True)
                    packed = pl.slice(weight, [1, 1, n_tile, k_tile], [column, kb, 0, 0])
                    wt = pl.reshape(packed, [n_tile, k_tile])
                    acc = pl.matmul_acc(acc, xt, wt, b_trans=True, init_cond=(kb == 0))
                valid = pl.tensor.set_validshape(acc, vm, n_tile)
                if n0 < first_dim:
                    mixed = pl.assemble(mixed, valid, [m0, n0])
                else:
                    z = pl.assemble(z, valid, [m0, n0 - first_dim])
        return mixed, z

    return gemm
