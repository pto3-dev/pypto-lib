# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -------------------------------------------------------------------------------------------------------
"""Composable QKV/O projection without diagnostic INT32 stores or NZ copies."""

from typing import Any

import pypto.language as pl

from .config import HIDDEN
from .linear import QKV_WIDTH

N_TILE = 256
K_TILE = 512


def make_projection_output(
    output_width: int,
    rows: int,
    *,
    inline: bool = False,
    n_tile: int = N_TILE,
    vector_epilogue: bool | None = None,
) -> Any:
    """Vectorize only the qualified B16/N256 tile; False retains a scalar control."""
    if not 1 <= rows <= 16:
        raise ValueError(f"Unsupported projection bucket: {rows}")
    if output_width not in (HIDDEN, QKV_WIDTH):
        raise ValueError(f"Unsupported Qwen30 projection width: {output_width}")
    if n_tile not in (128, 256):
        raise ValueError(f"Unsupported output projection N tile: {n_tile}")
    use_vector_epilogue = vector_epilogue is not False and rows == 16 and n_tile == 256
    decorator = pl.jit.inline if inline else pl.jit

    @decorator
    def qkv_project(
        x: pl.Tensor[[rows, HIDDEN], pl.INT8],
        input_scale: pl.Tensor[[rows, 1], pl.FP32],
        weight: pl.Tensor[[1, HIDDEN, output_width], pl.INT8, pl.NZ],
        weight_scale: pl.Tensor[[1, output_width], pl.FP32],
        y: pl.Out[pl.Tensor[[rows, output_width], pl.BF16]],
    ) -> pl.Tensor:
        for block in pl.spmd(output_width // n_tile, name_hint="w8a8_output_only", allow_early_resolve=True):
            pl.set_cache_policy(weight, pl.CachePolicy.BYPASS)
            for n0 in pl.unroll(0, output_width, n_tile):
                if block == n0 // n_tile:
                    acc = pl.create_tensor([rows, n_tile], dtype=pl.INT32)
                    for kb in pl.pipeline(HIDDEN // K_TILE, stage=2):
                        k0 = kb * K_TILE
                        lhs = pl.slice(x, [rows, K_TILE], [0, k0])
                        rhs = pl.reshape(pl.slice(weight, [1, K_TILE, n_tile], [0, k0, n0]), [K_TILE, n_tile])
                        acc = pl.matmul_acc(acc, lhs, rhs, init_cond=(kb == 0))
                    value = pl.cast(acc, pl.FP32, mode="rint")
                    if use_vector_epilogue:
                        expanded_scale = pl.col_expand_mul(
                            pl.full([rows, n_tile], dtype=pl.FP32, value=1.0),
                            weight_scale[:, n0 : n0 + n_tile],
                        )
                        scale_product = pl.row_expand_mul(expanded_scale, input_scale[:, :])
                        y[:, n0 : n0 + n_tile] = pl.cast(pl.mul(value, scale_product), pl.BF16, mode="rint")
                    else:
                        for row in pl.unroll(rows):
                            scale = pl.read(input_scale, [row, 0])
                            combined_scale = pl.mul(weight_scale[:, n0 : n0 + n_tile], scale)
                            y[row : row + 1, n0 : n0 + n_tile] = pl.cast(
                                pl.mul(value[row : row + 1, :], combined_scale), pl.BF16, mode="rint"
                            )
        return y

    @decorator
    def output_project(
        x: pl.Tensor[[rows, HIDDEN], pl.INT8],
        input_scale: pl.Tensor[[rows, 1], pl.FP32],
        weight: pl.Tensor[[1, HIDDEN, output_width], pl.INT8, pl.NZ],
        weight_scale: pl.Tensor[[1, output_width], pl.FP32],
        y: pl.Out[pl.Tensor[[rows, output_width], pl.BF16]],
    ) -> pl.Tensor:
        for block in pl.spmd(output_width // n_tile, name_hint="w8a8_output_only"):
            pl.set_cache_policy(weight, pl.CachePolicy.BYPASS)
            for n0 in pl.unroll(0, output_width, n_tile):
                if block == n0 // n_tile:
                    acc = pl.create_tensor([rows, n_tile], dtype=pl.INT32)
                    for kb in pl.pipeline(HIDDEN // K_TILE, stage=2):
                        k0 = kb * K_TILE
                        lhs = pl.slice(x, [rows, K_TILE], [0, k0])
                        rhs = pl.reshape(pl.slice(weight, [1, K_TILE, n_tile], [0, k0, n0]), [K_TILE, n_tile])
                        acc = pl.matmul_acc(acc, lhs, rhs, init_cond=(kb == 0))
                    value = pl.cast(acc, pl.FP32, mode="rint")
                    if use_vector_epilogue:
                        expanded_scale = pl.col_expand_mul(
                            pl.full([rows, n_tile], dtype=pl.FP32, value=1.0),
                            weight_scale[:, n0 : n0 + n_tile],
                        )
                        scale_product = pl.row_expand_mul(expanded_scale, input_scale[:, :])
                        y[:, n0 : n0 + n_tile] = pl.cast(pl.mul(value, scale_product), pl.BF16, mode="rint")
                    else:
                        for row in pl.unroll(rows):
                            scale = pl.read(input_scale, [row, 0])
                            combined_scale = pl.mul(weight_scale[:, n0 : n0 + n_tile], scale)
                            y[row : row + 1, n0 : n0 + n_tile] = pl.cast(
                                pl.mul(value[row : row + 1, :], combined_scale), pl.BF16, mode="rint"
                            )
        return y

    return qkv_project if output_width == QKV_WIDTH else output_project
