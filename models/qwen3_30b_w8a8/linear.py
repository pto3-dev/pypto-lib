# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -------------------------------------------------------------------------------------------------------
"""Rank-local W8A8 QKV and output projections using resident native NZ weights."""

from typing import Any

import pypto.language as pl

from .config import HIDDEN

HEAD_DIM = 128
LOCAL_Q_HEADS = 16
LOCAL_KV_HEADS = 2
Q_WIDTH = LOCAL_Q_HEADS * HEAD_DIM
KV_WIDTH = LOCAL_KV_HEADS * HEAD_DIM
QKV_WIDTH = Q_WIDTH + 2 * KV_WIDTH


def make_w8a8_projection(output_width: int, n_tile: int = 256, k_tile: int = 512, rows: int = 16) -> Any:
    """Specialize the projection and graph bucket shapes."""
    if not 1 <= rows <= 16:
        raise ValueError(f"Unsupported projection bucket: {rows}")
    if output_width % n_tile or HIDDEN % k_tile or n_tile % 32 or k_tile % 32:
        raise ValueError(f"Unsupported INT8 tiles: output={output_width}, N={n_tile}, K={k_tile}")

    @pl.jit
    def projection(
        x: pl.Tensor[[rows, HIDDEN], pl.INT8],
        input_scale: pl.Tensor[[rows, 1], pl.FP32],
        weight: pl.Tensor[[1, HIDDEN, output_width], pl.INT8, pl.NZ],
        weight_scale: pl.Tensor[[1, output_width], pl.FP32],
        acc_out: pl.Out[pl.Tensor[[rows, output_width], pl.INT32]],
        y: pl.Out[pl.Tensor[[rows, output_width], pl.BF16]],
    ) -> tuple[pl.Tensor, pl.Tensor]:
        for block in pl.spmd(output_width // n_tile, name_hint="w8a8_projection"):
            # Constant NZ offsets retain the frontend alignment/sign proof.
            for n0 in pl.unroll(0, output_width, n_tile):
                if block % (output_width // n_tile) == n0 // n_tile:
                    acc = pl.create_tensor([rows, n_tile], dtype=pl.INT32)
                    for kb in pl.pipeline(HIDDEN // k_tile, stage=2):
                        k0 = kb * k_tile
                        lhs = pl.slice(x, [rows, k_tile], [0, k0])
                        rhs = pl.reshape(pl.slice(weight, [1, k_tile, n_tile], [0, k0, n0]), [k_tile, n_tile])
                        acc = pl.matmul_acc(acc, lhs, rhs, init_cond=(kb == 0))
                    acc_out = pl.assemble(acc_out, acc, [0, n0])
                    value = pl.cast(acc, pl.FP32, mode="rint")
                    for row in pl.unroll(rows):
                        scale = pl.read(input_scale, [row, 0])
                        combined_scale = pl.mul(weight_scale[:, n0 : n0 + n_tile], scale)
                        result = pl.mul(value[row : row + 1, :], combined_scale)
                        y[row : row + 1, n0 : n0 + n_tile] = pl.cast(result, pl.BF16, mode="rint")
        return acc_out, y

    return projection


qkv_projection = make_w8a8_projection(QKV_WIDTH)
output_projection = make_w8a8_projection(HIDDEN)


def make_a1_w8a8_projection(output_width: int, input_width: int = HIDDEN) -> Any:
    """Specialize an A1 W8A8 projection with padded token rows and ND weights."""
    for name, width, alignment in (("output_width", output_width, 16), ("input_width", input_width, 32)):
        if isinstance(width, bool) or not isinstance(width, int) or width <= 0 or width % alignment:
            raise ValueError(f"A1 {name} must be a positive multiple of {alignment}, got {width}")

    # tiling
    M_TILE = 16
    N_TILE = 16
    K_TILE = 32

    @pl.jit.incore
    def projection_a1(
        x: pl.Tensor[[M_TILE, input_width], pl.INT8],
        input_scale: pl.Tensor[[1, M_TILE], pl.FP32],
        weight: pl.Tensor[[output_width, input_width], pl.INT8],
        weight_scale: pl.Tensor[[1, output_width], pl.FP32],
        acc_out: pl.Out[pl.Tensor[[M_TILE, output_width], pl.INT32]],
        y: pl.Out[pl.Tensor[[M_TILE, output_width], pl.BF16]],
    ):
        scale_row = pl.load(input_scale, [0, 0], [1, M_TILE])
        scale_column = pl.reshape(scale_row, [M_TILE, 1])
        for n0 in pl.range(0, output_width, N_TILE):
            lhs_first = pl.load(x, [0, 0], [M_TILE, K_TILE], target_memory=pl.MemorySpace.Mat)
            rhs_first = pl.load(weight, [n0, 0], [N_TILE, K_TILE], target_memory=pl.MemorySpace.Mat)
            acc = pl.matmul(lhs_first, pl.tile.transpose_view(rhs_first))
            for k0 in pl.range(K_TILE, input_width, K_TILE):
                lhs = pl.load(x, [0, k0], [M_TILE, K_TILE], target_memory=pl.MemorySpace.Mat)
                rhs = pl.load(weight, [n0, k0], [N_TILE, K_TILE], target_memory=pl.MemorySpace.Mat)
                acc = pl.matmul_acc(acc, lhs, pl.tile.transpose_view(rhs))
            pl.store(acc, [0, n0], acc_out)
            values = pl.cast(acc, pl.FP32, mode="none")
            channel_scale = pl.load(weight_scale, [0, n0], [1, N_TILE])
            scale_rows = pl.tile.row_expand(values, scale_column)
            scale_columns = pl.tile.col_expand(values, channel_scale)
            combined_scale = pl.mul(scale_columns, scale_rows)
            dequantized = pl.mul(values, combined_scale)
            rounded = pl.cast(dequantized, pl.BF16, mode="rint")
            pl.store(rounded, [0, n0], y)
        return acc_out, y

    return projection_a1
