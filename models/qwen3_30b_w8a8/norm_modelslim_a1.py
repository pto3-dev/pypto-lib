# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""A1 FP16 ModelSlim RMSNorm and biased AddRMSNorm."""

from typing import Any

import pypto.language as pl


def make_a1_modelslim_norm(*, fused_add: bool, rows: int = 1, reduction_width: int = 64,
                           balanced_reduction: bool = False, resident_row: bool = False) -> Any:
    """Build B1 or eight-core B20 Norm with explicit FP16 materialization boundaries."""
    if rows not in (1, 20):
        raise ValueError(f"A1 ModelSlim Norm requires rows=1 or 20, got {rows}")
    if reduction_width not in (64, 128, 256, 512):
        raise ValueError(f"A1 ModelSlim Norm reduction width must be 64/128/256/512, got {reduction_width}")
    if balanced_reduction and reduction_width != 64:
        raise ValueError("A1 balanced Norm reduces 32 paired 64-lane chunks")
    if resident_row:
        if rows != 20 or not fused_add or balanced_reduction or reduction_width != 64:
            raise ValueError("A1 resident Norm requires B20 fused-add and ascending 64-lane reduction")
        return _make_a1_resident_norm()
    # model
    HIDDEN = 2048
    EPS = 1e-6
    # tiling
    ROW_TILE = 8
    REDUCE_TILE = reduction_width

    @pl.jit.incore
    def modelslim_norm_a1(
        x: pl.Tensor[[rows, HIDDEN], pl.FP16],
        residual: pl.Tensor[[rows, HIDDEN], pl.FP16],
        weight: pl.Tensor[[1, HIDDEN], pl.FP16],
        bias: pl.Tensor[[1, HIDDEN], pl.FP16],
        y: pl.Out[pl.Tensor[[rows, HIDDEN], pl.FP16]],
        residual_out: pl.Out[pl.Tensor[[rows, HIDDEN], pl.FP16]],
    ):
        for row in pl.range(rows):
            if row % 8 == pl.tile.get_block_idx():
                if balanced_reduction:
                    full = pl.cast(pl.load(x, [row, 0], [1, HIDDEN]), pl.FP32, mode="none")
                    if fused_add:
                        skip_full = pl.cast(pl.load(residual, [row, 0], [1, HIDDEN]), pl.FP32, mode="none")
                        full = pl.cast(pl.cast(pl.add(full, skip_full), pl.FP16, mode="none"), pl.FP32, mode="none")
                    squared = pl.mul(pl.mul(full, full), 1.0 / HIDDEN)
                    for stage in pl.unroll(5):
                        stride = 1024 // (1 << stage)
                        for lane in pl.range(stride):
                            left_value = pl.read(squared, [0, lane])
                            right_value = pl.read(squared, [0, lane + stride])
                            pl.write(squared, [0, lane], left_value + right_value)
                    balanced_partial = pl.tile.full([ROW_TILE, REDUCE_TILE], dtype=pl.FP32, value=0.0)
                    for lane in pl.range(REDUCE_TILE):
                        value = pl.read(squared, [0, lane])
                        pl.write(balanced_partial, [0, lane], value)
                    balanced_scratch = pl.tile.create([ROW_TILE, REDUCE_TILE], dtype=pl.FP32)
                    balanced_variance = pl.tile.row_sum(balanced_partial, balanced_scratch)
                    variance_scalar = pl.read(balanced_variance, [0, 0])
                else:
                    first_half = pl.load(x, [row, 0], [ROW_TILE, REDUCE_TILE], valid_shape=[1, REDUCE_TILE])
                    first = pl.cast(first_half, pl.FP32, mode="none")
                    if fused_add:
                        first_skip_half = pl.load(residual, [row, 0], [ROW_TILE, REDUCE_TILE], valid_shape=[1, REDUCE_TILE])
                        first_skip = pl.cast(first_skip_half, pl.FP32, mode="none")
                        first_sum = pl.cast(pl.add(first, first_skip), pl.FP16, mode="none")
                        first = pl.cast(first_sum, pl.FP32, mode="none")
                    partial = pl.mul(pl.mul(first, first), 1.0 / HIDDEN)
                    for column in pl.range(REDUCE_TILE, HIDDEN, REDUCE_TILE):
                        chunk_half = pl.load(x, [row, column], [ROW_TILE, REDUCE_TILE], valid_shape=[1, REDUCE_TILE])
                        chunk = pl.cast(chunk_half, pl.FP32, mode="none")
                        if fused_add:
                            skip_half = pl.load(residual, [row, column], [ROW_TILE, REDUCE_TILE], valid_shape=[1, REDUCE_TILE])
                            chunk_skip = pl.cast(skip_half, pl.FP32, mode="none")
                            chunk_sum = pl.cast(pl.add(chunk, chunk_skip), pl.FP16, mode="none")
                            chunk = pl.cast(chunk_sum, pl.FP32, mode="none")
                        partial = pl.add(partial, pl.mul(pl.mul(chunk, chunk), 1.0 / HIDDEN))
                    scratch = pl.tile.create([ROW_TILE, REDUCE_TILE], dtype=pl.FP32)
                    reduced_variance = pl.tile.row_sum(partial, scratch)
                    variance_scalar = pl.read(reduced_variance, [0, 0])
                variance = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0)
                pl.write(variance, [0, 0], variance_scalar)
                root = pl.sqrt(pl.add(variance, EPS))
                inverse = pl.div(pl.add(pl.mul(root, 0.0), 1.0), root)
                inverse_scalar = pl.read(inverse, [0, 0])
                values = pl.cast(pl.load(x, [row, 0], [1, HIDDEN]), pl.FP32, mode="none")
                if fused_add:
                    skip = pl.cast(pl.load(residual, [row, 0], [1, HIDDEN]), pl.FP32, mode="none")
                    summed_half = pl.cast(pl.add(values, skip), pl.FP16, mode="none")
                else:
                    summed_half = pl.load(x, [row, 0], [1, HIDDEN])
                pl.store(summed_half, [row, 0], residual_out)
                summed = pl.cast(summed_half, pl.FP32, mode="none")
                normalized = pl.mul(summed, inverse_scalar)
                if fused_add:
                    normalized_half = pl.cast(normalized, pl.FP16, mode="none")
                    normalized = pl.cast(normalized_half, pl.FP32, mode="none")
                gamma = pl.cast(pl.load(weight, [0, 0], [1, HIDDEN]), pl.FP32, mode="none")
                weighted_half = pl.cast(pl.mul(normalized, gamma), pl.FP16, mode="none")
                weighted = pl.cast(weighted_half, pl.FP32, mode="none")
                beta = pl.cast(pl.load(bias, [0, 0], [1, HIDDEN]), pl.FP32, mode="none")
                output = pl.cast(pl.add(weighted, beta), pl.FP16, mode="none")
                pl.store(output, [row, 0], y)
        return y, residual_out

    return modelslim_norm_a1



def _make_a1_resident_norm() -> Any:
    """Build B20 residual addition and Norm over resident FP16 rows."""
    # model
    HIDDEN = 2048
    EPS = 1e-6

    @pl.jit.incore
    def modelslim_resident_norm_a1(
        x: pl.Tensor[[20, HIDDEN], pl.FP16],
        residual: pl.Tensor[[20, HIDDEN], pl.FP16],
        weight: pl.Tensor[[1, HIDDEN], pl.FP16],
        bias: pl.Tensor[[1, HIDDEN], pl.FP16],
        y: pl.Out[pl.Tensor[[20, HIDDEN], pl.FP16]],
        residual_out: pl.Out[pl.Tensor[[20, HIDDEN], pl.FP16]],
    ):
        gamma_half = pl.tile.load(weight, [0, 0], [1, HIDDEN])
        beta_half = pl.tile.load(bias, [0, 0], [1, HIDDEN])
        gamma = pl.tile.cast(gamma_half, pl.FP32, mode="none")
        beta = pl.tile.cast(beta_half, pl.FP32, mode="none")
        block = pl.tile.get_block_idx()
        for row in pl.range(20):
            if row % 8 == block:
                input_half = pl.tile.load(x, [row, 0], [1, HIDDEN])
                skip_half = pl.tile.load(residual, [row, 0], [1, HIDDEN])
                input_float = pl.tile.cast(input_half, pl.FP32, mode="none")
                skip_float = pl.tile.cast(skip_half, pl.FP32, mode="none")
                full_sum = pl.tile.add(input_float, skip_float)
                summed_half = pl.tile.cast(full_sum, pl.FP16, mode="none")
                values = pl.tile.cast(summed_half, pl.FP32, mode="none")
                square = pl.tile.mul(values, values)
                squared = pl.tile.muls(square, 1.0 / HIDDEN)
                scratch = pl.tile.create([1, HIDDEN], dtype=pl.FP32)
                reduced = pl.tile.row_sum(squared, scratch)
                variance_scalar = pl.tile.read(reduced, [0, 0])
                variance = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0)
                pl.tile.write(variance, [0, 0], variance_scalar)
                adjusted = pl.tile.adds(variance, EPS)
                root = pl.tile.sqrt(adjusted)
                zeros = pl.tile.muls(root, 0.0)
                ones = pl.tile.adds(zeros, 1.0)
                inverse = pl.tile.div(ones, root)
                inverse_scalar = pl.tile.read(inverse, [0, 0])
                pl.tile.store(summed_half, [row, 0], residual_out)
                normalized = pl.tile.muls(values, inverse_scalar)
                normalized_half = pl.tile.cast(normalized, pl.FP16, mode="none")
                normalized_float = pl.tile.cast(normalized_half, pl.FP32, mode="none")
                weighted = pl.tile.mul(normalized_float, gamma)
                weighted_half = pl.tile.cast(weighted, pl.FP16, mode="none")
                weighted_float = pl.tile.cast(weighted_half, pl.FP32, mode="none")
                biased = pl.tile.add(weighted_float, beta)
                output = pl.tile.cast(biased, pl.FP16, mode="none")
                pl.tile.store(output, [row, 0], y)
        return y, residual_out

    return modelslim_resident_norm_a1
