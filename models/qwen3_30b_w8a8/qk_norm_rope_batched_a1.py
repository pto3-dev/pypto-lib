# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""Batched A1 head processing with unchanged native FP16 arithmetic boundaries."""

from typing import Any

import pypto.language as pl

from .linear import HEAD_DIM

EPS = 1e-6


def make_a1_batched_head_norm_rope(heads: int, *, storage_dtype: pl.DataType = pl.BF16, rows: int = 1,
                                   resident_rows: bool = False) -> Any:
    """Normalize and rotate A1 heads using a selected 16-bit cosine/sine row.

    Four logical KV heads use eight physical input/output rows; consumers use
    the first four output rows. The caller supplies the selected cosine/sine
    row, with 64 cosine values followed by 64 sine values.
    """
    if isinstance(heads, bool) or not isinstance(heads, int) or heads not in (4, 8, 16, 32):
        raise ValueError(f"A1 head count must be 4, 8, 16 or 32, got {heads}")
    if storage_dtype not in (pl.FP16, pl.BF16):
        raise ValueError(f"A1 head storage must be FP16 or BF16, got {storage_dtype}")
    if rows not in (1, 20):
        raise ValueError(f"A1 head batching requires rows=1 or 20, got {rows}")
    if resident_rows:
        if storage_dtype != pl.FP16 or rows != 20 or heads not in (4, 32):
            raise ValueError("Resident direct heads require B20 FP16 and four or 32 heads")
        return _make_a1_resident_head_norm_rope(heads)
    # tiling
    HEAD_TILE = 8
    HALF_TILE = HEAD_DIM // 2
    padded_heads = max(heads, HEAD_TILE)
    narrowing_mode = "none" if storage_dtype == pl.FP16 else "rint"
    round_rope_products = storage_dtype == pl.FP16

    @pl.jit.incore
    def head_norm_rope_a1(
        x: pl.Tensor[[rows * padded_heads, HEAD_DIM], storage_dtype],
        weight: pl.Tensor[[1, HEAD_DIM], storage_dtype],
        cos_sin: pl.Tensor[[rows, HEAD_DIM], storage_dtype],
        norm: pl.Out[pl.Tensor[[rows * padded_heads, HEAD_DIM], storage_dtype]],
        rotated: pl.Out[pl.Tensor[[rows * padded_heads, HEAD_DIM], storage_dtype]],
    ):
        rope_rows = pl.cast(pl.load(cos_sin, [0, 0], [rows, HEAD_DIM]), pl.FP32, mode="none")
        for base in pl.range(0, rows * padded_heads, HEAD_TILE):
            if rows == 1 or (base // HEAD_TILE) % 8 == pl.tile.get_block_idx():
                row = base // padded_heads
                gamma_left = pl.cast(pl.load(weight, [0, 0], [1, HALF_TILE]), pl.FP32, mode="none")
                gamma_right = pl.cast(pl.load(weight, [0, HALF_TILE], [1, HALF_TILE]), pl.FP32, mode="none")
                cosine = pl.tile.create([1, HALF_TILE], dtype=pl.FP32)
                sine = pl.tile.create([1, HALF_TILE], dtype=pl.FP32)
                for lane in pl.range(HALF_TILE):
                    pl.write(cosine, [0, lane], pl.read(rope_rows, [row, lane]))
                    pl.write(sine, [0, lane], pl.read(rope_rows, [row, lane + HALF_TILE]))
                full_bf16 = pl.load(x, [base, 0], [HEAD_TILE, HEAD_DIM])
                full = pl.cast(full_bf16, pl.FP32, mode="none")
                square = pl.mul(full, full)
                scratch = pl.tile.create([HEAD_TILE, HEAD_DIM], dtype=pl.FP32)
                variance = pl.mul(pl.tile.row_sum(square, scratch), 1.0 / HEAD_DIM)
                root = pl.sqrt(pl.add(variance, EPS))
                ones = pl.add(pl.mul(root, 0.0), 1.0)
                inverse = pl.div(ones, root)
                left_bf16 = pl.load(x, [base, 0], [HEAD_TILE, HALF_TILE])
                right_bf16 = pl.load(x, [base, HALF_TILE], [HEAD_TILE, HALF_TILE])
                left = pl.cast(left_bf16, pl.FP32, mode="none")
                right = pl.cast(right_bf16, pl.FP32, mode="none")
                inverse_expanded = pl.tile.row_expand(left, inverse)
                left_gamma = pl.tile.col_expand(left, gamma_left)
                right_gamma = pl.tile.col_expand(right, gamma_right)
                left_normalized = pl.mul(left, inverse_expanded)
                right_normalized = pl.mul(right, inverse_expanded)
                left_weighted = pl.mul(left_normalized, left_gamma)
                right_weighted = pl.mul(right_normalized, right_gamma)
                left_norm = pl.cast(left_weighted, storage_dtype, mode=narrowing_mode)
                right_norm = pl.cast(right_weighted, storage_dtype, mode=narrowing_mode)
                pl.store(left_norm, [base, 0], norm)
                pl.store(right_norm, [base, HALF_TILE], norm)
                left_rounded = pl.cast(left_norm, pl.FP32, mode="none")
                right_rounded = pl.cast(right_norm, pl.FP32, mode="none")
                cosine_expanded = pl.tile.col_expand(left_rounded, cosine)
                sine_expanded = pl.tile.col_expand(right_rounded, sine)
                left_cos = pl.mul(left_rounded, cosine_expanded)
                right_sin = pl.mul(right_rounded, sine_expanded)
                right_cos = pl.mul(right_rounded, cosine_expanded)
                left_sin = pl.mul(left_rounded, sine_expanded)
                if round_rope_products:
                    left_cos_half = pl.cast(left_cos, pl.FP16, mode="none")
                    right_sin_half = pl.cast(right_sin, pl.FP16, mode="none")
                    right_cos_half = pl.cast(right_cos, pl.FP16, mode="none")
                    left_sin_half = pl.cast(left_sin, pl.FP16, mode="none")
                    left_cos = pl.cast(left_cos_half, pl.FP32, mode="none")
                    right_sin = pl.cast(right_sin_half, pl.FP32, mode="none")
                    right_cos = pl.cast(right_cos_half, pl.FP32, mode="none")
                    left_sin = pl.cast(left_sin_half, pl.FP32, mode="none")
                left_rotated = pl.cast(pl.sub(left_cos, right_sin), storage_dtype, mode=narrowing_mode)
                right_rotated = pl.cast(pl.add(right_cos, left_sin), storage_dtype, mode=narrowing_mode)
                pl.store(left_rotated, [base, 0], rotated)
                pl.store(right_rotated, [base, HALF_TILE], rotated)
        return norm, rotated

    return head_norm_rope_a1


def _make_a1_resident_head_norm_rope(heads: int) -> Any:
    """Preserve the selected B20 FP16 head ABI and every narrowing boundary."""
    if isinstance(heads, bool) or not isinstance(heads, int) or heads not in (4, 32):
        raise ValueError(f"Expected four KV heads or 32 query heads, got {heads}")
    padded_heads = max(heads, 8)
    HEAD_DIM = 128
    HALF_DIM = 64
    HEAD_TILE = 8
    ROWS = 20

    @pl.jit.incore
    def resident_head_norm_rope(
        x: pl.Tensor[[ROWS * padded_heads, HEAD_DIM], pl.FP16],
        weight: pl.Tensor[[1, HEAD_DIM], pl.FP16],
        cos_sin: pl.Tensor[[ROWS, HEAD_DIM], pl.FP16],
        norm: pl.Out[pl.Tensor[[ROWS * padded_heads, HEAD_DIM], pl.FP16]],
        rotated: pl.Out[pl.Tensor[[ROWS * padded_heads, HEAD_DIM], pl.FP16]],
    ):
        gamma_l_half = pl.tile.load(weight, [0, 0], [1, HALF_DIM])
        gamma_r_half = pl.tile.load(weight, [0, HALF_DIM], [1, HALF_DIM])
        gamma_l = pl.tile.cast(gamma_l_half, pl.FP32, mode="none")
        gamma_r = pl.tile.cast(gamma_r_half, pl.FP32, mode="none")
        shape = pl.tile.create([HEAD_TILE, HALF_DIM], dtype=pl.FP32)
        expanded_gamma_l = pl.tile.col_expand(shape, gamma_l)
        expanded_gamma_r = pl.tile.col_expand(shape, gamma_r)
        block = pl.tile.get_block_idx()
        for row_base in pl.range(0, ROWS * padded_heads, padded_heads):
            row = row_base // padded_heads
            for head_base in pl.range(0, padded_heads, HEAD_TILE):
                task = row_base // HEAD_TILE + head_base // HEAD_TILE
                if task % 8 == block:
                    base = row_base + head_base
                    raw = pl.tile.load(x, [base, 0], [HEAD_TILE, HEAD_DIM])
                    cos_half = pl.tile.load(cos_sin, [row, 0], [1, HALF_DIM])
                    sin_half = pl.tile.load(cos_sin, [row, HALF_DIM], [1, HALF_DIM])
                    full = pl.tile.cast(raw, pl.FP32, mode="none")
                    cosine = pl.tile.cast(cos_half, pl.FP32, mode="none")
                    sine = pl.tile.cast(sin_half, pl.FP32, mode="none")
                    square = pl.tile.mul(full, full)
                    scratch = pl.tile.create([HEAD_TILE, HEAD_DIM], dtype=pl.FP32)
                    reduced = pl.tile.row_sum(square, scratch)
                    mean = pl.tile.muls(reduced, 1.0 / HEAD_DIM)
                    adjusted = pl.tile.adds(mean, 1e-6)
                    root = pl.tile.sqrt(adjusted)
                    zero = pl.tile.muls(root, 0.0)
                    ones = pl.tile.adds(zero, 1.0)
                    inverse = pl.tile.div(ones, root)
                    half_l = pl.tile.extract(
                        raw, 0, 0, [HEAD_TILE, HALF_DIM], target_memory=pl.Mem.Vec
                    )
                    half_r = pl.tile.extract(
                        raw,
                        0,
                        HALF_DIM,
                        [HEAD_TILE, HALF_DIM],
                        target_memory=pl.Mem.Vec,
                    )
                    left = pl.tile.cast(half_l, pl.FP32, mode="none")
                    right = pl.tile.cast(half_r, pl.FP32, mode="none")
                    expanded_inverse = pl.tile.row_expand(left, inverse)
                    expanded_cosine = pl.tile.col_expand(left, cosine)
                    expanded_sine = pl.tile.col_expand(right, sine)
                    left_scaled = pl.tile.mul(left, expanded_inverse)
                    right_scaled = pl.tile.mul(right, expanded_inverse)
                    left_weighted = pl.tile.mul(left_scaled, expanded_gamma_l)
                    right_weighted = pl.tile.mul(right_scaled, expanded_gamma_r)
                    left_norm = pl.tile.cast(left_weighted, pl.FP16, mode="none")
                    right_norm = pl.tile.cast(right_weighted, pl.FP16, mode="none")
                    pl.tile.store(left_norm, [base, 0], norm)
                    pl.tile.store(right_norm, [base, HALF_DIM], norm)
                    rounded_l = pl.tile.cast(left_norm, pl.FP32, mode="none")
                    rounded_r = pl.tile.cast(right_norm, pl.FP32, mode="none")
                    left_cos = pl.tile.mul(rounded_l, expanded_cosine)
                    right_sin = pl.tile.mul(rounded_r, expanded_sine)
                    right_cos = pl.tile.mul(rounded_r, expanded_cosine)
                    left_sin = pl.tile.mul(rounded_l, expanded_sine)
                    left_cos_half = pl.tile.cast(left_cos, pl.FP16, mode="none")
                    right_sin_half = pl.tile.cast(right_sin, pl.FP16, mode="none")
                    right_cos_half = pl.tile.cast(right_cos, pl.FP16, mode="none")
                    left_sin_half = pl.tile.cast(left_sin, pl.FP16, mode="none")
                    left_cos_rounded = pl.tile.cast(left_cos_half, pl.FP32, mode="none")
                    right_sin_rounded = pl.tile.cast(
                        right_sin_half, pl.FP32, mode="none"
                    )
                    right_cos_rounded = pl.tile.cast(
                        right_cos_half, pl.FP32, mode="none"
                    )
                    left_sin_rounded = pl.tile.cast(left_sin_half, pl.FP32, mode="none")
                    rotated_l = pl.tile.cast(
                        pl.tile.sub(left_cos_rounded, right_sin_rounded),
                        pl.FP16,
                        mode="none",
                    )
                    rotated_r = pl.tile.cast(
                        pl.tile.add(right_cos_rounded, left_sin_rounded),
                        pl.FP16,
                        mode="none",
                    )
                    pl.tile.store(rotated_l, [base, 0], rotated)
                    pl.tile.store(rotated_r, [base, HALF_DIM], rotated)
        return norm, rotated

    return resident_head_norm_rope
