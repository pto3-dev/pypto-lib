# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""One A1 task for independent Q/K head branches, with unchanged FP16 boundaries."""

from typing import Any

import pypto.language as pl


def make_fused_qk_norm_rope(*, write_norm: bool = True) -> Any:
    """Reuse the selected arithmetic and partition; add no cross-core barrier."""
    HALF_DIM = 64
    HEAD_DIM = 128
    HEAD_TILE = 8

    @pl.jit.inline
    def apply_qk_norm_rope(
        query: pl.Tensor[[640, 128], pl.FP16],
        query_weight: pl.Tensor[[1, 128], pl.FP16],
        cos_sin: pl.Tensor[[20, 128], pl.FP16],
        query_norm: pl.InOut[pl.Tensor[[640, 128], pl.FP16]],
        query_rotated: pl.InOut[pl.Tensor[[640, 128], pl.FP16]],
        key: pl.Tensor[[160, 128], pl.FP16],
        key_weight: pl.Tensor[[1, 128], pl.FP16],
        key_norm: pl.InOut[pl.Tensor[[160, 128], pl.FP16]],
        key_rotated: pl.InOut[pl.Tensor[[160, 128], pl.FP16]],
    ):
        q_gamma_l_half = pl.tile.load(query_weight, [0, 0], [1, HALF_DIM])
        q_gamma_r_half = pl.tile.load(query_weight, [0, HALF_DIM], [1, HALF_DIM])
        q_gamma_l = pl.tile.cast(q_gamma_l_half, pl.FP32, mode="none")
        q_gamma_r = pl.tile.cast(q_gamma_r_half, pl.FP32, mode="none")
        q_shape = pl.tile.create([HEAD_TILE, HALF_DIM], dtype=pl.FP32)
        q_expanded_gamma_l = pl.tile.col_expand(q_shape, q_gamma_l)
        q_expanded_gamma_r = pl.tile.col_expand(q_shape, q_gamma_r)
        k_gamma_l_half = pl.tile.load(key_weight, [0, 0], [1, HALF_DIM])
        k_gamma_r_half = pl.tile.load(key_weight, [0, HALF_DIM], [1, HALF_DIM])
        k_gamma_l = pl.tile.cast(k_gamma_l_half, pl.FP32, mode="none")
        k_gamma_r = pl.tile.cast(k_gamma_r_half, pl.FP32, mode="none")
        k_shape = pl.tile.create([HEAD_TILE, HALF_DIM], dtype=pl.FP32)
        k_expanded_gamma_l = pl.tile.col_expand(k_shape, k_gamma_l)
        k_expanded_gamma_r = pl.tile.col_expand(k_shape, k_gamma_r)
        block = pl.tile.get_block_idx()
        for task in pl.range(100):
            if task % 8 == block:
                if task < 80:
                    raw = pl.tile.load(query, [task * 8, 0], [HEAD_TILE, HEAD_DIM])
                    cos_half = pl.tile.load(cos_sin, [task // 4, 0], [1, HALF_DIM])
                    sin_half = pl.tile.load(cos_sin, [task // 4, HALF_DIM], [1, HALF_DIM])
                    expanded_gamma_l = q_expanded_gamma_l
                    expanded_gamma_r = q_expanded_gamma_r
                else:
                    raw = pl.tile.load(key, [(task % 20) * 8, 0], [HEAD_TILE, HEAD_DIM])
                    cos_half = pl.tile.load(cos_sin, [task % 20, 0], [1, HALF_DIM])
                    sin_half = pl.tile.load(cos_sin, [task % 20, HALF_DIM], [1, HALF_DIM])
                    expanded_gamma_l = k_expanded_gamma_l
                    expanded_gamma_r = k_expanded_gamma_r
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
                if write_norm:
                    if task < 80:
                        pl.tile.store(left_norm, [task * 8, 0], query_norm)
                        pl.tile.store(right_norm, [task * 8, HALF_DIM], query_norm)
                    else:
                        pl.tile.store(left_norm, [(task % 20) * 8, 0], key_norm)
                        pl.tile.store(right_norm, [(task % 20) * 8, HALF_DIM], key_norm)
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
                if task < 80:
                    pl.tile.store(rotated_l, [task * 8, 0], query_rotated)
                    pl.tile.store(rotated_r, [task * 8, HALF_DIM], query_rotated)
                else:
                    pl.tile.store(rotated_l, [(task % 20) * 8, 0], key_rotated)
                    pl.tile.store(rotated_r, [(task % 20) * 8, HALF_DIM], key_rotated)

    if write_norm:
        @pl.jit.incore
        def fused_qk_norm_rope(
            query: pl.Tensor[[640, 128], pl.FP16],
            query_weight: pl.Tensor[[1, 128], pl.FP16],
            cos_sin: pl.Tensor[[20, 128], pl.FP16],
            query_norm: pl.Out[pl.Tensor[[640, 128], pl.FP16]],
            query_rotated: pl.Out[pl.Tensor[[640, 128], pl.FP16]],
            key: pl.Tensor[[160, 128], pl.FP16],
            key_weight: pl.Tensor[[1, 128], pl.FP16],
            key_norm: pl.Out[pl.Tensor[[160, 128], pl.FP16]],
            key_rotated: pl.Out[pl.Tensor[[160, 128], pl.FP16]],
        ):
            apply_qk_norm_rope(
                query, query_weight, cos_sin, query_norm, query_rotated,
                key, key_weight, key_norm, key_rotated,
            )
            return query_norm, query_rotated, key_norm, key_rotated

        return fused_qk_norm_rope

    @pl.jit.incore
    def fused_qk_rotated(
        query: pl.Tensor[[640, 128], pl.FP16],
        query_weight: pl.Tensor[[1, 128], pl.FP16],
        cos_sin: pl.Tensor[[20, 128], pl.FP16],
        query_rotated: pl.Out[pl.Tensor[[640, 128], pl.FP16]],
        key: pl.Tensor[[160, 128], pl.FP16],
        key_weight: pl.Tensor[[1, 128], pl.FP16],
        key_rotated: pl.Out[pl.Tensor[[160, 128], pl.FP16]],
    ):
        # The shared body skips the Norm outputs in this specialization.
        apply_qk_norm_rope(
            query, query_weight, cos_sin, query_rotated, query_rotated,
            key, key_weight, key_rotated, key_rotated,
        )
        return query_rotated, key_rotated

    return fused_qk_rotated
