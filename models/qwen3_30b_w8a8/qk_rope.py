# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -------------------------------------------------------------------------------------------------------
"""Q/K normalization and Neox RoPE on zero-copy flat head views."""

from typing import Any

import pypto.language as pl

from .linear import HEAD_DIM
from .quant import TOKENS_DYN

ROPE_POSITIONS = pl.dynamic("QWEN30_ROPE_POSITIONS")
PACKED_ROWS = pl.dynamic("QWEN30_PACKED_HEAD_ROWS")
Q_ROWS = pl.dynamic("QWEN30_Q_HEAD_ROWS")
KV_ROWS = pl.dynamic("QWEN30_KV_HEAD_ROWS")
EPS = 1e-6


def make_qk_norm_rope(*, fused: bool = False, inline: bool = False, diagnostics: bool = True) -> Any:
    """Select the observable native fusion boundary, without changing the unfused contract."""
    decorator = pl.jit.inline if inline else pl.jit

    @decorator
    def norm_rope(
        qkv: pl.Tensor[[PACKED_ROWS, HEAD_DIM], pl.BF16],
        q_weight: pl.Tensor[[1, HEAD_DIM], pl.BF16],
        k_weight: pl.Tensor[[1, HEAD_DIM], pl.BF16],
        positions: pl.Tensor[[TOKENS_DYN, 1], pl.INT64],
        cos_sin_cache: pl.Tensor[[ROPE_POSITIONS, HEAD_DIM], pl.BF16],
        q_norm: pl.Out[pl.Tensor[[Q_ROWS, HEAD_DIM], pl.BF16]],
        k_norm: pl.Out[pl.Tensor[[KV_ROWS, HEAD_DIM], pl.BF16]],
        q: pl.Out[pl.Tensor[[Q_ROWS, HEAD_DIM], pl.BF16]],
        k: pl.Out[pl.Tensor[[KV_ROWS, HEAD_DIM], pl.BF16]],
        v: pl.Out[pl.Tensor[[KV_ROWS, HEAD_DIM], pl.BF16]],
    ) -> tuple[pl.Tensor, pl.Tensor, pl.Tensor, pl.Tensor, pl.Tensor]:
        if not inline:
            qkv.bind_dynamic(0, PACKED_ROWS)
            positions.bind_dynamic(0, TOKENS_DYN)
            cos_sin_cache.bind_dynamic(0, ROPE_POSITIONS)
            q_norm.bind_dynamic(0, Q_ROWS)
            q.bind_dynamic(0, Q_ROWS)
            k_norm.bind_dynamic(0, KV_ROWS)
            k.bind_dynamic(0, KV_ROWS)
            v.bind_dynamic(0, KV_ROWS)
        tokens = pl.tensor.dim(positions, 0)
        for block in pl.spmd(tokens * 2, name_hint="q_norm_rope"):
            token = block // 2
            head = (block % 2) * 8
            packed_row = token * 20 + head
            output_row = token * 16 + head
            pos = pl.cast(pl.read(positions, [token, 0]), pl.INDEX)
            full = pl.cast(qkv[packed_row : packed_row + 8, :], pl.FP32)
            variance = pl.mul(pl.row_sum(pl.mul(full, full)), 1.0 / HEAD_DIM)
            inv = pl.rsqrt(pl.add(variance, EPS), high_precision=True)
            left = pl.cast(qkv[packed_row : packed_row + 8, :64], pl.FP32)
            right = pl.cast(qkv[packed_row : packed_row + 8, 64:], pl.FP32)
            left_norm = pl.cast(
                pl.col_expand_mul(pl.row_expand_mul(left, inv), pl.cast(q_weight[:, :64], pl.FP32)),
                pl.BF16,
                mode="rint",
            )
            right_norm = pl.cast(
                pl.col_expand_mul(pl.row_expand_mul(right, inv), pl.cast(q_weight[:, 64:], pl.FP32)),
                pl.BF16,
                mode="rint",
            )
            if diagnostics:
                q_norm[output_row : output_row + 8, :64] = left_norm
                q_norm[output_row : output_row + 8, 64:] = right_norm
            cos = pl.cast(cos_sin_cache[pos : pos + 1, :64], pl.FP32)
            sin = pl.cast(cos_sin_cache[pos : pos + 1, 64:], pl.FP32)
            if fused:
                left_f32 = pl.col_expand_mul(pl.row_expand_mul(left, inv), pl.cast(q_weight[:, :64], pl.FP32))
                right_f32 = pl.col_expand_mul(
                    pl.row_expand_mul(right, inv), pl.cast(q_weight[:, 64:], pl.FP32)
                )
            else:
                left_f32 = pl.cast(left_norm, pl.FP32)
                right_f32 = pl.cast(right_norm, pl.FP32)
            left_out = pl.cast(
                pl.sub(pl.col_expand_mul(left_f32, cos), pl.col_expand_mul(right_f32, sin)),
                pl.BF16,
                mode="rint",
            )
            right_out = pl.cast(
                pl.add(pl.col_expand_mul(right_f32, cos), pl.col_expand_mul(left_f32, sin)),
                pl.BF16,
                mode="rint",
            )
            q[output_row : output_row + 8, :64] = left_out
            q[output_row : output_row + 8, 64:] = right_out
        for token in pl.spmd(tokens, name_hint="k_norm_rope"):
            kv_packed_row = token * 20 + 16
            kv_output_row = token * 2
            kv_pos = pl.cast(pl.read(positions, [token, 0]), pl.INDEX)
            kv_full_valid = pl.cast(
                pl.slice(qkv, [8, HEAD_DIM], [kv_packed_row, 0], valid_shape=[2, HEAD_DIM]), pl.FP32
            )
            kv_full = pl.set_validshape(pl.fillpad(kv_full_valid, pad_value=pl.PadValue.zero), 8, HEAD_DIM)
            kv_variance = pl.mul(pl.row_sum(pl.mul(kv_full, kv_full)), 1.0 / HEAD_DIM)
            kv_inv = pl.rsqrt(pl.add(kv_variance, EPS), high_precision=True)
            kv_left_valid = pl.cast(pl.slice(qkv, [8, 64], [kv_packed_row, 0], valid_shape=[2, 64]), pl.FP32)
            kv_right_valid = pl.cast(
                pl.slice(qkv, [8, 64], [kv_packed_row, 64], valid_shape=[2, 64]), pl.FP32
            )
            kv_left = pl.set_validshape(pl.fillpad(kv_left_valid, pad_value=pl.PadValue.zero), 8, 64)
            kv_right = pl.set_validshape(pl.fillpad(kv_right_valid, pad_value=pl.PadValue.zero), 8, 64)
            kv_left_norm = pl.cast(
                pl.col_expand_mul(pl.row_expand_mul(kv_left, kv_inv), pl.cast(k_weight[:, :64], pl.FP32)),
                pl.BF16,
                mode="rint",
            )
            kv_right_norm = pl.cast(
                pl.col_expand_mul(pl.row_expand_mul(kv_right, kv_inv), pl.cast(k_weight[:, 64:], pl.FP32)),
                pl.BF16,
                mode="rint",
            )
            if diagnostics:
                k_norm = pl.assemble(k_norm, pl.set_validshape(kv_left_norm, 2, 64), [kv_output_row, 0])
                k_norm = pl.assemble(k_norm, pl.set_validshape(kv_right_norm, 2, 64), [kv_output_row, 64])
            kv_cos = pl.cast(cos_sin_cache[kv_pos : kv_pos + 1, :64], pl.FP32)
            kv_sin = pl.cast(cos_sin_cache[kv_pos : kv_pos + 1, 64:], pl.FP32)
            if fused:
                kv_left_f32 = pl.col_expand_mul(
                    pl.row_expand_mul(kv_left, kv_inv), pl.cast(k_weight[:, :64], pl.FP32)
                )
                kv_right_f32 = pl.col_expand_mul(
                    pl.row_expand_mul(kv_right, kv_inv), pl.cast(k_weight[:, 64:], pl.FP32)
                )
            else:
                kv_left_f32 = pl.cast(kv_left_norm, pl.FP32)
                kv_right_f32 = pl.cast(kv_right_norm, pl.FP32)
            kv_left_out = pl.cast(
                pl.sub(pl.col_expand_mul(kv_left_f32, kv_cos), pl.col_expand_mul(kv_right_f32, kv_sin)),
                pl.BF16,
                mode="rint",
            )
            kv_right_out = pl.cast(
                pl.add(pl.col_expand_mul(kv_right_f32, kv_cos), pl.col_expand_mul(kv_left_f32, kv_sin)),
                pl.BF16,
                mode="rint",
            )
            k = pl.assemble(k, pl.set_validshape(kv_left_out, 2, 64), [kv_output_row, 0])
            k = pl.assemble(k, pl.set_validshape(kv_right_out, 2, 64), [kv_output_row, 64])
            v[kv_output_row : kv_output_row + 2, :] = qkv[token * 20 + 18 : token * 20 + 20, :]
        return q_norm, k_norm, q, k, v

    return norm_rope


qk_norm_rope = make_qk_norm_rope()
qk_norm_rope_fused = make_qk_norm_rope(fused=True)


def make_a1_head_norm_rope(heads: int, *, storage_dtype: pl.DataType = pl.BF16) -> Any:
    """Normalize and rotate A1 heads using a selected 16-bit cosine/sine row.

    Four logical KV heads use eight physical input/output rows; consumers use
    the first four output rows. The caller supplies the selected cosine/sine
    row, with 64 cosine values followed by 64 sine values.
    """
    if isinstance(heads, bool) or not isinstance(heads, int) or heads not in (4, 8, 16, 32):
        raise ValueError(f"A1 head count must be 4, 8, 16 or 32, got {heads}")
    if storage_dtype not in (pl.FP16, pl.BF16):
        raise ValueError(f"A1 head storage must be FP16 or BF16, got {storage_dtype}")
    # tiling
    HEAD_TILE = 8
    HALF_TILE = HEAD_DIM // 2
    padded_heads = max(heads, HEAD_TILE)
    narrowing_mode = "none" if storage_dtype == pl.FP16 else "rint"
    round_rope_products = storage_dtype == pl.FP16

    @pl.jit.incore
    def head_norm_rope_a1(
        x: pl.Tensor[[padded_heads, HEAD_DIM], storage_dtype],
        weight: pl.Tensor[[1, HEAD_DIM], storage_dtype],
        cos_sin: pl.Tensor[[1, HEAD_DIM], storage_dtype],
        norm: pl.Out[pl.Tensor[[padded_heads, HEAD_DIM], storage_dtype]],
        rotated: pl.Out[pl.Tensor[[padded_heads, HEAD_DIM], storage_dtype]],
    ):
        gamma_left = pl.cast(pl.load(weight, [0, 0], [1, HALF_TILE]), pl.FP32, mode="none")
        gamma_right = pl.cast(pl.load(weight, [0, HALF_TILE], [1, HALF_TILE]), pl.FP32, mode="none")
        cosine = pl.cast(pl.load(cos_sin, [0, 0], [1, HALF_TILE]), pl.FP32, mode="none")
        sine = pl.cast(pl.load(cos_sin, [0, HALF_TILE], [1, HALF_TILE]), pl.FP32, mode="none")
        for base in pl.range(0, padded_heads, HEAD_TILE):
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
