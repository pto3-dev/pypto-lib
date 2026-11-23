# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -------------------------------------------------------------------------------------------------------
"""Candidate native fused AddRMSNorm/DynamicQuant contract, not unfused RMSNorm.

The fused path keeps the residual sum and weighted normalization in FP32 until
quantization; only its residual output is BF16. Do not substitute this operator
where the native graph consumes a separately materialized BF16 norm output.
"""

import pypto.language as pl
from typing import Any

from .config import HIDDEN
from .norm import RMS_EPS, NORM_ROWS
from .quant import TOKENS_DYN, INT8_MAX, HIDDEN_TILE, SCALE_COLS_DYN, SCALE_ROW_WIDTH, pack_row_scales


@pl.jit.extern(core_type="aiv", source="scalar_rstd.cpp")
def scalar_rstd_kernel(
    mean_plus_eps: pl.Tensor[[16, 1], pl.FP32],
    out: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
    rows: pl.Scalar[pl.INDEX],
) -> pl.Tensor[[16, 1], pl.FP32]: ...


@pl.jit.inline
def add_rms_statistics_impl(
    x: pl.Tensor[[TOKENS_DYN, HIDDEN], pl.BF16],
    residual: pl.Tensor[[TOKENS_DYN, HIDDEN], pl.BF16],
    residual_out: pl.Tensor[[TOKENS_DYN, HIDDEN], pl.BF16],
) -> tuple[pl.Tensor, pl.Tensor]:
    tokens = pl.tensor.dim(x, 0)
    statistics = pl.create_tensor([16, 1], dtype=pl.FP32)
    rstd_buffer = pl.create_tensor([16, 1], dtype=pl.FP32)
    for _block in pl.spmd(1, name_hint="hidden_add_rms_statistics"):
        for base in pl.range(0, tokens, NORM_ROWS):
            valid = pl.min(NORM_ROWS, tokens - base)
            values = pl.cast(
                pl.slice(x, [NORM_ROWS, HIDDEN], [base, 0], valid_shape=[valid, HIDDEN]), pl.FP32
            )
            skip = pl.cast(
                pl.slice(residual, [NORM_ROWS, HIDDEN], [base, 0], valid_shape=[valid, HIDDEN]), pl.FP32
            )
            summed = pl.add(values, skip)
            residual_out = pl.assemble(residual_out, pl.cast(summed, pl.BF16, mode="rint"), [base, 0])
            # CANN 8.5.1 ReduceSumHalfInterval folds the row in half down to
            # 64 FP32 lanes, then performs WholeReduceSum on those lanes.
            squared = pl.mul(summed, summed)
            fold1024 = pl.add(
                pl.slice(squared, [NORM_ROWS, 1024], [0, 0], valid_shape=[valid, 1024]),
                pl.slice(squared, [NORM_ROWS, 1024], [0, 1024], valid_shape=[valid, 1024]),
            )
            fold512 = pl.add(
                pl.slice(fold1024, [NORM_ROWS, 512], [0, 0], valid_shape=[valid, 512]),
                pl.slice(fold1024, [NORM_ROWS, 512], [0, 512], valid_shape=[valid, 512]),
            )
            fold256 = pl.add(
                pl.slice(fold512, [NORM_ROWS, 256], [0, 0], valid_shape=[valid, 256]),
                pl.slice(fold512, [NORM_ROWS, 256], [0, 256], valid_shape=[valid, 256]),
            )
            fold128 = pl.add(
                pl.slice(fold256, [NORM_ROWS, 128], [0, 0], valid_shape=[valid, 128]),
                pl.slice(fold256, [NORM_ROWS, 128], [0, 128], valid_shape=[valid, 128]),
            )
            fold64 = pl.add(
                pl.slice(fold128, [NORM_ROWS, 64], [0, 0], valid_shape=[valid, 64]),
                pl.slice(fold128, [NORM_ROWS, 64], [0, 64], valid_shape=[valid, 64]),
            )
            variance = pl.mul(pl.row_sum(fold64), 1.0 / HIDDEN)
            statistics = pl.assemble(statistics, pl.add(variance, RMS_EPS), [base, 0])
    rstd_buffer = scalar_rstd_kernel(statistics, rstd_buffer, tokens)
    return rstd_buffer, residual_out


@pl.jit.inline
def add_rms_quant_hidden_impl(
    x: pl.Tensor[[TOKENS_DYN, HIDDEN], pl.BF16],
    residual: pl.Tensor[[TOKENS_DYN, HIDDEN], pl.BF16],
    weight: pl.Tensor[[1, HIDDEN], pl.BF16],
    quant: pl.Tensor[[TOKENS_DYN, HIDDEN], pl.INT8],
    scale: pl.Tensor[[TOKENS_DYN, 1], pl.FP32],
    residual_out: pl.Tensor[[TOKENS_DYN, HIDDEN], pl.BF16],
) -> tuple[pl.Tensor, pl.Tensor, pl.Tensor]:
    tokens = pl.tensor.dim(x, 0)
    rstd_buffer, residual_out = add_rms_statistics_impl(x, residual, residual_out)
    for _block in pl.spmd(1, name_hint="hidden_add_rms_quant"):
        for base in pl.range(0, tokens, NORM_ROWS):
            valid = pl.min(NORM_ROWS, tokens - base)
            inv_rms = pl.slice(rstd_buffer, [NORM_ROWS, 1], [base, 0], valid_shape=[valid, 1])
            # Preserve the full-width variance reduction, but bound normalized
            # value liveness to 512 columns. Recompute after the max reduction.
            amax = pl.full([1, NORM_ROWS], dtype=pl.FP32, value=0.0)
            for col in pl.range(0, HIDDEN, HIDDEN_TILE):
                values_chunk = pl.cast(
                    pl.slice(x, [NORM_ROWS, HIDDEN_TILE], [base, col], valid_shape=[valid, HIDDEN_TILE]),
                    pl.FP32,
                )
                skip_chunk = pl.cast(
                    pl.slice(
                        residual, [NORM_ROWS, HIDDEN_TILE], [base, col], valid_shape=[valid, HIDDEN_TILE]
                    ),
                    pl.FP32,
                )
                normalized_chunk = pl.row_expand_mul(pl.add(values_chunk, skip_chunk), inv_rms)
                normalized_chunk = pl.col_expand_mul(
                    normalized_chunk, pl.cast(weight[:, col : col + HIDDEN_TILE], pl.FP32)
                )
                chunk_max = pl.reshape(pl.row_max(pl.abs(normalized_chunk)), [1, NORM_ROWS])
                amax = pl.maximum(amax, chunk_max)
            for lane in pl.range(valid):
                token = base + lane
                max_value = pl.read(amax, [0, lane])
                inverse_value = pl.div(INT8_MAX, max_value)
                rms_value = pl.read(inv_rms, [lane, 0])
                pl.write(scale, [token, 0], pl.div(1.0, inverse_value))
                for col in pl.range(0, HIDDEN, HIDDEN_TILE):
                    row_x = pl.cast(x[token : token + 1, col : col + HIDDEN_TILE], pl.FP32)
                    row_skip = pl.cast(residual[token : token + 1, col : col + HIDDEN_TILE], pl.FP32)
                    row = pl.mul(pl.add(row_x, row_skip), rms_value)
                    row = pl.mul(row, pl.cast(weight[:, col : col + HIDDEN_TILE], pl.FP32))
                    scaled = pl.set_validshape(pl.mul(row, inverse_value), 1, HIDDEN_TILE)
                    rounded = pl.cast(scaled, pl.INT32, mode="rint")
                    quant[token : token + 1, col : col + HIDDEN_TILE] = pl.cast(
                        pl.cast(rounded, pl.FP16), pl.INT8, mode="trunc"
                    )
    return quant, scale, residual_out


@pl.jit.inline
def add_rms_quant_retained_rows_impl(
    x: pl.Tensor[[TOKENS_DYN, HIDDEN], pl.BF16],
    residual: pl.Tensor[[TOKENS_DYN, HIDDEN], pl.BF16],
    weight: pl.Tensor[[1, HIDDEN], pl.BF16],
    quant: pl.Tensor[[TOKENS_DYN, HIDDEN], pl.INT8],
    scale: pl.Tensor[[TOKENS_DYN, SCALE_COLS_DYN], pl.FP32],
    residual_out: pl.Tensor[[TOKENS_DYN, HIDDEN], pl.BF16],
    blocks: pl.Scalar[pl.INDEX],
) -> tuple[pl.Tensor, pl.Tensor, pl.Tensor]:
    tokens = pl.tensor.dim(x, 0)
    rstd_buffer, residual_out = add_rms_statistics_impl(x, residual, residual_out)
    for block in pl.spmd(blocks, name_hint="hidden_add_rms_quant_retained"):
        gamma = pl.cast(weight, pl.FP32)
        for token in pl.range(block, tokens, blocks):
            values = pl.cast(x[token : token + 1, :], pl.FP32)
            skip = pl.cast(residual[token : token + 1, :], pl.FP32)
            rms_value = pl.read(rstd_buffer, [token, 0])
            normalized = pl.mul(pl.mul(pl.add(values, skip), rms_value), gamma)
            maximum = pl.reshape(pl.row_max(pl.reshape(pl.abs(normalized), [8, HIDDEN // 8])), [1, 8])
            max_value = pl.read(maximum, [0, 0])
            for lane in pl.unroll(1, 8):
                candidate = pl.read(maximum, [0, lane])
                if candidate > max_value:
                    max_value = candidate
            inverse = pl.div(INT8_MAX, max_value)
            pl.write(scale, [token, 0], pl.div(1.0, inverse))
            rounded = pl.cast(pl.mul(normalized, inverse), pl.INT32, mode="rint")
            quant = pl.assemble(quant, pl.cast(pl.cast(rounded, pl.FP16), pl.INT8, mode="trunc"), [token, 0])
    return quant, scale, residual_out


@pl.jit.inline
def add_rms_quant_retained_impl(
    x: pl.Tensor[[TOKENS_DYN, HIDDEN], pl.BF16],
    residual: pl.Tensor[[TOKENS_DYN, HIDDEN], pl.BF16],
    weight: pl.Tensor[[1, HIDDEN], pl.BF16],
    quant: pl.Tensor[[TOKENS_DYN, HIDDEN], pl.INT8],
    scale: pl.Tensor[[TOKENS_DYN, 1], pl.FP32],
    residual_out: pl.Tensor[[TOKENS_DYN, HIDDEN], pl.BF16],
) -> tuple[pl.Tensor, pl.Tensor, pl.Tensor]:
    quant_after, scale_after, residual_after = add_rms_quant_retained_rows_impl(
        x, residual, weight, quant, scale, residual_out, 1
    )
    return quant_after, scale_after, residual_after


@pl.jit.inline
def add_rms_quant_parallel_impl(
    x: pl.Tensor[[TOKENS_DYN, HIDDEN], pl.BF16],
    residual: pl.Tensor[[TOKENS_DYN, HIDDEN], pl.BF16],
    weight: pl.Tensor[[1, HIDDEN], pl.BF16],
    quant: pl.Tensor[[TOKENS_DYN, HIDDEN], pl.INT8],
    scale: pl.Tensor[[TOKENS_DYN, 1], pl.FP32],
    residual_out: pl.Tensor[[TOKENS_DYN, HIDDEN], pl.BF16],
) -> tuple[pl.Tensor, pl.Tensor, pl.Tensor]:
    tokens = pl.tensor.dim(x, 0)
    # Each row task owns a separate 64-byte FP32 cache line. Compact scales
    # would make adjacent tasks write the same line; pack them after the join.
    padded = pl.create_tensor([tokens, SCALE_ROW_WIDTH], dtype=pl.FP32)
    quant_after, padded_after, residual_after = add_rms_quant_retained_rows_impl(
        x, residual, weight, quant, padded, residual_out, tokens
    )
    scale_after = pack_row_scales(padded_after, scale)
    return quant_after, scale_after, residual_after


def make_add_rms_norm_quant(
    rows: int, *, retain_normalized: bool = True, row_parallel: bool | None = None
) -> Any:
    """Specialize Decode; auto-select parallel rows only for measured B=16.

    Explicit True/False supports qualification and matched serial comparisons.
    The composed Attention chain selects its own measured B=8/16 policy.
    """
    if not 1 <= rows <= 16:
        raise ValueError(f"Unsupported fused norm bucket: {rows}")
    if row_parallel is None:
        row_parallel = retain_normalized and rows == 16
    if row_parallel and not retain_normalized:
        raise ValueError("Row-parallel norm requires the retained epilogue")
    implementation = add_rms_quant_retained_impl if retain_normalized else add_rms_quant_hidden_impl
    if row_parallel:
        implementation = add_rms_quant_parallel_impl

    @pl.jit
    def add_rms_norm_quant_hidden(
        x: pl.Tensor[[rows, HIDDEN], pl.BF16],
        residual: pl.Tensor[[rows, HIDDEN], pl.BF16],
        weight: pl.Tensor[[1, HIDDEN], pl.BF16],
        quant: pl.Out[pl.Tensor[[rows, HIDDEN], pl.INT8]],
        scale: pl.Out[pl.Tensor[[rows, 1], pl.FP32]],
        residual_out: pl.Out[pl.Tensor[[rows, HIDDEN], pl.BF16]],
    ) -> tuple[pl.Tensor, pl.Tensor, pl.Tensor]:
        return implementation(x, residual, weight, quant, scale, residual_out)

    return add_rms_norm_quant_hidden
