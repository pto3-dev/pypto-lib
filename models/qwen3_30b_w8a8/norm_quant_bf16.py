# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""AddRMSNorm/Quant preserving the native unfused BF16 materialization boundary.

Unlike norm_quant.py's FP32 fused contract, this path rounds the residual sum,
normalized value before gamma, and weighted Norm output to BF16. Keeping that
output on chip is allowed; removing any of these rounding steps is not.
"""

from typing import Any

import pypto.language as pl

from .config import HIDDEN
from .norm import NATIVE_REDUCE_LANES, NORM_ROWS, RMS_EPS, add_rms_norm_hidden_impl
from .quant import INT8_MAX, dynamic_quantize_hidden_retained


def make_bf16_add_rms_norm_quant(
    rows: int, *, fused: bool = False, inline: bool = False, early: bool = False,
) -> Any:
    """Build a separate-task control or a retained-row single-task candidate."""
    if not 1 <= rows <= 16:
        raise ValueError(f"Expected BF16 norm bucket in [1,16], got {rows}")
    decorator = pl.jit.inline if inline else pl.jit
    variance_rows = NORM_ROWS

    if not fused:

        @decorator
        def bf16_norm_quant(
            x: pl.Tensor[[rows, HIDDEN], pl.BF16],
            residual: pl.Tensor[[rows, HIDDEN], pl.BF16],
            weight: pl.Tensor[[1, HIDDEN], pl.BF16],
            y: pl.Out[pl.Tensor[[rows, HIDDEN], pl.BF16]],
            residual_out: pl.Out[pl.Tensor[[rows, HIDDEN], pl.BF16]],
            quant: pl.Out[pl.Tensor[[rows, HIDDEN], pl.INT8]],
            scale: pl.Out[pl.Tensor[[rows, 1], pl.FP32]],
        ) -> tuple[pl.Tensor, pl.Tensor, pl.Tensor, pl.Tensor]:
            y_after, skip_after = add_rms_norm_hidden_impl(x, residual, weight, y, residual_out)
            quant_after, scale_after = dynamic_quantize_hidden_retained(y_after, quant, scale)
            return y_after, skip_after, quant_after, scale_after

        return bf16_norm_quant

    @decorator
    def bf16_norm_quant_fused(
        x: pl.Tensor[[rows, HIDDEN], pl.BF16],
        residual: pl.Tensor[[rows, HIDDEN], pl.BF16],
        weight: pl.Tensor[[1, HIDDEN], pl.BF16],
        y: pl.Out[pl.Tensor[[rows, HIDDEN], pl.BF16]],
        residual_out: pl.Out[pl.Tensor[[rows, HIDDEN], pl.BF16]],
        quant: pl.Out[pl.Tensor[[rows, HIDDEN], pl.INT8]],
        scale: pl.Out[pl.Tensor[[rows, 1], pl.FP32]],
    ) -> tuple[pl.Tensor, pl.Tensor, pl.Tensor, pl.Tensor]:
        # One owner writes compact scales, avoiding cross-core cache-line races.
        for _block in pl.spmd(1, name_hint="bf16_norm_quant_retained", allow_early_resolve=early):
            gamma = pl.cast(weight, pl.FP32, mode="none")
            for base in pl.unroll(0, rows, variance_rows):
                valid = pl.min(variance_rows, rows - base)
                # Accumulate the same 64 lanes in increasing column order.
                x_chunk = pl.slice(x, [variance_rows, NATIVE_REDUCE_LANES], [base, 0], valid_shape=[valid, NATIVE_REDUCE_LANES])
                skip_chunk = pl.slice(residual, [variance_rows, NATIVE_REDUCE_LANES], [base, 0], valid_shape=[valid, NATIVE_REDUCE_LANES])
                values = pl.cast(x_chunk, pl.FP32, mode="none")
                skip = pl.cast(skip_chunk, pl.FP32, mode="none")
                summed_bf16 = pl.cast(pl.add(values, skip), pl.BF16, mode="rint")
                residual_out[base : base + variance_rows, 0 : NATIVE_REDUCE_LANES] = summed_bf16
                summed = pl.cast(summed_bf16, pl.FP32, mode="none")
                partial = pl.mul(pl.mul(summed, summed), 1.0 / HIDDEN)
                for column in pl.range(NATIVE_REDUCE_LANES, HIDDEN, NATIVE_REDUCE_LANES):
                    chunk_x = pl.slice(x, [variance_rows, NATIVE_REDUCE_LANES], [base, column], valid_shape=[valid, NATIVE_REDUCE_LANES])
                    chunk_skip = pl.slice(residual, [variance_rows, NATIVE_REDUCE_LANES], [base, column], valid_shape=[valid, NATIVE_REDUCE_LANES])
                    chunk_x_fp32 = pl.cast(chunk_x, pl.FP32, mode="none")
                    chunk_skip_fp32 = pl.cast(chunk_skip, pl.FP32, mode="none")
                    chunk_sum = pl.add(chunk_x_fp32, chunk_skip_fp32)
                    chunk_bf16 = pl.cast(chunk_sum, pl.BF16, mode="rint")
                    residual_out[base : base + variance_rows, column : column + NATIVE_REDUCE_LANES] = chunk_bf16
                    chunk_fp32 = pl.cast(chunk_bf16, pl.FP32, mode="none")
                    chunk_square = pl.mul(pl.mul(chunk_fp32, chunk_fp32), 1.0 / HIDDEN)
                    partial = pl.add(partial, chunk_square)
                variance = pl.row_sum(partial)
                root_mean_square = pl.sqrt(pl.add(variance, RMS_EPS))
                ones = pl.add(pl.mul(root_mean_square, 0.0), 1.0)
                inverse_rms = pl.div(ones, root_mean_square)
                for offset in pl.range(valid):
                    token = base + offset
                    rms_value = pl.read(inverse_rms, [offset, 0])
                    row_x = pl.cast(x[token : token + 1, :], pl.FP32, mode="none")
                    row_skip = pl.cast(residual[token : token + 1, :], pl.FP32, mode="none")
                    row_sum_bf16 = pl.cast(pl.add(row_x, row_skip), pl.BF16, mode="rint")
                    row_sum = pl.cast(row_sum_bf16, pl.FP32, mode="none")
                    normalized_bf16 = pl.cast(pl.mul(row_sum, rms_value), pl.BF16, mode="rint")
                    weighted = pl.mul(pl.cast(normalized_bf16, pl.FP32, mode="none"), gamma)
                    output_bf16 = pl.cast(weighted, pl.BF16, mode="rint")
                    y = pl.assemble(y, output_bf16, [token, 0])
                    row = pl.cast(output_bf16, pl.FP32, mode="none")
                    maxima = pl.reshape(pl.row_max(pl.reshape(pl.abs(row), [8, HIDDEN // 8])), [1, 8])
                    maximum = pl.read(maxima, [0, 0])
                    for lane in pl.unroll(1, 8):
                        candidate = pl.read(maxima, [0, lane])
                        if candidate > maximum:
                            maximum = candidate
                    if maximum > 0.0:
                        pl.write(maxima, [0, 0], maximum)
                        inverses = pl.div(pl.full([1, 8], dtype=pl.FP32, value=INT8_MAX), maxima)
                        inverse = pl.read(inverses, [0, 0])
                        pl.write(scale, [token, 0], pl.div(1.0, inverse))
                        rounded = pl.cast(pl.mul(row, inverse), pl.INT32, mode="rint")
                        narrowed = pl.cast(pl.cast(rounded, pl.FP16, mode="none"), pl.INT8, mode="trunc")
                        quant = pl.assemble(quant, narrowed, [token, 0])
                    else:
                        zero_rounded = pl.cast(row, pl.INT32, mode="rint")
                        zero_half = pl.cast(zero_rounded, pl.FP16, mode="none")
                        zero_quant = pl.cast(zero_half, pl.INT8, mode="trunc")
                        quant = pl.assemble(quant, zero_quant, [token, 0])
                        pl.write(scale, [token, 0], 0.0)
        return y, residual_out, quant, scale

    return bf16_norm_quant_fused
