# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -------------------------------------------------------------------------------------------------------
"""BF16 hidden-state RMSNorm boundaries for Qwen3-30B Decode."""

from typing import Any

import pypto.language as pl

from .config import HIDDEN
from .quant import TOKENS_DYN

RMS_EPS = 1e-6
NORM_ROWS = 8
NATIVE_REDUCE_LANES = 64


def make_rms_norm_hidden_impl(*, early: bool = False) -> Any:
    """Specialize the producer-side scheduling policy."""
    @pl.jit.inline
    def rms_norm_hidden_impl(
        x: pl.Tensor[[TOKENS_DYN, HIDDEN], pl.BF16],
        weight: pl.Tensor[[1, HIDDEN], pl.BF16],
        y: pl.Tensor[[TOKENS_DYN, HIDDEN], pl.BF16],
    ) -> pl.Tensor:
        tokens = pl.tensor.dim(x, 0)
        for block in pl.spmd((tokens + NORM_ROWS - 1) // NORM_ROWS, name_hint="hidden_rms_norm", allow_early_resolve=early):
            base = block * NORM_ROWS
            valid = pl.min(NORM_ROWS, tokens - base)
            values = pl.cast(pl.slice(x, [NORM_ROWS, HIDDEN], [base, 0], valid_shape=[valid, HIDDEN]), pl.FP32)
            variance = pl.mul(pl.row_sum(pl.mul(values, values)), 1.0 / HIDDEN)
            inv_rms = pl.rsqrt(pl.add(variance, RMS_EPS), high_precision=True)
            values = pl.cast(pl.slice(x, [NORM_ROWS, HIDDEN], [base, 0], valid_shape=[valid, HIDDEN]), pl.FP32)
            normalized = pl.row_expand_mul(values, inv_rms)
            normalized = pl.col_expand_mul(normalized, pl.cast(weight, pl.FP32))
            y = pl.assemble(y, pl.cast(normalized, pl.BF16, mode="rint"), [base, 0])
        return y

    return rms_norm_hidden_impl


rms_norm_hidden_impl = make_rms_norm_hidden_impl()


@pl.jit
def rms_norm_hidden(
    x: pl.Tensor[[TOKENS_DYN, HIDDEN], pl.BF16],
    weight: pl.Tensor[[1, HIDDEN], pl.BF16],
    y: pl.Out[pl.Tensor[[TOKENS_DYN, HIDDEN], pl.BF16]],
) -> pl.Tensor:
    x.bind_dynamic(0, TOKENS_DYN)
    y.bind_dynamic(0, TOKENS_DYN)
    return rms_norm_hidden_impl(x, weight, y)


def make_add_rms_norm_hidden_impl(*, early: bool = False) -> Any:
    """Specialize the producer-side scheduling policy."""
    @pl.jit.inline
    def add_rms_norm_hidden_impl(
        x: pl.Tensor[[TOKENS_DYN, HIDDEN], pl.BF16],
        residual: pl.Tensor[[TOKENS_DYN, HIDDEN], pl.BF16],
        weight: pl.Tensor[[1, HIDDEN], pl.BF16],
        y: pl.Tensor[[TOKENS_DYN, HIDDEN], pl.BF16],
        residual_out: pl.Tensor[[TOKENS_DYN, HIDDEN], pl.BF16],
    ) -> tuple[pl.Tensor, pl.Tensor]:
        tokens = pl.tensor.dim(x, 0)
        for block in pl.spmd((tokens + NORM_ROWS - 1) // NORM_ROWS, name_hint="hidden_add_rms_norm", allow_early_resolve=early):
            base = block * NORM_ROWS
            valid = pl.min(NORM_ROWS, tokens - base)
            values = pl.cast(pl.slice(x, [NORM_ROWS, HIDDEN], [base, 0], valid_shape=[valid, HIDDEN]), pl.FP32)
            skip = pl.cast(
                pl.slice(residual, [NORM_ROWS, HIDDEN], [base, 0], valid_shape=[valid, HIDDEN]), pl.FP32
            )
            summed = pl.cast(pl.cast(pl.add(values, skip), pl.BF16, mode="rint"), pl.FP32)
            residual_out = pl.assemble(residual_out, pl.cast(summed, pl.BF16, mode="rint"), [base, 0])
            # Match native SingleN: accumulate 64 lanes in column order, then reduce.
            scaled_square = pl.mul(pl.mul(summed, summed), 1.0 / HIDDEN)
            partial = pl.slice(scaled_square, [NORM_ROWS, NATIVE_REDUCE_LANES], [0, 0],
                               valid_shape=[valid, NATIVE_REDUCE_LANES])
            for column in pl.range(NATIVE_REDUCE_LANES, HIDDEN, NATIVE_REDUCE_LANES):
                partial = pl.add(partial, pl.slice(scaled_square, [NORM_ROWS, NATIVE_REDUCE_LANES],
                                                   [0, column], valid_shape=[valid, NATIVE_REDUCE_LANES]))
            variance = pl.row_sum(partial)
            root_mean_square = pl.sqrt(pl.add(variance, RMS_EPS))
            ones = pl.add(pl.mul(root_mean_square, 0.0), 1.0)
            inv_rms = pl.div(ones, root_mean_square)
            values = pl.cast(pl.slice(x, [NORM_ROWS, HIDDEN], [base, 0], valid_shape=[valid, HIDDEN]), pl.FP32)
            skip = pl.cast(
                pl.slice(residual, [NORM_ROWS, HIDDEN], [base, 0], valid_shape=[valid, HIDDEN]), pl.FP32
            )
            summed = pl.cast(pl.cast(pl.add(values, skip), pl.BF16, mode="rint"), pl.FP32)
            normalized = pl.cast(pl.cast(pl.row_expand_mul(summed, inv_rms), pl.BF16, mode="rint"), pl.FP32)
            normalized = pl.col_expand_mul(normalized, pl.cast(weight, pl.FP32))
            y = pl.assemble(y, pl.cast(normalized, pl.BF16, mode="rint"), [base, 0])
        return y, residual_out

    return add_rms_norm_hidden_impl


add_rms_norm_hidden_impl = make_add_rms_norm_hidden_impl()


@pl.jit
def add_rms_norm_hidden(
    x: pl.Tensor[[TOKENS_DYN, HIDDEN], pl.BF16],
    residual: pl.Tensor[[TOKENS_DYN, HIDDEN], pl.BF16],
    weight: pl.Tensor[[1, HIDDEN], pl.BF16],
    y: pl.Out[pl.Tensor[[TOKENS_DYN, HIDDEN], pl.BF16]],
    residual_out: pl.Out[pl.Tensor[[TOKENS_DYN, HIDDEN], pl.BF16]],
) -> tuple[pl.Tensor, pl.Tensor]:
    x.bind_dynamic(0, TOKENS_DYN)
    residual.bind_dynamic(0, TOKENS_DYN)
    y.bind_dynamic(0, TOKENS_DYN)
    residual_out.bind_dynamic(0, TOKENS_DYN)
    return add_rms_norm_hidden_impl(x, residual, weight, y, residual_out)
