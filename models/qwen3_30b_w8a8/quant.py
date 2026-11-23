# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -------------------------------------------------------------------------------------------------------
"""Projection-input DynamicQuant; MoE routing quantization has a separate contract."""

from typing import Any

import pypto.language as pl

from .config import HIDDEN

TOKENS_DYN = pl.dynamic("QWEN30_QUANT_TOKENS")
SCALE_COLS_DYN = pl.dynamic("QWEN30_SCALE_COLS")
SCALE_ROW_WIDTH = 16
INT8_MAX = 127.0
TOKEN_TILE = 8
HIDDEN_TILE = 512
RETAINED_ROWS = 8


@pl.jit.inline
def dynamic_quantize_hidden(
    x: pl.Tensor[[TOKENS_DYN, HIDDEN], pl.BF16],
    quant: pl.Tensor[[TOKENS_DYN, HIDDEN], pl.INT8],
    scale: pl.Tensor[[TOKENS_DYN, 1], pl.FP32],
) -> tuple[pl.Tensor, pl.Tensor]:
    tokens = pl.tensor.dim(x, 0)
    for block in pl.spmd(1, name_hint="hidden_dynamic_quant"):
        for base in pl.range(block * TOKEN_TILE, tokens, TOKEN_TILE):
            valid_rows = pl.min(TOKEN_TILE, tokens - base)
            amax = pl.set_validshape(pl.full([1, TOKEN_TILE], dtype=pl.FP32, value=0.0), 1, valid_rows)
            for hidden_base in pl.range(0, HIDDEN, HIDDEN_TILE):
                rows = pl.cast(
                    pl.slice(
                        x,
                        [TOKEN_TILE, HIDDEN_TILE],
                        [base, hidden_base],
                        valid_shape=[valid_rows, HIDDEN_TILE],
                    ),
                    pl.FP32,
                )
                chunk_max = pl.reshape(pl.row_max(pl.abs(rows)), [1, TOKEN_TILE])
                amax = pl.maximum(amax, chunk_max)
            numerator = pl.set_validshape(
                pl.full([1, TOKEN_TILE], dtype=pl.FP32, value=INT8_MAX), 1, valid_rows
            )
            inverse_scale = pl.div(numerator, amax)
            for lane in pl.range(valid_rows):
                token = base + lane
                max_value = pl.read(amax, [0, lane])
                if max_value > 0.0:
                    inverse_value = pl.read(inverse_scale, [0, lane])
                    stored_scale = pl.div(1.0, inverse_value)
                    pl.write(scale, [token, 0], stored_scale)
                    for hidden_base in pl.range(0, HIDDEN, HIDDEN_TILE):
                        row = pl.cast(x[token : token + 1, hidden_base : hidden_base + HIDDEN_TILE], pl.FP32)
                        row = pl.set_validshape(row, 1, HIDDEN_TILE)
                        normalized = pl.mul(row, inverse_value)
                        # A2/A3 FP32 -> INT8 lowers through FP16. Round before
                        # narrowing so FP16 cannot turn near-half values into ties.
                        rounded = pl.cast(normalized, pl.INT32, mode="rint")
                        quant[token : token + 1, hidden_base : hidden_base + HIDDEN_TILE] = pl.cast(
                            pl.cast(rounded, pl.FP16), pl.INT8, mode="trunc"
                        )
                else:
                    zero_row = pl.cast(x[token : token + 1, :], pl.FP32)
                    zero_row = pl.set_validshape(zero_row, 1, HIDDEN)
                    quant[token : token + 1, :] = pl.cast(zero_row, pl.INT8, mode="rint")
                    pl.write(scale, [token, 0], 0.0)
    return quant, scale


@pl.jit
def projection_input_quant(
    x: pl.Tensor[[TOKENS_DYN, HIDDEN], pl.BF16],
    quant: pl.Out[pl.Tensor[[TOKENS_DYN, HIDDEN], pl.INT8]],
    scale: pl.Out[pl.Tensor[[TOKENS_DYN, 1], pl.FP32]],
) -> tuple[pl.Tensor, pl.Tensor]:
    x.bind_dynamic(0, TOKENS_DYN)
    quant.bind_dynamic(0, TOKENS_DYN)
    scale.bind_dynamic(0, TOKENS_DYN)
    return dynamic_quantize_hidden(x, quant, scale)


def make_dynamic_quantize_hidden_retained_rows(*, early: bool = False) -> Any:
    """Specialize the producer-side scheduling policy."""
    @pl.jit.inline
    def dynamic_quantize_hidden_retained_rows(
        x: pl.Tensor[[TOKENS_DYN, HIDDEN], pl.BF16],
        quant: pl.Tensor[[TOKENS_DYN, HIDDEN], pl.INT8],
        scale: pl.Tensor[[TOKENS_DYN, SCALE_COLS_DYN], pl.FP32],
        blocks: pl.Scalar[pl.INDEX],
    ) -> tuple[pl.Tensor, pl.Tensor]:
        """Retain one FP32 input row through amax and quantization; preserve vector division."""
        tokens = pl.tensor.dim(x, 0)
        for block in pl.spmd(blocks, name_hint="hidden_dynamic_quant_retained", allow_early_resolve=early):
            for token in pl.range(block, tokens, blocks):
                row = pl.cast(x[token : token + 1, :], pl.FP32)
                row = pl.set_validshape(row, 1, HIDDEN)
                amax = pl.reshape(pl.row_max(pl.reshape(pl.abs(row), [8, HIDDEN // 8])), [1, 8])
                max_value = pl.read(amax, [0, 0])
                for lane in pl.unroll(1, 8):
                    candidate = pl.read(amax, [0, lane])
                    if candidate > max_value:
                        max_value = candidate
                if max_value > 0.0:
                    # Keep division on the vector unit, unlike fused AddRMSNorm's scalar division.
                    pl.write(amax, [0, 0], max_value)
                    inverse_scale = pl.div(pl.full([1, 8], dtype=pl.FP32, value=INT8_MAX), amax)
                    inverse_value = pl.read(inverse_scale, [0, 0])
                    pl.write(scale, [token, 0], pl.div(1.0, inverse_value))
                    rounded = pl.cast(pl.mul(row, inverse_value), pl.INT32, mode="rint")
                    quant[token : token + 1, :] = pl.cast(pl.cast(rounded, pl.FP16), pl.INT8, mode="trunc")
                else:
                    quant[token : token + 1, :] = pl.cast(row, pl.INT8, mode="rint")
                    pl.write(scale, [token, 0], 0.0)
        return quant, scale

    return dynamic_quantize_hidden_retained_rows


dynamic_quantize_hidden_retained_rows = make_dynamic_quantize_hidden_retained_rows()


@pl.jit.inline
def pack_row_scales(
    padded: pl.Tensor[[TOKENS_DYN, SCALE_ROW_WIDTH], pl.FP32],
    compact: pl.Tensor[[TOKENS_DYN, 1], pl.FP32],
) -> pl.Tensor:
    tokens = pl.tensor.dim(compact, 0)
    for _block in pl.spmd(1, name_hint="pack_row_scales"):
        for token in pl.range(tokens):
            pl.write(compact, [token, 0], pl.read(padded, [token, 0]))
    return compact


def make_dynamic_quantize_hidden_retained(*, early: bool = False) -> Any:
    """Specialize the producer-side scheduling policy."""
    quantize_rows = make_dynamic_quantize_hidden_retained_rows(early=early)
    @pl.jit.inline
    def dynamic_quantize_hidden_retained(
        x: pl.Tensor[[TOKENS_DYN, HIDDEN], pl.BF16],
        quant: pl.Tensor[[TOKENS_DYN, HIDDEN], pl.INT8],
        scale: pl.Tensor[[TOKENS_DYN, 1], pl.FP32],
    ) -> tuple[pl.Tensor, pl.Tensor]:
        quant_after, scale_after = quantize_rows(x, quant, scale, 1)
        return quant_after, scale_after

    return dynamic_quantize_hidden_retained


dynamic_quantize_hidden_retained = make_dynamic_quantize_hidden_retained()


@pl.jit.inline
def dynamic_quantize_hidden_parallel(
    x: pl.Tensor[[TOKENS_DYN, HIDDEN], pl.BF16],
    quant: pl.Tensor[[TOKENS_DYN, HIDDEN], pl.INT8],
    scale: pl.Tensor[[TOKENS_DYN, 1], pl.FP32],
) -> tuple[pl.Tensor, pl.Tensor]:
    tokens = pl.tensor.dim(x, 0)
    padded = pl.create_tensor([tokens, SCALE_ROW_WIDTH], dtype=pl.FP32)
    quant_after, padded_after = dynamic_quantize_hidden_retained_rows(x, quant, padded, tokens)
    scale_after = pack_row_scales(padded_after, scale)
    return quant_after, scale_after


@pl.jit
def projection_input_quant_retained(
    x: pl.Tensor[[TOKENS_DYN, HIDDEN], pl.BF16],
    quant: pl.Out[pl.Tensor[[TOKENS_DYN, HIDDEN], pl.INT8]],
    scale: pl.Out[pl.Tensor[[TOKENS_DYN, 1], pl.FP32]],
) -> tuple[pl.Tensor, pl.Tensor]:
    x.bind_dynamic(0, TOKENS_DYN)
    quant.bind_dynamic(0, TOKENS_DYN)
    scale.bind_dynamic(0, TOKENS_DYN)
    return dynamic_quantize_hidden_retained(x, quant, scale)


@pl.jit
def projection_input_quant_parallel(
    x: pl.Tensor[[TOKENS_DYN, HIDDEN], pl.BF16],
    quant: pl.Out[pl.Tensor[[TOKENS_DYN, HIDDEN], pl.INT8]],
    scale: pl.Out[pl.Tensor[[TOKENS_DYN, 1], pl.FP32]],
) -> tuple[pl.Tensor, pl.Tensor]:
    x.bind_dynamic(0, TOKENS_DYN)
    quant.bind_dynamic(0, TOKENS_DYN)
    scale.bind_dynamic(0, TOKENS_DYN)
    return dynamic_quantize_hidden_parallel(x, quant, scale)


@pl.jit.inline
def dynamic_quantize_hidden_tiled(
    x: pl.Tensor[[TOKENS_DYN, HIDDEN], pl.BF16],
    quant: pl.Tensor[[TOKENS_DYN, HIDDEN], pl.INT8],
    scale: pl.Tensor[[TOKENS_DYN, 1], pl.FP32],
) -> tuple[pl.Tensor, pl.Tensor]:
    """Assign each core eight rows and one aligned vector store of compact scales."""
    tokens = pl.tensor.dim(x, 0)
    flat_scale = pl.reshape(scale, [1, tokens])
    for block in pl.spmd(
        (tokens + RETAINED_ROWS - 1) // RETAINED_ROWS, name_hint="hidden_dynamic_quant_tiled"
    ):
        for base in pl.range(
            block * RETAINED_ROWS, pl.min(tokens, (block + 1) * RETAINED_ROWS), RETAINED_ROWS
        ):
            valid_rows = pl.min(RETAINED_ROWS, tokens - base)
            amax = pl.set_validshape(pl.full([1, RETAINED_ROWS], dtype=pl.FP32, value=0.0), 1, valid_rows)
            for hidden_base in pl.range(0, HIDDEN, HIDDEN // 2):
                chunk = pl.cast(
                    pl.slice(
                        x,
                        [RETAINED_ROWS, HIDDEN // 2],
                        [base, hidden_base],
                        valid_shape=[valid_rows, HIDDEN // 2],
                    ),
                    pl.FP32,
                )
                chunk_max = pl.reshape(pl.row_max(pl.abs(chunk)), [1, RETAINED_ROWS])
                amax = pl.maximum(amax, chunk_max)
            numerator = pl.set_validshape(
                pl.full([1, RETAINED_ROWS], dtype=pl.FP32, value=INT8_MAX), 1, valid_rows
            )
            inverse_scale = pl.div(numerator, amax)
            stored_scales = pl.full([1, RETAINED_ROWS], dtype=pl.FP32, value=0.0)
            for lane in pl.range(valid_rows):
                max_value = pl.read(amax, [0, lane])
                if max_value > 0.0:
                    inverse_value = pl.read(inverse_scale, [0, lane])
                    pl.write(stored_scales, [0, lane], pl.div(1.0, inverse_value))
                else:
                    pl.write(inverse_scale, [0, lane], 0.0)
            flat_scale[:, base : base + RETAINED_ROWS] = pl.set_validshape(stored_scales, 1, valid_rows)
            inverse_rows = pl.reshape(inverse_scale, [RETAINED_ROWS, 1])
            for hidden_base in pl.range(0, HIDDEN, HIDDEN_TILE):
                values = pl.cast(
                    pl.slice(
                        x,
                        [RETAINED_ROWS, HIDDEN_TILE],
                        [base, hidden_base],
                        valid_shape=[valid_rows, HIDDEN_TILE],
                    ),
                    pl.FP32,
                )
                normalized = pl.set_validshape(
                    pl.row_expand_mul(values, inverse_rows), RETAINED_ROWS, HIDDEN_TILE
                )
                rounded = pl.cast(normalized, pl.INT32, mode="rint")
                narrowed = pl.cast(pl.cast(rounded, pl.FP16), pl.INT8, mode="trunc")
                quant[base : base + RETAINED_ROWS, hidden_base : hidden_base + HIDDEN_TILE] = (
                    pl.set_validshape(narrowed, valid_rows, HIDDEN_TILE)
                )
    return quant, scale


@pl.jit
def projection_input_quant_tiled(
    x: pl.Tensor[[TOKENS_DYN, HIDDEN], pl.BF16],
    quant: pl.Out[pl.Tensor[[TOKENS_DYN, HIDDEN], pl.INT8]],
    scale: pl.Out[pl.Tensor[[TOKENS_DYN, 1], pl.FP32]],
) -> tuple[pl.Tensor, pl.Tensor]:
    x.bind_dynamic(0, TOKENS_DYN)
    quant.bind_dynamic(0, TOKENS_DYN)
    scale.bind_dynamic(0, TOKENS_DYN)
    return dynamic_quantize_hidden_tiled(x, quant, scale)


def make_a1_projection_quant(rows: int, width: int) -> Any:
    """Quantize BF16 projection inputs with FP32 scale and RNE integer output."""
    if isinstance(rows, bool) or not isinstance(rows, int) or rows not in (1, 8, 16):
        raise ValueError(f"A1 projection quant rows must be 1, 8 or 16, got {rows}")
    if isinstance(width, bool) or not isinstance(width, int) or width not in (768, 2048, 4096):
        raise ValueError(f"A1 projection quant width must be 768, 2048 or 4096, got {width}")
    # tiling
    MAX_ROWS_TILE = 16

    @pl.jit.incore
    def projection_quant_a1(
        x: pl.Tensor[[rows, width], pl.BF16],
        quant: pl.Out[pl.Tensor[[rows, width], pl.INT8]],
        scale: pl.Out[pl.Tensor[[rows, 1], pl.FP32]],
    ):
        for token in pl.range(rows):
            input_bf16 = pl.load(x, [token, 0], [1, width])
            values = pl.cast(input_bf16, pl.FP32, mode="none")
            absolute = pl.abs(values)
            max_input = pl.reshape(absolute, [MAX_ROWS_TILE, width // MAX_ROWS_TILE])
            max_scratch = pl.tile.create([MAX_ROWS_TILE, width // MAX_ROWS_TILE], dtype=pl.FP32)
            max_column = pl.tile.row_max(max_input, max_scratch)
            maxima = pl.reshape(max_column, [1, MAX_ROWS_TILE])
            maximum = pl.read(maxima, [0, 0])
            for lane in pl.unroll(1, MAX_ROWS_TILE):
                candidate = pl.read(maxima, [0, lane])
                if candidate > maximum:
                    maximum = candidate
            if maximum > 0.0:
                pl.write(maxima, [0, 0], maximum)
                numerator = pl.tile.full([1, MAX_ROWS_TILE], dtype=pl.FP32, value=INT8_MAX)
                inverse_row = pl.div(numerator, maxima)
                inverse = pl.read(inverse_row, [0, 0])
                pl.write(scale, [token, 0], pl.div(1.0, inverse))
                normalized = pl.mul(values, inverse)
                rounded = pl.cast(normalized, pl.INT32, mode="rint")
                integer_carrier = pl.cast(rounded, pl.FP16, mode="none")
                quantized = pl.cast(integer_carrier, pl.INT8, mode="trunc")
                pl.store(quantized, [token, 0], quant)
            else:
                pl.write(scale, [token, 0], 0.0)
                rounded_zero = pl.cast(values, pl.INT32, mode="rint")
                zero_carrier = pl.cast(rounded_zero, pl.FP16, mode="none")
                quantized_zero = pl.cast(zero_carrier, pl.INT8, mode="trunc")
                pl.store(quantized_zero, [token, 0], quant)
        return quant, scale

    return projection_quant_a1
