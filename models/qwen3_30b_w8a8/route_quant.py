# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -------------------------------------------------------------------------------------------------------
"""BF16 MoE routing quantization: scalar amax/127 followed by vector division.

This is not projection DynamicQuant's reciprocal/multiply contract. Keep the
FP32 -> INT32 RNE boundary before the FP16 -> INT8 narrowing on A2/A3.
"""

from typing import Any

import pypto.language as pl

from .config import HIDDEN
from .quant import TOKENS_DYN, TOKEN_TILE, HIDDEN_TILE, INT8_MAX


def make_routing_quant_hidden_impl(*, early: bool = False) -> Any:
    """Specialize the producer-side scheduling policy."""
    @pl.jit.inline
    def routing_quant_hidden_impl(
        x: pl.Tensor[[TOKENS_DYN, HIDDEN], pl.BF16],
        quant: pl.Tensor[[TOKENS_DYN, HIDDEN], pl.INT8],
        scale: pl.Tensor[[TOKENS_DYN, 1], pl.FP32],
    ) -> tuple[pl.Tensor, pl.Tensor]:
        tokens = pl.tensor.dim(x, 0)
        # At most sixteen Decode rows share one 64-byte scale line. One writer
        # owns that line; eight-row reduction tiles are not independent writers.
        for _block in pl.spmd(1, name_hint="routing_dynamic_quant", allow_early_resolve=early):
            for base in pl.range(0, tokens, TOKEN_TILE):
                valid_rows = pl.min(TOKEN_TILE, tokens - base)
                amax = pl.set_validshape(pl.full([1, TOKEN_TILE], dtype=pl.FP32, value=0.0), 1, valid_rows)
                for hidden_base in pl.range(0, HIDDEN, HIDDEN_TILE):
                    values = pl.cast(
                        pl.slice(
                            x,
                            [TOKEN_TILE, HIDDEN_TILE],
                            [base, hidden_base],
                            valid_shape=[valid_rows, HIDDEN_TILE],
                        ),
                        pl.FP32,
                    )
                    amax = pl.maximum(amax, pl.reshape(pl.row_max(pl.abs(values)), [1, TOKEN_TILE]))
                for lane in pl.range(valid_rows):
                    token = base + lane
                    max_value = pl.read(amax, [0, lane])
                    stored_scale = pl.div(max_value, INT8_MAX)
                    pl.write(scale, [token, 0], stored_scale)
                    divisor = pl.mul(pl.full([1, HIDDEN_TILE], dtype=pl.FP32, value=1.0), stored_scale)
                    for hidden_base in pl.range(0, HIDDEN, HIDDEN_TILE):
                        row = pl.cast(x[token : token + 1, hidden_base : hidden_base + HIDDEN_TILE], pl.FP32)
                        normalized = pl.div(row, divisor)
                        rounded = pl.cast(normalized, pl.INT32, mode="rint")
                        quant[token : token + 1, hidden_base : hidden_base + HIDDEN_TILE] = pl.cast(
                            pl.cast(rounded, pl.FP16), pl.INT8, mode="trunc"
                        )
        return quant, scale

    return routing_quant_hidden_impl


routing_quant_hidden_impl = make_routing_quant_hidden_impl()


@pl.jit
def routing_input_quant(
    x: pl.Tensor[[TOKENS_DYN, HIDDEN], pl.BF16],
    quant: pl.Out[pl.Tensor[[TOKENS_DYN, HIDDEN], pl.INT8]],
    scale: pl.Out[pl.Tensor[[TOKENS_DYN, 1], pl.FP32]],
) -> tuple[pl.Tensor, pl.Tensor]:
    x.bind_dynamic(0, TOKENS_DYN)
    quant.bind_dynamic(0, TOKENS_DYN)
    scale.bind_dynamic(0, TOKENS_DYN)
    return routing_quant_hidden_impl(x, quant, scale)


@pl.jit.inline
def routing_quant_hidden_tiled_impl(
    x: pl.Tensor[[TOKENS_DYN, HIDDEN], pl.BF16],
    quant: pl.Tensor[[TOKENS_DYN, HIDDEN], pl.INT8],
    scale: pl.Tensor[[TOKENS_DYN, 1], pl.FP32],
) -> tuple[pl.Tensor, pl.Tensor]:
    tokens = pl.tensor.dim(x, 0)
    flat_scale = pl.reshape(scale, [1, tokens])
    for block in pl.spmd((tokens + TOKEN_TILE - 1) // TOKEN_TILE, name_hint="routing_quant_tiled"):
        base = block * TOKEN_TILE
        valid_rows = pl.min(TOKEN_TILE, tokens - base)
        amax = pl.set_validshape(pl.full([1, TOKEN_TILE], dtype=pl.FP32, value=0.0), 1, valid_rows)
        for hidden_base in pl.range(0, HIDDEN, 1024):
            max_values = pl.cast(
                pl.slice(x, [TOKEN_TILE, 1024], [base, hidden_base], valid_shape=[valid_rows, 1024]),
                pl.FP32,
            )
            amax = pl.maximum(amax, pl.reshape(pl.row_max(pl.abs(max_values)), [1, TOKEN_TILE]))
        scales = pl.full([1, TOKEN_TILE], dtype=pl.FP32, value=0.0)
        for lane in pl.range(valid_rows):
            pl.write(scales, [0, lane], pl.div(pl.read(amax, [0, lane]), INT8_MAX))
        flat_scale[:, base : base + TOKEN_TILE] = pl.set_validshape(scales, 1, valid_rows)
        divisors = pl.reshape(scales, [TOKEN_TILE, 1])
        for hidden_base in pl.range(0, HIDDEN, HIDDEN_TILE):
            values = pl.cast(
                pl.slice(
                    x,
                    [TOKEN_TILE, HIDDEN_TILE],
                    [base, hidden_base],
                    valid_shape=[valid_rows, HIDDEN_TILE],
                ),
                pl.FP32,
            )
            normalized = pl.set_validshape(pl.row_expand_div(values, divisors), TOKEN_TILE, HIDDEN_TILE)
            rounded = pl.cast(normalized, pl.INT32, mode="rint")
            narrowed = pl.cast(pl.cast(rounded, pl.FP16), pl.INT8, mode="trunc")
            quant[base : base + TOKEN_TILE, hidden_base : hidden_base + HIDDEN_TILE] = pl.set_validshape(
                narrowed, valid_rows, HIDDEN_TILE
            )
    return quant, scale


@pl.jit
def routing_input_quant_tiled(
    x: pl.Tensor[[TOKENS_DYN, HIDDEN], pl.BF16],
    quant: pl.Out[pl.Tensor[[TOKENS_DYN, HIDDEN], pl.INT8]],
    scale: pl.Out[pl.Tensor[[TOKENS_DYN, 1], pl.FP32]],
) -> tuple[pl.Tensor, pl.Tensor]:
    x.bind_dynamic(0, TOKENS_DYN)
    quant.bind_dynamic(0, TOKENS_DYN)
    scale.bind_dynamic(0, TOKENS_DYN)
    quant_after, scale_after = routing_quant_hidden_tiled_impl(x, quant, scale)
    return quant_after, scale_after
