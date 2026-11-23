# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""A1 B20 static quantization with column-partitioned vector clipping."""

from typing import Any

import pypto.language as pl


def make_a1_batched_modelslim_vector_quant(width: int) -> Any:
    """Quantize FP16 activation rows to signed INT8."""
    if width not in (2048, 4096):
        raise ValueError(f"Expected ModelSlim QKV/O input width 2048 or 4096, got {width}")
    COL_TILE = width // 8

    @pl.jit.incore
    def modelslim_b20_vector_quant_a1(
        x: pl.Tensor[[20, width], pl.FP16],
        reciprocal: pl.Tensor[[1, width], pl.FP16],
        offset: pl.Tensor[[1, width], pl.FP16],
        quant: pl.Out[pl.Tensor[[20, width], pl.INT8]],
    ):
        for column in pl.range(0, width, COL_TILE):
            if column // COL_TILE == pl.tile.get_block_idx():
                inverse_half = pl.load(reciprocal, [0, column], [1, COL_TILE])
                inverse = pl.cast(inverse_half, pl.FP32, mode="none")
                zero_half = pl.load(offset, [0, column], [1, COL_TILE])
                zero = pl.cast(zero_half, pl.FP32, mode="none")
                ceiling = pl.tile.full([1, COL_TILE], dtype=pl.FP32, value=127.0)
                floor = pl.tile.full([1, COL_TILE], dtype=pl.FP32, value=-128.0)
                for row in pl.range(20):
                    input_half = pl.load(x, [row, column], [1, COL_TILE])
                    values = pl.cast(input_half, pl.FP32, mode="none")
                    multiplied = pl.mul(values, inverse)
                    scaled = pl.add(multiplied, zero)
                    bounded_high = pl.minimum(scaled, ceiling)
                    bounded = pl.maximum(bounded_high, floor)
                    rounded = pl.cast(bounded, pl.INT32, mode="rint")
                    integral_float = pl.cast(rounded, pl.FP32, mode="none")
                    integral_half = pl.cast(integral_float, pl.FP16, mode="none")
                    narrowed = pl.cast(integral_half, pl.INT8, mode="trunc")
                    pl.store(narrowed, [row, column], quant)
        return quant

    return modelslim_b20_vector_quant_a1
