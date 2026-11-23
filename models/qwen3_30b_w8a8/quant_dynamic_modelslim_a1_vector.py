# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""Retained-row ModelSlim dynamic quantization expressed in PyPTO DSL."""

from typing import Any

import pypto.language as pl


def make_a1_modelslim_dynamic_quant(rows: int, width: int) -> Any:
    """Preserve the input quantizer's FP32 reciprocal and integer rounding."""
    if rows not in (1, 8, 20, 160) or width not in (768, 2048):
        raise ValueError(f"Unsupported ModelSlim dynamic quantization shape: {(rows, width)}")

    @pl.jit.incore
    def modelslim_dynamic_quant_a1(
        x: pl.Tensor[[rows, width], pl.FP16],
        quant: pl.Out[pl.Tensor[[rows, width], pl.INT8]],
        scale: pl.InOut[pl.Tensor[[rows, 16], pl.FP32]],
        status: pl.InOut[pl.Tensor[[rows, 16], pl.INT32]],
    ):
        block = pl.tile.get_block_idx()
        for row in pl.range(rows):
            if row % 8 == block:
                input_half = pl.load(x, [row, 0], [1, width])
                values = pl.cast(input_half, pl.FP32, mode="none")
                rows_of_values = pl.reshape(values, [8, width // 8])
                magnitude = pl.abs(rows_of_values)
                scratch = pl.tile.create([8, width // 8], dtype=pl.FP32)
                maximum_tile = pl.row_max(magnitude, scratch)
                maximum = pl.read(maximum_tile, [0, 0])
                for lane in pl.range(1, 8):
                    candidate = pl.read(maximum_tile, [lane, 0])
                    if candidate > maximum:
                        maximum = candidate
                # Input quantization multiplies by the rounded reciprocal;
                # SwiGLU quantization has a different division contract.
                dequant = maximum * (1.0 / 127.0)
                pl.write(scale, [row, 0], dequant)
                inverse = 0.0
                if dequant > 1e-7:
                    inverse = 1.0 / dequant
                scaled = pl.mul(values, inverse)
                rounded = pl.cast(scaled, pl.INT32, mode="rint")
                integral_float = pl.cast(rounded, pl.FP32, mode="none")
                integral_half = pl.cast(integral_float, pl.FP16, mode="none")
                narrowed = pl.cast(integral_half, pl.INT8, mode="trunc")
                pl.store(narrowed, [row, 0], quant)
                status_zero: pl.Scalar[pl.INT32] = 0
                pl.write(status, [row, 0], status_zero)
        return quant, scale, status

    return modelslim_dynamic_quant_a1
