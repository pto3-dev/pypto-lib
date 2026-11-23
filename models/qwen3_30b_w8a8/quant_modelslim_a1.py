# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""A1 FP16 ModelSlim static activation quantization."""

from typing import Any

import pypto.language as pl


def make_a1_modelslim_static_quant(width: int) -> Any:
    """Quantize one row using the native FP16 reciprocal scale and offset."""
    if width not in (2048, 4096):
        raise ValueError(f"Expected ModelSlim QKV/O input width 2048 or 4096, got {width}")
    # tiling
    COL_TILE = 128

    @pl.jit.incore
    def modelslim_static_quant_a1(
        x: pl.Tensor[[1, width], pl.FP16],
        reciprocal: pl.Tensor[[1, width], pl.FP16],
        offset: pl.Tensor[[1, width], pl.FP16],
        quant: pl.Out[pl.Tensor[[1, width], pl.INT8]],
    ):
        for column in pl.range(0, width, COL_TILE):
            values = pl.cast(pl.load(x, [0, column], [1, COL_TILE]), pl.FP32, mode="none")
            inverse = pl.cast(pl.load(reciprocal, [0, column], [1, COL_TILE]), pl.FP32, mode="none")
            zero_point = pl.cast(pl.load(offset, [0, column], [1, COL_TILE]), pl.FP32, mode="none")
            scaled = pl.add(pl.mul(values, inverse), zero_point)
            for lane in pl.range(COL_TILE):
                value = pl.read(scaled, [0, lane])
                if value > 127.0:
                    pl.write(scaled, [0, lane], 127.0)
                elif value < -128.0:
                    pl.write(scaled, [0, lane], -128.0)
            rounded = pl.cast(scaled, pl.INT32, mode="rint")
            integral_half = pl.cast(rounded, pl.FP16, mode="none")
            narrowed = pl.cast(integral_half, pl.INT8, mode="trunc")
            pl.store(narrowed, [0, column], quant)
        return quant

    return modelslim_static_quant_a1
