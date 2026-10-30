# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Explicit mathematical slice of production AddNormQuant, no communication.

A numerical diagnosis probe only, not part of a model's compilation registry.
Input is the already-rounded BF16 reduction. Native fused/separate comparisons
must report differences; this probe does not redefine the model's math.
"""

from pathlib import Path
from pypto import backend, ir
import pypto.language as pl

W = Path(__file__).resolve().parents[3]


@pl.function(type=pl.FunctionType.InCore)
def norm_math(
    x: pl.Tensor[[24, 5120], pl.BF16],
    residual: pl.Tensor[[24, 5120], pl.BF16],
    weight: pl.Tensor[[1, 5120], pl.BF16],
    out: pl.Out[pl.Tensor[[24, 5120], pl.BF16]],
    normalized: pl.Out[pl.Tensor[[24, 5120], pl.BF16]],
    quantized: pl.Out[pl.Tensor[[24, 5120], pl.INT8]],
    scales: pl.Out[pl.Tensor[[24, 1], pl.FP32]],
):
    row = pl.tile.get_block_idx()
    whole = pl.cast(pl.load(x, [row, 0], [1, 5120]), pl.FP32)
    previous = pl.cast(pl.load(residual, [row, 0], [1, 5120]), pl.FP32)
    rounded = pl.cast(pl.add(whole, previous), pl.BF16, mode="rint")
    out = pl.store(rounded, [row, 0], out)
    whole = pl.cast(rounded, pl.FP32)
    chunks = pl.tile.reshape(whole, [40, 128])
    temporary = pl.tile.create([40, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
    partials = pl.tile.row_sum(pl.mul(chunks, chunks), temporary)
    summation = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0)
    for group in pl.range(40):
        summation = pl.add(summation, pl.read(partials, [group, 0]))
    variance = pl.add(pl.mul(summation, 1.0 / 5120), 1e-6)
    inv = pl.tile.rsqrt(variance)
    inv = pl.mul(inv, pl.add(pl.mul(pl.mul(variance, pl.mul(inv, inv)), -0.5), 1.5))
    scalar_inv = pl.read(inv, [0, 0])
    gamma = pl.add(pl.cast(pl.load(weight, [0, 0], [1, 5120]), pl.FP32), 1.0)
    gamma_rounded = pl.cast(pl.cast(gamma, pl.BF16, mode="rint"), pl.FP32)
    scaled = pl.cast(pl.cast(pl.mul(whole, scalar_inv), pl.BF16, mode="rint"), pl.FP32)
    result = pl.cast(pl.mul(scaled, gamma_rounded), pl.BF16, mode="rint")
    normalized = pl.store(result, [row, 0], normalized)
    value = pl.cast(result, pl.FP32)
    quant_chunks = pl.tile.reshape(pl.abs(value), [8, 640])
    tmp = pl.tile.create([8, 640], pl.FP32)
    maximum = pl.row_max(quant_chunks, tmp)
    magnitude = pl.read(maximum, [0, 0])
    for part in pl.range(1, 8):
        partial = pl.read(maximum, [part, 0])
        if partial > magnitude:
            magnitude = partial
    if magnitude == 0.0:
        scaled_quant = pl.mul(value, 0.0)
    else:
        denominator = pl.mul(pl.tile.full([1, 8], pl.FP32, 1.0), magnitude)
        numerator = pl.tile.full([1, 8], pl.FP32, 127.0)
        ratio_tile = pl.div(numerator, denominator)
        ratio = pl.read(ratio_tile, [0, 0])
        scaled_quant = pl.mul(value, ratio)
    q32 = pl.cast(scaled_quant, pl.INT32, mode="rint")
    qhalf = pl.cast(q32, pl.FP16, mode="rint")
    q = pl.cast(qhalf, pl.INT8, mode="trunc")
    quantized = pl.store(q, [row, 0], quantized)
    scale = pl.mul(pl.tile.full([1, 8], pl.FP32, 1.0), magnitude * (1.0 / 127.0))
    scales = pl.store(pl.tile.set_validshape(scale, 1, 1), [row, 0], scales)
    return out, normalized, quantized, scales


@pl.function(type=pl.FunctionType.Orchestration)
def forward(
    x: pl.Tensor[[24, 5120], pl.BF16],
    residual: pl.Tensor[[24, 5120], pl.BF16],
    weight: pl.Tensor[[1, 5120], pl.BF16],
    out: pl.Out[pl.Tensor[[24, 5120], pl.BF16]],
    normalized: pl.Out[pl.Tensor[[24, 5120], pl.BF16]],
    quantized: pl.Out[pl.Tensor[[24, 5120], pl.INT8]],
    scales: pl.Out[pl.Tensor[[24, 1], pl.FP32]],
):
    with pl.spmd(24):
        out, normalized, quantized, scales = norm_math(
            x, residual, weight, out, normalized, quantized, scales
        )
    return out, normalized, quantized, scales


def main():
    backend.set_backend_type(backend.BackendType.Ascend910B)
    program = ir.Program([norm_math, forward], "NormMathDiagnosis", forward.span)
    ir.compile(program, platform="a2a3", output_dir=W / "build/norm_math_diagnosis", dump_passes=True)
    print("NORM_MATH_PROBE_COMPILED", flush=True)


if __name__ == "__main__":
    main()
