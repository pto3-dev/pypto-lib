# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Per-token attention reduction, sigmoid gate and dynamic INT8 quantization.

Each fused leaf owns six complete heads of one token. The production static
plan schedules those leaves independently. SPMD wrappers below are standalone
comparison entrypoints, not a requirement on the fused mathematical leaf.
"""

import pypto.language as pl


def make_gate_quant(fused=True):
    @pl.jit.incore
    def reduction(
        out: pl.Out[pl.Tensor[[24, 1536], pl.BF16]],
        partials: pl.Tensor[[4032, 256], pl.FP32],
        gate: pl.Tensor[[24, 1536], pl.BF16],
    ):
        for row in pl.range(pl.tile.get_block_idx(), 144, pl.tile.get_block_num()):
            acc = pl.tile.full([1, 256], pl.FP32, 0.0)
            for chunk in pl.range(28):
                acc = pl.add(acc, pl.load(partials, [chunk * 144 + row, 0], [1, 256]))
            rounded = pl.cast(pl.cast(acc, pl.BF16, mode="rint"), pl.FP32)
            g = pl.cast(pl.load(gate, [row // 6, row % 6 * 256], [1, 256]), pl.FP32)
            sigmoid = pl.recip(pl.add(pl.exp(pl.neg(g)), 1.0))
            sigmoid = pl.cast(pl.cast(sigmoid, pl.BF16, mode="rint"), pl.FP32)
            value = pl.cast(pl.mul(sigmoid, rounded), pl.BF16, mode="rint")
            out = pl.store(value, [row // 6, row % 6 * 256], out)
        return out

    @pl.jit.incore
    def quantization(
        x: pl.Tensor[[24, 1536], pl.BF16],
        q: pl.Out[pl.Tensor[[24, 1536], pl.INT8]],
        scales: pl.Out[pl.Tensor[[24, 1], pl.FP32]],
    ):
        for row in pl.range(pl.tile.get_block_idx(), 24, pl.tile.get_block_num()):
            value = pl.cast(pl.load(x, [row, 0], [1, 1536]), pl.FP32)
            chunks = pl.tile.reshape(pl.abs(value), [8, 192])
            tmp = pl.tile.create([8, 192], pl.FP32)
            maximum = pl.row_max(chunks, tmp)
            magnitude = pl.read(maximum, [0, 0])
            for part in pl.range(1, 8):
                partial = pl.read(maximum, [part, 0])
                if partial > magnitude:
                    magnitude = partial
            if magnitude == 0.0:
                normalized = pl.mul(value, 0.0)
            else:
                denominator = pl.mul(pl.tile.full([1, 8], pl.FP32, 1.0), magnitude)
                numerator = pl.tile.full([1, 8], pl.FP32, 127.0)
                ratio = pl.read(pl.div(numerator, denominator), [0, 0])
                normalized = pl.mul(value, ratio)
            rounded = pl.cast(normalized, pl.INT32, mode="rint")
            half = pl.cast(rounded, pl.FP16, mode="rint")
            quantized = pl.cast(half, pl.INT8, mode="trunc")
            q = pl.store(quantized, [row, 0], q)
            scale = pl.mul(pl.tile.full([1, 8], pl.FP32, 1.0), magnitude * (1.0 / 127.0))
            scales = pl.store(pl.tile.set_validshape(scale, 1, 1), [row, 0], scales)
        return q, scales

    @pl.jit.incore
    def fused_row(
        out: pl.Out[pl.Tensor[[24, 1536], pl.BF16]],
        partials: pl.Tensor[[4032, 256], pl.FP32],
        gate: pl.Tensor[[24, 1536], pl.BF16],
        q: pl.Out[pl.Tensor[[24, 1536], pl.INT8]],
        scales: pl.Out[pl.Tensor[[24, 1], pl.FP32]],
    ):
        row = pl.tile.get_block_idx()
        acc = pl.tile.full([6, 256], pl.FP32, 0.0)
        for chunk in pl.range(28):
            acc = pl.add(acc, pl.load(partials, [chunk * 144 + row * 6, 0], [6, 256]))
        rounded = pl.cast(pl.cast(acc, pl.BF16, mode="rint"), pl.FP32)
        g = pl.cast(pl.reshape(pl.load(gate, [row, 0], [1, 1536]), [6, 256]), pl.FP32)
        sigmoid = pl.recip(pl.add(pl.exp(pl.neg(g)), 1.0))
        sigmoid = pl.cast(pl.cast(sigmoid, pl.BF16, mode="rint"), pl.FP32)
        gated = pl.cast(pl.mul(sigmoid, rounded), pl.BF16, mode="rint")
        gated_row = pl.reshape(gated, [1, 1536])
        out = pl.store(gated_row, [row, 0], out)
        value = pl.cast(gated_row, pl.FP32)
        chunks = pl.tile.reshape(pl.abs(value), [8, 192])
        tmp = pl.tile.create([8, 192], pl.FP32)
        maximum = pl.row_max(chunks, tmp)
        magnitude = pl.read(maximum, [0, 0])
        for part in pl.range(1, 8):
            partial = pl.read(maximum, [part, 0])
            if partial > magnitude:
                magnitude = partial
        if magnitude == 0.0:
            normalized = pl.mul(value, 0.0)
        else:
            denominator = pl.mul(pl.tile.full([1, 8], pl.FP32, 1.0), magnitude)
            numerator = pl.tile.full([1, 8], pl.FP32, 127.0)
            ratio = pl.read(pl.div(numerator, denominator), [0, 0])
            normalized = pl.mul(value, ratio)
        rounded_q = pl.cast(normalized, pl.INT32, mode="rint")
        half = pl.cast(rounded_q, pl.FP16, mode="rint")
        quantized = pl.cast(half, pl.INT8, mode="trunc")
        q = pl.store(quantized, [row, 0], q)
        scale = pl.mul(pl.tile.full([1, 8], pl.FP32, 1.0), magnitude * (1.0 / 127.0))
        scales = pl.store(pl.tile.set_validshape(scale, 1, 1), [row, 0], scales)
        return out, q, scales

    @pl.jit
    def program(
        out: pl.Out[pl.Tensor[[24, 1536], pl.BF16]],
        partials: pl.Tensor[[4032, 256], pl.FP32],
        gate: pl.Tensor[[24, 1536], pl.BF16],
        q: pl.Out[pl.Tensor[[24, 1536], pl.INT8]],
        scales: pl.Out[pl.Tensor[[24, 1], pl.FP32]],
    ):
        available = pl.system.available_aiv_count()
        token_width = pl.min(available, 24)
        head_width = pl.min(available, 144)
        if fused:
            with pl.spmd(token_width, allow_early_resolve=True):
                out, q, scales = fused_row(out, partials, gate, q, scales)
        else:
            with pl.spmd(head_width, allow_early_resolve=True):
                out = reduction(out, partials, gate)
            with pl.spmd(token_width, allow_early_resolve=True):
                q, scales = quantization(out, q, scales)
        return out, q, scales

    return program
