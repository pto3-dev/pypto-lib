# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""FP16 310P paged Attention's ordered 128-token online softmax/PV contract."""

from typing import Any

import pypto.language as pl


def make_a1_modelslim_softmax_pv(capacity: int = 512) -> Any:
    """Preserve per-page FP16 centering and ordered FP32 rescaling of numerator/sum.

    Final FP16 numerator/denominator division is a separate native image. One
    staged GQA group contains eight query heads and one KV head, with pages of
    128 tokens. Invalid sequence positions already have masked QK scores.
    """
    if capacity not in (128, 256, 512):
        raise ValueError(f"Expected a 128-token page multiple up to 512, got {capacity}")
    PAGE = 128
    HEADS = 8

    @pl.jit.incore
    def modelslim_softmax_pv_a1(
        scores: pl.Tensor[[HEADS, capacity], pl.FP32],
        values: pl.Tensor[[capacity, PAGE], pl.FP16],
        lengths: pl.Tensor[[1, 8], pl.INT32],
        out: pl.Out[pl.Tensor[[HEADS, PAGE], pl.FP32]],
        sums: pl.Out[pl.Tensor[[1, HEADS], pl.FP32]],
    ):
        length_tile = pl.load(lengths, [0, 0], [1, 8])
        length = pl.read(length_tile, [0, 0])
        first = pl.load(scores, [0, 0], [HEADS, PAGE])
        max_scratch = pl.tile.create([HEADS, PAGE], dtype=pl.FP32)
        maximum = pl.tile.row_max(first, max_scratch)
        numerator = pl.tile.full([HEADS, PAGE], dtype=pl.FP32, value=0.0)
        total = pl.tile.full([HEADS, PAGE], dtype=pl.FP32, value=0.0)
        for base in pl.unroll(0, capacity, PAGE):
            if base < length:
                chunk = pl.load(scores, [0, base], [HEADS, PAGE])
                scratch = pl.tile.create([HEADS, PAGE], dtype=pl.FP32)
                next_maximum = pl.tile.row_max(chunk, scratch)
                if base == 0:
                    maximum = next_maximum
                    rescale = pl.exp(pl.mul(chunk, 0.0))
                else:
                    for head in pl.unroll(HEADS):
                        previous = pl.read(maximum, [head, 0])
                        candidate = pl.read(next_maximum, [head, 0])
                        if previous > candidate:
                            pl.write(next_maximum, [head, 0], previous)
                    old_expanded = pl.tile.row_expand(chunk, maximum)
                    new_expanded = pl.tile.row_expand(chunk, next_maximum)
                    delta = pl.sub(old_expanded, new_expanded)
                    delta_half = pl.cast(delta, pl.FP16, mode="none")
                    rescale = pl.exp(pl.cast(delta_half, pl.FP32, mode="none"))
                    maximum = next_maximum
                expanded_maximum = pl.tile.row_expand(chunk, maximum)
                centered = pl.sub(chunk, expanded_maximum)
                centered_half = pl.cast(centered, pl.FP16, mode="none")
                exponential = pl.exp(pl.cast(centered_half, pl.FP32, mode="none"))
                probability_half = pl.cast(exponential, pl.FP16, mode="none")
                probability = pl.cast(probability_half, pl.FP32, mode="none")
                block_sum = pl.tile.row_sum(exponential, scratch)
                expanded_sum = pl.tile.row_expand(chunk, block_sum)
                total = pl.add(pl.mul(total, rescale), expanded_sum)
                accumulator = pl.tile.full([HEADS, PAGE], dtype=pl.FP32, value=0.0)
                correction = pl.tile.full([HEADS, PAGE], dtype=pl.FP32, value=0.0)
                # Keep the DMA row index in a directly bounded range. The A1
                # static-DMA verifier cannot prove an added loop-index bound.
                for position in pl.range(capacity):
                    if position >= base and position < base + PAGE and position < length:
                        value_half = pl.load(values, [position, 0], [1, PAGE])
                        value = pl.cast(value_half, pl.FP32, mode="none")
                        expanded_value = pl.tile.col_expand(accumulator, value)
                        coefficient_row = pl.tile.full([1, HEADS], dtype=pl.FP32, value=0.0)
                        for query_head in pl.unroll(HEADS):
                            scalar = pl.read(probability, [query_head, position - base])
                            pl.write(coefficient_row, [0, query_head], scalar)
                        coefficient = pl.reshape(coefficient_row, [HEADS, 1])
                        expanded_coefficient = pl.tile.row_expand(accumulator, coefficient)
                        product = pl.mul(expanded_value, expanded_coefficient)
                        next_accumulator = pl.add(accumulator, product)
                        virtual_product = pl.sub(next_accumulator, accumulator)
                        virtual_accumulator = pl.sub(next_accumulator, virtual_product)
                        error_accumulator = pl.sub(accumulator, virtual_accumulator)
                        error_product = pl.sub(product, virtual_product)
                        error = pl.add(error_accumulator, error_product)
                        correction = pl.add(correction, error)
                        accumulator = next_accumulator
                    if position % 16 == 15:
                        accumulator = pl.add(accumulator, correction)
                        correction = pl.tile.full([HEADS, PAGE], dtype=pl.FP32, value=0.0)
                numerator = pl.add(pl.mul(numerator, rescale), accumulator)
        result_sums = pl.tile.full([1, HEADS], dtype=pl.FP32, value=1.0)
        if length > 0:
            for sum_head in pl.unroll(HEADS):
                sum_value = pl.read(total, [sum_head, 0])
                pl.write(result_sums, [0, sum_head], sum_value)
        pl.store(numerator, [0, 0], out)
        pl.store(result_sums, [0, 0], sums)
        return out, sums

    return modelslim_softmax_pv_a1
