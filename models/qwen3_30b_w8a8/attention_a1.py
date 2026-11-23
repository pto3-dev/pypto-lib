# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""A1 BF16 Decode attention over one staged GQA group."""

from pathlib import Path
from typing import Any

import pypto.language as pl

# model config
HEAD_DIM = 128
GROUP = 8
SCALE = HEAD_DIM ** -0.5


def _validate_capacity(capacity: int) -> None:
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity not in (64, 128, 256, 512):
        raise ValueError(f"A1 attention capacity must be 64, 128, 256 or 512, got {capacity}")


def _validate_storage(storage_dtype: pl.DataType) -> None:
    if storage_dtype not in (pl.FP16, pl.BF16):
        raise ValueError(f"A1 attention storage must be FP16 or BF16, got {storage_dtype}")


def make_a1_attention_qk(
    capacity: int = 512, *, storage_dtype: pl.DataType = pl.BF16,
    head_tile: int = GROUP, position_tiles: int = 1,
) -> Any:
    """Compute scaled FP32 QK scores from 16-bit input; mask past device length.

    Args:
        capacity: Fixed KV capacity.
        storage_dtype: Query and KV storage type.
        head_tile: Query heads per block; launch exactly ``GROUP // head_tile`` blocks.
        position_tiles: Independent KV position partitions; multiply the launch count by this value.
    """
    _validate_capacity(capacity)
    _validate_storage(storage_dtype)
    if type(head_tile) is not int or head_tile not in (1, GROUP):
        raise ValueError(f"A1 QK head_tile must be 1 or {GROUP}, got {head_tile}")
    if type(position_tiles) is not int or position_tiles not in (1, 8):
        raise ValueError(f"A1 QK position_tiles must be 1 or 8, got {position_tiles}")
    if head_tile != GROUP and position_tiles != 1:
        raise ValueError("A1 QK partitions either heads or positions, not both")
    native_fp16 = storage_dtype == pl.FP16
    HEAD_TILE = head_tile
    POSITION_TILE = capacity // position_tiles
    PARTITIONS = position_tiles

    @pl.jit.incore
    def qk_a1(
        query: pl.Tensor[[GROUP, HEAD_DIM], storage_dtype],
        keys: pl.Tensor[[capacity, HEAD_DIM], storage_dtype],
        lengths: pl.Tensor[[1, 8], pl.INT32],
        scores: pl.Out[pl.Tensor[[GROUP, capacity], pl.FP32]],
    ):
        head_offset = 0
        if HEAD_TILE != GROUP:
            head_offset = pl.tile.get_block_idx() % GROUP
        partition = 0
        if PARTITIONS != 1:
            partition = pl.tile.get_block_idx() % PARTITIONS
        position_base = partition * POSITION_TILE
        q_bf16 = pl.load(query, [0, 0], [GROUP, HEAD_DIM])
        q_fp32 = pl.cast(q_bf16, pl.FP32, mode="none")
        result = pl.tile.create([HEAD_TILE, POSITION_TILE], dtype=pl.FP32)
        length_tile = pl.load(lengths, [0, 0], [1, 8])
        length = pl.read(length_tile, [0, 0])
        for position in pl.range(capacity):
            if position < length and position >= position_base and position < position_base + POSITION_TILE:
                key_bf16 = pl.load(keys, [position, 0], [1, HEAD_DIM])
                key_fp32 = pl.cast(key_bf16, pl.FP32, mode="none")
                key_expanded = pl.tile.col_expand(q_fp32, key_fp32)
                product = pl.mul(q_fp32, key_expanded)
                # Sixteen physical lanes keep FP16 cast rows 32-byte aligned;
                # active query heads occupy the leading lanes.
                accumulator = pl.tile.full([1, 16], dtype=pl.FP32, value=0.0)
                for chunk in pl.range(0, HEAD_DIM, 16):
                    high = accumulator
                    low = pl.tile.full([1, 16], dtype=pl.FP32, value=0.0)
                    for lane in pl.range(16):
                        term_row = pl.tile.full([1, 16], dtype=pl.FP32, value=0.0)
                        for head in pl.unroll(HEAD_TILE):
                            term = pl.read(product, [head_offset + head, chunk + lane])
                            pl.write(term_row, [0, head], term)
                        next_high = pl.add(high, term_row)
                        virtual_term = pl.sub(next_high, high)
                        virtual_high = pl.sub(next_high, virtual_term)
                        error_high = pl.sub(high, virtual_high)
                        error_term = pl.sub(term_row, virtual_term)
                        error = pl.add(error_high, error_term)
                        low = pl.add(low, error)
                        high = next_high
                    accumulator = pl.add(high, low)
                if native_fp16:
                    accumulator_half = pl.cast(accumulator, pl.FP16, mode="none")
                    rounded_accumulator = pl.cast(accumulator_half, pl.FP32, mode="none")
                    scale_float = pl.tile.full([1, 16], dtype=pl.FP32, value=SCALE)
                    scale_half = pl.cast(scale_float, pl.FP16, mode="none")
                    scale_rounded = pl.cast(scale_half, pl.FP32, mode="none")
                    product_scaled = pl.mul(rounded_accumulator, scale_rounded)
                    scaled_half = pl.cast(product_scaled, pl.FP16, mode="none")
                    scaled = pl.cast(scaled_half, pl.FP32, mode="none")
                else:
                    scaled = pl.mul(accumulator, SCALE)
                for output_head in pl.unroll(HEAD_TILE):
                    value = pl.read(scaled, [0, output_head])
                    pl.write(result, [output_head, position % POSITION_TILE], value)
            elif position >= position_base and position < position_base + POSITION_TILE:
                for head in pl.unroll(HEAD_TILE):
                    pl.write(result, [head, position % POSITION_TILE], -3.4028234663852886e38)
        if PARTITIONS != 1:
            for destination_partition in pl.unroll(PARTITIONS):
                if partition == destination_partition:
                    pl.store(result, [0, destination_partition * POSITION_TILE], scores)
        elif HEAD_TILE == GROUP:
            pl.store(result, [0, 0], scores)
        else:
            for destination_head in pl.unroll(GROUP):
                if head_offset == destination_head:
                    pl.store(result, [destination_head, 0], scores)
        return scores

    return qk_a1


def make_a1_attention_softmax(capacity: int = 512) -> Any:
    """Compute BF16 unnormalized probabilities and sequential-64 FP32 row sums."""
    _validate_capacity(capacity)
    # tiling
    COL_TILE = 64

    @pl.jit.incore
    def softmax_a1(
        scores: pl.Tensor[[GROUP, capacity], pl.FP32],
        lengths: pl.Tensor[[1, 8], pl.INT32],
        probabilities: pl.Out[pl.Tensor[[GROUP, capacity], pl.BF16]],
        sums: pl.Out[pl.Tensor[[1, GROUP], pl.FP32]],
    ):
        length_tile = pl.load(lengths, [0, 0], [1, 8])
        length = pl.read(length_tile, [0, 0])
        if length > 0:
            first_scores = pl.load(scores, [0, 0], [GROUP, COL_TILE])
            max_scratch = pl.tile.create([GROUP, COL_TILE], dtype=pl.FP32)
            maximum = pl.tile.row_max(first_scores, max_scratch)
            for max_base in pl.range(COL_TILE, capacity, COL_TILE):
                max_chunk = pl.load(scores, [0, max_base], [GROUP, COL_TILE])
                chunk_max = pl.tile.row_max(max_chunk, max_scratch)
                for max_head in pl.unroll(GROUP):
                    previous_max = pl.read(maximum, [max_head, 0])
                    next_max = pl.read(chunk_max, [max_head, 0])
                    if next_max > previous_max:
                        pl.write(maximum, [max_head, 0], next_max)
            total_column = pl.mul(maximum, 0.0)
            total = pl.reshape(total_column, [1, GROUP])
            for base in pl.range(0, capacity, COL_TILE):
                chunk = pl.load(scores, [0, base], [GROUP, COL_TILE])
                expanded_maximum = pl.tile.row_expand(chunk, maximum)
                centered = pl.sub(chunk, expanded_maximum)
                exponential = pl.exp(centered)
                rounded = pl.cast(exponential, pl.BF16, mode="rint")
                pl.store(rounded, [0, base], probabilities)
                sum_scratch = pl.tile.create([GROUP, COL_TILE], dtype=pl.FP32)
                partial = pl.tile.row_sum(exponential, sum_scratch)
                partial_row = pl.reshape(partial, [1, GROUP])
                total = pl.add(total, partial_row)
            pl.store(total, [0, 0], sums)
        else:
            zeros = pl.tile.full([GROUP, capacity], dtype=pl.FP32, value=0.0)
            empty_probabilities = pl.cast(zeros, pl.BF16, mode="rint")
            pl.store(empty_probabilities, [0, 0], probabilities)
            ones = pl.tile.full([1, GROUP], dtype=pl.FP32, value=1.0)
            pl.store(ones, [0, 0], sums)
        return probabilities, sums

    return softmax_a1


def make_a1_attention_pv(capacity: int = 512, *, normalize: bool = True) -> Any:
    """Accumulate BF16 probabilities/values in FP32 with grouped-16 rounding."""
    _validate_capacity(capacity)
    if not isinstance(normalize, bool):
        raise ValueError(f"A1 PV normalize must be bool, got {normalize}")
    result_dtype = pl.BF16 if normalize else pl.FP32

    @pl.jit.incore
    def pv_a1(
        probabilities: pl.Tensor[[GROUP, capacity], pl.BF16],
        values: pl.Tensor[[capacity, HEAD_DIM], pl.BF16],
        sums: pl.Tensor[[1, GROUP], pl.FP32],
        lengths: pl.Tensor[[1, 8], pl.INT32],
        out: pl.Out[pl.Tensor[[GROUP, HEAD_DIM], result_dtype]],
    ):
        probability_bf16 = pl.load(probabilities, [0, 0], [GROUP, capacity])
        probability_fp32 = pl.cast(probability_bf16, pl.FP32, mode="none")
        accumulator = pl.tile.full([GROUP, HEAD_DIM], dtype=pl.FP32, value=0.0)
        correction = pl.tile.full([GROUP, HEAD_DIM], dtype=pl.FP32, value=0.0)
        length_tile = pl.load(lengths, [0, 0], [1, 8])
        length = pl.read(length_tile, [0, 0])
        for position in pl.range(capacity):
            if position < length:
                value_bf16 = pl.load(values, [position, 0], [1, HEAD_DIM])
                value_fp32 = pl.cast(value_bf16, pl.FP32, mode="none")
                expanded_value = pl.tile.col_expand(accumulator, value_fp32)
                coefficient_row = pl.tile.create([1, GROUP], dtype=pl.FP32)
                for head in pl.unroll(GROUP):
                    coefficient = pl.read(probability_fp32, [head, position])
                    pl.write(coefficient_row, [0, head], coefficient)
                coefficient_column = pl.reshape(coefficient_row, [GROUP, 1])
                expanded_coefficient = pl.tile.row_expand(accumulator, coefficient_column)
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
                correction = pl.tile.full([GROUP, HEAD_DIM], dtype=pl.FP32, value=0.0)
        if normalize:
            denominator_row = pl.load(sums, [0, 0], [1, GROUP])
            denominator = pl.reshape(denominator_row, [GROUP, 1])
            expanded_denominator = pl.tile.row_expand(accumulator, denominator)
            normalized = pl.div(accumulator, expanded_denominator)
            rounded = pl.cast(normalized, pl.BF16, mode="rint")
            pl.store(rounded, [0, 0], out)
        else:
            pl.store(accumulator, [0, 0], out)
        return out

    return pv_a1


def compile_a1_attention_normalize(
    pto_isa_root: str | Path, *, storage_dtype: pl.DataType = pl.BF16, build_dir: Path | None = None,
) -> bytes:
    """Compile per-head FP32 reciprocal refinement and 16-bit output rounding.

    ABI: FP32 numerator[8,128], FP32 sums[1,8], BF16 output[8,128].
    Prepare eight blocks, one per query head. Three explicitly ordered Newton
    refinements follow the native reciprocal seed; no payload-specific branch.
    """
    from simpler_setup import KernelCompiler

    _validate_storage(storage_dtype)
    source = Path(__file__).with_name("a1_attention_normalize.cpp")
    if storage_dtype == pl.FP16:
        if build_dir is None:
            raise ValueError("FP16 attention normalization requires an explicit build directory")
        build_dir.mkdir(parents=True, exist_ok=True)
        wrapper = build_dir / "attention_normalize_fp16.cpp"
        wrapper.write_text('#define PYPTO_A1_MODEL_FP16 1\n#include "a1_attention_normalize.cpp"\n')
        return KernelCompiler("a1").compile_single_launch(
            str(wrapper), str(pto_isa_root), extra_include_dirs=[str(source.parent)])
    return KernelCompiler("a1").compile_single_launch(str(source), str(pto_isa_root))
