# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""B20 FP16 expert combine and next-input Norm using public tile operations."""

import pypto.language as pl


@pl.jit.inline
def _combine_norm_impl(
    acc: pl.Tensor[[160, 2048], pl.INT32],
    input_scale: pl.Tensor[[160, 16], pl.FP32],
    ids: pl.Tensor[[20, 8], pl.INT32],
    weight_scale: pl.Tensor[[128, 2048], pl.FP32],
    route_weights: pl.Tensor[[20, 8], pl.FP32],
    expert_y: pl.InOut[pl.Tensor[[160, 2048], pl.FP16]],
    y: pl.InOut[pl.Tensor[[20, 2048], pl.FP16]],
    status: pl.InOut[pl.Tensor[[8, 16], pl.INT32]],
    residual: pl.Tensor[[20, 2048], pl.FP16],
    gamma: pl.Tensor[[1, 2048], pl.FP16],
    beta: pl.Tensor[[1, 2048], pl.FP16],
    norm_y: pl.InOut[pl.Tensor[[20, 2048], pl.FP16]],
    residual_out: pl.InOut[pl.Tensor[[20, 2048], pl.FP16]],
    reciprocal: pl.Tensor,
    offset: pl.Tensor,
    quant: pl.InOut[pl.Tensor],
    fuse_quant: pl.constexpr,
):
    block = pl.tile.get_block_idx()
    if block < 8:
        all_ids = pl.tile.load(ids, [0, 0], [20, 8])
        valid: pl.Scalar[pl.INT32] = 1
        invalid: pl.Scalar[pl.INT32] = 0
        success_code: pl.Scalar[pl.INT32] = 0
        invalid_code: pl.Scalar[pl.INT32] = 2
        for token in pl.range(20):
            for slot in pl.range(8):
                expert = pl.tile.read(all_ids, [token, slot])
                if expert < 0:
                    valid = invalid
                if expert >= 128:
                    valid = invalid
        if valid == 1:
            gamma_half = pl.tile.load(gamma, [0, 0], [1, 2048])
            beta_half = pl.tile.load(beta, [0, 0], [1, 2048])
            gamma_float = pl.tile.cast(gamma_half, pl.FP32, mode="none")
            beta_float = pl.tile.cast(beta_half, pl.FP32, mode="none")
            quant_inverse = gamma_float
            zero = beta_float
            ceiling = gamma_float
            floor = beta_float
            if fuse_quant:
                quant_inverse_half = pl.tile.load(reciprocal, [0, 0], [1, 2048])
                quant_inverse = pl.tile.cast(quant_inverse_half, pl.FP32, mode="none")
                offset_half = pl.tile.load(offset, [0, 0], [1, 2048])
                zero = pl.tile.cast(offset_half, pl.FP32, mode="none")
                ceiling = pl.tile.full([1, 2048], dtype=pl.FP32, value=127.0)
                floor = pl.tile.full([1, 2048], dtype=pl.FP32, value=-128.0)
            for route_base in pl.range(0, 160, 8):
                token = route_base // 8
                if token % 8 == block:
                    running = pl.tile.full([1, 2048], dtype=pl.FP32, value=0.0)
                    for slot in pl.range(8):
                        route = route_base + slot
                        expert = pl.tile.read(all_ids, [token, slot])
                        if expert >= 0:
                            if expert < 128:
                                integer = pl.tile.load(acc, [route, 0], [1, 2048])
                                channels = pl.tile.load(
                                    weight_scale, [expert, 0], [1, 2048]
                                )
                                activation = pl.tensor.read(input_scale, [route, 0])
                                route_weight = pl.tensor.read(
                                    route_weights, [token, slot]
                                )
                                values = pl.tile.cast(integer, pl.FP32, mode="none")
                                channel_values = pl.tile.mul(values, channels)
                                scaled_values = pl.tile.muls(channel_values, activation)
                                rounded = pl.tile.cast(
                                    scaled_values, pl.FP16, mode="none"
                                )
                                if not fuse_quant:
                                    pl.tile.store(rounded, [route, 0], expert_y)
                                expert_float = pl.tile.cast(
                                    rounded, pl.FP32, mode="none"
                                )
                                weighted = pl.tile.muls(expert_float, route_weight)
                                running = pl.tile.add(running, weighted)
                    summed_half = pl.tile.cast(running, pl.FP16, mode="none")
                    pl.tile.store(summed_half, [token, 0], y)
                    summed = pl.tile.cast(summed_half, pl.FP32, mode="none")
                    skip_half = pl.tile.load(residual, [token, 0], [1, 2048])
                    skip_float = pl.tile.cast(skip_half, pl.FP32, mode="none")
                    full_residual = pl.tile.add(summed, skip_float)
                    rounded_residual = pl.tile.cast(full_residual, pl.FP16, mode="none")
                    pl.tile.store(rounded_residual, [token, 0], residual_out)
                    norm_values = pl.tile.cast(rounded_residual, pl.FP32, mode="none")
                    squares = pl.tile.mul(norm_values, norm_values)
                    mean_squares = pl.tile.muls(squares, 1.0 / 2048.0)
                    scratch = pl.tile.create([1, 2048], dtype=pl.FP32)
                    reduced = pl.tile.row_sum(mean_squares, scratch)
                    variance_scalar = pl.tile.read(reduced, [0, 0])
                    variance = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0)
                    pl.tile.write(variance, [0, 0], variance_scalar)
                    adjusted = pl.tile.adds(variance, 1e-6)
                    root = pl.tile.sqrt(adjusted)
                    zeros = pl.tile.muls(root, 0.0)
                    ones = pl.tile.adds(zeros, 1.0)
                    inverse = pl.tile.div(ones, root)
                    inverse_scalar = pl.tile.read(inverse, [0, 0])
                    normalized = pl.tile.muls(norm_values, inverse_scalar)
                    normalized_half = pl.tile.cast(normalized, pl.FP16, mode="none")
                    normalized_float = pl.tile.cast(
                        normalized_half, pl.FP32, mode="none"
                    )
                    weighted_norm = pl.tile.mul(normalized_float, gamma_float)
                    weighted_half = pl.tile.cast(weighted_norm, pl.FP16, mode="none")
                    weighted_float = pl.tile.cast(weighted_half, pl.FP32, mode="none")
                    biased = pl.tile.add(weighted_float, beta_float)
                    output = pl.tile.cast(biased, pl.FP16, mode="none")
                    pl.tile.store(output, [token, 0], norm_y)
                    if fuse_quant:
                        quant_values = pl.tile.cast(output, pl.FP32, mode="none")
                        multiplied = pl.tile.mul(quant_values, quant_inverse)
                        scaled = pl.tile.add(multiplied, zero)
                        bounded_high = pl.tile.minimum(scaled, ceiling)
                        bounded = pl.tile.maximum(bounded_high, floor)
                        quant_rounded = pl.tile.cast(bounded, pl.INT32, mode="rint")
                        integral_float = pl.tile.cast(quant_rounded, pl.FP32, mode="none")
                        integral_half = pl.tile.cast(integral_float, pl.FP16, mode="none")
                        narrowed = pl.tile.cast(integral_half, pl.INT8, mode="trunc")
                        pl.tile.store(narrowed, [token, 0], quant)
            pl.tensor.write(status, [block, 0], success_code)
        else:
            pl.tensor.write(status, [block, 0], invalid_code)
    return expert_y, y, status, norm_y, residual_out


@pl.jit.incore
def resident_expert_combine_norm(
    acc: pl.Tensor[[160, 2048], pl.INT32],
    input_scale: pl.Tensor[[160, 16], pl.FP32],
    ids: pl.Tensor[[20, 8], pl.INT32],
    weight_scale: pl.Tensor[[128, 2048], pl.FP32],
    route_weights: pl.Tensor[[20, 8], pl.FP32],
    expert_y: pl.InOut[pl.Tensor[[160, 2048], pl.FP16]],
    y: pl.InOut[pl.Tensor[[20, 2048], pl.FP16]],
    status: pl.InOut[pl.Tensor[[8, 16], pl.INT32]],
    residual: pl.Tensor[[20, 2048], pl.FP16],
    gamma: pl.Tensor[[1, 2048], pl.FP16],
    beta: pl.Tensor[[1, 2048], pl.FP16],
    norm_y: pl.InOut[pl.Tensor[[20, 2048], pl.FP16]],
    residual_out: pl.InOut[pl.Tensor[[20, 2048], pl.FP16]],
):
    return _combine_norm_impl(
        acc, input_scale, ids, weight_scale, route_weights, expert_y, y, status,
        residual, gamma, beta, norm_y, residual_out, gamma, beta, residual_out, False,
    )


@pl.jit.incore
def resident_expert_combine_norm_quant(
    acc: pl.Tensor[[160, 2048], pl.INT32],
    input_scale: pl.Tensor[[160, 16], pl.FP32],
    ids: pl.Tensor[[20, 8], pl.INT32],
    weight_scale: pl.Tensor[[128, 2048], pl.FP32],
    route_weights: pl.Tensor[[20, 8], pl.FP32],
    expert_y: pl.InOut[pl.Tensor[[160, 2048], pl.FP16]],
    y: pl.InOut[pl.Tensor[[20, 2048], pl.FP16]],
    status: pl.InOut[pl.Tensor[[8, 16], pl.INT32]],
    residual: pl.Tensor[[20, 2048], pl.FP16],
    gamma: pl.Tensor[[1, 2048], pl.FP16],
    beta: pl.Tensor[[1, 2048], pl.FP16],
    norm_y: pl.InOut[pl.Tensor[[20, 2048], pl.FP16]],
    residual_out: pl.InOut[pl.Tensor[[20, 2048], pl.FP16]],
    reciprocal: pl.Tensor[[1, 2048], pl.FP16],
    offset: pl.Tensor[[1, 2048], pl.FP16],
    quant: pl.InOut[pl.Tensor[[20, 2048], pl.INT8]],
):
    _combined = _combine_norm_impl(
        acc, input_scale, ids, weight_scale, route_weights, expert_y, y, status,
        residual, gamma, beta, norm_y, residual_out, reciprocal, offset, quant, True,
    )
    return expert_y, y, status, norm_y, residual_out, quant


@pl.jit.incore
def resident_expert_combine(acc: pl.Tensor[[160, 2048], pl.INT32], input_scale: pl.Tensor[[160, 16], pl.FP32], ids: pl.Tensor[[20, 8], pl.INT32], weight_scale: pl.Tensor[[128, 2048], pl.FP32], route_weights: pl.Tensor[[20, 8], pl.FP32], expert_y: pl.InOut[pl.Tensor[[160, 2048], pl.FP16]], y: pl.InOut[pl.Tensor[[20, 2048], pl.FP16]], status: pl.InOut[pl.Tensor[[8, 16], pl.INT32]]):
    block = pl.tile.get_block_idx()
    if block < 8:
        all_ids = pl.tile.load(ids, [0, 0], [20, 8])
        valid: pl.Scalar[pl.INT32] = 1
        invalid: pl.Scalar[pl.INT32] = 0
        success_code: pl.Scalar[pl.INT32] = 0
        invalid_code: pl.Scalar[pl.INT32] = 2
        for token in pl.range(20):
            for slot in pl.range(8):
                expert = pl.tile.read(all_ids, [token, slot])
                if expert < 0:
                    valid = invalid
                if expert >= 128:
                    valid = invalid
        if valid == 1:
            for route_base in pl.range(0, 160, 8):
                token = route_base // 8
                if token % 8 == block:
                    running = pl.tile.full([1, 2048], dtype=pl.FP32, value=0.0)
                    for slot in pl.range(8):
                        route = route_base + slot
                        expert = pl.tile.read(all_ids, [token, slot])
                        if expert >= 0:
                            if expert < 128:
                                integer = pl.tile.load(acc, [route, 0], [1, 2048])
                                channels = pl.tile.load(weight_scale, [expert, 0], [1, 2048])
                                activation = pl.tensor.read(input_scale, [route, 0])
                                route_weight = pl.tensor.read(route_weights, [token, slot])
                                values = pl.tile.cast(integer, pl.FP32, mode='none')
                                channel_values = pl.tile.mul(values, channels)
                                scaled_values = pl.tile.muls(channel_values, activation)
                                rounded = pl.tile.cast(scaled_values, pl.FP16, mode='none')
                                pl.tile.store(rounded, [route, 0], expert_y)
                                expert_float = pl.tile.cast(rounded, pl.FP32, mode='none')
                                weighted = pl.tile.muls(expert_float, route_weight)
                                running = pl.tile.add(running, weighted)
                    summed_half = pl.tile.cast(running, pl.FP16, mode='none')
                    pl.tile.store(summed_half, [token, 0], y)
            pl.tensor.write(status, [block, 0], success_code)
        else:
            pl.tensor.write(status, [block, 0], invalid_code)
    return (expert_y, y, status)
