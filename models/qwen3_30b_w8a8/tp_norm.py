# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -------------------------------------------------------------------------------------------------------
"""TP2 BF16 reduction, residual addition and ordered RMSNorm for one decode row."""

import pypto.language as pl
import pypto.language.distributed as pld

from .config import HIDDEN
from .norm import NATIVE_REDUCE_LANES, NORM_ROWS, RMS_EPS
from .quant import INT8_MAX

# Communication slots.
TP_OPERATIONS = 96
SIGNAL_TILE = 32


@pl.jit.inline
def tp_add_rms_norm_hidden_impl(
    x: pl.Tensor[[1, HIDDEN], pl.BF16],
    residual: pl.Tensor[[1, HIDDEN], pl.BF16],
    weight: pl.Tensor[[1, HIDDEN], pl.BF16],
    reduced: pl.Tensor[[1, HIDDEN], pl.BF16],
    y: pl.Tensor[[1, HIDDEN], pl.BF16],
    residual_out: pl.Tensor[[1, HIDDEN], pl.BF16],
    data: pld.DistributedTensor[[TP_OPERATIONS, HIDDEN], pl.BF16],
    signal: pld.DistributedTensor[[TP_OPERATIONS * 4, SIGNAL_TILE], pl.INT32],
    epochs: pl.Tensor[[TP_OPERATIONS], pl.INT32],
    operation: pl.Scalar[pl.INT32],
) -> tuple[pl.Tensor, pl.Tensor, pl.Tensor]:
    for _block in pl.spmd(1, name_hint="tp_add_rms_norm_hidden", allow_early_resolve=True):
        ctx = pld.system.get_comm_ctx(data)
        rank = pld.system.rank(ctx)
        peer = 1 - rank
        epoch = pl.cast(pl.read(epochs, [operation]) + 1, pl.INT32)
        ready = operation * 4
        consumed = ready + 2

        # A slot cannot be overwritten until the peer has consumed its previous payload.
        pld.system.wait(signal, [consumed + peer, 0], epoch - 1, cmp=pld.WaitCmp.Ge)
        local_bf16 = pl.load(x, [0, 0], [1, HIDDEN])
        local = pl.cast(local_bf16, pl.FP32)
        data = pl.store(local_bf16, [operation, 0], data)
        pld.system.notify(signal, peer, [ready + rank, 0], epoch, op=pld.NotifyOp.Set)
        pld.system.wait(signal, [ready + peer, 0], epoch, cmp=pld.WaitCmp.Ge)
        other_bf16 = pld.tile.remote_load(data, peer, [operation, 0], [1, HIDDEN])
        other = pl.cast(other_bf16, pl.FP32)
        total = pl.add(local, other)
        narrowed = pl.cast(total, pl.BF16, mode="rint")
        reduced = pl.store(narrowed, [0, 0], reduced)
        pld.system.notify(signal, peer, [consumed + rank, 0], epoch, op=pld.NotifyOp.Set)

        values = pl.cast(pl.load(reduced, [0, 0], [NORM_ROWS, HIDDEN], valid_shape=[1, HIDDEN]), pl.FP32)
        skip = pl.cast(pl.load(residual, [0, 0], [NORM_ROWS, HIDDEN], valid_shape=[1, HIDDEN]), pl.FP32)
        summed = pl.cast(pl.cast(pl.add(values, skip), pl.BF16, mode="rint"), pl.FP32)
        residual_out = pl.store(pl.cast(summed, pl.BF16, mode="rint"), [0, 0], residual_out)
        scaled_square = pl.mul(pl.mul(summed, summed), 1.0 / HIDDEN)
        partial = pl.slice(scaled_square, [NORM_ROWS, NATIVE_REDUCE_LANES], [0, 0], valid_shape=[1, NATIVE_REDUCE_LANES])
        for column in pl.range(NATIVE_REDUCE_LANES, HIDDEN, NATIVE_REDUCE_LANES):
            chunk = pl.slice(scaled_square, [NORM_ROWS, NATIVE_REDUCE_LANES], [0, column], valid_shape=[1, NATIVE_REDUCE_LANES])
            partial = pl.add(partial, chunk)
        scratch = pl.create_tile([NORM_ROWS, NATIVE_REDUCE_LANES], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        variance = pl.row_sum(partial, scratch)
        root_mean_square = pl.sqrt(pl.add(variance, RMS_EPS))
        ones = pl.add(pl.mul(root_mean_square, 0.0), 1.0)
        inv_rms = pl.div(ones, root_mean_square)
        values = pl.cast(pl.load(reduced, [0, 0], [NORM_ROWS, HIDDEN], valid_shape=[1, HIDDEN]), pl.FP32)
        skip = pl.cast(pl.load(residual, [0, 0], [NORM_ROWS, HIDDEN], valid_shape=[1, HIDDEN]), pl.FP32)
        summed = pl.cast(pl.cast(pl.add(values, skip), pl.BF16, mode="rint"), pl.FP32)
        normalized = pl.cast(pl.cast(pl.row_expand_mul(summed, inv_rms), pl.BF16, mode="rint"), pl.FP32)
        gamma = pl.cast(pl.load(weight, [0, 0], [1, HIDDEN]), pl.FP32)
        normalized = pl.col_expand_mul(normalized, gamma)
        y = pl.store(pl.cast(normalized, pl.BF16, mode="rint"), [0, 0], y)
        pl.write(epochs, [operation], epoch)
    return reduced, y, residual_out


@pl.jit
def tp_add_rms_norm_hidden(
    x: pl.Tensor[[1, HIDDEN], pl.BF16],
    residual: pl.Tensor[[1, HIDDEN], pl.BF16],
    weight: pl.Tensor[[1, HIDDEN], pl.BF16],
    reduced: pl.Out[pl.Tensor[[1, HIDDEN], pl.BF16]],
    y: pl.Out[pl.Tensor[[1, HIDDEN], pl.BF16]],
    residual_out: pl.Out[pl.Tensor[[1, HIDDEN], pl.BF16]],
    data: pl.InOut[pld.DistributedTensor[[TP_OPERATIONS, HIDDEN], pl.BF16]],
    signal: pl.InOut[pld.DistributedTensor[[TP_OPERATIONS * 4, SIGNAL_TILE], pl.INT32]],
    epochs: pl.InOut[pl.Tensor[[TP_OPERATIONS], pl.INT32]],
    operation: pl.Scalar[pl.INT32],
) -> tuple[pl.Tensor, pl.Tensor, pl.Tensor]:
    return tp_add_rms_norm_hidden_impl(
        x, residual, weight, reduced, y, residual_out,
        data, signal, epochs, operation,
    )


@pl.jit.inline
def tp_reduce_hidden_impl(
    x: pl.Tensor[[1, HIDDEN], pl.BF16],
    y: pl.Tensor[[1, HIDDEN], pl.BF16],
    data: pld.DistributedTensor[[TP_OPERATIONS, HIDDEN], pl.BF16],
    signal: pld.DistributedTensor[[TP_OPERATIONS * 4, SIGNAL_TILE], pl.INT32],
    epochs: pl.Tensor[[TP_OPERATIONS], pl.INT32],
    operation: pl.Scalar[pl.INT32],
) -> pl.Tensor:
    for _block in pl.spmd(1, name_hint="tp_reduce_hidden", allow_early_resolve=True):
        ctx = pld.system.get_comm_ctx(data)
        rank = pld.system.rank(ctx)
        peer = 1 - rank
        epoch = pl.cast(pl.read(epochs, [operation]) + 1, pl.INT32)
        ready = operation * 4
        consumed = ready + 2
        pld.system.wait(signal, [consumed + peer, 0], epoch - 1, cmp=pld.WaitCmp.Ge)
        values = pl.load(x, [0, 0], [1, HIDDEN])
        local = pl.cast(values, pl.FP32)
        data = pl.store(values, [operation, 0], data)
        pld.system.notify(signal, peer, [ready + rank, 0], epoch, op=pld.NotifyOp.Set)
        pld.system.wait(signal, [ready + peer, 0], epoch, cmp=pld.WaitCmp.Ge)
        other_bf16 = pld.tile.remote_load(data, peer, [operation, 0], [1, HIDDEN])
        other = pl.cast(other_bf16, pl.FP32)
        total = pl.add(local, other)
        narrowed = pl.cast(total, pl.BF16, mode="rint")
        y = pl.store(narrowed, [0, 0], y)
        pld.system.notify(signal, peer, [consumed + rank, 0], epoch, op=pld.NotifyOp.Set)
        pl.write(epochs, [operation], epoch)
    return y


@pl.jit
def tp_reduce_hidden(
    x: pl.Tensor[[1, HIDDEN], pl.BF16],
    y: pl.Out[pl.Tensor[[1, HIDDEN], pl.BF16]],
    data: pl.InOut[pld.DistributedTensor[[TP_OPERATIONS, HIDDEN], pl.BF16]],
    signal: pl.InOut[pld.DistributedTensor[[TP_OPERATIONS * 4, SIGNAL_TILE], pl.INT32]],
    epochs: pl.InOut[pl.Tensor[[TP_OPERATIONS], pl.INT32]],
    operation: pl.Scalar[pl.INT32],
) -> pl.Tensor:
    return tp_reduce_hidden_impl(x, y, data, signal, epochs, operation)


@pl.jit.inline
def tp_add_rms_quant_hidden_impl(
    x: pl.Tensor[[1, HIDDEN], pl.BF16],
    residual: pl.Tensor[[1, HIDDEN], pl.BF16],
    weight: pl.Tensor[[1, HIDDEN], pl.BF16],
    reduced: pl.Tensor[[1, HIDDEN], pl.BF16],
    y: pl.Tensor[[1, HIDDEN], pl.BF16],
    residual_out: pl.Tensor[[1, HIDDEN], pl.BF16],
    quant: pl.Tensor[[1, HIDDEN], pl.INT8],
    scale: pl.Tensor[[1, 1], pl.FP32],
    data: pld.DistributedTensor[[TP_OPERATIONS, HIDDEN], pl.BF16],
    signal: pld.DistributedTensor[[TP_OPERATIONS * 4, SIGNAL_TILE], pl.INT32],
    epochs: pl.Tensor[[TP_OPERATIONS], pl.INT32],
    operation: pl.Scalar[pl.INT32],
) -> tuple[pl.Tensor, pl.Tensor, pl.Tensor, pl.Tensor, pl.Tensor]:
    for _block in pl.spmd(1, name_hint="tp_bf16_norm_quant_retained", allow_early_resolve=True):
        ctx = pld.system.get_comm_ctx(data)
        rank = pld.system.rank(ctx)
        peer = 1 - rank
        epoch = pl.cast(pl.read(epochs, [operation]) + 1, pl.INT32)
        ready = operation * 4
        consumed = ready + 2
        pld.system.wait(signal, [consumed + peer, 0], epoch - 1, cmp=pld.WaitCmp.Ge)
        local_bf16 = pl.load(x, [0, 0], [1, HIDDEN])
        local = pl.cast(local_bf16, pl.FP32)
        data = pl.store(local_bf16, [operation, 0], data)
        pld.system.notify(signal, peer, [ready + rank, 0], epoch, op=pld.NotifyOp.Set)
        pld.system.wait(signal, [ready + peer, 0], epoch, cmp=pld.WaitCmp.Ge)
        other_bf16 = pld.tile.remote_load(data, peer, [operation, 0], [1, HIDDEN])
        other = pl.cast(other_bf16, pl.FP32)
        total = pl.add(local, other)
        narrowed = pl.cast(total, pl.BF16, mode="rint")
        reduced = pl.store(narrowed, [0, 0], reduced)
        pld.system.notify(signal, peer, [consumed + rank, 0], epoch, op=pld.NotifyOp.Set)

        # BF16 residual sum and native ordered variance.
        gamma = pl.cast(pl.load(weight, [0, 0], [1, HIDDEN]), pl.FP32)
        values = pl.cast(pl.load(reduced, [0, 0], [NORM_ROWS, HIDDEN], valid_shape=[1, HIDDEN]), pl.FP32)
        skip = pl.cast(pl.load(residual, [0, 0], [NORM_ROWS, HIDDEN], valid_shape=[1, HIDDEN]), pl.FP32)
        summed_bf16 = pl.cast(pl.add(values, skip), pl.BF16, mode="rint")
        residual_out = pl.store(summed_bf16, [0, 0], residual_out)
        summed = pl.cast(summed_bf16, pl.FP32)
        scaled_square = pl.mul(pl.mul(summed, summed), 1.0 / HIDDEN)
        partial = pl.slice(scaled_square, [NORM_ROWS, NATIVE_REDUCE_LANES], [0, 0],
                           valid_shape=[1, NATIVE_REDUCE_LANES])
        for column in pl.range(NATIVE_REDUCE_LANES, HIDDEN, NATIVE_REDUCE_LANES):
            chunk = pl.slice(scaled_square, [NORM_ROWS, NATIVE_REDUCE_LANES], [0, column],
                             valid_shape=[1, NATIVE_REDUCE_LANES])
            partial = pl.add(partial, chunk)
        scratch = pl.create_tile([NORM_ROWS, NATIVE_REDUCE_LANES], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        variance = pl.row_sum(partial, scratch)
        root_mean_square = pl.sqrt(pl.add(variance, RMS_EPS))
        ones = pl.add(pl.mul(root_mean_square, 0.0), 1.0)
        inverse_rms = pl.div(ones, root_mean_square)
        rms_value = pl.read(inverse_rms, [0, 0])
        row_x = pl.cast(pl.load(reduced, [0, 0], [1, HIDDEN]), pl.FP32)
        row_skip = pl.cast(pl.load(residual, [0, 0], [1, HIDDEN]), pl.FP32)
        row_sum = pl.cast(pl.cast(pl.add(row_x, row_skip), pl.BF16, mode="rint"), pl.FP32)
        normalized_bf16 = pl.cast(pl.mul(row_sum, rms_value), pl.BF16, mode="rint")
        weighted = pl.mul(pl.cast(normalized_bf16, pl.FP32), gamma)
        output_bf16 = pl.cast(weighted, pl.BF16, mode="rint")
        y = pl.store(output_bf16, [0, 0], y)

        # Retained BF16 output quantization and zero-row scale.
        row = pl.cast(output_bf16, pl.FP32)
        maximum_input = pl.reshape(pl.abs(row), [8, HIDDEN // 8])
        maximum_scratch = pl.create_tile([8, HIDDEN // 8], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        maxima = pl.reshape(pl.row_max(maximum_input, maximum_scratch), [1, 8])
        maximum = pl.read(maxima, [0, 0])
        for lane in pl.unroll(1, 8):
            candidate = pl.read(maxima, [0, lane])
            if candidate > maximum:
                maximum = candidate
        if maximum > 0.0:
            pl.write(maxima, [0, 0], maximum)
            numerator = pl.tile.full([1, 8], dtype=pl.FP32, value=INT8_MAX)
            inverses = pl.div(numerator, maxima)
            inverse = pl.read(inverses, [0, 0])
            pl.write(scale, [0, 0], pl.div(1.0, inverse))
            rounded = pl.cast(pl.mul(row, inverse), pl.INT32, mode="rint")
            quantized = pl.cast(pl.cast(rounded, pl.FP16), pl.INT8, mode="trunc")
            quant = pl.store(quantized, [0, 0], quant)
        else:
            quant = pl.store(pl.cast(row, pl.INT8, mode="rint"), [0, 0], quant)
            pl.write(scale, [0, 0], 0.0)
        pl.write(epochs, [operation], epoch)
    return reduced, y, residual_out, quant, scale


@pl.jit
def tp_add_rms_quant_hidden(
    x: pl.Tensor[[1, HIDDEN], pl.BF16],
    residual: pl.Tensor[[1, HIDDEN], pl.BF16],
    weight: pl.Tensor[[1, HIDDEN], pl.BF16],
    reduced: pl.Out[pl.Tensor[[1, HIDDEN], pl.BF16]],
    y: pl.Out[pl.Tensor[[1, HIDDEN], pl.BF16]],
    residual_out: pl.Out[pl.Tensor[[1, HIDDEN], pl.BF16]],
    quant: pl.Out[pl.Tensor[[1, HIDDEN], pl.INT8]],
    scale: pl.Out[pl.Tensor[[1, 1], pl.FP32]],
    data: pl.InOut[pld.DistributedTensor[[TP_OPERATIONS, HIDDEN], pl.BF16]],
    signal: pl.InOut[pld.DistributedTensor[[TP_OPERATIONS * 4, SIGNAL_TILE], pl.INT32]],
    epochs: pl.InOut[pl.Tensor[[TP_OPERATIONS], pl.INT32]],
    operation: pl.Scalar[pl.INT32],
) -> tuple[pl.Tensor, pl.Tensor, pl.Tensor, pl.Tensor, pl.Tensor]:
    return tp_add_rms_quant_hidden_impl(
        x, residual, weight, reduced, y, residual_out, quant, scale,
        data, signal, epochs, operation,
    )
