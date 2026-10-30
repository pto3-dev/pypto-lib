# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""collectives operators. Explicit PyPTO functions; no source generator at build time.

Migrated from the validated L1K2560 graph; tuning is audited separately.
"""

import pypto.language as pl
import pypto.language.distributed as pld


@pl.function(type=pl.FunctionType.InCore)
def reduce_embedding(
    x: pl.Tensor[[24, 5120], pl.BF16],
    out: pl.Out[pl.Tensor[[24, 5120], pl.BF16]],
    data: pl.InOut[pld.DistributedTensor[[3096, 5120], pl.BF16]],
    signal: pl.InOut[pld.DistributedTensor[[516, 1], pl.INT32]],
    metadata: pl.Tensor[[4], pl.INT32],
    operation: pl.Scalar[pl.INT32],
) -> tuple[
    pl.Tensor[[24, 5120], pl.BF16],
    pld.DistributedTensor[[3096, 5120], pl.BF16],
    pld.DistributedTensor[[516, 1], pl.INT32],
]:
    ctx = pld.get_comm_ctx(data)
    rank = pld.rank(ctx)
    epoch_tile = pl.tile.load(metadata, [0], [8], valid_shape=[4], target_memory=pl.Mem.Vec)
    epoch = pl.read(epoch_tile, [1])
    for row in pl.range(24):
        local = pl.cast(pl.load(x, [row, 0], [1, 5120]), pl.FP32)
        data = pl.store(pl.cast(local, pl.BF16, mode="rint"), [operation * 24 + row, 0], data)
    for peer in pl.range(4):
        if peer != rank:
            pld.system.notify(
                signal, peer=peer, offsets=[operation * 4 + rank, 0], value=1, op=pld.NotifyOp.AtomicAdd
            )
    for peer in pl.range(4):
        if peer != rank:
            pld.system.wait(signal, offsets=[operation * 4 + peer, 0], expected=epoch, cmp=pld.WaitCmp.Ge)
    for row in pl.range(24):
        total = pl.tile.full([1, 5120], dtype=pl.FP32, value=0.0)
        for peer in pl.range(4):
            if peer == rank:
                value = pl.load(data, [operation * 24 + row, 0], [1, 5120])
            else:
                value = pld.tile.remote_load(
                    data, peer=peer, offsets=[operation * 24 + row, 0], shape=[1, 5120]
                )
            total = pl.add(total, pl.cast(value, pl.FP32))
        narrowed = pl.cast(total, pl.BF16, mode="rint")
        out = pl.store(narrowed, [row, 0], out)
    return (out, data, signal)


@pl.function(type=pl.FunctionType.InCore)
def reduce_partial(
    x: pl.Tensor[[24, 5120], pl.FP32],
    out: pl.Out[pl.Tensor[[24, 5120], pl.BF16]],
    data: pl.InOut[pld.DistributedTensor[[3096, 5120], pl.BF16]],
    signal: pl.InOut[pld.DistributedTensor[[516, 1], pl.INT32]],
    metadata: pl.Tensor[[4], pl.INT32],
    operation: pl.Scalar[pl.INT32],
) -> tuple[
    pl.Tensor[[24, 5120], pl.BF16],
    pld.DistributedTensor[[3096, 5120], pl.BF16],
    pld.DistributedTensor[[516, 1], pl.INT32],
]:
    ctx = pld.get_comm_ctx(data)
    rank = pld.rank(ctx)
    epoch_tile = pl.tile.load(metadata, [0], [8], valid_shape=[4], target_memory=pl.Mem.Vec)
    epoch = pl.read(epoch_tile, [1])
    for row in pl.range(24):
        local = pl.load(x, [row, 0], [1, 5120])
        local = pl.cast(pl.cast(local, pl.BF16, mode="rint"), pl.FP32)
        data = pl.store(pl.cast(local, pl.BF16, mode="rint"), [operation * 24 + row, 0], data)
    for peer in pl.range(4):
        if peer != rank:
            pld.system.notify(
                signal, peer=peer, offsets=[operation * 4 + rank, 0], value=1, op=pld.NotifyOp.AtomicAdd
            )
    for peer in pl.range(4):
        if peer != rank:
            pld.system.wait(signal, offsets=[operation * 4 + peer, 0], expected=epoch, cmp=pld.WaitCmp.Ge)
    for row in pl.range(24):
        total = pl.tile.full([1, 5120], dtype=pl.FP32, value=0.0)
        for peer in pl.range(4):
            if peer == rank:
                value = pl.load(data, [operation * 24 + row, 0], [1, 5120])
            else:
                value = pld.tile.remote_load(
                    data, peer=peer, offsets=[operation * 24 + row, 0], shape=[1, 5120]
                )
            total = pl.add(total, pl.cast(value, pl.FP32))
        narrowed = pl.cast(total, pl.BF16, mode="rint")
        out = pl.store(narrowed, [row, 0], out)
    return (out, data, signal)


@pl.function(type=pl.FunctionType.InCore)
def gather_logits_kernel(
    local: pl.Tensor[[24, 62080], pl.BF16],
    out: pl.Out[pl.Tensor[[24, 248320], pl.FP32]],
    data: pl.InOut[pld.DistributedTensor[[24, 62080], pl.BF16]],
    signal: pl.InOut[pld.DistributedTensor[[768, 32], pl.INT32]],
    metadata: pl.Tensor[[4], pl.INT32],
) -> tuple[
    pl.Tensor[[24, 248320], pl.FP32],
    pld.DistributedTensor[[24, 62080], pl.BF16],
    pld.DistributedTensor[[768, 32], pl.INT32],
]:
    pl.func_attr({"static_block_mapping": True})
    ctx = pld.get_comm_ctx(data)
    rank = pld.rank(ctx)
    epoch_tile = pl.tile.load(metadata, [0], [8], valid_shape=[4], target_memory=pl.Mem.Vec)
    epoch = pl.read(epoch_tile, [1])
    lane = pl.tile.get_block_idx()
    lanes = pl.tile.get_block_num()
    for peer in pl.range(4):
        if peer != rank:
            pld.system.wait(
                signal, offsets=[(4 + peer) * 96 + lane, 0], expected=epoch - 1, cmp=pld.WaitCmp.Ge
            )
    for job in pl.range(lane, 2328, lanes):
        row = job // 97
        col = job % 97 * 640
        value = pl.load(local, [row, col], [1, 640])
        data = pl.store(value, [row, col], data)
    for peer in pl.range(4):
        if peer != rank:
            pld.system.notify(
                signal, peer=peer, offsets=[rank * 96 + lane, 0], value=epoch, op=pld.NotifyOp.Set
            )
    for peer in pl.range(4):
        if peer != rank:
            pld.system.wait(signal, offsets=[peer * 96 + lane, 0], expected=epoch, cmp=pld.WaitCmp.Ge)
    for job in pl.range(lane, 2328, lanes):
        row = job // 97
        col = job % 97 * 640
        for peer in pl.range(4):
            if peer == rank:
                value = pl.load(data, [row, col], [1, 640])
            else:
                value = pld.tile.remote_load(data, peer=peer, offsets=[row, col], shape=[1, 640])
            out = pl.store(pl.cast(value, pl.FP32), [row, peer * 62080 + col], out)
    for peer in pl.range(4):
        if peer != rank:
            pld.system.notify(
                signal, peer=peer, offsets=[(4 + rank) * 96 + lane, 0], value=epoch, op=pld.NotifyOp.Set
            )
    return (out, data, signal)


@pl.function(type=pl.FunctionType.Inline)
def gather_logits(
    local: pl.Tensor[[24, 62080], pl.BF16],
    out: pl.Out[pl.Tensor[[24, 248320], pl.FP32]],
    data: pl.InOut[pld.DistributedTensor[[24, 62080], pl.BF16]],
    signal: pl.InOut[pld.DistributedTensor[[768, 32], pl.INT32]],
    metadata: pl.Tensor[[4], pl.INT32],
) -> tuple[
    pl.Tensor[[24, 248320], pl.FP32],
    pld.DistributedTensor[[24, 62080], pl.BF16],
    pld.DistributedTensor[[768, 32], pl.INT32],
]:
    available = pl.system.available_aiv_count()
    width = pl.min(available, 96)
    with pl.spmd(width, allow_early_resolve=True):
        out, data, signal = gather_logits_kernel(local, out, data, signal, metadata)
    return (out, data, signal)


@pl.function(type=pl.FunctionType.InCore)
def reduce_add_kernel(
    x: pl.Tensor[[24, 5120], pl.FP32],
    residual: pl.Tensor[[24, 5120], pl.BF16],
    out: pl.Out[pl.Tensor[[24, 5120], pl.BF16]],
    data: pl.InOut[pld.DistributedTensor[[3096, 5120], pl.BF16]],
    signal: pl.InOut[pld.DistributedTensor[[516, 1], pl.INT32]],
    metadata: pl.Tensor[[4], pl.INT32],
    operation: pl.Scalar[pl.INT32],
) -> tuple[
    pl.Tensor[[24, 5120], pl.BF16],
    pld.DistributedTensor[[3096, 5120], pl.BF16],
    pld.DistributedTensor[[516, 1], pl.INT32],
]:
    pl.func_attr({"static_block_mapping": True})
    ctx = pld.get_comm_ctx(data)
    rank = pld.rank(ctx)
    epoch = pl.read(metadata, [1])
    lane = pl.tile.get_block_idx()
    width = pl.tile.get_block_num()
    for row in pl.range(lane, 24, width):
        local = pl.load(x, [row, 0], [1, 5120])
        local_bf16 = pl.cast(local, pl.BF16, mode="rint")
        data = pl.store(local_bf16, [operation * 24 + row, 0], data)
    for peer in pl.range(4):
        if peer != rank:
            pld.system.notify(
                signal, peer=peer, offsets=[operation * 4 + rank, 0], value=1, op=pld.NotifyOp.AtomicAdd
            )
    for peer in pl.range(4):
        if peer != rank:
            pld.system.wait(
                signal, offsets=[operation * 4 + peer, 0], expected=epoch * width, cmp=pld.WaitCmp.Ge
            )
    for row in pl.range(lane, 24, width):
        total = pl.tile.full([1, 5120], dtype=pl.FP32, value=0.0)
        for peer in pl.range(4):
            if peer == rank:
                value = pl.load(data, [operation * 24 + row, 0], [1, 5120])
            else:
                value = pld.tile.remote_load(
                    data, peer=peer, offsets=[operation * 24 + row, 0], shape=[1, 5120]
                )
            total = pl.add(total, pl.cast(value, pl.FP32))
        reduced = pl.cast(total, pl.BF16, mode="rint")
        previous = pl.load(residual, [row, 0], [1, 5120])
        added = pl.add(pl.cast(previous, pl.FP32), pl.cast(reduced, pl.FP32))
        rounded = pl.cast(added, pl.BF16, mode="rint")
        out = pl.store(rounded, [row, 0], out)
    return (out, data, signal)


@pl.function(type=pl.FunctionType.Inline)
def reduce_add(
    x: pl.Tensor[[24, 5120], pl.FP32],
    residual: pl.Tensor[[24, 5120], pl.BF16],
    out: pl.Out[pl.Tensor[[24, 5120], pl.BF16]],
    data: pl.InOut[pld.DistributedTensor[[3096, 5120], pl.BF16]],
    signal: pl.InOut[pld.DistributedTensor[[516, 1], pl.INT32]],
    metadata: pl.Tensor[[4], pl.INT32],
    operation: pl.Scalar[pl.INT32],
) -> tuple[
    pl.Tensor[[24, 5120], pl.BF16],
    pld.DistributedTensor[[3096, 5120], pl.BF16],
    pld.DistributedTensor[[516, 1], pl.INT32],
]:
    available = pl.system.available_aiv_count()
    width = pl.min(available, 24)
    with pl.spmd(width, allow_early_resolve=True):
        out, data, signal = reduce_add_kernel(x, residual, out, data, signal, metadata, operation)
    return (out, data, signal)


@pl.function(type=pl.FunctionType.InCore)
def reduce_add_norm_kernel(
    x: pl.Tensor[[24, 5120], pl.FP32],
    residual: pl.Tensor[[24, 5120], pl.BF16],
    out: pl.Out[pl.Tensor[[24, 5120], pl.BF16]],
    weight: pl.Tensor[[1, 5120], pl.BF16],
    normalized: pl.Out[pl.Tensor[[24, 5120], pl.BF16]],
    data: pl.InOut[pld.DistributedTensor[[3096, 5120], pl.BF16]],
    signal: pl.InOut[pld.DistributedTensor[[516, 1], pl.INT32]],
    metadata: pl.Tensor[[4], pl.INT32],
    operation: pl.Scalar[pl.INT32],
) -> tuple[
    pl.Tensor[[24, 5120], pl.BF16],
    pl.Tensor[[24, 5120], pl.BF16],
    pld.DistributedTensor[[3096, 5120], pl.BF16],
    pld.DistributedTensor[[516, 1], pl.INT32],
]:
    pl.func_attr({"static_block_mapping": True})
    ctx = pld.get_comm_ctx(data)
    rank = pld.rank(ctx)
    epoch = pl.read(metadata, [1])
    lane = pl.tile.get_block_idx()
    width = pl.tile.get_block_num()
    for row in pl.range(lane, 24, width):
        local = pl.load(x, [row, 0], [1, 5120])
        local_bf16 = pl.cast(local, pl.BF16, mode="rint")
        data = pl.store(local_bf16, [operation * 24 + row, 0], data)
    for peer in pl.range(4):
        if peer != rank:
            pld.system.notify(
                signal, peer=peer, offsets=[operation * 4 + rank, 0], value=1, op=pld.NotifyOp.AtomicAdd
            )
    for peer in pl.range(4):
        if peer != rank:
            pld.system.wait(
                signal, offsets=[operation * 4 + peer, 0], expected=epoch * width, cmp=pld.WaitCmp.Ge
            )
    for row in pl.range(lane, 24, width):
        total = pl.tile.full([1, 5120], dtype=pl.FP32, value=0.0)
        for peer in pl.range(4):
            if peer == rank:
                value = pl.load(data, [operation * 24 + row, 0], [1, 5120])
            else:
                value = pld.tile.remote_load(
                    data, peer=peer, offsets=[operation * 24 + row, 0], shape=[1, 5120]
                )
            total = pl.add(total, pl.cast(value, pl.FP32))
        reduced = pl.cast(total, pl.BF16, mode="rint")
        previous = pl.load(residual, [row, 0], [1, 5120])
        added = pl.add(pl.cast(previous, pl.FP32), pl.cast(reduced, pl.FP32))
        rounded = pl.cast(added, pl.BF16, mode="rint")
        out = pl.store(rounded, [row, 0], out)
        whole = pl.cast(rounded, pl.FP32)
        chunks = pl.tile.reshape(whole, [40, 128])
        temporary = pl.tile.create([40, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        partials = pl.tile.row_sum(pl.mul(chunks, chunks), temporary)
        summation = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0)
        for group in pl.range(40):
            summation = pl.add(summation, pl.read(partials, [group, 0]))
        variance = pl.add(pl.mul(summation, 1.0 / 5120), 1e-06)
        inv = pl.tile.rsqrt(variance)
        inv = pl.mul(inv, pl.add(pl.mul(pl.mul(variance, pl.mul(inv, inv)), -0.5), 1.5))
        scalar_inv = pl.read(inv, [0, 0])
        gamma = pl.add(pl.cast(pl.load(weight, [0, 0], [1, 5120]), pl.FP32), 1.0)
        gamma_rounded = pl.cast(pl.cast(gamma, pl.BF16, mode="rint"), pl.FP32)
        scaled = pl.cast(pl.cast(pl.mul(whole, scalar_inv), pl.BF16, mode="rint"), pl.FP32)
        result = pl.cast(pl.mul(scaled, gamma_rounded), pl.BF16, mode="rint")
        normalized = pl.store(result, [row, 0], normalized)
    return (out, normalized, data, signal)


@pl.function(type=pl.FunctionType.Inline)
def reduce_add_norm(
    x: pl.Tensor[[24, 5120], pl.FP32],
    residual: pl.Tensor[[24, 5120], pl.BF16],
    out: pl.Out[pl.Tensor[[24, 5120], pl.BF16]],
    weight: pl.Tensor[[1, 5120], pl.BF16],
    normalized: pl.Out[pl.Tensor[[24, 5120], pl.BF16]],
    data: pl.InOut[pld.DistributedTensor[[3096, 5120], pl.BF16]],
    signal: pl.InOut[pld.DistributedTensor[[516, 1], pl.INT32]],
    metadata: pl.Tensor[[4], pl.INT32],
    operation: pl.Scalar[pl.INT32],
) -> tuple[
    pl.Tensor[[24, 5120], pl.BF16],
    pl.Tensor[[24, 5120], pl.BF16],
    pld.DistributedTensor[[3096, 5120], pl.BF16],
    pld.DistributedTensor[[516, 1], pl.INT32],
]:
    available = pl.system.available_aiv_count()
    width = pl.min(available, 24)
    with pl.spmd(width, allow_early_resolve=True):
        out, normalized, data, signal = reduce_add_norm_kernel(
            x, residual, out, weight, normalized, data, signal, metadata, operation
        )
    return (out, normalized, data, signal)


@pl.function(type=pl.FunctionType.InCore)
def publish_hidden(
    x: pl.Tensor[[24, 5120], pl.BF16], out: pl.Out[pl.Tensor[[24, 5120], pl.BF16]]
) -> pl.Tensor[[24, 5120], pl.BF16]:
    for row in pl.range(24):
        value = pl.load(x, [row, 0], [1, 5120])
        out = pl.store(value, [row, 0], out)
    return out


@pl.function(type=pl.FunctionType.InCore)
def graph_stop_fence(controls: pl.InOut[pl.Tensor[[1, 8165], pl.INT32]]) -> pl.Tensor[[1, 8165], pl.INT32]:
    value = pl.load(controls, [0, 0], [1, 8168], [1, 8165])
    controls = pl.store(value, [0, 0], controls)
    return controls


@pl.function(type=pl.FunctionType.InCore)
def reduce_add_norm_quant_kernel(
    x: pl.Tensor[[24, 5120], pl.FP32],
    residual: pl.Tensor[[24, 5120], pl.BF16],
    out: pl.Out[pl.Tensor[[24, 5120], pl.BF16]],
    weight: pl.Tensor[[1, 5120], pl.BF16],
    normalized: pl.Out[pl.Tensor[[24, 5120], pl.BF16]],
    data: pl.InOut[pld.DistributedTensor[[3096, 5120], pl.BF16]],
    signal: pl.InOut[pld.DistributedTensor[[516, 1], pl.INT32]],
    metadata: pl.Tensor[[4], pl.INT32],
    operation: pl.Scalar[pl.INT32],
    quantized: pl.Out[pl.Tensor[[24, 5120], pl.INT8]],
    activation_scales: pl.Out[pl.Tensor[[24, 1], pl.FP32]],
) -> tuple[
    pl.Tensor[[24, 5120], pl.BF16],
    pl.Tensor[[24, 5120], pl.BF16],
    pld.DistributedTensor[[3096, 5120], pl.BF16],
    pld.DistributedTensor[[516, 1], pl.INT32],
    pl.Tensor[[24, 5120], pl.INT8],
    pl.Tensor[[24, 1], pl.FP32],
]:
    pl.func_attr({"static_block_mapping": True})
    ctx = pld.get_comm_ctx(data)
    rank = pld.rank(ctx)
    epoch = pl.read(metadata, [1])
    lane = pl.tile.get_block_idx()
    width = pl.tile.get_block_num()
    for row in pl.range(lane, 24, width):
        local = pl.load(x, [row, 0], [1, 5120])
        local_bf16 = pl.cast(local, pl.BF16, mode="rint")
        data = pl.store(local_bf16, [operation * 24 + row, 0], data)
    for peer in pl.range(4):
        if peer != rank:
            pld.system.notify(
                signal, peer=peer, offsets=[operation * 4 + rank, 0], value=1, op=pld.NotifyOp.AtomicAdd
            )
    for peer in pl.range(4):
        if peer != rank:
            pld.system.wait(
                signal, offsets=[operation * 4 + peer, 0], expected=epoch * width, cmp=pld.WaitCmp.Ge
            )
    for row in pl.range(lane, 24, width):
        total = pl.tile.full([1, 5120], dtype=pl.FP32, value=0.0)
        for peer in pl.range(4):
            if peer == rank:
                value = pl.load(data, [operation * 24 + row, 0], [1, 5120])
            else:
                value = pld.tile.remote_load(
                    data, peer=peer, offsets=[operation * 24 + row, 0], shape=[1, 5120]
                )
            total = pl.add(total, pl.cast(value, pl.FP32))
        reduced = pl.cast(total, pl.BF16, mode="rint")
        previous = pl.load(residual, [row, 0], [1, 5120])
        added = pl.add(pl.cast(previous, pl.FP32), pl.cast(reduced, pl.FP32))
        rounded = pl.cast(added, pl.BF16, mode="rint")
        out = pl.store(rounded, [row, 0], out)
        whole = pl.cast(rounded, pl.FP32)
        chunks = pl.tile.reshape(whole, [40, 128])
        temporary = pl.tile.create([40, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        partials = pl.tile.row_sum(pl.mul(chunks, chunks), temporary)
        summation = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0)
        for group in pl.range(40):
            summation = pl.add(summation, pl.read(partials, [group, 0]))
        variance = pl.add(pl.mul(summation, 1.0 / 5120), 1e-06)
        inv = pl.tile.rsqrt(variance)
        inv = pl.mul(inv, pl.add(pl.mul(pl.mul(variance, pl.mul(inv, inv)), -0.5), 1.5))
        scalar_inv = pl.read(inv, [0, 0])
        gamma = pl.add(pl.cast(pl.load(weight, [0, 0], [1, 5120]), pl.FP32), 1.0)
        gamma_rounded = pl.cast(pl.cast(gamma, pl.BF16, mode="rint"), pl.FP32)
        scaled = pl.cast(pl.cast(pl.mul(whole, scalar_inv), pl.BF16, mode="rint"), pl.FP32)
        result = pl.cast(pl.mul(scaled, gamma_rounded), pl.BF16, mode="rint")
        normalized = pl.store(result, [row, 0], normalized)
        quant_value = pl.cast(result, pl.FP32)
        quant_chunks = pl.tile.reshape(pl.abs(quant_value), [8, 640])
        quant_tmp = pl.tile.create([8, 640], pl.FP32)
        quant_maximum = pl.row_max(quant_chunks, quant_tmp)
        quant_magnitude = pl.read(quant_maximum, [0, 0])
        for quant_part in pl.range(1, 8):
            quant_partial = pl.read(quant_maximum, [quant_part, 0])
            if quant_partial > quant_magnitude:
                quant_magnitude = quant_partial
        if quant_magnitude == 0.0:
            quant_normalized = pl.mul(quant_value, 0.0)
        else:
            quant_denominator = pl.mul(pl.tile.full([1, 8], pl.FP32, 1.0), quant_magnitude)
            quant_numerator = pl.tile.full([1, 8], pl.FP32, 127.0)
            quant_ratio_tile = pl.div(quant_numerator, quant_denominator)
            quant_ratio = pl.read(quant_ratio_tile, [0, 0])
            quant_normalized = pl.mul(quant_value, quant_ratio)
        quant_rounded = pl.cast(quant_normalized, pl.INT32, mode="rint")
        quant_half = pl.cast(quant_rounded, pl.FP16, mode="rint")
        quant_quantized = pl.cast(quant_half, pl.INT8, mode="trunc")
        quantized = pl.store(quant_quantized, [row, 0], quantized)
        quant_scale_tile = pl.mul(pl.tile.full([1, 8], pl.FP32, 1.0), quant_magnitude * (1.0 / 127.0))
        activation_scales = pl.store(
            pl.tile.set_validshape(quant_scale_tile, 1, 1), [row, 0], activation_scales
        )
    return (out, normalized, data, signal, quantized, activation_scales)


@pl.function(type=pl.FunctionType.Inline)
def reduce_add_norm_quant(
    x: pl.Tensor[[24, 5120], pl.FP32],
    residual: pl.Tensor[[24, 5120], pl.BF16],
    out: pl.Out[pl.Tensor[[24, 5120], pl.BF16]],
    weight: pl.Tensor[[1, 5120], pl.BF16],
    normalized: pl.Out[pl.Tensor[[24, 5120], pl.BF16]],
    data: pl.InOut[pld.DistributedTensor[[3096, 5120], pl.BF16]],
    signal: pl.InOut[pld.DistributedTensor[[516, 1], pl.INT32]],
    metadata: pl.Tensor[[4], pl.INT32],
    operation: pl.Scalar[pl.INT32],
    quantized: pl.Out[pl.Tensor[[24, 5120], pl.INT8]],
    activation_scales: pl.Out[pl.Tensor[[24, 1], pl.FP32]],
) -> tuple[
    pl.Tensor[[24, 5120], pl.BF16],
    pl.Tensor[[24, 5120], pl.BF16],
    pld.DistributedTensor[[3096, 5120], pl.BF16],
    pld.DistributedTensor[[516, 1], pl.INT32],
    pl.Tensor[[24, 5120], pl.INT8],
    pl.Tensor[[24, 1], pl.FP32],
]:
    available = pl.system.available_aiv_count()
    width = pl.min(available, 24)
    with pl.spmd(width, allow_early_resolve=True):
        out, normalized, data, signal, quantized, activation_scales = reduce_add_norm_quant_kernel(
            x,
            residual,
            out,
            weight,
            normalized,
            data,
            signal,
            metadata,
            operation,
            quantized,
            activation_scales,
        )
    return (out, normalized, data, signal, quantized, activation_scales)
