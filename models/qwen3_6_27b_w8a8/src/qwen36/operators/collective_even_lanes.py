# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Explicit even-operation lane protocol:48 AddNorm +16 Add per step.

Must select BOTH kernels together. Slots (lane+1)*8+peer are disjoint
from odd NormQuant slots lane*8+4+peer and embedding slots0..3.
Expected=(epoch-1)*64+operation//2 for operations2,4,...,128.
Same math, rounding, data windows and queried AIV bound as reference.
"""

import pypto.language as pl
import pypto.language.distributed as pld


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
                signal, peer=peer, offsets=[(lane + 1) * 8 + rank, 0], value=1, op=pld.NotifyOp.AtomicAdd
            )
    for peer in pl.range(4):
        if peer != rank:
            pld.system.wait(
                signal,
                offsets=[(lane + 1) * 8 + peer, 0],
                expected=(epoch - 1) * 64 + operation // 2,
                cmp=pld.WaitCmp.Ge,
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
                signal, peer=peer, offsets=[(lane + 1) * 8 + rank, 0], value=1, op=pld.NotifyOp.AtomicAdd
            )
    for peer in pl.range(4):
        if peer != rank:
            pld.system.wait(
                signal,
                offsets=[(lane + 1) * 8 + peer, 0],
                expected=(epoch - 1) * 64 + operation // 2,
                cmp=pld.WaitCmp.Ge,
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
