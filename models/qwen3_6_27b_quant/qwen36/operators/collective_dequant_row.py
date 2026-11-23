# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Dequantization fused with the existing row-ready TP4 collective.

Only the local publication input changes from BF16-rounded FP32 to INT32+scales.
BF16 publication, signals, residual and norm stay explicit. This candidate uses
the measured native deterministic TP4 BF16 pair tree (0+1)+(2+3), with rounding
at both pair sums and the final sum. It is not a universal backend policy.
"""
import pypto.language as pl
import pypto.language.distributed as pld

@pl.function(type=pl.FunctionType.InCore)
def reduce_add_dequant_kernel(x: pl.Tensor[[24, 5120], pl.INT32], residual: pl.Tensor[[24, 5120], pl.BF16], out: pl.Out[pl.Tensor[[24, 5120], pl.BF16]], data: pl.InOut[pld.DistributedTensor[[3096, 5120], pl.BF16]], signal: pl.InOut[pld.DistributedTensor[[12900, 32], pl.INT32]], metadata: pl.Tensor[[4], pl.INT32], operation: pl.Scalar[pl.INT32], weight_scale: pl.Tensor[[16, 5120], pl.FP32], activation_scale: pl.Tensor[[24, 1], pl.FP32], weight_group: pl.Scalar[pl.INT64]) -> tuple[pl.Tensor[[24, 5120], pl.BF16], pld.DistributedTensor[[3096, 5120], pl.BF16], pld.DistributedTensor[[12900, 32], pl.INT32]]:
    pl.func_attr({'static_block_mapping': True})
    ctx = pld.get_comm_ctx(data)
    rank = pld.rank(ctx)
    epoch = pl.read(metadata, [1])
    lane = pl.tile.get_block_idx()
    width = pl.tile.get_block_num()
    for row in pl.range(lane, 24, width):
        local_i32 = pl.load(x, [row, 0], [1, 5120])
        ws = pl.load(weight_scale, [weight_group, 0], [1, 5120])
        xs_column = pl.tile.load(activation_scale, [0, 0], [32, 1], valid_shape=[24, 1], target_memory=pl.Mem.Vec)
        xs = pl.read(xs_column, [row, 0])
        combined_scale = pl.mul(ws, xs)
        local_fp32 = pl.mul(pl.cast(local_i32, pl.FP32), combined_scale)
        local_bf16 = pl.cast(local_fp32, pl.BF16, mode='rint')
        data = pl.store(local_bf16, [operation * 24 + row, 0], data)
        # Prefix516 belongs to legacy single-block reductions. Layer rows use
        # independent128-byte-separated signals: no cross-operation alias.
        for peer in pl.range(4):
            if peer != rank:
                pld.system.notify(signal, peer=peer, offsets=[516 + operation * 96 + rank * 24 + row, 0], value=epoch, op=pld.NotifyOp.Set)
        for peer in pl.range(4):
            if peer != rank:
                pld.system.wait(signal, offsets=[516 + operation * 96 + peer * 24 + row, 0], expected=epoch, cmp=pld.WaitCmp.Ge)
        value0 = pld.tile.remote_load(data, peer=pl.const(0, pl.INT32), offsets=[operation * 24 + row, 0], shape=[1, 5120])
        value1 = pld.tile.remote_load(data, peer=pl.const(1, pl.INT32), offsets=[operation * 24 + row, 0], shape=[1, 5120])
        value2 = pld.tile.remote_load(data, peer=pl.const(2, pl.INT32), offsets=[operation * 24 + row, 0], shape=[1, 5120])
        value3 = pld.tile.remote_load(data, peer=pl.const(3, pl.INT32), offsets=[operation * 24 + row, 0], shape=[1, 5120])
        pair01 = pl.cast(pl.add(pl.cast(value0, pl.FP32), pl.cast(value1, pl.FP32)), pl.BF16, mode='rint')
        pair23 = pl.cast(pl.add(pl.cast(value2, pl.FP32), pl.cast(value3, pl.FP32)), pl.BF16, mode='rint')
        total = pl.add(pl.cast(pair01, pl.FP32), pl.cast(pair23, pl.FP32))
        reduced = pl.cast(total, pl.BF16, mode='rint')
        previous = pl.load(residual, [row, 0], [1, 5120])
        added = pl.add(pl.cast(previous, pl.FP32), pl.cast(reduced, pl.FP32))
        rounded = pl.cast(added, pl.BF16, mode='rint')
        out = pl.store(rounded, [row, 0], out)
    return (out, data, signal)


@pl.function(type=pl.FunctionType.Inline)
def reduce_add_dequant(x: pl.Tensor[[24, 5120], pl.INT32], residual: pl.Tensor[[24, 5120], pl.BF16], out: pl.Out[pl.Tensor[[24, 5120], pl.BF16]], data: pl.InOut[pld.DistributedTensor[[3096, 5120], pl.BF16]], signal: pl.InOut[pld.DistributedTensor[[12900, 32], pl.INT32]], metadata: pl.Tensor[[4], pl.INT32], operation: pl.Scalar[pl.INT32], weight_scale: pl.Tensor[[16, 5120], pl.FP32], activation_scale: pl.Tensor[[24, 1], pl.FP32], weight_group: pl.Scalar[pl.INT64]) -> tuple[pl.Tensor[[24, 5120], pl.BF16], pld.DistributedTensor[[3096, 5120], pl.BF16], pld.DistributedTensor[[12900, 32], pl.INT32]]:
    available = pl.system.available_aiv_count()
    width = pl.min(available, 24)
    with pl.spmd(width, allow_early_resolve=True):
        out, data, signal = reduce_add_dequant_kernel(x, residual, out, data, signal, metadata, operation, weight_scale, activation_scale, weight_group)
    return (out, data, signal)


@pl.function(type=pl.FunctionType.InCore)
def reduce_add_norm_dequant_kernel(x: pl.Tensor[[24, 5120], pl.INT32], residual: pl.Tensor[[24, 5120], pl.BF16], out: pl.Out[pl.Tensor[[24, 5120], pl.BF16]], weight: pl.Tensor[[1, 5120], pl.BF16], normalized: pl.Out[pl.Tensor[[24, 5120], pl.BF16]], data: pl.InOut[pld.DistributedTensor[[3096, 5120], pl.BF16]], signal: pl.InOut[pld.DistributedTensor[[12900, 32], pl.INT32]], metadata: pl.Tensor[[4], pl.INT32], operation: pl.Scalar[pl.INT32], weight_scale: pl.Tensor[[16, 5120], pl.FP32], activation_scale: pl.Tensor[[24, 1], pl.FP32], weight_group: pl.Scalar[pl.INT64]) -> tuple[pl.Tensor[[24, 5120], pl.BF16], pl.Tensor[[24, 5120], pl.BF16], pld.DistributedTensor[[3096, 5120], pl.BF16], pld.DistributedTensor[[12900, 32], pl.INT32]]:
    pl.func_attr({'static_block_mapping': True})
    ctx = pld.get_comm_ctx(data)
    rank = pld.rank(ctx)
    epoch = pl.read(metadata, [1])
    lane = pl.tile.get_block_idx()
    width = pl.tile.get_block_num()
    for row in pl.range(lane, 24, width):
        local_i32 = pl.load(x, [row, 0], [1, 5120])
        ws = pl.load(weight_scale, [weight_group, 0], [1, 5120])
        xs_column = pl.tile.load(activation_scale, [0, 0], [32, 1], valid_shape=[24, 1], target_memory=pl.Mem.Vec)
        xs = pl.read(xs_column, [row, 0])
        combined_scale = pl.mul(ws, xs)
        local_fp32 = pl.mul(pl.cast(local_i32, pl.FP32), combined_scale)
        local_bf16 = pl.cast(local_fp32, pl.BF16, mode='rint')
        data = pl.store(local_bf16, [operation * 24 + row, 0], data)
        # Prefix516 belongs to legacy single-block reductions. Layer rows use
        # independent128-byte-separated signals: no cross-operation alias.
        for peer in pl.range(4):
            if peer != rank:
                pld.system.notify(signal, peer=peer, offsets=[516 + operation * 96 + rank * 24 + row, 0], value=epoch, op=pld.NotifyOp.Set)
        for peer in pl.range(4):
            if peer != rank:
                pld.system.wait(signal, offsets=[516 + operation * 96 + peer * 24 + row, 0], expected=epoch, cmp=pld.WaitCmp.Ge)
        total = pl.tile.full([1, 5120], dtype=pl.FP32, value=0.0)
        # Preserve the measured four-peer prefetch in readable DSL: issue
        # every load before consuming values in the original rank order.
        value0 = pld.tile.remote_load(data, peer=pl.const(0, pl.INT32), offsets=[operation * 24 + row, 0], shape=[1, 5120])
        value1 = pld.tile.remote_load(data, peer=pl.const(1, pl.INT32), offsets=[operation * 24 + row, 0], shape=[1, 5120])
        value2 = pld.tile.remote_load(data, peer=pl.const(2, pl.INT32), offsets=[operation * 24 + row, 0], shape=[1, 5120])
        value3 = pld.tile.remote_load(data, peer=pl.const(3, pl.INT32), offsets=[operation * 24 + row, 0], shape=[1, 5120])
        pair01 = pl.cast(pl.add(pl.cast(value0, pl.FP32), pl.cast(value1, pl.FP32)), pl.BF16, mode='rint')
        pair23 = pl.cast(pl.add(pl.cast(value2, pl.FP32), pl.cast(value3, pl.FP32)), pl.BF16, mode='rint')
        total = pl.add(pl.cast(pair01, pl.FP32), pl.cast(pair23, pl.FP32))
        reduced = pl.cast(total, pl.BF16, mode='rint')
        previous = pl.load(residual, [row, 0], [1, 5120])
        added = pl.add(pl.cast(previous, pl.FP32), pl.cast(reduced, pl.FP32))
        rounded = pl.cast(added, pl.BF16, mode='rint')
        out = pl.store(rounded, [row, 0], out)
        whole = pl.cast(rounded, pl.FP32)
        # Source-derived residual Norm reduction, preserving native FP32 order.
        terms = pl.tile.muls(pl.tile.mul(whole, whole), 1.0 / 5120)
        lanes = pl.tile.full([1, 64], dtype=pl.FP32, value=0.0)
        for repeat in pl.range(80):
            values = pl.tile.slice(terms, [1, 64], [0, repeat * 64])
            lanes = pl.tile.add(values, lanes)
        temporary = pl.tile.create([1, 64], pl.FP32, target_memory=pl.Mem.Vec)
        summation = pl.tile.row_sum(lanes, temporary, physical_rows=8)
        variance = pl.tile.adds(summation, 1e-06)
        inv = pl.recip(pl.sqrt(variance))
        scalar_inv = pl.read(inv, [0, 0])
        gamma = pl.add(pl.cast(pl.load(weight, [0, 0], [1, 5120]), pl.FP32), 1.0)
        gamma_rounded = pl.cast(pl.cast(gamma, pl.BF16, mode='rint'), pl.FP32)
        scaled = pl.cast(pl.cast(pl.mul(whole, scalar_inv), pl.BF16, mode='rint'), pl.FP32)
        result = pl.cast(pl.mul(scaled, gamma_rounded), pl.BF16, mode='rint')
        normalized = pl.store(result, [row, 0], normalized)
    return (out, normalized, data, signal)


@pl.function(type=pl.FunctionType.Inline)
def reduce_add_norm_dequant(x: pl.Tensor[[24, 5120], pl.INT32], residual: pl.Tensor[[24, 5120], pl.BF16], out: pl.Out[pl.Tensor[[24, 5120], pl.BF16]], weight: pl.Tensor[[1, 5120], pl.BF16], normalized: pl.Out[pl.Tensor[[24, 5120], pl.BF16]], data: pl.InOut[pld.DistributedTensor[[3096, 5120], pl.BF16]], signal: pl.InOut[pld.DistributedTensor[[12900, 32], pl.INT32]], metadata: pl.Tensor[[4], pl.INT32], operation: pl.Scalar[pl.INT32], weight_scale: pl.Tensor[[16, 5120], pl.FP32], activation_scale: pl.Tensor[[24, 1], pl.FP32], weight_group: pl.Scalar[pl.INT64]) -> tuple[pl.Tensor[[24, 5120], pl.BF16], pl.Tensor[[24, 5120], pl.BF16], pld.DistributedTensor[[3096, 5120], pl.BF16], pld.DistributedTensor[[12900, 32], pl.INT32]]:
    available = pl.system.available_aiv_count()
    width = pl.min(available, 24)
    with pl.spmd(width, allow_early_resolve=True):
        out, normalized, data, signal = reduce_add_norm_dequant_kernel(x, residual, out, weight, normalized, data, signal, metadata, operation, weight_scale, activation_scale, weight_group)
    return (out, normalized, data, signal)


OPERATORS=(reduce_add_dequant_kernel,reduce_add_dequant,reduce_add_norm_dequant_kernel,reduce_add_norm_dequant)
