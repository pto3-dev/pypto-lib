# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""B20 QKV/O projection with packed-channel FP16 dequantization."""

from typing import Any

import pypto.language as pl


def make_a1_modelslim_projection(kind: str) -> Any:
    if kind == "o":
        return _make_o_projection()
    if kind != "qkv":
        raise ValueError(f"Expected qkv, got {kind}")
    inner, width = 2048, 5120
    channel_tile = 128

    @pl.jit.incore
    def projection(
        x: pl.Tensor[[32, inner], pl.INT8],
        weight: pl.Tensor[[1, width, inner], pl.INT8, pl.NZ],
        bias: pl.Tensor[[1, width], pl.INT32],
        scales: pl.Tensor[[1, width], pl.INT64],
        output: pl.Out[pl.Tensor[[20, width], pl.FP16]],
        status: pl.InOut[pl.Tensor[[8, 16], pl.INT32]],
    ):
        block = pl.tile.get_block_idx()
        for column in pl.range(0, width, channel_tile):
            if (column // channel_tile) % 8 == block:
                b = pl.tile.load(weight, [0, column, 0], [1, channel_tile, 256], target_memory=pl.Mem.Mat)
                right = pl.tile.extract(pl.tile.transpose_view(pl.tile.reshape(b, [channel_tile, 256])),
                                        0, 0, [256, channel_tile], target_memory=pl.Mem.Right)
                a = pl.tile.load(x, [0, 0], [16, 256], target_memory=pl.Mem.Mat)
                tail = pl.tile.load(x, [16, 0], [16, 256], valid_shape=[4, 256], target_memory=pl.Mem.Mat)
                full_acc = pl.tile.matmul(pl.tile.extract(a, 0, 0, [16, 256], target_memory=pl.Mem.Left), right)
                tail_acc = pl.tile.matmul(pl.tile.extract(tail, 0, 0, [16, 256], target_memory=pl.Mem.Left), right)
                for k in pl.pipeline(256, inner, 256, stage=2):
                    b = pl.tile.load(weight, [0, column, k], [1, channel_tile, 256], target_memory=pl.Mem.Mat)
                    right = pl.tile.extract(pl.tile.transpose_view(pl.tile.reshape(b, [channel_tile, 256])),
                                            0, 0, [256, channel_tile], target_memory=pl.Mem.Right)
                    a = pl.tile.load(x, [0, k], [16, 256], target_memory=pl.Mem.Mat)
                    tail = pl.tile.load(x, [16, k], [16, 256], valid_shape=[4, 256], target_memory=pl.Mem.Mat)
                    left = pl.tile.extract(a, 0, 0, [16, 256], target_memory=pl.Mem.Left)
                    left_tail = pl.tile.extract(tail, 0, 0, [16, 256], target_memory=pl.Mem.Left)
                    full_acc = pl.tile.matmul_acc(full_acc, left, right)
                    tail_acc = pl.tile.matmul_acc(tail_acc, left_tail, right)
                for panel in pl.range(0, channel_tile, 16):
                    channel = column + panel
                    channel_bias = pl.tile.load(bias, [0, channel], [1, 16])
                    channel_scales = pl.tile.load(scales, [0, channel], [1, 16])
                    values = pl.tile.extract(full_acc, 0, panel, [16, 16], target_memory=pl.Mem.Vec)
                    full = pl.tile.add(values, pl.tile.col_expand(values, channel_bias))
                    pl.tile.store(pl.tile.dequantize_channel(full, channel_scales), [0, channel], output)
                    values_tail = pl.tile.extract(tail_acc, 0, panel, [16, 16], target_memory=pl.Mem.Vec)
                    biased_tail = pl.tile.add(values_tail, pl.tile.col_expand(values_tail, channel_bias))
                    pl.tile.store(pl.tile.dequantize_channel(biased_tail, channel_scales), [16, channel], output)
        zero: pl.Scalar[pl.INT32] = 0
        pl.write(status, [block, 0], zero)
        return output, status

    return projection


def _make_o_projection() -> Any:
    """Reuse each Right tile across the full and tail row groups."""
    @pl.jit.incore
    def projection(
        x: pl.Tensor[[32, 4096], pl.INT8],
        weight: pl.Tensor[[1, 2048, 4096], pl.INT8, pl.NZ],
        bias: pl.Tensor[[1, 2048], pl.INT32],
        scales: pl.Tensor[[1, 2048], pl.INT64],
        output: pl.Out[pl.Tensor[[20, 2048], pl.FP16]],
        status: pl.InOut[pl.Tensor[[8, 16], pl.INT32]],
    ):
        block = pl.tile.get_block_idx()
        for column in pl.range(0, 2048, 256):
            if column // 256 == block:
                b = pl.tile.load(weight, [0, column, 0], [1, 256, 64], target_memory=pl.Mem.Mat)
                right = pl.tile.extract(pl.tile.transpose_view(pl.tile.reshape(b, [256, 64])),
                                        0, 0, [64, 256], target_memory=pl.Mem.Right)
                a = pl.tile.load(x, [0, 0], [16, 64], target_memory=pl.Mem.Mat)
                tail = pl.tile.load(x, [16, 0], [16, 64], valid_shape=[4, 64], target_memory=pl.Mem.Mat)
                full_acc = pl.tile.matmul(pl.tile.extract(a, 0, 0, [16, 64], target_memory=pl.Mem.Left), right)
                tail_acc = pl.tile.matmul(pl.tile.extract(tail, 0, 0, [16, 64], target_memory=pl.Mem.Left), right)
                for k in pl.pipeline(64, 4096, 64, stage=2):
                    b = pl.tile.load(weight, [0, column, k], [1, 256, 64], target_memory=pl.Mem.Mat)
                    right = pl.tile.extract(pl.tile.transpose_view(pl.tile.reshape(b, [256, 64])),
                                            0, 0, [64, 256], target_memory=pl.Mem.Right)
                    a = pl.tile.load(x, [0, k], [16, 64], target_memory=pl.Mem.Mat)
                    tail = pl.tile.load(x, [16, k], [16, 64], valid_shape=[4, 64], target_memory=pl.Mem.Mat)
                    left = pl.tile.extract(a, 0, 0, [16, 64], target_memory=pl.Mem.Left)
                    left_tail = pl.tile.extract(tail, 0, 0, [16, 64], target_memory=pl.Mem.Left)
                    full_acc = pl.tile.matmul_acc(full_acc, left, right)
                    tail_acc = pl.tile.matmul_acc(tail_acc, left_tail, right)
                for panel in pl.range(0, 256, 16):
                    channel = column + panel
                    channel_bias = pl.tile.load(bias, [0, channel], [1, 16])
                    channel_scales = pl.tile.load(scales, [0, channel], [1, 16])
                    values = pl.tile.extract(full_acc, 0, panel, [16, 16], target_memory=pl.Mem.Vec)
                    full = pl.tile.add(values, pl.tile.col_expand(values, channel_bias))
                    pl.tile.store(pl.tile.dequantize_channel(full, channel_scales), [0, channel], output)
                    values_tail = pl.tile.extract(tail_acc, 0, panel, [16, 16], target_memory=pl.Mem.Vec)
                    biased_tail = pl.tile.add(values_tail, pl.tile.col_expand(values_tail, channel_bias))
                    pl.tile.store(pl.tile.dequantize_channel(biased_tail, channel_scales), [16, channel], output)
        zero: pl.Scalar[pl.INT32] = 0
        pl.write(status, [block, 0], zero)
        return output, status

    return projection
