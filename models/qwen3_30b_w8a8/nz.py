# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -------------------------------------------------------------------------------------------------------
"""Native-NZ expert projections with borrowed rank-local weight banks."""

from typing import Any

import pypto.language as pl

from .config import EXPERTS, HIDDEN, LOCAL_INTERMEDIATE, TOP_K, W13_WIDTH
from .swiglu import expert_swiglu_quant_batched_impl
from .w13 import ROUTES_DYN

W13_N_TILE = 256
W13_K_TILE = 512
W13_INNER_K_TILE = 128
W13_M_TILE = 16
W2_N_TILE = 512
W2_K_TILE = 128


def make_w13_nz_token_inputs(rows: int, *, inline: bool = False) -> Any:
    """Read token-major quantized inputs without materializing top-k route copies."""
    if not 1 <= rows <= 16:
        raise ValueError(f"Unsupported direct expert input bucket: {rows}")
    routes = rows * TOP_K
    early = rows == 1
    decorator = pl.jit.inline if inline else pl.jit

    @decorator
    def w13_nz_token_inputs(
        x: pl.Tensor[[rows, HIDDEN], pl.INT8],
        expert_ids: pl.Tensor[[1, routes], pl.INT32],
        weight: pl.Tensor[[EXPERTS, HIDDEN, W13_WIDTH], pl.INT8, pl.NZ],
        out: pl.Out[pl.Tensor[[routes, W13_WIDTH], pl.INT32]],
    ) -> pl.Tensor:
        for route in pl.spmd(routes, name_hint="w13_nz_token_inputs", allow_early_resolve=early):
            pl.set_cache_policy(weight, pl.CachePolicy.BYPASS)
            token = route // TOP_K
            expert = pl.cast(pl.read(expert_ids, [0, route]), pl.INDEX)
            activation = pl.tile.load(x, [token, 0], [W13_M_TILE, HIDDEN], valid_shape=[1, HIDDEN], target_memory=pl.Mem.Mat)
            for n0 in pl.unroll(0, W13_WIDTH, W13_N_TILE):
                acc = pl.tile.create([W13_M_TILE, W13_N_TILE], pl.INT32, target_memory=pl.Mem.Acc)
                for k0 in pl.pipeline(0, HIDDEN, W13_K_TILE, stage=2):
                    weight3 = pl.tile.load(weight, [expert, k0, n0], [1, W13_K_TILE, W13_N_TILE], target_memory=pl.Mem.Mat)
                    rhs = pl.tile.reshape(weight3, [W13_K_TILE, W13_N_TILE])
                    for kk in pl.pipeline(0, W13_K_TILE, W13_INNER_K_TILE, stage=2):
                        lhs = pl.tile.extract(activation, 0, k0 + kk, [W13_M_TILE, W13_INNER_K_TILE], target_memory=pl.Mem.Left)
                        right = pl.tile.extract(rhs, kk, 0, [W13_INNER_K_TILE, W13_N_TILE], target_memory=pl.Mem.Right)
                        acc = pl.tile.matmul_acc(acc, lhs, right, init_cond=(k0 == 0 and kk == 0))
                valid = pl.tile.set_validshape(acc, 1, W13_N_TILE)
                out = pl.tile.store(valid, [route, n0], out)
        return out

    return w13_nz_token_inputs


@pl.jit.inline
def w13_nz_impl(
    x: pl.Tensor[[ROUTES_DYN, HIDDEN], pl.INT8],
    expert_ids: pl.Tensor[[1, ROUTES_DYN], pl.INT32],
    weight: pl.Tensor[[EXPERTS, HIDDEN, W13_WIDTH], pl.INT8, pl.NZ],
    out: pl.Tensor[[ROUTES_DYN, W13_WIDTH], pl.INT32],
) -> pl.Tensor:
    routes = pl.tensor.dim(x, 0)
    for route in pl.spmd(routes, name_hint="w13_nz"):
        expert = pl.cast(pl.read(expert_ids, [0, route]), pl.INDEX)
        for n0 in pl.unroll(0, W13_WIDTH, W13_N_TILE):
            acc = pl.create_tensor([1, W13_N_TILE], dtype=pl.INT32)
            for kb in pl.pipeline(HIDDEN // W13_K_TILE, stage=2):
                k0 = kb * W13_K_TILE
                weight3 = pl.slice(weight, [1, W13_K_TILE, W13_N_TILE], [expert, k0, n0])
                weight2 = pl.reshape(weight3, [W13_K_TILE, W13_N_TILE])
                acc = pl.matmul_acc(
                    acc, x[route : route + 1, k0 : k0 + W13_K_TILE], weight2, init_cond=(kb == 0)
                )
            out[route : route + 1, n0 : n0 + W13_N_TILE] = acc
    return out


@pl.jit.inline
def w2_nz_impl(
    x: pl.Tensor[[ROUTES_DYN, LOCAL_INTERMEDIATE], pl.INT8],
    expert_ids: pl.Tensor[[1, ROUTES_DYN], pl.INT32],
    weight: pl.Tensor[[EXPERTS, LOCAL_INTERMEDIATE, HIDDEN], pl.INT8, pl.NZ],
    out: pl.Tensor[[ROUTES_DYN, HIDDEN], pl.INT32],
) -> pl.Tensor:
    x.bind_dynamic(0, ROUTES_DYN)
    expert_ids.bind_dynamic(1, ROUTES_DYN)
    out.bind_dynamic(0, ROUTES_DYN)
    routes = pl.tensor.dim(x, 0)
    for route in pl.spmd(routes, name_hint="w2_nz"):
        expert = pl.cast(pl.read(expert_ids, [0, route]), pl.INDEX)
        for n0 in pl.unroll(0, HIDDEN, 512):
            acc = pl.create_tensor([1, 512], dtype=pl.INT32)
            for kb in pl.pipeline(LOCAL_INTERMEDIATE // 128, stage=2):
                k0 = kb * 128
                weight3 = pl.slice(weight, [1, 128, 512], [expert, k0, n0])
                weight2 = pl.reshape(weight3, [128, 512])
                acc = pl.matmul_acc(acc, x[route : route + 1, k0 : k0 + 128], weight2, init_cond=(kb == 0))
            out[route : route + 1, n0 : n0 + 512] = acc
    return out


@pl.jit.inline
def w2_nz_dequant_impl(
    x: pl.Tensor[[ROUTES_DYN, LOCAL_INTERMEDIATE], pl.INT8],
    scale: pl.Tensor[[ROUTES_DYN, 1], pl.FP32],
    expert_ids: pl.Tensor[[1, ROUTES_DYN], pl.INT32],
    weight: pl.Tensor[[EXPERTS, LOCAL_INTERMEDIATE, HIDDEN], pl.INT8, pl.NZ],
    weight_scale: pl.Tensor[[EXPERTS, HIDDEN], pl.FP32],
    out: pl.Tensor[[ROUTES_DYN, HIDDEN], pl.INT32],
    y: pl.Tensor[[ROUTES_DYN, HIDDEN], pl.BF16],
) -> tuple[pl.Tensor, pl.Tensor]:
    x.bind_dynamic(0, ROUTES_DYN)
    scale.bind_dynamic(0, ROUTES_DYN)
    expert_ids.bind_dynamic(1, ROUTES_DYN)
    out.bind_dynamic(0, ROUTES_DYN)
    y.bind_dynamic(0, ROUTES_DYN)
    routes = pl.tensor.dim(x, 0)
    for route in pl.spmd(routes, name_hint="w2_nz_dequant"):
        expert = pl.cast(pl.read(expert_ids, [0, route]), pl.INDEX)
        activation_scale = pl.read(scale, [route, 0])
        for n0 in pl.unroll(0, HIDDEN, W2_N_TILE):
            acc = pl.create_tensor([1, W2_N_TILE], dtype=pl.INT32)
            for kb in pl.pipeline(LOCAL_INTERMEDIATE // W2_K_TILE, stage=2):
                k0 = kb * W2_K_TILE
                weight3 = pl.slice(weight, [1, W2_K_TILE, W2_N_TILE], [expert, k0, n0])
                weight2 = pl.reshape(weight3, [W2_K_TILE, W2_N_TILE])
                acc = pl.matmul_acc(
                    acc, x[route : route + 1, k0 : k0 + W2_K_TILE], weight2, init_cond=(kb == 0)
                )
            out[route : route + 1, n0 : n0 + W2_N_TILE] = acc
            value = pl.cast(acc, pl.FP32)
            value = pl.mul(value, weight_scale[expert : expert + 1, n0 : n0 + W2_N_TILE])
            value = pl.mul(value, activation_scale)
            y[route : route + 1, n0 : n0 + W2_N_TILE] = pl.cast(value, pl.BF16, mode="rint")
    return out, y


@pl.jit.inline
def w2_nz_output_impl(
    x: pl.Tensor[[ROUTES_DYN, LOCAL_INTERMEDIATE], pl.INT8],
    scale: pl.Tensor[[ROUTES_DYN, 1], pl.FP32],
    expert_ids: pl.Tensor[[1, ROUTES_DYN], pl.INT32],
    weight: pl.Tensor[[EXPERTS, LOCAL_INTERMEDIATE, HIDDEN], pl.INT8, pl.NZ],
    weight_scale: pl.Tensor[[EXPERTS, HIDDEN], pl.FP32],
    y: pl.Tensor[[ROUTES_DYN, HIDDEN], pl.BF16],
) -> pl.Tensor:
    """Deployment epilogue without writing the diagnostic INT32 W2 buffer."""
    routes = pl.tensor.dim(x, 0)
    for route in pl.spmd(routes, name_hint="w2_nz_output"):
        expert = pl.cast(pl.read(expert_ids, [0, route]), pl.INDEX)
        activation_scale = pl.read(scale, [route, 0])
        for n0 in pl.unroll(0, HIDDEN, W2_N_TILE):
            acc = pl.create_tensor([1, W2_N_TILE], dtype=pl.INT32)
            for kb in pl.pipeline(LOCAL_INTERMEDIATE // W2_K_TILE, stage=2):
                k0 = kb * W2_K_TILE
                weight3 = pl.slice(weight, [1, W2_K_TILE, W2_N_TILE], [expert, k0, n0])
                weight2 = pl.reshape(weight3, [W2_K_TILE, W2_N_TILE])
                acc = pl.matmul_acc(
                    acc, x[route : route + 1, k0 : k0 + W2_K_TILE], weight2, init_cond=(kb == 0)
                )
            value = pl.cast(acc, pl.FP32)
            value = pl.mul(value, weight_scale[expert : expert + 1, n0 : n0 + W2_N_TILE])
            value = pl.mul(value, activation_scale)
            y[route : route + 1, n0 : n0 + W2_N_TILE] = pl.cast(value, pl.BF16, mode="rint")
    return y


@pl.jit
def w13_swiglu_chip_nz(
    x: pl.Tensor[[ROUTES_DYN, HIDDEN], pl.INT8],
    input_scale: pl.Tensor[[ROUTES_DYN, 1], pl.FP32],
    expert_ids: pl.Tensor[[1, ROUTES_DYN], pl.INT32],
    w13: pl.Tensor[[EXPERTS, HIDDEN, W13_WIDTH], pl.INT8, pl.NZ],
    weight_scale: pl.Tensor[[EXPERTS, W13_WIDTH], pl.FP32],
    acc: pl.Out[pl.Tensor[[ROUTES_DYN, W13_WIDTH], pl.INT32]],
    quant: pl.Out[pl.Tensor[[ROUTES_DYN, LOCAL_INTERMEDIATE], pl.INT8]],
    scale: pl.Out[pl.Tensor[[ROUTES_DYN, 1], pl.FP32]],
) -> tuple[pl.Tensor, pl.Tensor, pl.Tensor]:
    x.bind_dynamic(0, ROUTES_DYN)
    input_scale.bind_dynamic(0, ROUTES_DYN)
    expert_ids.bind_dynamic(1, ROUTES_DYN)
    acc.bind_dynamic(0, ROUTES_DYN)
    quant.bind_dynamic(0, ROUTES_DYN)
    scale.bind_dynamic(0, ROUTES_DYN)
    acc_after = w13_nz_impl(x, expert_ids, w13, acc)
    quant_after, scale_after = expert_swiglu_quant_batched_impl(
        acc_after, input_scale, expert_ids, weight_scale, quant, scale
    )
    return acc_after, quant_after, scale_after


@pl.jit
def w2_dequant_chip_nz(
    quant: pl.Tensor[[ROUTES_DYN, LOCAL_INTERMEDIATE], pl.INT8],
    scale: pl.Tensor[[ROUTES_DYN, 1], pl.FP32],
    expert_ids: pl.Tensor[[1, ROUTES_DYN], pl.INT32],
    w2: pl.Tensor[[EXPERTS, LOCAL_INTERMEDIATE, HIDDEN], pl.INT8, pl.NZ],
    w2_scale: pl.Tensor[[EXPERTS, HIDDEN], pl.FP32],
    acc2: pl.Out[pl.Tensor[[ROUTES_DYN, HIDDEN], pl.INT32]],
    y: pl.Out[pl.Tensor[[ROUTES_DYN, HIDDEN], pl.BF16]],
) -> tuple[pl.Tensor, pl.Tensor]:
    quant.bind_dynamic(0, ROUTES_DYN)
    scale.bind_dynamic(0, ROUTES_DYN)
    expert_ids.bind_dynamic(1, ROUTES_DYN)
    acc2.bind_dynamic(0, ROUTES_DYN)
    y.bind_dynamic(0, ROUTES_DYN)
    return w2_nz_dequant_impl(quant, scale, expert_ids, w2, w2_scale, acc2, y)


@pl.jit
def expert_path_chip_nz(
    x: pl.Tensor[[ROUTES_DYN, HIDDEN], pl.INT8],
    input_scale: pl.Tensor[[ROUTES_DYN, 1], pl.FP32],
    expert_ids: pl.Tensor[[1, ROUTES_DYN], pl.INT32],
    w13: pl.Tensor[[EXPERTS, HIDDEN, W13_WIDTH], pl.INT8, pl.NZ],
    weight_scale: pl.Tensor[[EXPERTS, W13_WIDTH], pl.FP32],
    w2: pl.Tensor[[EXPERTS, LOCAL_INTERMEDIATE, HIDDEN], pl.INT8, pl.NZ],
    w2_scale: pl.Tensor[[EXPERTS, HIDDEN], pl.FP32],
    acc: pl.Out[pl.Tensor[[ROUTES_DYN, W13_WIDTH], pl.INT32]],
    quant: pl.Out[pl.Tensor[[ROUTES_DYN, LOCAL_INTERMEDIATE], pl.INT8]],
    scale: pl.Out[pl.Tensor[[ROUTES_DYN, 1], pl.FP32]],
    acc2: pl.Out[pl.Tensor[[ROUTES_DYN, HIDDEN], pl.INT32]],
    y: pl.Out[pl.Tensor[[ROUTES_DYN, HIDDEN], pl.BF16]],
) -> tuple[pl.Tensor, pl.Tensor, pl.Tensor, pl.Tensor, pl.Tensor]:
    x.bind_dynamic(0, ROUTES_DYN)
    input_scale.bind_dynamic(0, ROUTES_DYN)
    expert_ids.bind_dynamic(1, ROUTES_DYN)
    acc.bind_dynamic(0, ROUTES_DYN)
    quant.bind_dynamic(0, ROUTES_DYN)
    scale.bind_dynamic(0, ROUTES_DYN)
    acc2.bind_dynamic(0, ROUTES_DYN)
    y.bind_dynamic(0, ROUTES_DYN)
    acc_after = w13_nz_impl(x, expert_ids, w13, acc)
    quant_after, scale_after = expert_swiglu_quant_batched_impl(
        acc_after, input_scale, expert_ids, weight_scale, quant, scale
    )
    acc2_after, y_after = w2_nz_dequant_impl(quant_after, scale_after, expert_ids, w2, w2_scale, acc2, y)
    return acc_after, quant_after, scale_after, acc2_after, y_after
