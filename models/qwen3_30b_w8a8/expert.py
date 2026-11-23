# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -------------------------------------------------------------------------------------------------------
"""One resident rank-local expert-path callable; excludes router/combine/TP reduction."""

import pypto.language as pl

from .config import EXPERTS, HIDDEN, LOCAL_INTERMEDIATE, W13_WIDTH
from .swiglu import expert_swiglu_quant, expert_swiglu_quant_batched_impl
from .w13 import ROUTES_DYN, w13_nt, w13_nt_impl
from .w2 import w2_nt, w2_nt_impl

DEQUANT_TILE = 512


@pl.jit.inline
def w2_dequant(
    acc2: pl.Tensor[[ROUTES_DYN, HIDDEN], pl.INT32],
    scale: pl.Tensor[[ROUTES_DYN, 1], pl.FP32],
    expert_ids: pl.Tensor[[1, ROUTES_DYN], pl.INT32],
    w2_scale: pl.Tensor[[EXPERTS, HIDDEN], pl.FP32],
    y: pl.Out[pl.Tensor[[ROUTES_DYN, HIDDEN], pl.BF16]],
) -> pl.Tensor:
    acc2.bind_dynamic(0, ROUTES_DYN)
    scale.bind_dynamic(0, ROUTES_DYN)
    expert_ids.bind_dynamic(1, ROUTES_DYN)
    y.bind_dynamic(0, ROUTES_DYN)
    routes = pl.tensor.dim(acc2, 0)
    for block in pl.spmd(routes * (HIDDEN // DEQUANT_TILE), name_hint="w2_dequant"):
        route = block // (HIDDEN // DEQUANT_TILE)
        n0 = (block % (HIDDEN // DEQUANT_TILE)) * DEQUANT_TILE
        expert = pl.cast(pl.read(expert_ids, [0, route]), pl.INDEX)
        activation_scale = pl.read(scale, [route, 0])
        value = pl.cast(acc2[route : route + 1, n0 : n0 + DEQUANT_TILE], pl.FP32)
        value = pl.mul(value, w2_scale[expert : expert + 1, n0 : n0 + DEQUANT_TILE])
        value = pl.mul(value, activation_scale)
        y[route : route + 1, n0 : n0 + DEQUANT_TILE] = pl.cast(value, pl.BF16, mode="rint")
    return y


@pl.jit.host
def resident_expert_path_multirun(
    x: pl.Tensor[[ROUTES_DYN, HIDDEN], pl.INT8],
    input_scale: pl.Tensor[[ROUTES_DYN, 1], pl.FP32],
    expert_ids: pl.Tensor[[1, ROUTES_DYN], pl.INT32],
    w13: pl.Tensor[[EXPERTS, W13_WIDTH, HIDDEN], pl.INT8],
    weight_scale: pl.Tensor[[EXPERTS, W13_WIDTH], pl.FP32],
    w2: pl.Tensor[[EXPERTS, HIDDEN, LOCAL_INTERMEDIATE], pl.INT8],
    w2_scale: pl.Tensor[[EXPERTS, HIDDEN], pl.FP32],
    acc: pl.Out[pl.Tensor[[ROUTES_DYN, W13_WIDTH], pl.INT32]],
    quant: pl.Out[pl.Tensor[[ROUTES_DYN, LOCAL_INTERMEDIATE], pl.INT8]],
    scale: pl.Out[pl.Tensor[[ROUTES_DYN, 1], pl.FP32]],
    acc2: pl.Out[pl.Tensor[[ROUTES_DYN, HIDDEN], pl.INT32]],
    y: pl.Out[pl.Tensor[[ROUTES_DYN, HIDDEN], pl.BF16]],
):
    x.bind_dynamic(0, ROUTES_DYN)
    input_scale.bind_dynamic(0, ROUTES_DYN)
    expert_ids.bind_dynamic(1, ROUTES_DYN)
    acc.bind_dynamic(0, ROUTES_DYN)
    quant.bind_dynamic(0, ROUTES_DYN)
    scale.bind_dynamic(0, ROUTES_DYN)
    acc2.bind_dynamic(0, ROUTES_DYN)
    y.bind_dynamic(0, ROUTES_DYN)
    acc_after = w13_nt(x, expert_ids, w13, acc, device=0)
    quant_after, scale_after = expert_swiglu_quant(
        acc_after, input_scale, expert_ids, weight_scale, quant, scale, device=0
    )
    acc2_after = w2_nt(quant_after, expert_ids, w2, acc2, device=0)
    w2_dequant_chip(acc2_after, scale_after, expert_ids, w2_scale, y, device=0)


@pl.jit
def w2_dequant_chip(
    acc2: pl.Tensor[[ROUTES_DYN, HIDDEN], pl.INT32],
    scale: pl.Tensor[[ROUTES_DYN, 1], pl.FP32],
    expert_ids: pl.Tensor[[1, ROUTES_DYN], pl.INT32],
    w2_scale: pl.Tensor[[EXPERTS, HIDDEN], pl.FP32],
    y: pl.Out[pl.Tensor[[ROUTES_DYN, HIDDEN], pl.BF16]],
) -> pl.Tensor:
    return w2_dequant(acc2, scale, expert_ids, w2_scale, y)


@pl.jit
def expert_path_chip(
    x: pl.Tensor[[ROUTES_DYN, HIDDEN], pl.INT8],
    input_scale: pl.Tensor[[ROUTES_DYN, 1], pl.FP32],
    expert_ids: pl.Tensor[[1, ROUTES_DYN], pl.INT32],
    w13: pl.Tensor[[EXPERTS, W13_WIDTH, HIDDEN], pl.INT8],
    weight_scale: pl.Tensor[[EXPERTS, W13_WIDTH], pl.FP32],
    w2: pl.Tensor[[EXPERTS, HIDDEN, LOCAL_INTERMEDIATE], pl.INT8],
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
    acc_after = w13_nt_impl(x, expert_ids, w13, acc)
    quant_after, scale_after = expert_swiglu_quant_batched_impl(
        acc_after, input_scale, expert_ids, weight_scale, quant, scale
    )
    acc2_after = w2_nt_impl(quant_after, expert_ids, w2, acc2)
    y_after = w2_dequant(acc2_after, scale_after, expert_ids, w2_scale, y)
    return acc_after, quant_after, scale_after, acc2_after, y_after


@pl.jit.host
def resident_expert_path(
    x: pl.Tensor[[ROUTES_DYN, HIDDEN], pl.INT8],
    input_scale: pl.Tensor[[ROUTES_DYN, 1], pl.FP32],
    expert_ids: pl.Tensor[[1, ROUTES_DYN], pl.INT32],
    w13: pl.Tensor[[EXPERTS, W13_WIDTH, HIDDEN], pl.INT8],
    weight_scale: pl.Tensor[[EXPERTS, W13_WIDTH], pl.FP32],
    w2: pl.Tensor[[EXPERTS, HIDDEN, LOCAL_INTERMEDIATE], pl.INT8],
    w2_scale: pl.Tensor[[EXPERTS, HIDDEN], pl.FP32],
    acc: pl.Out[pl.Tensor[[ROUTES_DYN, W13_WIDTH], pl.INT32]],
    quant: pl.Out[pl.Tensor[[ROUTES_DYN, LOCAL_INTERMEDIATE], pl.INT8]],
    scale: pl.Out[pl.Tensor[[ROUTES_DYN, 1], pl.FP32]],
    acc2: pl.Out[pl.Tensor[[ROUTES_DYN, HIDDEN], pl.INT32]],
    y: pl.Out[pl.Tensor[[ROUTES_DYN, HIDDEN], pl.BF16]],
):
    x.bind_dynamic(0, ROUTES_DYN)
    input_scale.bind_dynamic(0, ROUTES_DYN)
    expert_ids.bind_dynamic(1, ROUTES_DYN)
    acc.bind_dynamic(0, ROUTES_DYN)
    quant.bind_dynamic(0, ROUTES_DYN)
    scale.bind_dynamic(0, ROUTES_DYN)
    acc2.bind_dynamic(0, ROUTES_DYN)
    y.bind_dynamic(0, ROUTES_DYN)
    expert_path_chip(
        x, input_scale, expert_ids, w13, weight_scale, w2, w2_scale, acc, quant, scale, acc2, y, device=0
    )
