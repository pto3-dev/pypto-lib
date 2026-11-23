# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -------------------------------------------------------------------------------------------------------
"""Materialize token-major routes for direct resident expert-bank addressing."""

import pypto.language as pl

from .config import HIDDEN, TOP_K
from .quant import TOKENS_DYN
from .w13 import ROUTES_DYN


@pl.jit.inline
def dispatch_quantized_routes_impl(
    x: pl.Tensor[[TOKENS_DYN, HIDDEN], pl.INT8],
    input_scale: pl.Tensor[[TOKENS_DYN, 1], pl.FP32],
    ids: pl.Tensor[[TOKENS_DYN, TOP_K], pl.INT32],
    route_x: pl.Tensor[[ROUTES_DYN, HIDDEN], pl.INT8],
    route_scale: pl.Tensor[[ROUTES_DYN, 1], pl.FP32],
    route_ids: pl.Tensor[[1, ROUTES_DYN], pl.INT32],
) -> tuple[pl.Tensor, pl.Tensor, pl.Tensor]:
    routes = pl.tensor.dim(route_x, 0)
    for route in pl.spmd(routes, name_hint="route_materialize"):
        token = route // TOP_K
        route_x[route : route + 1, :] = x[token : token + 1, :]
    for _block in pl.spmd(1, name_hint="route_metadata"):
        for route in pl.range(routes):
            token = route // TOP_K
            index = route % TOP_K
            pl.write(route_scale, [route, 0], pl.read(input_scale, [token, 0]))
            pl.write(route_ids, [0, route], pl.read(ids, [token, index]))
    return route_x, route_scale, route_ids


@pl.jit
def dispatch_quantized_routes(
    x: pl.Tensor[[TOKENS_DYN, HIDDEN], pl.INT8],
    input_scale: pl.Tensor[[TOKENS_DYN, 1], pl.FP32],
    ids: pl.Tensor[[TOKENS_DYN, TOP_K], pl.INT32],
    route_x: pl.Out[pl.Tensor[[ROUTES_DYN, HIDDEN], pl.INT8]],
    route_scale: pl.Out[pl.Tensor[[ROUTES_DYN, 1], pl.FP32]],
    route_ids: pl.Out[pl.Tensor[[1, ROUTES_DYN], pl.INT32]],
) -> tuple[pl.Tensor, pl.Tensor, pl.Tensor]:
    x.bind_dynamic(0, TOKENS_DYN)
    input_scale.bind_dynamic(0, TOKENS_DYN)
    ids.bind_dynamic(0, TOKENS_DYN)
    route_x.bind_dynamic(0, ROUTES_DYN)
    route_scale.bind_dynamic(0, ROUTES_DYN)
    route_ids.bind_dynamic(1, ROUTES_DYN)
    return dispatch_quantized_routes_impl(x, input_scale, ids, route_x, route_scale, route_ids)


@pl.jit.inline
def dispatch_quantized_routes_grouped_impl(
    x: pl.Tensor[[TOKENS_DYN, HIDDEN], pl.INT8],
    input_scale: pl.Tensor[[TOKENS_DYN, 1], pl.FP32],
    ids: pl.Tensor[[TOKENS_DYN, TOP_K], pl.INT32],
    route_x: pl.Tensor[[ROUTES_DYN, HIDDEN], pl.INT8],
    route_scale: pl.Tensor[[ROUTES_DYN, 1], pl.FP32],
    route_ids: pl.Tensor[[1, ROUTES_DYN], pl.INT32],
) -> tuple[pl.Tensor, pl.Tensor, pl.Tensor]:
    tokens = pl.tensor.dim(x, 0)
    routes = pl.tensor.dim(route_x, 0)
    flat_ids = pl.reshape(ids, [1, routes])
    flat_scale = pl.reshape(route_scale, [1, routes])
    for block in pl.spmd((tokens + 1) // 2, name_hint="route_grouped"):
        token_base = block * 2
        route_base = token_base * TOP_K
        valid_tokens = pl.min(2, tokens - token_base)
        valid_routes = valid_tokens * TOP_K
        scales = pl.full([1, 16], dtype=pl.FP32, value=0.0)
        for lane in pl.range(valid_tokens):
            token = token_base + lane
            row = x[token : token + 1, :]
            value = pl.read(input_scale, [token, 0])
            for index in pl.range(TOP_K):
                route = token * TOP_K + index
                route_x[route : route + 1, :] = row
                pl.write(scales, [0, lane * TOP_K + index], value)
        flat_scale[:, route_base : route_base + 16] = pl.set_validshape(scales, 1, valid_routes)
        route_ids[:, route_base : route_base + 16] = pl.slice(
            flat_ids, [1, 16], [0, route_base], valid_shape=[1, valid_routes]
        )
    return route_x, route_scale, route_ids


@pl.jit
def dispatch_quantized_routes_grouped(
    x: pl.Tensor[[TOKENS_DYN, HIDDEN], pl.INT8],
    input_scale: pl.Tensor[[TOKENS_DYN, 1], pl.FP32],
    ids: pl.Tensor[[TOKENS_DYN, TOP_K], pl.INT32],
    route_x: pl.Out[pl.Tensor[[ROUTES_DYN, HIDDEN], pl.INT8]],
    route_scale: pl.Out[pl.Tensor[[ROUTES_DYN, 1], pl.FP32]],
    route_ids: pl.Out[pl.Tensor[[1, ROUTES_DYN], pl.INT32]],
) -> tuple[pl.Tensor, pl.Tensor, pl.Tensor]:
    x.bind_dynamic(0, TOKENS_DYN)
    input_scale.bind_dynamic(0, TOKENS_DYN)
    ids.bind_dynamic(0, TOKENS_DYN)
    route_x.bind_dynamic(0, ROUTES_DYN)
    route_scale.bind_dynamic(0, ROUTES_DYN)
    route_ids.bind_dynamic(1, ROUTES_DYN)
    x_after, scale_after, ids_after = dispatch_quantized_routes_grouped_impl(
        x, input_scale, ids, route_x, route_scale, route_ids
    )
    return x_after, scale_after, ids_after
