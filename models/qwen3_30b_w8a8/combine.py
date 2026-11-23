# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -------------------------------------------------------------------------------------------------------
"""Top-8 weighted rank-local expert combine before the native TP collective."""

import pypto.language as pl

from .config import HIDDEN, TOP_K
from .quant import TOKENS_DYN
from .w13 import ROUTES_DYN

COMBINE_TILE = 512


@pl.jit.inline
def combine_routes_impl(
    expert_y: pl.Tensor[[ROUTES_DYN, HIDDEN], pl.BF16],
    weights: pl.Tensor[[TOKENS_DYN, TOP_K], pl.FP32],
    y: pl.Tensor[[TOKENS_DYN, HIDDEN], pl.BF16],
) -> pl.Tensor:
    tiles = HIDDEN // COMBINE_TILE
    tokens = pl.tensor.dim(y, 0)
    for block in pl.spmd(tokens * tiles, name_hint="combine_routes"):
        token = block // tiles
        col = (block % tiles) * COMBINE_TILE
        acc = pl.full([1, COMBINE_TILE], dtype=pl.FP32, value=0.0)
        for index in pl.range(TOP_K):
            route = token * TOP_K + index
            value = pl.cast(expert_y[route : route + 1, col : col + COMBINE_TILE], pl.FP32)
            weight = pl.read(weights, [token, index])
            acc = pl.add(acc, pl.mul(value, weight))
        y[token : token + 1, col : col + COMBINE_TILE] = pl.cast(acc, pl.BF16, mode="rint")
    return y


@pl.jit
def combine_routes(
    expert_y: pl.Tensor[[ROUTES_DYN, HIDDEN], pl.BF16],
    weights: pl.Tensor[[TOKENS_DYN, TOP_K], pl.FP32],
    y: pl.Out[pl.Tensor[[TOKENS_DYN, HIDDEN], pl.BF16]],
) -> pl.Tensor:
    expert_y.bind_dynamic(0, ROUTES_DYN)
    weights.bind_dynamic(0, TOKENS_DYN)
    y.bind_dynamic(0, TOKENS_DYN)
    return combine_routes_impl(expert_y, weights, y)
