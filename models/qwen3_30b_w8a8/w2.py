# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -------------------------------------------------------------------------------------------------------
"""Device-indexed rank-local expert Down projection with exact INT32 output."""

import pypto.language as pl

from .config import EXPERTS, HIDDEN, LOCAL_INTERMEDIATE
from .w13 import ROUTES_DYN

N_TILE = 512
K_TILE = 128
assert HIDDEN % N_TILE == 0 and LOCAL_INTERMEDIATE % K_TILE == 0


@pl.jit.inline
def w2_nt_impl(
    x: pl.Tensor[[ROUTES_DYN, LOCAL_INTERMEDIATE], pl.INT8],
    expert_ids: pl.Tensor[[1, ROUTES_DYN], pl.INT32],
    weight: pl.Tensor[[EXPERTS, HIDDEN, LOCAL_INTERMEDIATE], pl.INT8],
    out: pl.Out[pl.Tensor[[ROUTES_DYN, HIDDEN], pl.INT32]],
) -> pl.Tensor:
    x.bind_dynamic(0, ROUTES_DYN)
    expert_ids.bind_dynamic(1, ROUTES_DYN)
    out.bind_dynamic(0, ROUTES_DYN)
    routes = pl.tensor.dim(x, 0)
    weight_flat = pl.reshape(weight, [EXPERTS * HIDDEN, LOCAL_INTERMEDIATE])
    for block in pl.spmd(routes * (HIDDEN // N_TILE), name_hint="w2_nt"):
        route = block // (HIDDEN // N_TILE)
        n0 = (block % (HIDDEN // N_TILE)) * N_TILE
        expert = pl.cast(pl.read(expert_ids, [0, route]), pl.INDEX)
        weight_base = expert * HIDDEN + n0
        acc = pl.create_tensor([1, N_TILE], dtype=pl.INT32)
        for kb in pl.pipeline(LOCAL_INTERMEDIATE // K_TILE, stage=2):
            k0 = kb * K_TILE
            acc = pl.matmul_acc(
                acc,
                x[route : route + 1, k0 : k0 + K_TILE],
                weight_flat[weight_base : weight_base + N_TILE, k0 : k0 + K_TILE],
                b_trans=True,
                init_cond=(kb == 0),
            )
        out[route : route + 1, n0 : n0 + N_TILE] = acc
    return out


@pl.jit
def w2_nt(
    x: pl.Tensor[[ROUTES_DYN, LOCAL_INTERMEDIATE], pl.INT8],
    expert_ids: pl.Tensor[[1, ROUTES_DYN], pl.INT32],
    weight: pl.Tensor[[EXPERTS, HIDDEN, LOCAL_INTERMEDIATE], pl.INT8],
    out: pl.Out[pl.Tensor[[ROUTES_DYN, HIDDEN], pl.INT32]],
) -> pl.Tensor:
    return w2_nt_impl(x, expert_ids, weight, out)


@pl.jit.host
def resident_w2_nt(
    x: pl.Tensor[[ROUTES_DYN, LOCAL_INTERMEDIATE], pl.INT8],
    expert_ids: pl.Tensor[[1, ROUTES_DYN], pl.INT32],
    weight: pl.Tensor[[EXPERTS, HIDDEN, LOCAL_INTERMEDIATE], pl.INT8],
    out: pl.Out[pl.Tensor[[ROUTES_DYN, HIDDEN], pl.INT32]],
):
    x.bind_dynamic(0, ROUTES_DYN)
    expert_ids.bind_dynamic(1, ROUTES_DYN)
    out.bind_dynamic(0, ROUTES_DYN)
    w2_nt(x, expert_ids, weight, out, device=0)
