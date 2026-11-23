# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -------------------------------------------------------------------------------------------------------
"""Device-indexed expert Gate/Up GEMM; no routing or dequantization changes."""

import pypto.language as pl

from .config import EXPERTS, HIDDEN, W13_WIDTH

ROUTES_DYN = pl.dynamic("W13_ROUTES")
N_TILE = 256
K_TILE = 512
assert HIDDEN % K_TILE == 0 and W13_WIDTH % N_TILE == 0


@pl.jit
def w13_nd(
    x: pl.Tensor[[ROUTES_DYN, HIDDEN], pl.INT8],
    expert_ids: pl.Tensor[[1, ROUTES_DYN], pl.INT32],
    weight: pl.Tensor[[EXPERTS, HIDDEN, W13_WIDTH], pl.INT8],
    out: pl.Out[pl.Tensor[[ROUTES_DYN, W13_WIDTH], pl.INT32]],
) -> pl.Tensor:
    """Baseline: weights in [E,K,N] order, indexed by the device expert IDs."""
    x.bind_dynamic(0, ROUTES_DYN)
    expert_ids.bind_dynamic(1, ROUTES_DYN)
    out.bind_dynamic(0, ROUTES_DYN)
    routes = pl.tensor.dim(x, 0)
    weight_flat = pl.reshape(weight, [EXPERTS * HIDDEN, W13_WIDTH])
    for block in pl.spmd(routes * (W13_WIDTH // N_TILE), name_hint="w13_nd"):
        route = block // (W13_WIDTH // N_TILE)
        n0 = (block % (W13_WIDTH // N_TILE)) * N_TILE
        expert = pl.cast(pl.read(expert_ids, [0, route]), pl.INDEX)
        weight_base = expert * HIDDEN
        acc = pl.create_tensor([1, N_TILE], dtype=pl.INT32)
        for kb in pl.pipeline(HIDDEN // K_TILE, stage=2):
            k0 = kb * K_TILE
            acc = pl.matmul_acc(
                acc,
                x[route : route + 1, k0 : k0 + K_TILE],
                weight_flat[weight_base + k0 : weight_base + k0 + K_TILE, n0 : n0 + N_TILE],
                init_cond=(kb == 0),
            )
        out[route : route + 1, n0 : n0 + N_TILE] = acc
    return out


@pl.jit.inline
def w13_nt_impl(
    x: pl.Tensor[[ROUTES_DYN, HIDDEN], pl.INT8],
    expert_ids: pl.Tensor[[1, ROUTES_DYN], pl.INT32],
    weight: pl.Tensor[[EXPERTS, W13_WIDTH, HIDDEN], pl.INT8],
    out: pl.Out[pl.Tensor[[ROUTES_DYN, W13_WIDTH], pl.INT32]],
) -> pl.Tensor:
    """Candidate: K-contiguous [E,N,K] weights packed once before dispatch."""
    x.bind_dynamic(0, ROUTES_DYN)
    expert_ids.bind_dynamic(1, ROUTES_DYN)
    out.bind_dynamic(0, ROUTES_DYN)
    routes = pl.tensor.dim(x, 0)
    weight_flat = pl.reshape(weight, [EXPERTS * W13_WIDTH, HIDDEN])
    for block in pl.spmd(routes * (W13_WIDTH // N_TILE), name_hint="w13_nt"):
        route = block // (W13_WIDTH // N_TILE)
        n0 = (block % (W13_WIDTH // N_TILE)) * N_TILE
        expert = pl.cast(pl.read(expert_ids, [0, route]), pl.INDEX)
        weight_base = expert * W13_WIDTH + n0
        acc = pl.create_tensor([1, N_TILE], dtype=pl.INT32)
        for kb in pl.pipeline(HIDDEN // K_TILE, stage=2):
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
def w13_nt(
    x: pl.Tensor[[ROUTES_DYN, HIDDEN], pl.INT8],
    expert_ids: pl.Tensor[[1, ROUTES_DYN], pl.INT32],
    weight: pl.Tensor[[EXPERTS, W13_WIDTH, HIDDEN], pl.INT8],
    out: pl.Out[pl.Tensor[[ROUTES_DYN, W13_WIDTH], pl.INT32]],
) -> pl.Tensor:
    return w13_nt_impl(x, expert_ids, weight, out)


@pl.jit.host
def resident_w13_nd(
    x: pl.Tensor[[ROUTES_DYN, HIDDEN], pl.INT8],
    expert_ids: pl.Tensor[[1, ROUTES_DYN], pl.INT32],
    weight: pl.Tensor[[EXPERTS, HIDDEN, W13_WIDTH], pl.INT8],
    out: pl.Out[pl.Tensor[[ROUTES_DYN, W13_WIDTH], pl.INT32]],
):
    x.bind_dynamic(0, ROUTES_DYN)
    expert_ids.bind_dynamic(1, ROUTES_DYN)
    out.bind_dynamic(0, ROUTES_DYN)
    w13_nd(x, expert_ids, weight, out, device=0)


@pl.jit.host
def resident_w13_nt(
    x: pl.Tensor[[ROUTES_DYN, HIDDEN], pl.INT8],
    expert_ids: pl.Tensor[[1, ROUTES_DYN], pl.INT32],
    weight: pl.Tensor[[EXPERTS, W13_WIDTH, HIDDEN], pl.INT8],
    out: pl.Out[pl.Tensor[[ROUTES_DYN, W13_WIDTH], pl.INT32]],
):
    x.bind_dynamic(0, ROUTES_DYN)
    expert_ids.bind_dynamic(1, ROUTES_DYN)
    out.bind_dynamic(0, ROUTES_DYN)
    w13_nt(x, expert_ids, weight, out, device=0)
