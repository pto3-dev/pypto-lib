# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -------------------------------------------------------------------------------------------------------
"""Candidate fused W13/SwiGLU/quant epilogue with cache-line-owned scale rows."""

from typing import Any

import pypto.language as pl

from .config import EXPERTS, HIDDEN, LOCAL_INTERMEDIATE, TOP_K, W13_WIDTH
from .nz import W2_K_TILE, W2_N_TILE

SCALE_COLUMNS = 16


def make_parallel_swiglu_quant(
    routes: int, *, inline: bool = False, token_input_scale: bool = False
) -> Any:
    """Parallel vector epilogues with one cache-line-owned scale row per route."""
    if not 8 <= routes <= 128 or routes % 8:
        raise ValueError(f"Unsupported parallel SwiGLU route count: {routes}")
    routes_per_scale = TOP_K if token_input_scale else 1
    scale_rows = routes // routes_per_scale
    early = routes == TOP_K and token_input_scale
    decorator = pl.jit.inline if inline else pl.jit

    @decorator
    def parallel_swiglu_quant(
        acc: pl.Tensor[[routes, W13_WIDTH], pl.INT32],
        input_scale: pl.Tensor[[scale_rows, 1], pl.FP32],
        expert_ids: pl.Tensor[[1, routes], pl.INT32],
        weight_scale: pl.Tensor[[EXPERTS, W13_WIDTH], pl.FP32],
        quant: pl.Out[pl.Tensor[[routes, LOCAL_INTERMEDIATE], pl.INT8]],
        scale_pad: pl.Out[pl.Tensor[[routes, SCALE_COLUMNS], pl.FP32]],
    ) -> tuple[pl.Tensor, pl.Tensor]:
        for route in pl.spmd(routes, name_hint="expert_swiglu_parallel", allow_early_resolve=early):
            expert = pl.cast(pl.read(expert_ids, [0, route]), pl.INDEX)
            activation_scale = pl.read(input_scale, [route // routes_per_scale, 0])
            gate = pl.mul(pl.cast(acc[route : route + 1, :LOCAL_INTERMEDIATE], pl.FP32), activation_scale)
            up = pl.mul(
                pl.cast(acc[route : route + 1, LOCAL_INTERMEDIATE:W13_WIDTH], pl.FP32), activation_scale
            )
            gate = pl.mul(gate, weight_scale[expert : expert + 1, :LOCAL_INTERMEDIATE])
            up = pl.mul(up, weight_scale[expert : expert + 1, LOCAL_INTERMEDIATE:W13_WIDTH])
            silu = pl.div(gate, pl.add(pl.exp(pl.neg(gate)), 1.0))
            activated = pl.set_validshape(pl.mul(up, silu), 1, LOCAL_INTERMEDIATE)
            reduction_rows = pl.col_expand_mul(
                pl.full([8, LOCAL_INTERMEDIATE], dtype=pl.FP32, value=1.0), pl.abs(activated)
            )
            maximum = pl.read(pl.row_max(reduction_rows), [0, 0])
            stored_scale = maximum / 127.0
            # Each block writes a whole 64-byte line; no cross-block scalar stores.
            scale_pad[route : route + 1, :] = pl.mul(
                pl.full([1, SCALE_COLUMNS], dtype=pl.FP32, value=1.0), stored_scale
            )
            if maximum > 0.0:
                normalized = pl.set_validshape(pl.mul(activated, 1.0 / stored_scale), 1, LOCAL_INTERMEDIATE)
                quant[route : route + 1, :] = pl.cast(
                    pl.cast(normalized, pl.FP16, mode="rint"), pl.INT8, mode="rint"
                )
            else:
                quant[route : route + 1, :] = pl.cast(activated, pl.INT8, mode="trunc")
        return quant, scale_pad

    return parallel_swiglu_quant


def make_fused_expert_impls(
    routes: int, *, w13_k_tile: int = 256, w13_stages: int = 2, token_input_scale: bool = False
) -> tuple[Any, Any]:
    if not 8 <= routes <= 128 or routes % 8:
        raise ValueError(f"Unsupported fused expert route count: {routes}")
    if (w13_k_tile, w13_stages) not in ((256, 2), (512, 1)):
        raise ValueError(f"Unsupported fused W13 pipeline: {(w13_k_tile, w13_stages)}")
    routes_per_input = TOP_K if token_input_scale else 1
    input_rows = routes // routes_per_input
    early = routes == TOP_K and token_input_scale

    @pl.jit.inline
    def w13_swiglu_output(
        x: pl.Tensor[[input_rows, HIDDEN], pl.INT8],
        input_scale: pl.Tensor[[input_rows, 1], pl.FP32],
        expert_ids: pl.Tensor[[1, routes], pl.INT32],
        weight: pl.Tensor[[EXPERTS, HIDDEN, W13_WIDTH], pl.INT8, pl.NZ],
        weight_scale: pl.Tensor[[EXPERTS, W13_WIDTH], pl.FP32],
        quant: pl.Tensor[[routes, LOCAL_INTERMEDIATE], pl.INT8],
        scale_pad: pl.Tensor[[routes, SCALE_COLUMNS], pl.FP32],
    ) -> tuple[pl.Tensor, pl.Tensor]:
        for route in pl.spmd(routes, name_hint="w13_swiglu_output"):
            expert = pl.cast(pl.read(expert_ids, [0, route]), pl.INDEX)
            input_row = route // routes_per_input
            activation_scale = pl.read(input_scale, [input_row, 0])
            acc = pl.create_tensor([1, W13_WIDTH], dtype=pl.INT32)
            for kb in pl.pipeline(HIDDEN // w13_k_tile, stage=w13_stages):
                k0 = kb * w13_k_tile
                rhs = pl.reshape(
                    pl.slice(weight, [1, w13_k_tile, W13_WIDTH], [expert, k0, 0]), [w13_k_tile, W13_WIDTH]
                )
                acc = pl.matmul_acc(acc, x[input_row : input_row + 1, k0 : k0 + w13_k_tile], rhs, init_cond=(kb == 0))
            for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                values = pl.cast(pl.aiv_shard(acc), pl.FP32, mode="rint")
                gate = pl.mul(pl.slice(values, [1, LOCAL_INTERMEDIATE], [0, 0]), activation_scale)
                up = pl.mul(
                    pl.slice(values, [1, LOCAL_INTERMEDIATE], [0, LOCAL_INTERMEDIATE]),
                    activation_scale,
                )
                gate = pl.mul(gate, pl.slice(weight_scale, [1, LOCAL_INTERMEDIATE], [expert, 0]))
                up = pl.mul(up, pl.slice(weight_scale, [1, LOCAL_INTERMEDIATE], [expert, LOCAL_INTERMEDIATE]))
                silu = pl.div(gate, pl.add(pl.exp(pl.neg(gate)), 1.0))
                activated = pl.set_validshape(pl.mul(up, silu), 1, LOCAL_INTERMEDIATE)
                reduction_rows = pl.col_expand_mul(
                    pl.full([8, LOCAL_INTERMEDIATE], dtype=pl.FP32, value=1.0), pl.abs(activated)
                )
                amax = pl.row_max(reduction_rows)
                maximum = pl.read(amax, [0, 0])
                stored_scale = maximum / 127.0
                line = pl.mul(pl.full([1, SCALE_COLUMNS], dtype=pl.FP32, value=1.0), stored_scale)
                if aiv_id == 0:
                    scale_pad[route : route + 1, :] = line
                if maximum > 0.0:
                    normalized = pl.set_validshape(
                        pl.mul(activated, 1.0 / stored_scale), 1, LOCAL_INTERMEDIATE
                    )
                    narrowed = pl.cast(pl.cast(normalized, pl.FP16, mode="rint"), pl.INT8, mode="rint")
                    if aiv_id == 0:
                        quant[route : route + 1, :] = narrowed
                else:
                    if aiv_id == 0:
                        quant[route : route + 1, :] = pl.cast(activated, pl.INT8, mode="trunc")
        return quant, scale_pad

    @pl.jit.inline
    def w2_padded_output(
        x: pl.Tensor[[routes, LOCAL_INTERMEDIATE], pl.INT8],
        scale_pad: pl.Tensor[[routes, SCALE_COLUMNS], pl.FP32],
        expert_ids: pl.Tensor[[1, routes], pl.INT32],
        weight: pl.Tensor[[EXPERTS, LOCAL_INTERMEDIATE, HIDDEN], pl.INT8, pl.NZ],
        weight_scale: pl.Tensor[[EXPERTS, HIDDEN], pl.FP32],
        y: pl.Tensor[[routes, HIDDEN], pl.BF16],
    ) -> pl.Tensor:
        for route in pl.spmd(routes, name_hint="w2_padded_output", allow_early_resolve=early):
            pl.set_cache_policy(weight, pl.CachePolicy.BYPASS)
            expert = pl.cast(pl.read(expert_ids, [0, route]), pl.INDEX)
            activation_scale = pl.read(scale_pad, [route, 0])
            for n0 in pl.unroll(0, HIDDEN, W2_N_TILE):
                acc = pl.create_tensor([1, W2_N_TILE], dtype=pl.INT32)
                for kb in pl.pipeline(LOCAL_INTERMEDIATE // W2_K_TILE, stage=2):
                    k0 = kb * W2_K_TILE
                    rhs = pl.reshape(
                        pl.slice(weight, [1, W2_K_TILE, W2_N_TILE], [expert, k0, n0]), [W2_K_TILE, W2_N_TILE]
                    )
                    acc = pl.matmul_acc(
                        acc, x[route : route + 1, k0 : k0 + W2_K_TILE], rhs, init_cond=(kb == 0)
                    )
                value = pl.cast(acc, pl.FP32, mode="rint")
                value = pl.mul(value, weight_scale[expert : expert + 1, n0 : n0 + W2_N_TILE])
                y[route : route + 1, n0 : n0 + W2_N_TILE] = pl.cast(
                    pl.mul(value, activation_scale), pl.BF16, mode="rint"
                )
        return y

    return w13_swiglu_output, w2_padded_output


def make_w2_route_combine(rows: int, *, n_tile: int = 512) -> Any:
    """Retain each rounded expert result and the native top8 FP32 sum order."""
    if rows != 1 or n_tile not in (256, 512):
        raise ValueError(f"Fused W2/combine requires rows=1 and N=256 or 512, got {rows}, {n_tile}")
    routes = rows * TOP_K
    n_blocks = HIDDEN // n_tile

    @pl.jit.inline
    def w2_route_combine(
        x: pl.Tensor[[routes, LOCAL_INTERMEDIATE], pl.INT8],
        scale_pad: pl.Tensor[[routes, SCALE_COLUMNS], pl.FP32],
        expert_ids: pl.Tensor[[1, routes], pl.INT32],
        route_weights: pl.Tensor[[rows, TOP_K], pl.FP32],
        weight: pl.Tensor[[EXPERTS, LOCAL_INTERMEDIATE, HIDDEN], pl.INT8, pl.NZ],
        weight_scale: pl.Tensor[[EXPERTS, HIDDEN], pl.FP32],
        y: pl.Tensor[[rows, HIDDEN], pl.BF16],
    ) -> pl.Tensor:
        for block in pl.spmd(rows * n_blocks, name_hint="w2_route_combine"):
            token = 0
            n0 = block * n_tile
            combined = pl.full([1, n_tile], dtype=pl.FP32, value=0.0)
            for index in pl.range(TOP_K):
                route = token * TOP_K + index
                expert = pl.cast(pl.read(expert_ids, [0, route]), pl.INDEX)
                activation_scale = pl.read(scale_pad, [route, 0])
                acc = pl.create_tensor([1, n_tile], dtype=pl.INT32)
                for kb in pl.pipeline(LOCAL_INTERMEDIATE // W2_K_TILE, stage=2):
                    k0 = kb * W2_K_TILE
                    rhs = pl.reshape(
                        pl.slice(weight, [1, W2_K_TILE, n_tile], [expert, k0, n0]), [W2_K_TILE, n_tile]
                    )
                    acc = pl.matmul_acc(
                        acc, x[route : route + 1, k0 : k0 + W2_K_TILE], rhs, init_cond=(kb == 0)
                    )
                value = pl.mul(pl.cast(acc, pl.FP32, mode="rint"),
                               weight_scale[expert : expert + 1, n0 : n0 + n_tile])
                rounded = pl.cast(pl.mul(value, activation_scale), pl.BF16, mode="rint")
                probability = pl.read(route_weights, [token, index])
                combined = pl.add(combined, pl.mul(pl.cast(rounded, pl.FP32), probability))
            y[token : token + 1, n0 : n0 + n_tile] = pl.cast(combined, pl.BF16, mode="rint")
        return y

    return w2_route_combine
