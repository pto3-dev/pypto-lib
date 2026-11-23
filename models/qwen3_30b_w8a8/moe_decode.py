# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -------------------------------------------------------------------------------------------------------
"""Rank-local MoE Decode in one callable; native TP collective stays outside."""

from typing import Any

import pypto.language as pl

from .combine import combine_routes_impl
from .config import EXPERTS, HIDDEN, LOCAL_INTERMEDIATE, TOP_K, W13_WIDTH
from .dispatch import dispatch_quantized_routes_impl, dispatch_quantized_routes_grouped_impl
from .nz import make_w13_nz_token_inputs, w13_nz_impl, w2_nz_output_impl
from .route_quant import make_routing_quant_hidden_impl, routing_quant_hidden_tiled_impl
from .router import make_compensated_router, make_router_topk_impl
from .swiglu import expert_swiglu_quant_batched_impl
from .expert_fusion import make_fused_expert_impls, make_parallel_swiglu_quant, make_w2_route_combine, SCALE_COLUMNS
from .expert_grouped import make_grouped_expert


def _expert_epilogue_policy(
    rows: int, fused_experts: bool | None, grouped_experts: bool, parallel_swiglu: bool | None
) -> tuple[bool, bool]:
    if parallel_swiglu is None:
        # B1 passed the complete captured-state and composed-performance gates.
        parallel_swiglu = rows == 1 and not grouped_experts and fused_experts is not True
    if fused_experts is None:
        fused_experts = rows == 2 and not grouped_experts and not parallel_swiglu
    if fused_experts and grouped_experts:
        raise ValueError("Grouped and fused expert candidates are mutually exclusive")
    if parallel_swiglu and (fused_experts or grouped_experts):
        raise ValueError("Parallel SwiGLU requires unfused, ungrouped experts")
    return fused_experts, parallel_swiglu


def _direct_expert_input_policy(
    rows: int, parallel_swiglu: bool, grouped_dispatch: bool | None, requested: bool | None
) -> bool:
    if requested is None:
        requested = rows == 1 and parallel_swiglu and grouped_dispatch is not True
    if requested and not parallel_swiglu:
        raise ValueError("Direct expert inputs require parallel_swiglu=True without fused/grouped experts")
    if requested and grouped_dispatch:
        raise ValueError("Direct expert inputs cannot also select grouped dispatch")
    return requested


def make_moe_decode(
    rows: int,
    router_k_tile: int = 32,
    *,
    inline: bool = False,
    fused_experts: bool | None = None,
    parallel_swiglu: bool | None = None,
    direct_expert_inputs: bool | None = None,
    grouped_experts: bool = False,
    router_sum_rows: int | None = None,
    router_sum_stages: int | None = None,
    router_partials_per_block: int | None = None,
    tiled_routing_quant: bool | None = None,
    grouped_dispatch: bool | None = None,
    moe_fusion: str = "none",
) -> Any:
    if not 1 <= rows <= 16:
        raise ValueError(f"Unsupported MoE Decode bucket: {rows}")
    fused_experts, parallel_swiglu = _expert_epilogue_policy(
        rows, fused_experts, grouped_experts, parallel_swiglu
    )
    direct_expert_inputs = _direct_expert_input_policy(
        rows, parallel_swiglu, grouped_dispatch, direct_expert_inputs
    )
    if moe_fusion not in ("none", "w13", "w2", "w13_w2"):
        raise ValueError(f"Unsupported MoE fusion mode: {moe_fusion}")
    if moe_fusion != "none" and (rows != 1 or not direct_expert_inputs):
        raise ValueError("MoE fusion currently requires rows=1 and direct expert inputs")
    fuse_direct_w13 = moe_fusion in ("w13", "w13_w2")
    fuse_route_combine = moe_fusion in ("w2", "w13_w2")
    if tiled_routing_quant is None:
        tiled_routing_quant = rows == 16
    if grouped_dispatch is None:
        grouped_dispatch = rows == 16 and not direct_expert_inputs
    if router_sum_stages is None:
        # Only B1/K32 has passed the complete captured-state and composed-speed gates.
        router_sum_stages = 2 if rows == 1 and router_k_tile == 32 else 1
    if router_partials_per_block is None:
        # Coalesce independent K32 products; keep the compensated sum order unchanged.
        # Only B1/K32 has passed all captured states and the composed-speed gate.
        router_partials_per_block = 4 if rows == 1 and router_k_tile == 32 else 1
    routes = rows * TOP_K
    compensated = make_compensated_router(
        rows, inline=True, partial_k=router_k_tile, sum_rows=router_sum_rows,
        sum_stages=router_sum_stages,
        partials_per_block=router_partials_per_block,
    )
    w13_swiglu_output, w2_padded_output = make_fused_expert_impls(
        routes, token_input_scale=direct_expert_inputs
    )
    w2_route_combine = make_w2_route_combine(1)
    parallel_swiglu_output = make_parallel_swiglu_quant(
        routes, inline=True, token_input_scale=direct_expert_inputs
    )
    direct_w13 = make_w13_nz_token_inputs(rows, inline=True)
    grouped_expert_output = make_grouped_expert(routes, diagnostic=False, inline=True, combine=True)
    topk = make_router_topk_impl(early=rows == 1)
    quantize = make_routing_quant_hidden_impl(early=rows == 1)
    if tiled_routing_quant:
        quantize = routing_quant_hidden_tiled_impl
    dispatch = dispatch_quantized_routes_grouped_impl if grouped_dispatch else dispatch_quantized_routes_impl

    decorator = pl.jit.inline if inline else pl.jit

    @decorator
    def moe_decode(
        x: pl.Tensor[[rows, HIDDEN], pl.BF16],
        gate: pl.Tensor[[EXPERTS, HIDDEN], pl.BF16],
        w13: pl.Tensor[[EXPERTS, HIDDEN, W13_WIDTH], pl.INT8, pl.NZ],
        w13_scale: pl.Tensor[[EXPERTS, W13_WIDTH], pl.FP32],
        w2: pl.Tensor[[EXPERTS, LOCAL_INTERMEDIATE, HIDDEN], pl.INT8, pl.NZ],
        w2_scale: pl.Tensor[[EXPERTS, HIDDEN], pl.FP32],
        y: pl.Out[pl.Tensor[[rows, HIDDEN], pl.BF16]],
    ) -> pl.Tensor:
        logits = pl.create_tensor([rows, EXPERTS], dtype=pl.BF16)
        ids = pl.create_tensor([rows, TOP_K], dtype=pl.INT32)
        weights = pl.create_tensor([rows, TOP_K], dtype=pl.FP32)
        quant_x = pl.create_tensor([rows, HIDDEN], dtype=pl.INT8)
        input_scale = pl.create_tensor([rows, 1], dtype=pl.FP32)
        acc13 = pl.create_tensor([routes, W13_WIDTH], dtype=pl.INT32)
        quant13 = pl.create_tensor([routes, LOCAL_INTERMEDIATE], dtype=pl.INT8)
        expert_y = pl.create_tensor([routes, HIDDEN], dtype=pl.BF16)
        logits = compensated(x, gate, logits)
        ids, weights = topk(logits, ids, weights)
        quant_x, input_scale = quantize(x, quant_x, input_scale)
        if grouped_experts:
            grouped_expert_output(quant_x, input_scale, ids, weights, w13, w13_scale, w2, w2_scale, y)
        elif direct_expert_inputs:
            direct_ids = pl.reshape(ids, [1, routes])
            scale_pad = pl.create_tensor([routes, SCALE_COLUMNS], dtype=pl.FP32)
            if fuse_direct_w13:
                parallel_quant, parallel_scale = w13_swiglu_output(
                    quant_x, input_scale, direct_ids, w13, w13_scale, quant13, scale_pad
                )
            else:
                acc13 = direct_w13(quant_x, direct_ids, w13, acc13)
                parallel_quant, parallel_scale = parallel_swiglu_output(
                    acc13, input_scale, direct_ids, w13_scale, quant13, scale_pad
                )
            if fuse_route_combine:
                w2_route_combine(parallel_quant, parallel_scale, direct_ids, weights, w2, w2_scale, y)
            else:
                expert_y = w2_padded_output(parallel_quant, parallel_scale, direct_ids, w2, w2_scale, expert_y)
                combine_routes_impl(expert_y, weights, y)
        else:
            route_x = pl.create_tensor([routes, HIDDEN], dtype=pl.INT8)
            route_scale = pl.create_tensor([routes, 1], dtype=pl.FP32)
            route_ids = pl.create_tensor([1, routes], dtype=pl.INT32)
            route_x, route_scale, route_ids = dispatch(
                quant_x, input_scale, ids, route_x, route_scale, route_ids
            )
            if fused_experts:
                scale_pad = pl.create_tensor([routes, SCALE_COLUMNS], dtype=pl.FP32)
                fused_quant, fused_scale = w13_swiglu_output(
                    route_x, route_scale, route_ids, w13, w13_scale, quant13, scale_pad
                )
                expert_y = w2_padded_output(fused_quant, fused_scale, route_ids, w2, w2_scale, expert_y)
            elif parallel_swiglu:
                scale_pad = pl.create_tensor([routes, SCALE_COLUMNS], dtype=pl.FP32)
                acc13 = w13_nz_impl(route_x, route_ids, w13, acc13)
                parallel_quant, parallel_scale = parallel_swiglu_output(
                    acc13, route_scale, route_ids, w13_scale, quant13, scale_pad
                )
                expert_y = w2_padded_output(parallel_quant, parallel_scale, route_ids, w2, w2_scale, expert_y)
            else:
                scale13 = pl.create_tensor([routes, 1], dtype=pl.FP32)
                acc13 = w13_nz_impl(route_x, route_ids, w13, acc13)
                quant13, scale13 = expert_swiglu_quant_batched_impl(
                    acc13, route_scale, route_ids, w13_scale, quant13, scale13
                )
                expert_y = w2_nz_output_impl(quant13, scale13, route_ids, w2, w2_scale, expert_y)
            combine_routes_impl(expert_y, weights, y)
        return y

    return moe_decode
