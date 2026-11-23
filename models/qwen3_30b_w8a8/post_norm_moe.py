# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -------------------------------------------------------------------------------------------------------
"""Post-attention residual RMSNorm and rank-local MoE in one callable.

The input and output residual must have distinct storage. Attention TP reduction
and expert-output TP reduction are independently selected at construction.
"""

from typing import Any

import pypto.language as pl
import pypto.language.distributed as pld

from .config import EXPERTS, HIDDEN, LOCAL_INTERMEDIATE, W13_WIDTH
from .moe_decode import make_moe_decode
from .norm import make_add_rms_norm_hidden_impl
from .tp_norm import SIGNAL_TILE, TP_OPERATIONS, tp_add_rms_norm_hidden_impl, tp_reduce_hidden_impl


def make_post_norm_moe(
    rows: int,
    router_k_tile: int = 32,
    *,
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
    tp_reduction: bool = False,
    expert_tp_reduction: bool = False,
    inline: bool = False,
) -> Any:
    if tp_reduction and rows != 1:
        raise ValueError(f"Fused TP reduction requires rows=1, got {rows}")
    if expert_tp_reduction and not tp_reduction:
        raise ValueError("Expert TP reduction requires attention TP reduction and its communication arguments")
    normalize = make_add_rms_norm_hidden_impl(early=rows == 1)
    moe_decode = make_moe_decode(
        rows,
        router_k_tile,
        inline=True,
        fused_experts=fused_experts,
        parallel_swiglu=parallel_swiglu,
        direct_expert_inputs=direct_expert_inputs,
        grouped_experts=grouped_experts,
        router_sum_rows=router_sum_rows,
        router_sum_stages=router_sum_stages,
        router_partials_per_block=router_partials_per_block,
        tiled_routing_quant=tiled_routing_quant,
        grouped_dispatch=grouped_dispatch,
        moe_fusion=moe_fusion,
    )

    decorator = pl.jit.inline if inline else pl.jit
    if tp_reduction:
        @decorator
        def tp_post_norm_moe(
            x: pl.Tensor[[1, HIDDEN], pl.BF16],
            residual: pl.Tensor[[1, HIDDEN], pl.BF16],
            norm_weight: pl.Tensor[[1, HIDDEN], pl.BF16],
            gate: pl.Tensor[[EXPERTS, HIDDEN], pl.BF16],
            w13: pl.Tensor[[EXPERTS, HIDDEN, W13_WIDTH], pl.INT8, pl.NZ],
            w13_scale: pl.Tensor[[EXPERTS, W13_WIDTH], pl.FP32],
            w2: pl.Tensor[[EXPERTS, LOCAL_INTERMEDIATE, HIDDEN], pl.INT8, pl.NZ],
            w2_scale: pl.Tensor[[EXPERTS, HIDDEN], pl.FP32],
            y: pl.Out[pl.Tensor[[1, HIDDEN], pl.BF16]],
            residual_out: pl.Out[pl.Tensor[[1, HIDDEN], pl.BF16]],
            data: pl.InOut[pld.DistributedTensor[[TP_OPERATIONS, HIDDEN], pl.BF16]],
            signal: pl.InOut[pld.DistributedTensor[[TP_OPERATIONS * 4, SIGNAL_TILE], pl.INT32]],
            epochs: pl.InOut[pl.Tensor[[TP_OPERATIONS], pl.INT32]],
            operation: pl.Scalar[pl.INT32],
        ) -> tuple[pl.Tensor, pl.Tensor]:
            reduced = pl.create_tensor([1, HIDDEN], dtype=pl.BF16)
            normalized = pl.create_tensor([1, HIDDEN], dtype=pl.BF16)
            _reduced_after, normalized_after, residual_after = tp_add_rms_norm_hidden_impl(
                x, residual, norm_weight, reduced, normalized, residual_out,
                data, signal, epochs, operation,
            )
            if expert_tp_reduction:
                local_y = pl.create_tensor([1, HIDDEN], dtype=pl.BF16)
                local_after = moe_decode(normalized_after, gate, w13, w13_scale, w2, w2_scale, local_y)
                expert_operation = operation + 1
                y_after = tp_reduce_hidden_impl(local_after, y, data, signal, epochs, expert_operation)
            else:
                y_after = moe_decode(normalized_after, gate, w13, w13_scale, w2, w2_scale, y)
            return y_after, residual_after

        return tp_post_norm_moe

    @decorator
    def post_norm_moe(
        x: pl.Tensor[[rows, HIDDEN], pl.BF16],
        residual: pl.Tensor[[rows, HIDDEN], pl.BF16],
        norm_weight: pl.Tensor[[1, HIDDEN], pl.BF16],
        gate: pl.Tensor[[EXPERTS, HIDDEN], pl.BF16],
        w13: pl.Tensor[[EXPERTS, HIDDEN, W13_WIDTH], pl.INT8, pl.NZ],
        w13_scale: pl.Tensor[[EXPERTS, W13_WIDTH], pl.FP32],
        w2: pl.Tensor[[EXPERTS, LOCAL_INTERMEDIATE, HIDDEN], pl.INT8, pl.NZ],
        w2_scale: pl.Tensor[[EXPERTS, HIDDEN], pl.FP32],
        y: pl.Out[pl.Tensor[[rows, HIDDEN], pl.BF16]],
        residual_out: pl.Out[pl.Tensor[[rows, HIDDEN], pl.BF16]],
    ) -> tuple[pl.Tensor, pl.Tensor]:
        normalized = pl.create_tensor([rows, HIDDEN], dtype=pl.BF16)
        normalized_after, residual_after = normalize(
            x, residual, norm_weight, normalized, residual_out
        )
        y_after = moe_decode(normalized_after, gate, w13, w13_scale, w2, w2_scale, y)
        return y_after, residual_after

    return post_norm_moe
