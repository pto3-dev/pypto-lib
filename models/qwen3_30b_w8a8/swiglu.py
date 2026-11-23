# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -------------------------------------------------------------------------------------------------------
"""Expert dequantization/SwiGLU/quantization; native-oracle validation is required."""

import pypto.language as pl

from .config import EXPERTS, LOCAL_INTERMEDIATE, W13_WIDTH
from .w13 import ROUTES_DYN

SCALE_LINE_ROWS = 16  # Each group owns a 64-byte FP32 scale cache line.


@pl.jit.inline
def expert_swiglu_quant_impl(
    acc: pl.Tensor[[ROUTES_DYN, W13_WIDTH], pl.INT32],
    input_scale: pl.Tensor[[ROUTES_DYN, 1], pl.FP32],
    expert_ids: pl.Tensor[[1, ROUTES_DYN], pl.INT32],
    weight_scale: pl.Tensor[[EXPERTS, W13_WIDTH], pl.FP32],
    quant: pl.Out[pl.Tensor[[ROUTES_DYN, LOCAL_INTERMEDIATE], pl.INT8]],
    scale: pl.Out[pl.Tensor[[ROUTES_DYN, 1], pl.FP32]],
) -> tuple[pl.Tensor, pl.Tensor]:
    acc.bind_dynamic(0, ROUTES_DYN)
    input_scale.bind_dynamic(0, ROUTES_DYN)
    expert_ids.bind_dynamic(1, ROUTES_DYN)
    quant.bind_dynamic(0, ROUTES_DYN)
    scale.bind_dynamic(0, ROUTES_DYN)
    routes = pl.tensor.dim(acc, 0)
    for group in pl.spmd((routes + SCALE_LINE_ROWS - 1) // SCALE_LINE_ROWS, name_hint="expert_swiglu"):
        base = group * SCALE_LINE_ROWS
        for route in pl.range(base, pl.min(routes, base + SCALE_LINE_ROWS)):
            expert = pl.cast(pl.read(expert_ids, [0, route]), pl.INDEX)
            activation_scale = pl.read(input_scale, [route, 0])
            gate = pl.cast(acc[route : route + 1, :LOCAL_INTERMEDIATE], pl.FP32)
            up = pl.cast(acc[route : route + 1, LOCAL_INTERMEDIATE:W13_WIDTH], pl.FP32)
            gate = pl.mul(gate, activation_scale)
            up = pl.mul(up, activation_scale)
            gate = pl.mul(gate, weight_scale[expert : expert + 1, :LOCAL_INTERMEDIATE])
            up = pl.mul(up, weight_scale[expert : expert + 1, LOCAL_INTERMEDIATE:W13_WIDTH])
            silu = pl.div(gate, pl.add(pl.exp(pl.neg(gate)), 1.0))
            activated = pl.set_validshape(pl.mul(up, silu), 1, LOCAL_INTERMEDIATE)
            # A2/A3 column-major FP32 reduction tiles need eight aligned rows.
            reduction_rows = pl.col_expand_mul(
                pl.full([8, LOCAL_INTERMEDIATE], dtype=pl.FP32, value=1.0), pl.abs(activated)
            )
            amax = pl.row_max(reduction_rows)
            max_value = pl.read(amax, [0, 0])
            stored_scale = max_value / 127.0
            pl.write(scale, [route, 0], stored_scale)
            if max_value > 0.0:
                inverse_scale = 1.0 / stored_scale
                normalized = pl.set_validshape(pl.mul(activated, inverse_scale), 1, LOCAL_INTERMEDIATE)
                # The native fused GMM epilogue rounds through FP16, unlike DynamicQuant.
                quant[route : route + 1, :] = pl.cast(
                    pl.cast(normalized, pl.FP16, mode="rint"), pl.INT8, mode="rint"
                )
            else:
                quant[route : route + 1, :] = pl.cast(activated, pl.INT8, mode="trunc")
    return quant, scale


@pl.jit
def expert_swiglu_quant(
    acc: pl.Tensor[[ROUTES_DYN, W13_WIDTH], pl.INT32],
    input_scale: pl.Tensor[[ROUTES_DYN, 1], pl.FP32],
    expert_ids: pl.Tensor[[1, ROUTES_DYN], pl.INT32],
    weight_scale: pl.Tensor[[EXPERTS, W13_WIDTH], pl.FP32],
    quant: pl.Out[pl.Tensor[[ROUTES_DYN, LOCAL_INTERMEDIATE], pl.INT8]],
    scale: pl.Out[pl.Tensor[[ROUTES_DYN, 1], pl.FP32]],
) -> tuple[pl.Tensor, pl.Tensor]:
    return expert_swiglu_quant_impl(acc, input_scale, expert_ids, weight_scale, quant, scale)


@pl.jit.inline
def expert_swiglu_quant_batched_impl(
    acc: pl.Tensor[[ROUTES_DYN, W13_WIDTH], pl.INT32],
    input_scale: pl.Tensor[[ROUTES_DYN, 1], pl.FP32],
    expert_ids: pl.Tensor[[1, ROUTES_DYN], pl.INT32],
    weight_scale: pl.Tensor[[EXPERTS, W13_WIDTH], pl.FP32],
    quant: pl.Tensor,
    scale: pl.Tensor,
) -> tuple[pl.Tensor, pl.Tensor]:
    """Reduce eight real activation rows together instead of duplicating each row."""
    routes = pl.tensor.dim(acc, 0)
    for group in pl.spmd((routes + SCALE_LINE_ROWS - 1) // SCALE_LINE_ROWS, name_hint="expert_swiglu_batch"):
        group_base = group * SCALE_LINE_ROWS
        for base in pl.range(group_base, pl.min(routes, group_base + SCALE_LINE_ROWS), 8):
            valid_rows = pl.min(8, routes - base)
            activated_rows = pl.full([8, LOCAL_INTERMEDIATE], dtype=pl.FP32, value=0.0)
            for lane in pl.range(valid_rows):
                route = base + lane
                expert = pl.cast(pl.read(expert_ids, [0, route]), pl.INDEX)
                activation_scale = pl.read(input_scale, [route, 0])
                gate = pl.mul(pl.cast(acc[route : route + 1, :LOCAL_INTERMEDIATE], pl.FP32), activation_scale)
                up = pl.mul(
                    pl.cast(acc[route : route + 1, LOCAL_INTERMEDIATE:W13_WIDTH], pl.FP32), activation_scale
                )
                gate = pl.mul(gate, weight_scale[expert : expert + 1, :LOCAL_INTERMEDIATE])
                up = pl.mul(up, weight_scale[expert : expert + 1, LOCAL_INTERMEDIATE:W13_WIDTH])
                silu = pl.div(gate, pl.add(pl.exp(pl.neg(gate)), 1.0))
                activated_rows[lane : lane + 1, :] = pl.mul(up, silu)
            amax = pl.row_max(pl.abs(activated_rows))
            for lane in pl.range(valid_rows):
                route = base + lane
                max_value = pl.read(amax, [lane, 0])
                stored_scale = max_value / 127.0
                pl.write(scale, [route, 0], stored_scale)
                row = pl.slice(
                    activated_rows,
                    [1, LOCAL_INTERMEDIATE],
                    [lane, 0],
                    valid_shape=[1, LOCAL_INTERMEDIATE],
                )
                if max_value > 0.0:
                    normalized = pl.mul(row, 1.0 / stored_scale)
                    quant[route : route + 1, :] = pl.cast(
                        pl.cast(normalized, pl.FP16, mode="rint"), pl.INT8, mode="rint"
                    )
                else:
                    quant[route : route + 1, :] = pl.cast(row, pl.INT8, mode="trunc")
    return quant, scale


@pl.jit
def expert_swiglu_quant_batched(
    acc: pl.Tensor[[ROUTES_DYN, W13_WIDTH], pl.INT32],
    input_scale: pl.Tensor[[ROUTES_DYN, 1], pl.FP32],
    expert_ids: pl.Tensor[[1, ROUTES_DYN], pl.INT32],
    weight_scale: pl.Tensor[[EXPERTS, W13_WIDTH], pl.FP32],
    quant: pl.Out[pl.Tensor[[ROUTES_DYN, LOCAL_INTERMEDIATE], pl.INT8]],
    scale: pl.Out[pl.Tensor[[ROUTES_DYN, 1], pl.FP32]],
) -> tuple[pl.Tensor, pl.Tensor]:
    acc.bind_dynamic(0, ROUTES_DYN)
    input_scale.bind_dynamic(0, ROUTES_DYN)
    expert_ids.bind_dynamic(1, ROUTES_DYN)
    quant.bind_dynamic(0, ROUTES_DYN)
    scale.bind_dynamic(0, ROUTES_DYN)
    return expert_swiglu_quant_batched_impl(acc, input_scale, expert_ids, weight_scale, quant, scale)
