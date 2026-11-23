# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -------------------------------------------------------------------------------------------------------
"""Opt-in grouped expert computation with device-side stable route permutation."""

from typing import Any

import pypto.language as pl

from .config import EXPERTS, HIDDEN, LOCAL_INTERMEDIATE, TOP_K, W13_WIDTH
from .swiglu import expert_swiglu_quant_batched_impl

ROW_TILE = 16
COPY_ROWS = 8
META_COLUMNS = 16
W13_N_TILE = 384
W13_K_TILE = 512
W2_N_TILE = 512
W2_K_TILE = 128


def make_grouped_expert(
    routes: int, *, diagnostic: bool = True, inline: bool = False, combine: bool = False
) -> Any:
    """Build an expert chain preserving the caller's original route order."""
    if routes < 8 or routes > 128 or routes % 8:
        raise ValueError(f"Expected a route count divisible by 8 in [8, 128], got {routes}")
    if combine and diagnostic:
        raise ValueError("Combined grouped experts expose only the combined output")
    input_rows = routes // TOP_K if combine else routes

    @pl.jit.inline
    def encode_routes(
        ids: pl.Tensor[[1, routes], pl.INT32],
        metadata: pl.Tensor[[routes, META_COLUMNS], pl.INT32],
        permutation: pl.Tensor[[1, routes], pl.INT32],
        inverse: pl.Tensor[[1, routes], pl.INT32],
        sorted_ids: pl.Tensor[[1, routes], pl.INT32],
    ) -> tuple[pl.Tensor, pl.Tensor, pl.Tensor, pl.Tensor]:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="expert_route_encode"):
            counts = pl.array.create(EXPERTS, pl.INT32)
            cursors = pl.array.create(EXPERTS, pl.INT32)
            for expert in pl.range(EXPERTS):
                counts[expert] = 0
            id_tile = pl.add(pl.slice(ids, [1, routes], [0, 0]), 0)
            for route in pl.range(routes):
                expert = pl.cast(pl.read(id_tile, [0, route]), pl.INDEX)
                counts[expert] = counts[expert] + 1
            meta_tile = pl.full([routes, META_COLUMNS], dtype=pl.INT32, value=0)
            offset = pl.cast(0, pl.INT32)
            group = pl.cast(0, pl.INDEX)
            for expert in pl.range(EXPERTS):
                count = counts[expert]
                cursors[expert] = offset
                if count > 0:
                    pl.write(meta_tile, [group, 0], pl.cast(expert, pl.INT32))
                    pl.write(meta_tile, [group, 1], offset)
                    pl.write(meta_tile, [group, 2], count)
                    group = group + 1
                offset = offset + count
            perm_tile = pl.full([1, routes], dtype=pl.INT32, value=0)
            inverse_tile = pl.full([1, routes], dtype=pl.INT32, value=0)
            sorted_tile = pl.full([1, routes], dtype=pl.INT32, value=0)
            for route in pl.range(routes):
                expert = pl.cast(pl.read(id_tile, [0, route]), pl.INDEX)
                position = pl.cast(cursors[expert], pl.INDEX)
                pl.write(perm_tile, [0, position], pl.cast(route, pl.INT32))
                pl.write(inverse_tile, [0, route], pl.cast(position, pl.INT32))
                pl.write(sorted_tile, [0, position], pl.cast(expert, pl.INT32))
                cursors[expert] = cursors[expert] + 1
            metadata[:, :] = meta_tile
            permutation[:, :] = perm_tile
            inverse[:, :] = inverse_tile
            sorted_ids[:, :] = sorted_tile
        return metadata, permutation, inverse, sorted_ids

    @pl.jit.inline
    def gather_routes(
        x: pl.Tensor[[input_rows, HIDDEN], pl.INT8],
        scales: pl.Tensor[[input_rows, 1], pl.FP32],
        permutation: pl.Tensor[[1, routes], pl.INT32],
        sorted_x: pl.Tensor[[routes, HIDDEN], pl.INT8],
        sorted_scale: pl.Tensor[[routes, 1], pl.FP32],
    ) -> tuple[pl.Tensor, pl.Tensor]:
        scale_flat = pl.reshape(sorted_scale, [1, routes])
        for group in pl.spmd(routes // COPY_ROWS, name_hint="expert_route_gather"):
            base = group * COPY_ROWS
            scale_tile = pl.full([1, COPY_ROWS], dtype=pl.FP32, value=0.0)
            for lane in pl.range(COPY_ROWS):
                position = base + lane
                source = pl.cast(pl.read(permutation, [0, position]), pl.INDEX)
                if combine:
                    source = source // TOP_K
                sorted_x[position : position + 1, :] = x[source : source + 1, :]
                pl.write(scale_tile, [0, lane], pl.read(scales, [source, 0]))
            scale_flat[:, base : base + COPY_ROWS] = scale_tile
        return sorted_x, sorted_scale

    @pl.jit.inline
    def grouped_w13(
        x: pl.Tensor[[routes, HIDDEN], pl.INT8],
        metadata: pl.Tensor[[routes, META_COLUMNS], pl.INT32],
        weight: pl.Tensor[[EXPERTS, HIDDEN, W13_WIDTH], pl.INT8, pl.NZ],
        out: pl.Tensor[[routes, W13_WIDTH], pl.INT32],
    ) -> pl.Tensor:
        for group in pl.spmd(routes, name_hint="expert_grouped_w13"):
            expert = pl.cast(pl.read(metadata, [group, 0]), pl.INDEX)
            start = pl.cast(pl.read(metadata, [group, 1]), pl.INDEX)
            count = pl.cast(pl.read(metadata, [group, 2]), pl.INDEX)
            for row in pl.range(0, count, ROW_TILE):
                valid_rows = pl.min(ROW_TILE, count - row)
                for n0 in pl.unroll(0, W13_WIDTH, W13_N_TILE):
                    acc = pl.create_tensor([ROW_TILE, W13_N_TILE], dtype=pl.INT32)
                    for kb in pl.pipeline(HIDDEN // W13_K_TILE, stage=2):
                        k0 = kb * W13_K_TILE
                        lhs = pl.slice(
                            x, [ROW_TILE, W13_K_TILE], [start + row, k0], valid_shape=[valid_rows, W13_K_TILE]
                        )
                        rhs = pl.reshape(
                            pl.slice(weight, [1, W13_K_TILE, W13_N_TILE], [expert, k0, n0]),
                            [W13_K_TILE, W13_N_TILE],
                        )
                        acc = pl.matmul_acc(acc, lhs, rhs, init_cond=(kb == 0))
                    out = pl.assemble(out, pl.set_validshape(acc, valid_rows, W13_N_TILE), [start + row, n0])
        return out

    @pl.jit.inline
    def grouped_w2(
        x: pl.Tensor[[routes, LOCAL_INTERMEDIATE], pl.INT8],
        scales: pl.Tensor[[routes, 1], pl.FP32],
        metadata: pl.Tensor[[routes, META_COLUMNS], pl.INT32],
        weight: pl.Tensor[[EXPERTS, LOCAL_INTERMEDIATE, HIDDEN], pl.INT8, pl.NZ],
        weight_scale: pl.Tensor[[EXPERTS, HIDDEN], pl.FP32],
        out: pl.Tensor[[routes, HIDDEN], pl.INT32],
        y: pl.Tensor[[routes, HIDDEN], pl.BF16],
    ) -> tuple[pl.Tensor, pl.Tensor]:
        for group in pl.spmd(routes, name_hint="expert_grouped_w2"):
            expert = pl.cast(pl.read(metadata, [group, 0]), pl.INDEX)
            start = pl.cast(pl.read(metadata, [group, 1]), pl.INDEX)
            count = pl.cast(pl.read(metadata, [group, 2]), pl.INDEX)
            for row in pl.range(0, count, ROW_TILE):
                valid_rows = pl.min(ROW_TILE, count - row)
                activation_scales = pl.slice(
                    scales, [ROW_TILE, 1], [start + row, 0], valid_shape=[valid_rows, 1]
                )
                for n0 in pl.unroll(0, HIDDEN, W2_N_TILE):
                    acc = pl.create_tensor([ROW_TILE, W2_N_TILE], dtype=pl.INT32)
                    for kb in pl.pipeline(LOCAL_INTERMEDIATE // W2_K_TILE, stage=2):
                        k0 = kb * W2_K_TILE
                        lhs = pl.slice(
                            x, [ROW_TILE, W2_K_TILE], [start + row, k0], valid_shape=[valid_rows, W2_K_TILE]
                        )
                        rhs = pl.reshape(
                            pl.slice(weight, [1, W2_K_TILE, W2_N_TILE], [expert, k0, n0]),
                            [W2_K_TILE, W2_N_TILE],
                        )
                        acc = pl.matmul_acc(acc, lhs, rhs, init_cond=(kb == 0))
                    if diagnostic:
                        out = pl.assemble(
                            out, pl.set_validshape(acc, valid_rows, W2_N_TILE), [start + row, n0]
                        )
                    value = pl.col_expand_mul(
                        pl.cast(acc, pl.FP32), weight_scale[expert : expert + 1, n0 : n0 + W2_N_TILE]
                    )
                    value = pl.row_expand_mul(value, activation_scales)
                    result = pl.cast(value, pl.BF16, mode="rint")
                    y = pl.assemble(y, pl.set_validshape(result, valid_rows, W2_N_TILE), [start + row, n0])
        return out, y

    @pl.jit.inline
    def grouped_expert_impl(
        x: pl.Tensor[[input_rows, HIDDEN], pl.INT8],
        input_scale: pl.Tensor[[input_rows, 1], pl.FP32],
        expert_ids: pl.Tensor[[1, routes], pl.INT32],
        route_weights: pl.Tensor,
        w13: pl.Tensor[[EXPERTS, HIDDEN, W13_WIDTH], pl.INT8, pl.NZ],
        weight_scale: pl.Tensor[[EXPERTS, W13_WIDTH], pl.FP32],
        w2: pl.Tensor[[EXPERTS, LOCAL_INTERMEDIATE, HIDDEN], pl.INT8, pl.NZ],
        w2_scale: pl.Tensor[[EXPERTS, HIDDEN], pl.FP32],
        acc: pl.Tensor,
        quant: pl.Tensor,
        scale: pl.Tensor,
        acc2: pl.Tensor,
        y: pl.Tensor,
    ) -> tuple[pl.Tensor, pl.Tensor, pl.Tensor, pl.Tensor, pl.Tensor]:
        metadata = pl.create_tensor([routes, META_COLUMNS], dtype=pl.INT32)
        permutation = pl.create_tensor([1, routes], dtype=pl.INT32)
        inverse = pl.create_tensor([1, routes], dtype=pl.INT32)
        sorted_ids = pl.create_tensor([1, routes], dtype=pl.INT32)
        metadata, permutation, inverse, sorted_ids = encode_routes(
            expert_ids, metadata, permutation, inverse, sorted_ids
        )
        sorted_x = pl.create_tensor([routes, HIDDEN], dtype=pl.INT8)
        sorted_input_scale = pl.create_tensor([routes, 1], dtype=pl.FP32)
        sorted_x, sorted_input_scale = gather_routes(
            x, input_scale, permutation, sorted_x, sorted_input_scale
        )
        sorted_acc = pl.create_tensor([routes, W13_WIDTH], dtype=pl.INT32)
        sorted_acc = grouped_w13(sorted_x, metadata, w13, sorted_acc)
        sorted_quant = pl.create_tensor([routes, LOCAL_INTERMEDIATE], dtype=pl.INT8)
        sorted_scale = pl.create_tensor([routes, 1], dtype=pl.FP32)
        sorted_quant, sorted_scale = expert_swiglu_quant_batched_impl(
            sorted_acc, sorted_input_scale, sorted_ids, weight_scale, sorted_quant, sorted_scale
        )
        sorted_acc2 = pl.create_tensor([routes, HIDDEN], dtype=pl.INT32)
        sorted_y = pl.create_tensor([routes, HIDDEN], dtype=pl.BF16)
        sorted_acc2, sorted_y = grouped_w2(
            sorted_quant, sorted_scale, metadata, w2, w2_scale, sorted_acc2, sorted_y
        )
        if combine:
            for block in pl.spmd(input_rows * 4, name_hint="expert_grouped_combine"):
                token = block // 4
                col = (block % 4) * 512
                total = pl.full([1, 512], dtype=pl.FP32, value=0.0)
                for index in pl.range(TOP_K):
                    original_route = token * TOP_K + index
                    source = pl.cast(pl.read(inverse, [0, original_route]), pl.INDEX)
                    value = pl.cast(sorted_y[source : source + 1, col : col + 512], pl.FP32)
                    route_weight = pl.read(route_weights, [token, index])
                    total = pl.add(total, pl.mul(value, route_weight))
                y[token : token + 1, col : col + 512] = pl.cast(total, pl.BF16, mode="rint")
        else:
            scale_flat = pl.reshape(scale, [1, routes])
            for group in pl.spmd(routes // COPY_ROWS, name_hint="expert_route_restore"):
                base = group * COPY_ROWS
                scale_tile = pl.full([1, COPY_ROWS], dtype=pl.FP32, value=0.0)
                for lane in pl.range(COPY_ROWS):
                    target = base + lane
                    source = pl.cast(pl.read(inverse, [0, target]), pl.INDEX)
                    y[target : target + 1, :] = sorted_y[source : source + 1, :]
                    if diagnostic:
                        acc[target : target + 1, :] = sorted_acc[source : source + 1, :]
                        quant[target : target + 1, :] = sorted_quant[source : source + 1, :]
                        acc2[target : target + 1, :] = sorted_acc2[source : source + 1, :]
                        pl.write(scale_tile, [0, lane], pl.read(sorted_scale, [source, 0]))
                if diagnostic:
                    scale_flat[:, base : base + COPY_ROWS] = scale_tile
        return acc, quant, scale, acc2, y

    decorator = pl.jit.inline if inline else pl.jit
    if diagnostic:

        @decorator
        def grouped_expert(
            x: pl.Tensor[[routes, HIDDEN], pl.INT8],
            input_scale: pl.Tensor[[routes, 1], pl.FP32],
            expert_ids: pl.Tensor[[1, routes], pl.INT32],
            w13: pl.Tensor[[EXPERTS, HIDDEN, W13_WIDTH], pl.INT8, pl.NZ],
            weight_scale: pl.Tensor[[EXPERTS, W13_WIDTH], pl.FP32],
            w2: pl.Tensor[[EXPERTS, LOCAL_INTERMEDIATE, HIDDEN], pl.INT8, pl.NZ],
            w2_scale: pl.Tensor[[EXPERTS, HIDDEN], pl.FP32],
            acc: pl.Out[pl.Tensor[[routes, W13_WIDTH], pl.INT32]],
            quant: pl.Out[pl.Tensor[[routes, LOCAL_INTERMEDIATE], pl.INT8]],
            scale: pl.Out[pl.Tensor[[routes, 1], pl.FP32]],
            acc2: pl.Out[pl.Tensor[[routes, HIDDEN], pl.INT32]],
            y: pl.Out[pl.Tensor[[routes, HIDDEN], pl.BF16]],
        ) -> tuple[pl.Tensor, pl.Tensor, pl.Tensor, pl.Tensor, pl.Tensor]:
            unused_weights = pl.create_tensor([input_rows, TOP_K], dtype=pl.FP32)
            return grouped_expert_impl(
                x,
                input_scale,
                expert_ids,
                unused_weights,
                w13,
                weight_scale,
                w2,
                w2_scale,
                acc,
                quant,
                scale,
                acc2,
                y,
            )

        return grouped_expert

    if combine:

        @decorator
        def grouped_expert_combine(
            x: pl.Tensor[[input_rows, HIDDEN], pl.INT8],
            input_scale: pl.Tensor[[input_rows, 1], pl.FP32],
            ids: pl.Tensor[[input_rows, TOP_K], pl.INT32],
            weights: pl.Tensor[[input_rows, TOP_K], pl.FP32],
            w13: pl.Tensor[[EXPERTS, HIDDEN, W13_WIDTH], pl.INT8, pl.NZ],
            weight_scale: pl.Tensor[[EXPERTS, W13_WIDTH], pl.FP32],
            w2: pl.Tensor[[EXPERTS, LOCAL_INTERMEDIATE, HIDDEN], pl.INT8, pl.NZ],
            w2_scale: pl.Tensor[[EXPERTS, HIDDEN], pl.FP32],
            y: pl.Out[pl.Tensor[[input_rows, HIDDEN], pl.BF16]],
        ) -> pl.Tensor:
            flat_ids = pl.reshape(ids, [1, routes])
            acc = pl.create_tensor([routes, W13_WIDTH], dtype=pl.INT32)
            quant = pl.create_tensor([routes, LOCAL_INTERMEDIATE], dtype=pl.INT8)
            scale = pl.create_tensor([routes, 1], dtype=pl.FP32)
            acc2 = pl.create_tensor([routes, HIDDEN], dtype=pl.INT32)
            unused_acc, unused_quant, unused_scale, unused_acc2, result = grouped_expert_impl(
                x, input_scale, flat_ids, weights, w13, weight_scale, w2, w2_scale, acc, quant, scale, acc2, y
            )
            return result

        return grouped_expert_combine

    @decorator
    def grouped_expert_output(
        x: pl.Tensor[[routes, HIDDEN], pl.INT8],
        input_scale: pl.Tensor[[routes, 1], pl.FP32],
        expert_ids: pl.Tensor[[1, routes], pl.INT32],
        w13: pl.Tensor[[EXPERTS, HIDDEN, W13_WIDTH], pl.INT8, pl.NZ],
        weight_scale: pl.Tensor[[EXPERTS, W13_WIDTH], pl.FP32],
        w2: pl.Tensor[[EXPERTS, LOCAL_INTERMEDIATE, HIDDEN], pl.INT8, pl.NZ],
        w2_scale: pl.Tensor[[EXPERTS, HIDDEN], pl.FP32],
        y: pl.Out[pl.Tensor[[routes, HIDDEN], pl.BF16]],
    ) -> pl.Tensor:
        acc = pl.create_tensor([routes, W13_WIDTH], dtype=pl.INT32)
        quant = pl.create_tensor([routes, LOCAL_INTERMEDIATE], dtype=pl.INT8)
        scale = pl.create_tensor([routes, 1], dtype=pl.FP32)
        acc2 = pl.create_tensor([routes, HIDDEN], dtype=pl.INT32)
        unused_weights = pl.create_tensor([input_rows, TOP_K], dtype=pl.FP32)
        unused_acc, unused_quant, unused_scale, unused_acc2, result = grouped_expert_impl(
            x,
            input_scale,
            expert_ids,
            unused_weights,
            w13,
            weight_scale,
            w2,
            w2_scale,
            acc,
            quant,
            scale,
            acc2,
            y,
        )
        return result

    return grouped_expert_output
