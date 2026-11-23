# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""Whole M16 expert groups with resident FP16 activation and INT8 W2 input."""

import pypto.language as pl


@pl.jit.incore
def fused_expert_b20(
    input: pl.Tensor[[20, 2048], pl.INT8],
    w13_weight: pl.Tensor[[12582912, 32], pl.INT8],
    ids: pl.Tensor[[20, 8], pl.INT32],
    expert_counts: pl.Tensor[[1, 128], pl.INT32],
    route_indices: pl.Tensor[[128, 160], pl.INT32],
    channel_scale: pl.Tensor[[128, 3072], pl.INT32],
    input_scale: pl.Tensor[[20, 16], pl.FP32],
    w13_status: pl.InOut[pl.Tensor[[160, 16], pl.INT32]],
    quant: pl.InOut[pl.Tensor[[160, 768], pl.INT8]],
    scale: pl.InOut[pl.Tensor[[160, 16], pl.FP32]],
    quant_status: pl.InOut[pl.Tensor[[160, 16], pl.INT32]],
    w2_weight: pl.Tensor[[6291456, 32], pl.INT8],
    output: pl.InOut[pl.Tensor[[160, 2048], pl.INT32]],
    w2_status: pl.InOut[pl.Tensor[[160, 16], pl.INT32]],
):
    block = pl.tile.get_block_idx()
    bank13 = pl.tensor.view(w13_weight, shape=[128, 1536, 2048], layout=pl.NZ)
    bank2 = pl.tensor.view(w2_weight, shape=[128, 2048, 768], layout=pl.NZ)
    routes = pl.tile.load(ids, [0, 0], [20, 8], target_memory=pl.Mem.Vec)
    counts = pl.tile.load(expert_counts, [0, 0], [1, 128])
    input_scales = pl.tile.load(input_scale, [0, 0], [20, 8], target_memory=pl.Mem.Vec)
    first_bad = pl.tile.create([1, 8], pl.INT32, target_memory=pl.Mem.Vec)
    for slot in pl.range(8):
        first_invalid: pl.Scalar[pl.INT32] = 20
        for token in pl.range(20):
            expert_id = pl.tile.read(routes, [token, slot])
            if expert_id < 0 or expert_id >= 128:
                if token < first_invalid:
                    first_invalid = pl.cast(token, pl.INT32)
                if slot == block:
                    bad_route = token * 8 + slot
                    invalid: pl.Scalar[pl.INT32] = 2
                    pl.tensor.write(w13_status, [bad_route, 0], invalid)
                    pl.tensor.write(w2_status, [bad_route, 0], invalid)
        pl.tile.write(first_bad, [0, slot], first_invalid)
        if slot == block:
            if first_invalid < 20:
                bad_route = pl.cast(first_invalid, pl.INDEX) * 8 + slot
                invalid: pl.Scalar[pl.INT32] = 2
                pl.tensor.write(quant_status, [bad_route, 0], invalid)

    group_index: pl.Scalar[pl.INDEX] = 0
    for expert in pl.range(128):
        count = pl.cast(pl.tile.read(counts, [0, expert]), pl.INDEX)
        if count > 0:
            if count <= 160:
                for first in pl.range(0, 160, 16):
                    if first < count:
                        group = group_index
                        group_index = group_index + 1
                        if group % 8 == block:
                            activation = pl.tile.create([16, 2048], pl.INT8, target_memory=pl.Mem.Mat)
                            selected = pl.tile.create([1, 16], pl.INT32, target_memory=pl.Mem.Vec)
                            listed = pl.tile.load(route_indices, [expert, first], [1, 16])
                            for lane in pl.range(16):
                                absent: pl.Scalar[pl.INT32] = -1
                                pl.tile.write(selected, [0, lane], absent)
                                if first + lane < count:
                                    value = pl.tile.read(listed, [0, lane])
                                    route = pl.cast(value, pl.INDEX)
                                    if route >= 0:
                                        if route < 160:
                                            pl.tile.write(selected, [0, lane], value)
                                            activation = pl.tile.gather_row(
                                                activation, input, [lane, 0], [route // 8, 0], [1, 2048]
                                            )

                            activated_group = pl.tile.create([16, 768], pl.FP16, target_memory=pl.Mem.Vec)
                            for n_tile in pl.range(2):
                                column = n_tile * 768
                                acc13 = pl.tile.create([16, 768], pl.INT32, target_memory=pl.Mem.Acc)
                                for k in pl.pipeline(0, 2048, 128, stage=2):
                                    weight_nk = pl.tile.load(
                                        bank13,
                                        [expert, column, k],
                                        [1, 768, 128],
                                        target_memory=pl.Mem.Mat,
                                    )
                                    weight_rows = pl.tile.reshape(weight_nk, [768, 128])
                                    weight_kn = pl.tile.transpose_view(weight_rows)
                                    for inner in pl.range(2):
                                        inner_k = inner * 64
                                        global_k = k + inner_k
                                        left = pl.tile.extract(
                                            activation, 0, global_k, [16, 64], target_memory=pl.Mem.Left
                                        )
                                        right = pl.tile.extract(
                                            weight_kn, inner_k, 0, [64, 768], target_memory=pl.Mem.Right
                                        )
                                        acc13 = pl.tile.matmul_acc(
                                            acc13, left, right, init_cond=(global_k == 0)
                                        )
                                words = pl.tile.load(
                                    channel_scale,
                                    [expert, column * 2],
                                    [1, 1536],
                                    target_memory=pl.Mem.Vec,
                                )
                                packed_scale = pl.tile.reinterpret_view(words, pl.INT64, shape=[1, 768])
                                packed = pl.tile.dequantize_channel(acc13, packed_scale)
                                for lane in pl.range(16):
                                    value = pl.tile.read(selected, [0, lane])
                                    if value >= 0:
                                        if value < 160:
                                            route = pl.cast(value, pl.INDEX)
                                            projected_row13 = pl.tile.extract(
                                                packed, lane, 0, [1, 768], target_memory=pl.Mem.Vec
                                            )
                                            gate_half = pl.tile.extract(
                                                projected_row13, 0, 0, [1, 384], target_memory=pl.Mem.Vec
                                            )
                                            up_half = pl.tile.extract(
                                                projected_row13,
                                                0,
                                                384,
                                                [1, 384],
                                                target_memory=pl.Mem.Vec,
                                            )
                                            factor = pl.tile.read(input_scales, [route // 8, 0])
                                            gate = pl.tile.cast(gate_half, pl.FP32, mode="none")
                                            up = pl.tile.cast(up_half, pl.FP32, mode="none")
                                            gate_scaled = pl.tile.muls(gate, factor)
                                            up_scaled = pl.tile.muls(up, factor)
                                            gate_round = pl.tile.cast(gate_scaled, pl.FP16, mode="none")
                                            up_round = pl.tile.cast(up_scaled, pl.FP16, mode="none")
                                            gate_fp32 = pl.tile.cast(gate_round, pl.FP32, mode="none")
                                            up_fp32 = pl.tile.cast(up_round, pl.FP32, mode="none")
                                            negative = pl.tile.muls(gate_fp32, -1.0)
                                            exponential = pl.tile.exp(negative)
                                            denominator = pl.tile.adds(exponential, 1.0)
                                            silu = pl.tile.div(gate_fp32, denominator)
                                            swiglu = pl.tile.mul(silu, up_fp32)
                                            activation_half = pl.tile.cast(swiglu, pl.FP16, mode="none")
                                            activated_group = pl.tile.assemble(
                                                activated_group, activation_half, [lane, n_tile * 384]
                                            )
                                            if n_tile == 0:
                                                success: pl.Scalar[pl.INT32] = 0
                                                pl.tensor.write(w13_status, [route, 0], success)

                            for chunk in pl.range(0, 16, 4):
                                half = pl.tile.extract(
                                    activated_group, chunk, 0, [4, 768], target_memory=pl.Mem.Vec
                                )
                                activated = pl.tile.cast(half, pl.FP32, mode="none")
                                magnitude = pl.tile.abs(activated)
                                scratch = pl.tile.create([4, 768], pl.FP32)
                                maxima = pl.tile.row_max(magnitude, scratch)
                                inverse_factors = pl.tile.create([4, 1], pl.FP32)
                                for row in pl.range(4):
                                    inverse: pl.Scalar[pl.FP32] = 0.0
                                    value = pl.tile.read(selected, [0, chunk + row])
                                    if value >= 0:
                                        if value < 160:
                                            route = pl.cast(value, pl.INDEX)
                                            limit = pl.cast(
                                                pl.tile.read(first_bad, [0, route % 8]), pl.INDEX
                                            )
                                            if route // 8 < limit:
                                                maximum = pl.tile.read(maxima, [row, 0])
                                                dequant = maximum / 127.0
                                                if dequant > 1e-7:
                                                    inverse = 1.0 / dequant
                                                pl.tensor.write(scale, [route, 0], dequant)
                                                success: pl.Scalar[pl.INT32] = 0
                                                pl.tensor.write(quant_status, [route, 0], success)
                                    pl.tile.write(inverse_factors, [row, 0], inverse)
                                scaled = pl.tile.row_expand_mul(activated, inverse_factors)
                                rounded = pl.tile.cast(scaled, pl.INT32, mode="rint")
                                integral_float = pl.tile.cast(rounded, pl.FP32, mode="none")
                                integral_half = pl.tile.cast(integral_float, pl.FP16, mode="none")
                                activated_group = pl.tile.assemble(
                                    activated_group, integral_half, [chunk, 0]
                                )
                                for row in pl.range(4):
                                    value = pl.tile.read(selected, [0, chunk + row])
                                    if value >= 0:
                                        if value < 160:
                                            route = pl.cast(value, pl.INDEX)
                                            limit = pl.cast(
                                                pl.tile.read(first_bad, [0, route % 8]), pl.INDEX
                                            )
                                            if route // 8 < limit:
                                                integral_row = pl.tile.extract(
                                                    integral_half,
                                                    row,
                                                    0,
                                                    [1, 768],
                                                    target_memory=pl.Mem.Vec,
                                                )
                                                quant_row = pl.tile.cast(
                                                    integral_row, pl.INT8, mode="trunc"
                                                )
                                                pl.tile.store(quant_row, [route, 0], quant)
                            quant_vector = pl.tile.cast(activated_group, pl.INT8, mode="trunc")
                            quant_mat = pl.tile.move(
                                quant_vector,
                                target_memory=pl.Mem.Mat,
                                blayout=pl.TileLayout.col_major,
                                slayout=pl.TileLayout.row_major,
                            )
                            for lane in pl.range(16):
                                value = pl.tile.read(selected, [0, lane])
                                if value >= 0:
                                    if value < 160:
                                        route = pl.cast(value, pl.INDEX)
                                        limit = pl.cast(pl.tile.read(first_bad, [0, route % 8]), pl.INDEX)
                                        if route // 8 >= limit:
                                            quant_mat = pl.tile.gather_row(
                                                quant_mat, quant, [lane, 0], [route, 0], [1, 768]
                                            )

                            for n_tile in pl.range(4):
                                column = n_tile * 512
                                acc2 = pl.tile.create([16, 512], pl.INT32, target_memory=pl.Mem.Acc)
                                for k in pl.pipeline(0, 768, 64, stage=2):
                                    k_tile = k // 64
                                    left2 = pl.tile.extract(
                                        quant_mat, 0, k, [16, 64], target_memory=pl.Mem.Left
                                    )
                                    weight_nk2 = pl.tile.load(
                                        bank2, [expert, column, k], [1, 512, 64], target_memory=pl.Mem.Mat
                                    )
                                    weight_rows2 = pl.tile.reshape(weight_nk2, [512, 64])
                                    weight_kn2 = pl.tile.transpose_view(weight_rows2)
                                    right2 = pl.tile.extract(
                                        weight_kn2, 0, 0, [64, 512], target_memory=pl.Mem.Right
                                    )
                                    acc2 = pl.tile.matmul_acc(
                                        acc2, left2, right2, init_cond=(k_tile == 0)
                                    )
                                packed2 = pl.tile.move(
                                    acc2,
                                    target_memory=pl.Mem.Vec,
                                    blayout=pl.TileLayout.col_major,
                                    slayout=pl.TileLayout.row_major,
                                )
                                for lane in pl.range(16):
                                    value = pl.tile.read(selected, [0, lane])
                                    if value >= 0:
                                        if value < 160:
                                            route = pl.cast(value, pl.INDEX)
                                            output_row2 = pl.tile.extract(
                                                packed2, lane, 0, [1, 512], target_memory=pl.Mem.Vec
                                            )
                                            pl.tile.store(output_row2, [route, column], output)
                                            if n_tile == 0:
                                                success: pl.Scalar[pl.INT32] = 0
                                                pl.tensor.write(w2_status, [route, 0], success)
    return w13_status, quant, scale, quant_status, output, w2_status
