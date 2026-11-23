# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""Paired W13/SwiGLU with the original FP16 rounding and route guards.

Weights and channel scales must pair gate/up in blocks of 384. The existing
1536-column carrier retains its ABI; only its first 768 columns are written.
"""

from typing import Any

import pypto.language as pl


def make_w13_swiglu_b20(*, column_partition: bool) -> Any:
    """Gather once per owned group; reuse Mat16x2048 across N768/K64 steps."""
    owner_group_stride = 2 if column_partition else 1
    owner_column_stride = 1 if column_partition else 0

    @pl.jit.incore
    def w13_b20(
        input: pl.Tensor[[20, 2048], pl.INT8],
        weight: pl.Tensor[[12582912, 32], pl.INT8],
        ids: pl.Tensor[[20, 8], pl.INT32],
        expert_counts: pl.Tensor[[1, 128], pl.INT32],
        route_indices: pl.Tensor[[128, 160], pl.INT32],
        output: pl.InOut[pl.Tensor[[160, 1536], pl.FP16]],
        channel_scale: pl.Tensor[[128, 3072], pl.INT32],
        input_scale: pl.Tensor[[20, 16], pl.FP32],
        status: pl.InOut[pl.Tensor[[160, 16], pl.INT32]],
    ):
        block = pl.tile.get_block_idx()
        bank = pl.tensor.view(weight, shape=[128, 1536, 2048], layout=pl.NZ)
        routes = pl.tile.load(ids, [0, 0], [20, 8], target_memory=pl.Mem.Vec)
        for route in pl.range(160):
            if route % 8 == block:
                expert_id = pl.tile.read(routes, [route // 8, route % 8])
                if expert_id < 0:
                    invalid: pl.Scalar[pl.INT32] = 2
                    pl.tensor.write(status, [route, 0], invalid)
                elif expert_id >= 128:
                    invalid: pl.Scalar[pl.INT32] = 2
                    pl.tensor.write(status, [route, 0], invalid)

        counts = pl.tile.load(expert_counts, [0, 0], [1, 128])
        input_scales = pl.tile.load(input_scale, [0, 0], [20, 8], target_memory=pl.Mem.Vec)

        group_index: pl.Scalar[pl.INDEX] = 0
        for expert in pl.range(128):
            count = pl.cast(pl.tile.read(counts, [0, expert]), pl.INDEX)
            if count > 0:
                if count <= 160:
                    for first in pl.range(0, 160, 16):
                        if first < count:
                            group = group_index
                            group_index = group_index + 1
                            participates: pl.Scalar[pl.BOOL] = False
                            for n_tile in pl.range(2):
                                owner = ((group * owner_group_stride + n_tile * owner_column_stride) + ((group * owner_group_stride + n_tile * owner_column_stride) // 8) * 4) % 8
                                if block == owner:
                                    participates = True
                            if participates:
                                # One resident activation per participating block/group:
                                # [K/32,M16,32], 32 KiB, reused for every owned N tile.
                                activation = pl.tile.create([16, 2048], pl.INT8, target_memory=pl.Mem.Mat)
                                listed = pl.tile.load(route_indices, [expert, first], [1, 16])
                                for lane in pl.range(16):
                                    if first + lane < count:
                                        route_value = pl.tile.read(listed, [0, lane])
                                        route = pl.cast(route_value, pl.INDEX)
                                        if route >= 0:
                                            if route < 160:
                                                activation = pl.tile.gather_row(
                                                    activation, input, [lane, 0], [route // 8, 0], [1, 2048],
                                                )

                                # Unselected physical rows are never scattered. Their
                                # contents do not mix with selected rows in integer MAD.
                                # Do not add an extra GM read or write for those lanes.
                                for n_tile in pl.range(2):
                                    owner = ((group * owner_group_stride + n_tile * owner_column_stride) + ((group * owner_group_stride + n_tile * owner_column_stride) // 8) * 4) % 8
                                    if block == owner:
                                        column = n_tile * 768
                                        acc = pl.tile.create([16, 768], pl.INT32, target_memory=pl.Mem.Acc)
                                        # Load two K64 weight slices into L1 per transfer.
                                        # Keep the original ascending sequence of K64 MADs.
                                        for k in pl.pipeline(0, 2048, 128, stage=2):
                                            b_nk = pl.tile.load(
                                                bank, [expert, column, k], [1, 768, 128],
                                                target_memory=pl.Mem.Mat,
                                            )
                                            b_nk_2d = pl.tile.reshape(b_nk, [768, 128])
                                            b_kn = pl.tile.transpose_view(b_nk_2d)
                                            for inner in pl.range(2):
                                                inner_k = inner * 64
                                                global_k = k + inner_k
                                                left = pl.tile.extract(
                                                    activation, 0, global_k, [16, 64],
                                                    target_memory=pl.Mem.Left,
                                                )
                                                right = pl.tile.extract(
                                                    b_kn, inner_k, 0, [64, 768],
                                                    target_memory=pl.Mem.Right,
                                                )
                                                acc = pl.tile.matmul_acc(
                                                    acc, left, right, init_cond=(global_k == 0)
                                                )

                                        # Keep [low FP32 bits, high zero] words unchanged.
                                        # This alias is on-chip only; external descriptors
                                        # keep their original INT32 dtype and shape.
                                        scale_words = pl.tile.load(
                                            channel_scale, [expert, column * 2], [1, 1536],
                                            target_memory=pl.Mem.Vec,
                                        )
                                        packed_scale = pl.tile.reinterpret_view(
                                            scale_words, pl.INT64, shape=[1, 768],
                                        )
                                        # Native channel dequant: Acc16x768 -> FP16
                                        # NZ Vec16x512, 512-byte fractals, native VDEQ16.
                                        # Do not replace with cast/mul/cast or fold in the
                                        # per-token input scale from the downstream SwiGLU.
                                        packed = pl.tile.dequantize_channel(acc, packed_scale)
                                        for lane in pl.range(16):
                                            if first + lane < count:
                                                selected_route = pl.tile.read(listed, [0, lane])
                                                if selected_route >= 0:
                                                    if selected_route < 160:
                                                        output_row = pl.cast(selected_route, pl.INDEX)
                                                        output_row_tile = pl.tile.extract(
                                                            packed, lane, 0, [1, 768], target_memory=pl.Mem.Vec,
                                                        )
                                                        gate_half = pl.tile.extract(
                                                            output_row_tile, 0, 0, [1, 384], target_memory=pl.Mem.Vec,
                                                        )
                                                        up_half = pl.tile.extract(
                                                            output_row_tile, 0, 384, [1, 384], target_memory=pl.Mem.Vec,
                                                        )
                                                        factor = pl.tile.read(input_scales, [output_row // 8, 0])
                                                        gate = pl.tile.cast(gate_half, pl.FP32, mode="none")
                                                        up = pl.tile.cast(up_half, pl.FP32, mode="none")
                                                        rounded_gate = pl.tile.cast(pl.tile.muls(gate, factor), pl.FP16, mode="none")
                                                        rounded_up = pl.tile.cast(pl.tile.muls(up, factor), pl.FP16, mode="none")
                                                        gate_values = pl.tile.cast(rounded_gate, pl.FP32, mode="none")
                                                        up_values = pl.tile.cast(rounded_up, pl.FP32, mode="none")
                                                        negative = pl.tile.muls(gate_values, -1.0)
                                                        exponential = pl.tile.exp(negative)
                                                        denominator = pl.tile.adds(exponential, 1.0)
                                                        swiglu = pl.tile.mul(pl.tile.div(gate_values, denominator), up_values)
                                                        activation_half = pl.tile.cast(swiglu, pl.FP16, mode="none")
                                                        pl.tile.store(activation_half, [output_row, n_tile * 384], output)
                                                        if n_tile == 0:
                                                            success: pl.Scalar[pl.INT32] = 0
                                                            pl.tensor.write(status, [output_row, 0], success)
        return output, status

    return w13_b20


@pl.jit.incore
def quantize_swiglu_b20(
    input: pl.Tensor[[160, 1536], pl.FP16],
    input_scale: pl.Tensor[[20, 16], pl.FP32],
    ids: pl.Tensor[[20, 8], pl.INT32],
    weight_scale: pl.Tensor[[128, 3072], pl.INT32],
    quant: pl.InOut[pl.Tensor[[160, 768], pl.INT8]],
    scale: pl.InOut[pl.Tensor[[160, 16], pl.FP32]],
    status: pl.InOut[pl.Tensor[[160, 16], pl.INT32]],
):
    """Preserve four-route batches and stop each lane at its first invalid ID.

    Weight scale remains an unused ABI input on this channel-dequantized path.
    Quant is InOut because routes at and after the first invalid ID retain their
    previous bytes. Scale/status also retain all untouched carrier columns.
    """
    input_view = pl.tensor.view(input, [20, 8, 1536])
    quant_view = pl.tensor.view(quant, [20, 8, 768])
    scale_view = pl.tensor.view(scale, [20, 8, 16])
    status_view = pl.tensor.view(status, [20, 8, 16])
    block = pl.tile.get_block_idx()

    # A constant lane loop supplies explicit bounds for every GM offset.
    for lane in pl.range(0, 8):
        if lane == block:
            ids_tile = pl.tile.load(ids, [0, 0], [20, 8])
            first_invalid: pl.Scalar[pl.INDEX] = 20
            for token_id in pl.range(0, 20):
                expert = pl.read(ids_tile, [token_id, lane])
                if expert < 0 or expert >= 128:
                    if token_id < first_invalid:
                        first_invalid = token_id

            for token in pl.range(0, 20, 4):
                if token <= first_invalid:
                    # Only the first 32 bytes of each 64-byte carrier are
                    # transferred, matching the grouped native path.
                    scale_carrier_nd = pl.tile.load(scale_view, [token, lane, 0], [4, 1, 8])
                    status_carrier_nd = pl.tile.load(status_view, [token, lane, 0], [4, 1, 8])
                    scale_carrier = pl.reshape(scale_carrier_nd, [4, 8])
                    status_carrier = pl.reshape(status_carrier_nd, [4, 8])

                    # When the first group row is already invalid, no input
                    # arithmetic or quant writes are needed; only status 2.
                    if token < first_invalid:
                        half_tile = pl.tile.load(input_view, [token, lane, 0], [4, 1, 768])
                        activated = pl.cast(pl.reshape(half_tile, [4, 768]), pl.FP32, mode="none")

                        magnitude = pl.abs(activated)
                        maximum_scratch = pl.tile.create([4, 768], dtype=pl.FP32)
                        half_maxima = pl.row_max(magnitude, maximum_scratch)
                        inverse_factors = pl.tile.create([4, 1], dtype=pl.FP32)
                        for row in pl.range(0, 4):
                            inverse: pl.Scalar[pl.FP32] = 0.0
                            if token + row < first_invalid:
                                maximum = pl.read(half_maxima, [row, 0])
                                # SwiGLU divides by 127.0; do not substitute the
                                # input quantizer's rounded reciprocal multiply.
                                dequant = maximum / 127.0
                                if dequant > 1e-7:
                                    inverse = 1.0 / dequant
                                pl.write(scale_carrier, [row, 0], dequant)
                                success: pl.Scalar[pl.INT32] = 0
                                pl.write(status_carrier, [row, 0], success)
                            pl.write(inverse_factors, [row, 0], inverse)

                        scaled = pl.row_expand_mul(activated, inverse_factors)
                        rounded = pl.cast(scaled, pl.INT32, mode="rint")
                        integral_float = pl.cast(rounded, pl.FP32, mode="none")
                        integral_half = pl.cast(integral_float, pl.FP16, mode="none")
                        integral_routes = pl.reshape(integral_half, [4, 768])

                        if token + 4 <= first_invalid:
                            grouped_quant = pl.cast(integral_routes, pl.INT8, mode="trunc")
                            pl.tile.store(
                                grouped_quant,
                                [token, lane, 0],
                                quant_view,
                                shapes=[4, 1, 768],
                            )
                        else:
                            # A1 Vec extraction supports FP16, not INT8. Extract
                            # the valid integral FP16 rows before narrowing.
                            for row in pl.range(0, 4):
                                if token + row < first_invalid:
                                    integral_row = pl.tile.extract(
                                        integral_routes,
                                        row,
                                        0,
                                        shape=[1, 768],
                                        target_memory=pl.Mem.Vec,
                                    )
                                    row_quant = pl.cast(integral_row, pl.INT8, mode="trunc")
                                    pl.tile.store(row_quant, [(token + row) * 8 + lane, 0], quant)

                    # Publish the first bad status after all valid-prefix quant
                    # writes. Its scale/quant and all later route data persist.
                    for row in pl.range(0, 4):
                        if token + row == first_invalid:
                            invalid: pl.Scalar[pl.INT32] = 2
                            pl.write(status_carrier, [row, 0], invalid)
                    pl.tile.store(scale_carrier, [token, lane, 0], scale_view, shapes=[4, 1, 8])
                    pl.tile.store(status_carrier, [token, lane, 0], status_view, shapes=[4, 1, 8])

    return quant, scale, status
