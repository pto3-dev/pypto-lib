# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""Local B20 SwiGLU DSL draft with grouped routes and native invalid-ID writes.

The fixed ABI describes the real packed buffers. Internal GM views retain the
route axis; padded scalar columns supply fused FP32 per-route scaling. Compiler
and numerical validation are pending; this module does not launch the kernel.
"""

import pypto.language as pl


@pl.jit.incore
def swiglu_quant_b20(
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
            input_scales = pl.tile.load(input_scale, [0, 0], [20, 8])
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
                        gate_half = pl.tile.load(input_view, [token, lane, 0], [4, 1, 768])
                        up_half = pl.tile.load(input_view, [token, lane, 768], [4, 1, 768])
                        gate_half_rows = pl.reshape(gate_half, [4, 768])
                        up_half_rows = pl.reshape(up_half, [4, 768])
                        gate = pl.cast(gate_half_rows, pl.FP32, mode="none")
                        up = pl.cast(up_half_rows, pl.FP32, mode="none")

                        # Logical four-row factors use A1's padded packed
                        # column allocation; only four values are consumed.
                        factors = pl.tile.create([4, 1], dtype=pl.FP32)
                        for row in pl.range(0, 4):
                            activation_scale = pl.read(input_scales, [token + row, 0])
                            pl.write(factors, [row, 0], activation_scale)
                        scaled_gate = pl.row_expand_mul(gate, factors)
                        scaled_up = pl.row_expand_mul(up, factors)

                        # Keep the native FP16 round points after input scaling.
                        # A1's qualified default FP32->FP16 conversion uses
                        # the same nearest vconv_f322f16 as native CAST_RINT.
                        rounded_gate = pl.cast(scaled_gate, pl.FP16, mode="none")
                        rounded_up = pl.cast(scaled_up, pl.FP16, mode="none")
                        gate_values = pl.cast(rounded_gate, pl.FP32, mode="none")
                        up_values = pl.cast(rounded_up, pl.FP32, mode="none")
                        negative_gate = pl.mul(gate_values, -1.0)
                        exponential = pl.exp(negative_gate)
                        denominator = pl.add(exponential, 1.0)
                        silu = pl.div(gate_values, denominator)
                        activation = pl.mul(silu, up_values)
                        activation_half = pl.cast(activation, pl.FP16, mode="none")
                        activated = pl.cast(activation_half, pl.FP32, mode="none")

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
