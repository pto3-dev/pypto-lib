# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""Public DSL B20 expert routing with native FP16 normalization.

Preserves FP16 logits and rounded probabilities, native winner ordering, serial
FP32 selected-sum accumulation and FP16 normalization. Tensor ABI is
(logits, ids, weights, status); scalar status publication preserves padding.
"""

import pypto.language as pl


@pl.jit.incore
def topk_b20(
    logits: pl.Tensor[[20, 128], pl.FP16],
    ids: pl.Out[pl.Tensor[[20, 8], pl.INT32]],
    weights: pl.Out[pl.Tensor[[20, 8], pl.FP32]],
    status: pl.InOut[pl.Tensor[[20, 16], pl.INT32]],
):
    """Select eight rounded probabilities with native fused-routing semantics.

    Uses the fixed B20 four-tensor ABI and eight logical blocks. Status column0
    is consumed after kernel completion; all metadata padding remains intact.
    """
    block = pl.tile.get_block_idx()
    for token in pl.range(0, 20):
        if token % 8 == block:
            input_half = pl.tile.load(logits, [token, 0], [1, 128])
            values = pl.cast(input_half, pl.FP32, mode="none")

            # One logical row replaces eight identical native expansion rows.
            # The FP32 scalar result needs existing packed-column padding.
            maximum_scratch = pl.tile.create([1, 128], dtype=pl.FP32)
            maximum_tile = pl.tile.row_max(values, maximum_scratch)
            maximum = pl.read(maximum_tile, [0, 0])
            centered = pl.add(values, -maximum)
            # FusedNativeRouting=true does not round centered logits to FP16.
            exponential = pl.exp(centered)
            total_scratch = pl.tile.create([1, 128], dtype=pl.FP32)
            total_tile = pl.tile.row_sum(exponential, total_scratch)
            total = pl.read(total_tile, [0, 0])
            inverse_total: pl.Scalar[pl.FP32] = 1.0 / total
            probability_float = pl.mul(exponential, inverse_total)
            # A1's qualified FP32->FP16 none mode uses the native nearest vconv.
            probability_half = pl.cast(probability_float, pl.FP16, mode="none")
            probability = pl.cast(probability_half, pl.FP32, mode="none")

            # Preserve original rounded probabilities while masking a private
            # copy. Allocation/liveness must keep these two buffers distinct.
            masked = pl.mul(probability, 1.0)
            index_row = pl.tile.create([1, 8], dtype=pl.INT32)
            selected = pl.tile.full([1, 16], dtype=pl.FP32, value=0.0)
            selected_sum: pl.Scalar[pl.FP32] = 0.0
            for slot in pl.range(0, 8):
                argmax_scratch = pl.tile.create([1, 128], dtype=pl.FP32)
                # Capability gate: A1 must reproduce two independent 64-lane
                # vcmax calls and strict right>left selection inside this op.
                index_tile = pl.tile.row_argmax(masked, argmax_scratch)
                best = pl.read(index_tile, [0, 0])
                best_index: pl.Scalar[pl.INDEX] = pl.cast(best, pl.INDEX)
                selected_probability = pl.read(probability, [0, best_index])
                pl.write(index_row, [0, slot], best)
                pl.write(selected, [0, slot], selected_probability)
                # Keep eight serial FP32 additions in native winner order.
                selected_sum = selected_sum + selected_probability
                pl.write(masked, [0, best_index], -1.0)

            divisor = pl.tile.create([1, 16], dtype=pl.FP32)
            for slot in pl.range(0, 16):
                pl.write(divisor, [0, slot], selected_sum)
            # Native normalization divides in FP16 after rounding BOTH operands.
            # Do not use FP32 division or reciprocal multiplication here.
            selected_half = pl.cast(selected, pl.FP16, mode="none")
            divisor_half = pl.cast(divisor, pl.FP16, mode="none")
            normalized_half = pl.div(selected_half, divisor_half)
            normalized = pl.cast(normalized_half, pl.FP32, mode="none")
            weight_row = pl.tile.extract(normalized, 0, 0, shape=[1, 8], target_memory=pl.Mem.Vec)
            pl.tile.store(weight_row, [token, 0], weights)
            pl.tile.store(index_row, [token, 0], ids)

            # Update the native status cell directly; padding is never read or
            # written. Consumption remains after complete kernel execution.
            success: pl.Scalar[pl.INT32] = 0
            pl.tensor.write(status, [token, 0], success)

    return ids, weights, status
