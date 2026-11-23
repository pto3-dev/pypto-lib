# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""Fused B20 ModelSlim Attention over native FP16 paged KV storage.

The factory specializes internal read-only PA views to the actual allocation
page count. External cache descriptors remain [physical_pages, 32, 128, 16];
the seven-argument entry preserves native FP16 boundaries and page recurrence.
"""

from typing import Any

import pypto.language as pl


def make_attention_b20(*, physical_pages: int, fuse_o_quant: bool = False) -> Any:
    """Specialize to the actual PA allocation's page count, without repacking it."""
    if isinstance(physical_pages, bool) or not isinstance(physical_pages, int):
        raise TypeError("physical_pages must be the actual integer PA allocation extent")
    if not 1 <= physical_pages <= ((1 << 32) - 1) // (32 * 128 * 16):
        raise ValueError("physical_pages must keep the full PA Tensor within uint32 elements")

    page_heads = physical_pages * 4

    @pl.jit.inline
    def attention_body(
        query: pl.Tensor[[80, 16, 128], pl.FP16],
        keys: pl.Tensor[[physical_pages, 32, 128, 16], pl.FP16],
        values: pl.Tensor[[physical_pages, 32, 128, 16], pl.FP16],
        lengths: pl.Tensor[[20, 8], pl.INT32],
        output: pl.InOut[pl.Tensor],
        block_table: pl.Tensor[[20, 32], pl.INT32],
        reciprocal: pl.Tensor,
        offset: pl.Tensor,
        status: pl.InOut[pl.Tensor[[80, 16], pl.INT32]],
    ):
        block = pl.tile.get_block_idx()
        # Existing PA borrowing uses native_pa_caches=(1, 2), not NZ weights.
        # Each logical batch is one KV-head page, physically [8, 128, 16].
        key_bank = pl.tensor.view(keys, shape=[page_heads, 128, 128], layout=pl.NZ)
        value_bank = pl.tensor.view(values, shape=[page_heads, 128, 128], layout=pl.NZ)
        query_rows = pl.tensor.view(query, shape=[1280, 128])
        length_rows = pl.tile.load(lengths, [0, 0], [20, 8], target_memory=pl.Mem.Vec)
        table_rows = pl.tile.load(block_table, [0, 0], [20, 32], target_memory=pl.Mem.Vec)

        # One exact 0..127 row per core, independent of GM data or extra ABI.
        positions = pl.tile.create([1, 128], pl.FP32, target_memory=pl.Mem.Vec)
        for column in pl.range(128):
            pl.tile.write(positions, [0, column], pl.cast(pl.cast(column, pl.INT32), pl.FP32))
        geometry = pl.tile.create([8, 128], pl.FP32, target_memory=pl.Mem.Vec)
        position_rows = pl.tile.col_expand(geometry, positions)

        if block >= 0:
            if block < 8:
                for task in pl.range(80):
                    if task % 8 == block:
                        request = task // 4
                        kv_head = task % 4
                        length_i32 = pl.tile.read(length_rows, [request, 0])
                        if length_i32 < 0:
                            bad_length: pl.Scalar[pl.INT32] = 1
                            pl.tensor.write(status, [task, 0], bad_length)
                        elif length_i32 > 4096:
                            bad_length: pl.Scalar[pl.INT32] = 1
                            pl.tensor.write(status, [task, 0], bad_length)
                        else:
                            length = pl.cast(length_i32, pl.INDEX)
                            page_count = (length + 127) // 128
                            valid_table: pl.Scalar[pl.BOOL] = True
                            for check_page in pl.range(page_count):
                                checked = pl.tile.read(table_rows, [request, check_page])
                                if checked < 0:
                                    valid_table = False
                                elif checked >= physical_pages:
                                    valid_table = False
                            if not valid_table:
                                bad_page: pl.Scalar[pl.INT32] = 2
                                pl.tensor.write(status, [task, 0], bad_page)
                            else:
                                if length == 0:
                                    result = pl.tile.cast(pl.tile.full([8, 128], pl.FP32, 0.0), pl.FP16, mode="none")
                                    if fuse_o_quant:
                                        # Preserve the FP16 divide output before the existing O quantization.
                                        o_column = kv_head * 1024
                                        o_inverse = pl.tile.cast(pl.tile.load(reciprocal, [0, o_column], [1, 1024]), pl.FP32, mode="none")
                                        o_zero = pl.tile.cast(pl.tile.load(offset, [0, o_column], [1, 1024]), pl.FP32, mode="none")
                                        o_values_fp32 = pl.tile.cast(pl.tile.reshape(result, [1, 1024]), pl.FP32, mode="none")
                                        o_scaled = pl.tile.add(pl.tile.mul(o_values_fp32, o_inverse), o_zero)
                                        o_ceiling = pl.tile.full([1, 1024], pl.FP32, 127.0)
                                        o_floor = pl.tile.full([1, 1024], pl.FP32, -128.0)
                                        o_bounded = pl.tile.maximum(pl.tile.minimum(o_scaled, o_ceiling), o_floor)
                                        o_rounded = pl.tile.cast(o_bounded, pl.INT32, mode="rint")
                                        o_integral = pl.tile.cast(pl.tile.cast(o_rounded, pl.FP32, mode="none"), pl.FP16, mode="none")
                                        o_narrowed = pl.tile.cast(o_integral, pl.INT8, mode="trunc")
                                        pl.tile.store(o_narrowed, [request, o_column], output)
                                    else:
                                        # Ordinary ND output alias; no physical layout
                                        # reinterpretation and no extra external tensor.
                                        output_rows = pl.tensor.view(output, shape=[640, 128])
                                        pl.tile.store(result, [task * 8, 0], output_rows)
                                    success: pl.Scalar[pl.INT32] = 0
                                    pl.tensor.write(status, [task, 0], success)
                                else:
                                    q_mat = pl.tile.load(
                                        query_rows, [task * 16, 0], [16, 128], target_memory=pl.Mem.Mat
                                    )
                                    q_left = pl.tile.extract(q_mat, 0, 0, [16, 128], target_memory=pl.Mem.Left)
                                    numerator = pl.tile.full([8, 128], pl.FP32, 0.0)
                                    total = pl.tile.reshape(pl.tile.full([1, 8], pl.FP32, 0.0), [8, 1])
                                    old_maximum = pl.tile.reshape(pl.tile.full([1, 8], pl.FP32, 0.0), [8, 1])
                                    first_physical = pl.tile.read(table_rows, [request, 0])
                                    first_bank = (pl.cast(first_physical, pl.INDEX) % physical_pages) * 4 + kv_head
                                    prefetched_k = pl.tile.load(key_bank, [first_bank, 0, 0], [1, 128, 128], target_memory=pl.Mem.Mat)
                                    first_v_rows = length - (0) * 128
                                    if first_v_rows > 128:
                                        first_v_rows = 128
                                    if first_v_rows >= 0:
                                        if first_v_rows <= 128:
                                            first_v_source = pl.tile.load(value_bank, [first_bank, 0, 0], [1, 128, 128], target_memory=pl.Mem.Mat)
                                            first_v_full = pl.tile.reshape(first_v_source, [128, 128])
                                            first_v_valid = pl.tile.set_validshape(first_v_full, first_v_rows, 128)
                                            prefetched_v = pl.tile.fillpad_inplace(first_v_valid, pad_value=0)
                                            for page_index in pl.range(page_count):
                                                page_base = page_index * 128
                                                k_rows = pl.tile.reshape(prefetched_k, [128, 128])
                                                k_transposed = pl.tile.transpose_view(k_rows)
                                                k_right = pl.tile.extract(k_transposed, 0, 0, [128, 128], target_memory=pl.Mem.Right)
                                                v_prefetched = prefetched_v
                                                qk_acc = pl.tile.matmul(q_left, k_right)
                                                next_page = page_index + 1
                                                if next_page < page_count:
                                                    next_k_physical = pl.tile.read(table_rows, [request, next_page])
                                                    next_k_bank = (pl.cast(next_k_physical, pl.INDEX) % physical_pages) * 4 + kv_head
                                                    prefetched_k = pl.tile.load(key_bank, [next_k_bank, 0, 0], [1, 128, 128], target_memory=pl.Mem.Mat)
                                                qk_nz = pl.tile.move(
                                                    qk_acc,
                                                    pl.Mem.Vec,
                                                    blayout=pl.TileLayout.col_major,
                                                    slayout=pl.TileLayout.row_major,
                                                )
                                                qk = pl.tile.extract(
                                                    qk_nz, 0, 0, [8, 128], target_memory=pl.Mem.Vec
                                                )
                                                qk_rounded = pl.tile.cast(
                                                    pl.tile.cast(qk, pl.FP16, mode="none"), pl.FP32, mode="none"
                                                )
                                                scaled = pl.tile.muls(qk_rounded, 0.08837890625)
                                                scores = pl.tile.cast(
                                                    pl.tile.cast(scaled, pl.FP16, mode="none"),
                                                    pl.FP32,
                                                    mode="none",
                                                )

                                                valid_rows: pl.Scalar[pl.INDEX] = 128
                                                if length - page_base < 128:
                                                    valid_rows = length - page_base
                                                # Bounds on the actual phi result dominate every use.
                                                # Both hold for each executed, prevalidated page.
                                                if valid_rows >= 0:
                                                    if valid_rows <= 128:
                                                        # A full page selects every score unchanged.
                                                        # Keep the native sentinel mask on partial pages.
                                                        if valid_rows < 128:
                                                            score_mask = pl.tile.cmps(
                                                                position_rows,
                                                                pl.cast(pl.cast(valid_rows, pl.INT32), pl.FP32),
                                                                cmp_type=2,
                                                            )
                                                            scores = pl.tile.select(
                                                                score_mask, scores, -3.402823466e38
                                                            )
                                                        reduction_scratch = pl.tile.create([8, 128], pl.FP32)
                                                        page_maximum = pl.tile.row_max(scores, reduction_scratch)
                                                        if page_index == 0:
                                                            maximum = page_maximum
                                                            delta = pl.tile.sub(page_maximum, page_maximum)
                                                        else:
                                                            maximum = pl.tile.maximum(old_maximum, page_maximum)
                                                            delta = pl.tile.sub(old_maximum, maximum)
                                                        delta_row = pl.tile.reshape(delta, [1, 8])
                                                        delta16 = pl.tile.fillpad_expand(
                                                            delta_row, [1, 16], pad_value=0
                                                        )
                                                        delta_float = pl.tile.cast(
                                                            pl.tile.cast(delta16, pl.FP16, mode="none"),
                                                            pl.FP32,
                                                            mode="none",
                                                        )
                                                        factor16 = pl.tile.exp(delta_float)
                                                        factor8 = pl.tile.extract(
                                                            factor16, 0, 0, [1, 8], target_memory=pl.Mem.Vec
                                                        )
                                                        factors = pl.tile.reshape(factor8, [8, 1])
                                                        # One 32-byte factor block per row avoids a full-width copy.
                                                        row_blocks = pl.tile.create([8, 8], pl.FP32)
                                                        rescale = pl.tile.row_expand(row_blocks, factors)
                                                        maximum_blocks = pl.tile.row_expand(row_blocks, maximum)
                                                        centered = pl.tile.row_expand_sub(scores, maximum_blocks)
                                                        centered_float = pl.tile.cast(
                                                            pl.tile.cast(centered, pl.FP16, mode="none"),
                                                            pl.FP32,
                                                            mode="none",
                                                        )
                                                        exponential = pl.tile.exp(centered_float)
                                                        page_sum = pl.tile.row_sum(exponential, reduction_scratch)
                                                        probability = pl.tile.cast(
                                                            exponential, pl.FP16, mode="none"
                                                        )
                                                        probability16 = pl.tile.fillpad_expand(
                                                            probability, [16, 128], pad_value=0
                                                        )
                                                        total = pl.tile.add(pl.tile.mul(total, factors), page_sum)
                                                        old_maximum = maximum
                                                        probability_mat = pl.tile.move(
                                                            probability16,
                                                            pl.Mem.Mat,
                                                            blayout=pl.TileLayout.row_major,
                                                            slayout=pl.TileLayout.row_major,
                                                        )
                                                        probability_left = pl.tile.extract(
                                                            probability_mat,
                                                            0,
                                                            0,
                                                            [16, 128],
                                                            target_memory=pl.Mem.Left,
                                                        )
                                                        v_zeroed = v_prefetched
                                                        v_right = pl.tile.move(
                                                            v_zeroed,
                                                            pl.Mem.Right,
                                                            blayout=pl.TileLayout.row_major,
                                                            slayout=pl.TileLayout.col_major,
                                                        )
                                                        pv_acc = pl.tile.matmul(probability_left, v_right)
                                                        if next_page < page_count:
                                                            next_v_physical = pl.tile.read(table_rows, [request, next_page])
                                                            next_v_bank = (pl.cast(next_v_physical, pl.INDEX) % physical_pages) * 4 + kv_head
                                                            next_v_rows = length - (next_page) * 128
                                                            if next_v_rows > 128:
                                                                next_v_rows = 128
                                                            if next_v_rows >= 0:
                                                                if next_v_rows <= 128:
                                                                    next_v_source = pl.tile.load(value_bank, [next_v_bank, 0, 0], [1, 128, 128], target_memory=pl.Mem.Mat)
                                                                    next_v_full = pl.tile.reshape(next_v_source, [128, 128])
                                                                    next_v_valid = pl.tile.set_validshape(next_v_full, next_v_rows, 128)
                                                                    prefetched_v = pl.tile.fillpad_inplace(next_v_valid, pad_value=0)
                                                        pv_nz = pl.tile.move(
                                                            pv_acc,
                                                            pl.Mem.Vec,
                                                            blayout=pl.TileLayout.col_major,
                                                            slayout=pl.TileLayout.row_major,
                                                        )
                                                        page_value = pl.tile.extract(
                                                            pv_nz, 0, 0, [8, 128], target_memory=pl.Mem.Vec
                                                        )
                                                        numerator = pl.tile.add(
                                                            pl.tile.row_expand_mul(numerator, rescale), page_value
                                                        )
                                            denominator = pl.tile.row_expand(numerator, total)
                                            denominator_half = pl.tile.cast(denominator, pl.FP16, mode="none")
                                            numerator_half = pl.tile.cast(numerator, pl.FP16, mode="none")
                                            result = pl.tile.div(numerator_half, denominator_half)
                                            if fuse_o_quant:
                                                # Preserve the FP16 divide output before the existing O quantization.
                                                o_column = kv_head * 1024
                                                o_inverse = pl.tile.cast(pl.tile.load(reciprocal, [0, o_column], [1, 1024]), pl.FP32, mode="none")
                                                o_zero = pl.tile.cast(pl.tile.load(offset, [0, o_column], [1, 1024]), pl.FP32, mode="none")
                                                o_values_fp32 = pl.tile.cast(pl.tile.reshape(result, [1, 1024]), pl.FP32, mode="none")
                                                o_scaled = pl.tile.add(pl.tile.mul(o_values_fp32, o_inverse), o_zero)
                                                o_ceiling = pl.tile.full([1, 1024], pl.FP32, 127.0)
                                                o_floor = pl.tile.full([1, 1024], pl.FP32, -128.0)
                                                o_bounded = pl.tile.maximum(pl.tile.minimum(o_scaled, o_ceiling), o_floor)
                                                o_rounded = pl.tile.cast(o_bounded, pl.INT32, mode="rint")
                                                o_integral = pl.tile.cast(pl.tile.cast(o_rounded, pl.FP32, mode="none"), pl.FP16, mode="none")
                                                o_narrowed = pl.tile.cast(o_integral, pl.INT8, mode="trunc")
                                                pl.tile.store(o_narrowed, [request, o_column], output)
                                            else:
                                                # Ordinary ND output alias; no physical layout
                                                # reinterpretation and no extra external tensor.
                                                output_rows = pl.tensor.view(output, shape=[640, 128])
                                                pl.tile.store(result, [task * 8, 0], output_rows)
                                            success: pl.Scalar[pl.INT32] = 0
                                            pl.tensor.write(status, [task, 0], success)
        return output, status

    if fuse_o_quant:
        @pl.jit.incore
        def attention_o_quant_b20(
            query: pl.Tensor[[80, 16, 128], pl.FP16],
            keys: pl.Tensor[[physical_pages, 32, 128, 16], pl.FP16],
            values: pl.Tensor[[physical_pages, 32, 128, 16], pl.FP16],
            lengths: pl.Tensor[[20, 8], pl.INT32],
            output: pl.InOut[pl.Tensor[[20, 4096], pl.INT8]],
            block_table: pl.Tensor[[20, 32], pl.INT32],
            reciprocal: pl.Tensor[[1, 4096], pl.FP16],
            offset: pl.Tensor[[1, 4096], pl.FP16],
            status: pl.InOut[pl.Tensor[[80, 16], pl.INT32]],
        ):
            return attention_body(query, keys, values, lengths, output, block_table, reciprocal, offset, status)

        return attention_o_quant_b20

    @pl.jit.incore
    def attention_b20(
        query: pl.Tensor[[80, 16, 128], pl.FP16],
        keys: pl.Tensor[[physical_pages, 32, 128, 16], pl.FP16],
        values: pl.Tensor[[physical_pages, 32, 128, 16], pl.FP16],
        lengths: pl.Tensor[[20, 8], pl.INT32],
        output: pl.InOut[pl.Tensor[[20, 4096], pl.FP16]],
        block_table: pl.Tensor[[20, 32], pl.INT32],
        status: pl.InOut[pl.Tensor[[80, 16], pl.INT32]],
    ):
        return attention_body(query, keys, values, lengths, output, block_table, lengths, lengths, status)

    return attention_b20
