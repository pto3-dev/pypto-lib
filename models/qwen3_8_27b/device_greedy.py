# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exact BF16 local-vocabulary argmax and finite checking on PTO vector cores."""

import pypto.language as pl


def make_device_greedy(vocab=124160, tile=1024, inline=False):
    chunks = (vocab + tile - 1) // tile
    expected_chunks = float(chunks)
    chunk_pad = (chunks + 31) // 32 * 32
    decorate = pl.jit.inline if inline else pl.jit

    @decorate
    def greedy(x: pl.Tensor[[1, vocab], pl.BF16], out: pl.Out[pl.Tensor[[1, 8], pl.FP32]]):
        partial = pl.create_tensor([chunks, 8], pl.FP32)
        available = pl.system.available_aiv_count()
        launches = pl.min(available, chunks)
        for core in pl.spmd(launches, name_hint="vocab_partial_argmax"):
            for job in pl.range(core, chunks, pl.tile.get_block_num()):
                start = job * tile
                valid = pl.min(tile, vocab - start)
                values = pl.cast(
                    pl.slice(x, [8, tile], [0, start], [1, valid], pad_value=pl.PadValue.min, clamp=True),
                    pl.FP32,
                )
                values = pl.add(values, 0.0)
                maximum = pl.row_max(values)
                index = pl.row_argmax(values)
                check_values = pl.cast(
                    pl.slice(x, [8, tile], [0, start], [1, valid], pad_value=0, clamp=True), pl.FP32
                )
                check = pl.row_sum(pl.mul(check_values, 1e-30))
                total = pl.read(check, [0, 0])
                result = pl.full([1, 8], dtype=pl.FP32, value=0.0)
                if total == total and total < 1e30 and total > -1e30:
                    pl.write(result, [0, 0], pl.read(maximum, [0, 0]))
                    pl.write(
                        result, [0, 1], pl.cast(pl.cast(pl.read(index, [0, 0]) + start, pl.INT32), pl.FP32)
                    )
                    pl.write(result, [0, 2], 1.0)
                else:
                    pl.write(result, [0, 1], -1.0)
                partial = pl.assemble(partial, result, [job, 0])
        for job in pl.spmd(1, name_hint="vocab_final_argmax"):
            scores = pl.slice(
                partial, [chunk_pad, 8], [0, 0], [chunks, 1], pad_value=pl.PadValue.min, clamp=True
            )
            flags = pl.slice(partial, [chunk_pad, 8], [0, 2], [chunks, 1], pad_value=0, clamp=True)
            final_index = pl.col_argmax(scores)
            total = pl.read(pl.col_sum(flags), [0, 0])
            result = pl.full([1, 8], dtype=pl.FP32, value=0.0)
            if total == expected_chunks:
                winner = pl.read(final_index, [0, 0])
                pl.write(result, [0, 0], pl.read(partial, [winner, 0]))
                pl.write(result, [0, 1], pl.read(partial, [winner, 1]))
                pl.write(result, [0, 2], 1.0)
            else:
                pl.write(result, [0, 1], -1.0)
            out = pl.assemble(out, result, [0, 0])
        return out

    return greedy
