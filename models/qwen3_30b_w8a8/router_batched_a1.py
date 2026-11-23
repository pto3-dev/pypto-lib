# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""Public DSL Router candidate retaining both L0B weight operands across row groups.

Compile make_router_b20() through the model compiler helper and launch eight
blocks. ABI: input FP16[20,2048], weight FP16[2048,128], output FP16[20,128],
status INT32[8,16]. This module performs no compilation or device execution.
"""

from typing import Any

import pypto.language as pl


def make_router_b20() -> Any:
    """Reuse two 32 KiB Right operands without changing either group's MAD order."""

    @pl.jit.incore
    def router_b20(
        input: pl.Tensor[[20, 2048], pl.FP16],
        weight: pl.Tensor[[2048, 128], pl.FP16],
        output: pl.Out[pl.Tensor[[20, 128], pl.FP16]],
        status: pl.InOut[pl.Tensor[[8, 16], pl.INT32]],
    ):
        block = pl.tile.get_block_idx()
        for lane in pl.range(8):
            if block == lane:
                column = lane * 16

                # Rows 0..15: physical M16, literal valid M16, K0 then K1024.
                full_a0 = pl.tile.load(
                    input, [0, 0], [16, 1024],
                    valid_shape=[16, 1024], target_memory=pl.Mem.Mat,
                )
                full_b0 = pl.tile.load(
                    weight, [0, column], [1024, 16], target_memory=pl.Mem.Mat,
                )
                full_l0 = pl.tile.extract(full_a0, 0, 0, [16, 1024], target_memory=pl.Mem.Left)
                full_r0 = pl.tile.extract(full_b0, 0, 0, [1024, 16], target_memory=pl.Mem.Right)
                full_acc = pl.tile.matmul(full_l0, full_r0)
                full_a1 = pl.tile.load(
                    input, [0, 1024], [16, 1024],
                    valid_shape=[16, 1024], target_memory=pl.Mem.Mat,
                )
                full_b1 = pl.tile.load(
                    weight, [1024, column], [1024, 16], target_memory=pl.Mem.Mat,
                )
                full_l1 = pl.tile.extract(full_a1, 0, 0, [16, 1024], target_memory=pl.Mem.Left)
                full_r1 = pl.tile.extract(full_b1, 0, 0, [1024, 16], target_memory=pl.Mem.Right)
                full_acc = pl.tile.matmul_acc(full_acc, full_l1, full_r1)
                full_values = pl.tile.extract(full_acc, 0, 0, [16, 16], target_memory=pl.Mem.Vec)
                # Supported A1 mode; require the native vconv_f322f16 instruction.
                full_rounded = pl.tile.cast(full_values, pl.FP16, mode="none")
                pl.tile.store(full_rounded, [0, column], output)

                # Retain full_r0/full_r1 in the two 32 KiB L0B slots.
                # Rows 16..19: physical M16, literal valid M4; never read rows20+.
                tail_a0 = pl.tile.load(
                    input, [16, 0], [16, 1024],
                    valid_shape=[4, 1024], target_memory=pl.Mem.Mat,
                )
                tail_l0 = pl.tile.extract(tail_a0, 0, 0, [16, 1024], target_memory=pl.Mem.Left)
                tail_acc = pl.tile.matmul(tail_l0, full_r0)
                tail_a1 = pl.tile.load(
                    input, [16, 1024], [16, 1024],
                    valid_shape=[4, 1024], target_memory=pl.Mem.Mat,
                )
                tail_l1 = pl.tile.extract(tail_a1, 0, 0, [16, 1024], target_memory=pl.Mem.Left)
                tail_acc = pl.tile.matmul_acc(tail_acc, tail_l1, full_r1)
                tail_values = pl.tile.extract(tail_acc, 0, 0, [16, 16], target_memory=pl.Mem.Vec)
                tail_rounded = pl.tile.cast(tail_values, pl.FP16, mode="none")
                pl.tile.store(tail_rounded, [16, column], output)

                # Both groups precede the sole status write; preserve columns1..15.
                zero: pl.Scalar[pl.INT32] = 0
                pl.write(status, [lane, 0], zero)
        return output, status

    return router_b20
