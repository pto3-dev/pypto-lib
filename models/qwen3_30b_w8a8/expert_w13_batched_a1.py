# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""Public B20 grouped-channel W13 over the resident NZ expert bank.

The selected cold-build profile uses column-partitioned M16/N512 with K128 L1 loads and ascending K64 MADs.
This is an isolated transfer-granularity trial. Keep direct Acc VDEQ16 FP16 rounding, readonly packed
INT32 scale-word ABI and route/tail guards identical to measured V2.
"""

from typing import Any

import pypto.language as pl


def make_w13_b20(*, column_partition: bool) -> Any:
    """Gather once per owned group; reuse Mat16x2048 across N512/K64 steps."""
    owner_group_stride = 3 if column_partition else 1
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
                            for n_tile in pl.range(3):
                                owner = (group * owner_group_stride + n_tile * owner_column_stride) % 8
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
                                for n_tile in pl.range(3):
                                    owner = (group * owner_group_stride + n_tile * owner_column_stride) % 8
                                    if block == owner:
                                        column = n_tile * 512
                                        acc = pl.tile.create([16, 512], pl.INT32, target_memory=pl.Mem.Acc)
                                        # Load two K64 weight slices into L1 per transfer.
                                        # Keep the original ascending sequence of K64 MADs.
                                        for k in pl.pipeline(0, 2048, 128, stage=2):
                                            b_nk = pl.tile.load(
                                                bank, [expert, column, k], [1, 512, 128],
                                                target_memory=pl.Mem.Mat,
                                            )
                                            b_nk_2d = pl.tile.reshape(b_nk, [512, 128])
                                            b_kn = pl.tile.transpose_view(b_nk_2d)
                                            for inner in pl.range(2):
                                                inner_k = inner * 64
                                                global_k = k + inner_k
                                                left = pl.tile.extract(
                                                    activation, 0, global_k, [16, 64],
                                                    target_memory=pl.Mem.Left,
                                                )
                                                right = pl.tile.extract(
                                                    b_kn, inner_k, 0, [64, 512],
                                                    target_memory=pl.Mem.Right,
                                                )
                                                acc = pl.tile.matmul_acc(
                                                    acc, left, right, init_cond=(global_k == 0)
                                                )

                                        # Keep [low FP32 bits, high zero] words unchanged.
                                        # This alias is on-chip only; external descriptors
                                        # keep their original INT32 dtype and shape.
                                        scale_words = pl.tile.load(
                                            channel_scale, [expert, column * 2], [1, 1024],
                                            target_memory=pl.Mem.Vec,
                                        )
                                        packed_scale = pl.tile.reinterpret_view(
                                            scale_words, pl.INT64, shape=[1, 512],
                                        )
                                        # Required bounded extension: Acc16x512 -> FP16
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
                                                            packed, lane, 0, [1, 512], target_memory=pl.Mem.Vec,
                                                        )
                                                        pl.tile.store(output_row_tile, [output_row, column], output)
                                                        if n_tile == 0:
                                                            success: pl.Scalar[pl.INT32] = 0
                                                            pl.tensor.write(status, [output_row, 0], success)
        return output, status

    return w13_b20
