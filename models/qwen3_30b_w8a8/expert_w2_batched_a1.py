# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""Public B20 W2 projection over the resident NZ expert bank.

The default preserves one owner per group. An explicit columns_per_owner
selects a smaller column group without changing integer accumulation. Keep raw INT32 accumulation, partial-group output guards
and route/tail guards identical to the measured V8 prototype.
Expert counts and ordered route lists are supplied by the shared metadata leaf.
"""

from typing import Any

import pypto.language as pl


def make_w2_b20(*, column_partition: bool, columns_per_owner: int | None = None) -> Any:
    """Reuse each gathered M16 activation for 1, 2, or 4 N512 column tiles."""
    if columns_per_owner is not None and column_partition:
        raise ValueError("Choose column_partition or columns_per_owner, not both")
    columns_per_owner = (1 if column_partition else 4) if columns_per_owner is None else columns_per_owner
    if type(columns_per_owner) is not int or columns_per_owner not in (1, 2, 4):
        raise ValueError("columns_per_owner must be 1, 2, or 4")
    owner_group_stride = 4 // columns_per_owner
    owner_column_divisor = columns_per_owner

    @pl.jit.incore
    def w2_b20(
        input: pl.Tensor[[160, 768], pl.INT8],
        weight: pl.Tensor[[6291456, 32], pl.INT8],
        ids: pl.Tensor[[20, 8], pl.INT32],
        expert_counts: pl.Tensor[[1, 128], pl.INT32],
        route_indices: pl.Tensor[[128, 160], pl.INT32],
        output: pl.InOut[pl.Tensor[[160, 2048], pl.INT32]],
        status: pl.InOut[pl.Tensor[[160, 16], pl.INT32]],
    ):
        block = pl.tile.get_block_idx()
        bank = pl.tensor.view(weight, shape=[128, 2048, 768], layout=pl.NZ)
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
                            for n_tile in pl.range(4):
                                owner = (group * owner_group_stride + n_tile // owner_column_divisor) % 8
                                if block == owner:
                                    participates = True
                            if participates:
                                # One resident activation per participating block/group:
                                # [K/32,M16,32], 12 KiB, reused for every owned N tile.
                                activation = pl.tile.create([16, 768], pl.INT8, target_memory=pl.Mem.Mat)
                                selected = pl.tile.create([1, 16], pl.INT32, target_memory=pl.Mem.Vec)
                                for lane in pl.range(16):
                                    absent: pl.Scalar[pl.INT32] = -1
                                    pl.tile.write(selected, [0, lane], absent)

                                listed = pl.tile.load(route_indices, [expert, first], [1, 16])
                                for lane in pl.range(16):
                                    if first + lane < count:
                                        route_value = pl.tile.read(listed, [0, lane])
                                        route = pl.cast(route_value, pl.INDEX)
                                        if route >= 0:
                                            if route < 160:
                                                pl.tile.write(selected, [0, lane], route_value)
                                                activation = pl.tile.gather_row(
                                                    activation, input, [lane, 0], [route, 0], [1, 768],
                                                )

                                # Unselected physical rows are never scattered. Their
                                # contents do not mix with selected rows in integer MAD.
                                # Do not add an extra GM read or write for those lanes.
                                for n_tile in pl.range(4):
                                    owner = (group * owner_group_stride + n_tile // owner_column_divisor) % 8
                                    if block == owner:
                                        column = n_tile * 512
                                        acc = pl.tile.create([16, 512], pl.INT32, target_memory=pl.Mem.Acc)
                                        for k in pl.pipeline(0, 768, 64, stage=2):
                                            k_tile = k // 64
                                            left = pl.tile.extract(
                                                activation, 0, k, [16, 64], target_memory=pl.Mem.Left,
                                            )
                                            b_nk = pl.tile.load(
                                                bank, [expert, column, k], [1, 512, 64],
                                                target_memory=pl.Mem.Mat,
                                            )
                                            # Public on-chip singleton rank reduction;
                                            # verify Mat/NZ orientation through the actual pipeline.
                                            # The GM [expert,N,K] alias is unchanged.
                                            b_nk_2d = pl.tile.reshape(b_nk, [512, 64])
                                            b_kn = pl.tile.transpose_view(b_nk_2d)
                                            right = pl.tile.extract(
                                                b_kn, 0, 0, [64, 512], target_memory=pl.Mem.Right,
                                            )
                                            acc = pl.tile.matmul_acc(acc, left, right, init_cond=(k_tile == 0))

                                        # Preserve the complete raw INT32 NZ accumulator once.
                                        # Backend/ISA materialize the logical row with strided UB DMA.
                                        packed = pl.tile.move(
                                            acc, target_memory=pl.Mem.Vec,
                                            blayout=pl.TileLayout.col_major,
                                            slayout=pl.TileLayout.row_major,
                                        )
                                        for lane in pl.range(16):
                                            selected_route = pl.tile.read(selected, [0, lane])
                                            if selected_route >= 0:
                                                if selected_route < 160:
                                                    output_row = pl.cast(selected_route, pl.INDEX)
                                                    output_row_tile = pl.tile.extract(
                                                        packed, lane, 0, [1, 512], target_memory=pl.Mem.Vec,
                                                    )
                                                    pl.tile.store(output_row_tile, [output_row, column], output)
                                        if n_tile == 0:
                                            for lane in pl.range(16):
                                                selected_route = pl.tile.read(selected, [0, lane])
                                                if selected_route >= 0:
                                                    if selected_route < 160:
                                                        status_row = pl.cast(selected_route, pl.INDEX)
                                                        success: pl.Scalar[pl.INT32] = 0
                                                        pl.tensor.write(status, [status_row, 0], success)
        return output, status

    return w2_b20
