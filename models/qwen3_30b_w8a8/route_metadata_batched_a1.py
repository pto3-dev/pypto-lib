# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""Per-layer routing metadata shared by W13 and W2, rebuilt every Decode."""

import pypto.language as pl


@pl.jit.incore
def route_metadata_b20(
    ids: pl.Tensor[[20, 8], pl.INT32],
    expert_counts: pl.Out[pl.Tensor[[1, 128], pl.INT32]],
    route_indices: pl.InOut[pl.Tensor[[128, 160], pl.INT32]],
    status: pl.InOut[pl.Tensor[[8, 16], pl.INT32]],
):
    block = pl.tile.get_block_idx()
    for owner in pl.range(8):
        if owner == block:
            routes = pl.tile.load(ids, [0, 0], [20, 8])
            counts = pl.tile.create([1, 16], dtype=pl.INT32)
            zero: pl.Scalar[pl.INT32] = 0
            one: pl.Scalar[pl.INT32] = 1
            invalid: pl.Scalar[pl.INT32] = 2
            result: pl.Scalar[pl.INT32] = 0
            for local in pl.range(16):
                pl.tile.write(counts, [0, local], zero)
            for route in pl.range(160):
                expert = pl.tile.read(routes, [route // 8, route % 8])
                if expert < 0:
                    result = invalid
                elif expert >= 128:
                    result = invalid
                elif expert // 16 == owner:
                    local = pl.cast(expert % 16, pl.INDEX)
                    count = pl.tile.read(counts, [0, local])
                    slot = pl.cast(count, pl.INDEX)
                    expert_index = pl.cast(expert, pl.INDEX)
                    route_value = pl.cast(route, pl.INT32)
                    pl.tensor.write(route_indices, [expert_index, slot], route_value)
                    pl.tile.write(counts, [0, local], count + one)
            # Each block owns one aligned contiguous 64-byte count segment.
            pl.tile.store(counts, [0, owner * 16], expert_counts)
            pl.tensor.write(status, [owner, 0], result)
    return expert_counts, route_indices, status
