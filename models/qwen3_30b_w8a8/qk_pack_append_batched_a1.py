# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""B20 Q/K packing and native physical PA cache append using public DSL operations."""

import pypto.language as pl

def make_pack_qk_append_b20(physical_pages=None):
    cache_pages = pl.dynamic("CACHE_PAGES") if physical_pages is None else physical_pages

    @pl.jit.incore
    def pack_qk_append_b20(
        q: pl.Tensor[[640, 128], pl.FP16],
        k: pl.Tensor[[160, 128], pl.FP16],
        packed_q: pl.Out[pl.Tensor[[80, 16, 128], pl.FP16]],
        packed_k: pl.Out[pl.Tensor[[20, 512], pl.FP16]],
        status: pl.InOut[pl.Tensor[[80, 16], pl.INT32]],
        values: pl.Tensor[[20, 512], pl.FP16],
        key_cache: pl.InOut[pl.Tensor[[cache_pages, 32, 128, 16], pl.FP16]],
        value_cache: pl.InOut[pl.Tensor[[cache_pages, 32, 128, 16], pl.FP16]],
        slots: pl.Tensor[[20, 8], pl.INT32],
        append_status: pl.InOut[pl.Tensor[[20, 16], pl.INT32]],
    ):
        zero = pl.tile.full([8, 128], dtype=pl.FP16, value=0.0)
        slot_rows = pl.tile.load(slots, [0, 0], [20, 8])
        pages = pl.dim(key_cache, 0)
        block = pl.tile.get_block_idx()
        for group in pl.range(0, 40):
            if group % 8 == block:
                row = group // 2
                first_head = (group % 2) * 2
                slot = pl.tile.read(slot_rows, [row, 0])
                page = slot // 128
                code: pl.Scalar[pl.INT32] = 0
                if slot >= 0:
                    if page >= pages:
                        invalid: pl.Scalar[pl.INT32] = 1
                        code = invalid
                    else:
                        for other in pl.range(0, 20):
                            if other != row:
                                other_slot = pl.tile.read(slot_rows, [other, 0])
                                if other_slot == slot:
                                    duplicate: pl.Scalar[pl.INT32] = 3
                                    code = duplicate
                for delta in pl.range(0, 2):
                    head = first_head + delta
                    task = row * 4 + head
                    query = pl.tile.load(q, [task * 8, 0], [8, 128])
                    key = pl.tile.load(k, [row * 8 + head, 0], [1, 128])
                    pl.tile.store(query, [task, 0, 0], packed_q, shapes=[1, 8, 128])
                    pl.tile.store(zero, [task, 8, 0], packed_q, shapes=[1, 8, 128])
                    pl.tile.store(key, [row, head * 128], packed_k)
                    if slot >= 0:
                        if page < pages:
                            if code == 0:
                                value = pl.tile.load(values, [row, head * 128], [1, 128])
                                key_rows = pl.reshape(key, [8, 16])
                                value_rows = pl.reshape(value, [8, 16])
                                time = slot % 128
                                pl.tile.store(
                                    key_rows,
                                    [page, head * 8, time, 0],
                                    key_cache,
                                    shapes=[1, 8, 1, 16],
                                )
                                pl.tile.store(
                                    value_rows,
                                    [page, head * 8, time, 0],
                                    value_cache,
                                    shapes=[1, 8, 1, 16],
                                )
                    success: pl.Scalar[pl.INT32] = 0
                    pl.write(status, [task, 0], success)
                if first_head == 0:
                    pl.write(append_status, [row, 0], code)
        return packed_q, packed_k, status, key_cache, value_cache, append_status

    return pack_qk_append_b20


pack_qk_append_b20 = make_pack_qk_append_b20()
