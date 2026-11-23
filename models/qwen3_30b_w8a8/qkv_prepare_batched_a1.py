# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""Bit-preserving B20 QKV and dynamic RoPE-table preparation in PyPTO DSL."""

import pypto.language as pl

ROPE_ROWS = pl.dynamic("ROPE_ROWS")


@pl.jit.incore
def prepare_qkv_rows(
    qkv: pl.Tensor[[20, 5120], pl.FP16],
    positions: pl.Tensor[[20, 8], pl.INT32],
    table: pl.Tensor[[ROPE_ROWS, 128], pl.FP16],
    queries: pl.Out[pl.Tensor[[640, 128], pl.FP16]],
    keys: pl.Out[pl.Tensor[[160, 128], pl.FP16]],
    values: pl.Out[pl.Tensor[[20, 512], pl.FP16]],
    rope_rows: pl.Out[pl.Tensor[[20, 128], pl.FP16]],
    status: pl.InOut[pl.Tensor[[20, 16], pl.INT32]],
):
    zero = pl.tile.full([4, 128], dtype=pl.FP16, value=0.0)
    table_rows = pl.dim(table, 0)
    block = pl.tile.get_block_idx()
    for query_base in pl.range(0, 640, 32):
        row = query_base // 32
        if row % 8 == block:
            position = pl.read(positions, [row, 0])
            code: pl.Scalar[pl.INT32] = 1
            if position >= 0:
                if position < table_rows:
                    query_flat = pl.tile.load(qkv, [row, 0], [1, 4096])
                    query = pl.reshape(query_flat, [32, 128])
                    pl.tile.store(query, [query_base, 0], queries)
                    key_flat = pl.tile.load(qkv, [row, 4096], [1, 512])
                    key = pl.reshape(key_flat, [4, 128])
                    value = pl.tile.load(qkv, [row, 4608], [1, 512])
                    rope = pl.tile.load(table, [position, 0], [1, 128])
                    key_base = query_base // 4
                    pl.tile.store(key, [key_base, 0], keys)
                    pl.tile.store(zero, [key_base + 4, 0], keys)
                    pl.tile.store(value, [row, 0], values)
                    pl.tile.store(rope, [row, 0], rope_rows)
                    valid_code: pl.Scalar[pl.INT32] = 0
                    code = valid_code
            pl.write(status, [row, 0], code)
    return queries, keys, values, rope_rows, status
