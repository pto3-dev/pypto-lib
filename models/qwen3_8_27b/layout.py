# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""PTO embedding gather and packed projection splits."""

import pypto.language as pl

TOKENS = pl.dynamic("tokens")


def make_embedding(vocab_size=248320, hidden=5120):
    @pl.jit
    def gather(
        ids: pl.Tensor[[TOKENS], pl.INT64],
        weight: pl.Tensor[[vocab_size, hidden], pl.BF16],
        out: pl.Out[pl.Tensor[[TOKENS, hidden], pl.BF16]],
    ):
        for job in pl.spmd(pl.tensor.dim(ids, 0) * (hidden // 128), name_hint="token_embedding"):
            token = job // (hidden // 128)
            c0 = job % (hidden // 128) * 128
            word = pl.read(ids, [token])
            value = pl.slice(weight, [1, 128], [word, c0])
            out = pl.assemble(out, value, [token, c0])
        return out

    return gather


embedding = make_embedding()


def make_gdn_unpack(key_heads, value_heads, inline=False, tokens=TOKENS):
    channels = (2 * key_heads + value_heads) * 128

    decorate = pl.jit.inline if inline else pl.jit

    @decorate
    def unpack(
        x: pl.Tensor[[tokens, channels], pl.BF16],
        q: pl.Out[pl.Tensor[[tokens, key_heads, 128], pl.BF16]],
        k: pl.Out[pl.Tensor[[tokens, key_heads, 128], pl.BF16]],
        v: pl.Out[pl.Tensor[[tokens, value_heads, 128], pl.BF16]],
    ):
        qf = pl.reshape(q, [pl.tensor.dim(x, 0), key_heads * 128])
        kf = pl.reshape(k, [pl.tensor.dim(x, 0), key_heads * 128])
        vf = pl.reshape(v, [pl.tensor.dim(x, 0), value_heads * 128])
        tokens = pl.tensor.dim(x, 0)
        for job in pl.spmd(tokens * (2 * key_heads + value_heads), name_hint="gdn_split"):
            token = job // (2 * key_heads + value_heads)
            head = job % (2 * key_heads + value_heads)
            value = pl.slice(x, [1, 128], [token, head * 128])
            if head < key_heads:
                qf = pl.assemble(qf, value, [token, head * 128])
            elif head < 2 * key_heads:
                kf = pl.assemble(kf, value, [token, (head - key_heads) * 128])
            else:
                vf = pl.assemble(vf, value, [token, (head - 2 * key_heads) * 128])
        return q, k, v

    return unpack


def make_query_gate_unpack(heads):
    @pl.jit
    def unpack(
        x: pl.Tensor[[TOKENS, heads * 512], pl.BF16],
        q: pl.Out[pl.Tensor[[TOKENS, heads, 256], pl.BF16]],
        gate: pl.Out[pl.Tensor[[TOKENS, heads * 256], pl.BF16]],
    ):
        qf = pl.reshape(q, [pl.tensor.dim(x, 0), heads * 256])
        for job in pl.spmd(pl.tensor.dim(x, 0) * heads, name_hint="query_gate_split"):
            token = job // heads
            head = job % heads
            query = pl.slice(x, [1, 256], [token, head * 512])
            gating = pl.slice(x, [1, 256], [token, head * 512 + 256])
            qf = pl.assemble(qf, query, [token, head * 256])
            gate = pl.assemble(gate, gating, [token, head * 256])
        return q, gate

    return unpack
