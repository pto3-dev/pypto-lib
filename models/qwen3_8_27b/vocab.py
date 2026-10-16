# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Vocabulary-parallel embedding with disjoint token ownership."""

import pypto.language as pl

TOKENS = pl.dynamic("tokens")


def make_embedding(rank, vocab=124160, hidden=5120):
    @pl.jit
    def gather(
        ids: pl.Tensor[[TOKENS], pl.INT64],
        weight: pl.Tensor[[vocab, hidden], pl.BF16],
        out: pl.Out[pl.Tensor[[TOKENS, hidden], pl.BF16]],
    ):
        for job in pl.spmd(pl.tensor.dim(ids, 0) * (hidden // 128), name_hint="tp_embedding"):
            token = job // (hidden // 128)
            c0 = job % (hidden // 128) * 128
            word = pl.read(ids, [token]) - rank * vocab
            if word >= 0 and word < vocab:
                value = pl.slice(weight, [1, 128], [word, c0])
                out = pl.assemble(out, value, [token, c0])
            else:
                out = pl.assemble(out, pl.full([1, 128], dtype=pl.BF16, value=0.0), [token, c0])
        return out

    return gather
