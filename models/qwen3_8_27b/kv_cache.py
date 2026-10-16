# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Direct append into native Ascend BSND paged KV storage."""

import pypto.language as pl

TOKENS = pl.dynamic("tokens")
CACHE_ROWS = pl.dynamic("cache_rows")


def make_append(kv_heads, head_dim=256):
    width = kv_heads * head_dim

    @pl.jit
    def append(
        k: pl.Tensor[[TOKENS, width], pl.BF16],
        v: pl.Tensor[[TOKENS, width], pl.BF16],
        key_cache: pl.InOut[pl.Tensor[[CACHE_ROWS, width], pl.BF16]],
        value_cache: pl.InOut[pl.Tensor[[CACHE_ROWS, width], pl.BF16]],
        slots: pl.Tensor[[TOKENS], pl.INT64],
    ):
        for job in pl.spmd(pl.tensor.dim(k, 0) * (width // 128), name_hint="paged_kv_append"):
            token = job // (width // 128)
            c0 = job % (width // 128) * 128
            slot = pl.read(slots, [token])
            if slot >= 0:
                kt = pl.slice(k, [1, 128], [token, c0])
                vt = pl.slice(v, [1, 128], [token, c0])
                key_cache = pl.assemble(key_cache, kt, [slot, c0])
                value_cache = pl.assemble(value_cache, vt, [slot, c0])
        return key_cache, value_cache

    return append
