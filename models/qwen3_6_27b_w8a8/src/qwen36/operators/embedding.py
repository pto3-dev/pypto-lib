# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""embedding operators. Explicit PyPTO functions; no source generator at build time.

Migrated from the validated L1K2560 graph; tuning is audited separately.
"""

import pypto.language as pl


@pl.function(type=pl.FunctionType.InCore)
def embedding0_kernel(
    ids: pl.Tensor[[24], pl.INT64],
    weight: pl.Tensor[[62080, 5120], pl.BF16],
    out: pl.Tensor[[24, 5120], pl.BF16],
) -> pl.Tensor[[24, 5120], pl.BF16]:
    pl.func_attr({"static_block_mapping": True})
    job = pl.tile.get_block_idx()
    token = job // (5120 // 128)
    c0 = job % (5120 // 128) * 128
    word = pl.read(ids, [token]) - 0 * 62080
    if word >= 0 and word < 62080:
        value = pl.slice(weight, [1, 128], [word, c0])
        out = pl.assemble(out, value, [token, c0])
    else:
        out = pl.assemble(out, pl.full([1, 128], dtype=pl.BF16, value=0.0), [token, c0])
    return out


@pl.function(type=pl.FunctionType.Inline)
def embedding0(
    ids: pl.Tensor[[24], pl.INT64],
    weight: pl.Tensor[[62080, 5120], pl.BF16],
    out: pl.Tensor[[24, 5120], pl.BF16],
) -> pl.Tensor[[24, 5120], pl.BF16]:
    with pl.spmd(24 * (5120 // 128), allow_early_resolve=True):
        out = embedding0_kernel(ids, weight, out)
    return out


@pl.function(type=pl.FunctionType.InCore)
def embedding1_kernel(
    ids: pl.Tensor[[24], pl.INT64],
    weight: pl.Tensor[[62080, 5120], pl.BF16],
    out: pl.Tensor[[24, 5120], pl.BF16],
) -> pl.Tensor[[24, 5120], pl.BF16]:
    pl.func_attr({"static_block_mapping": True})
    job = pl.tile.get_block_idx()
    token = job // (5120 // 128)
    c0 = job % (5120 // 128) * 128
    word = pl.read(ids, [token]) - 1 * 62080
    if word >= 0 and word < 62080:
        value = pl.slice(weight, [1, 128], [word, c0])
        out = pl.assemble(out, value, [token, c0])
    else:
        out = pl.assemble(out, pl.full([1, 128], dtype=pl.BF16, value=0.0), [token, c0])
    return out


@pl.function(type=pl.FunctionType.Inline)
def embedding1(
    ids: pl.Tensor[[24], pl.INT64],
    weight: pl.Tensor[[62080, 5120], pl.BF16],
    out: pl.Tensor[[24, 5120], pl.BF16],
) -> pl.Tensor[[24, 5120], pl.BF16]:
    with pl.spmd(24 * (5120 // 128), allow_early_resolve=True):
        out = embedding1_kernel(ids, weight, out)
    return out


@pl.function(type=pl.FunctionType.InCore)
def embedding2_kernel(
    ids: pl.Tensor[[24], pl.INT64],
    weight: pl.Tensor[[62080, 5120], pl.BF16],
    out: pl.Tensor[[24, 5120], pl.BF16],
) -> pl.Tensor[[24, 5120], pl.BF16]:
    pl.func_attr({"static_block_mapping": True})
    job = pl.tile.get_block_idx()
    token = job // (5120 // 128)
    c0 = job % (5120 // 128) * 128
    word = pl.read(ids, [token]) - 2 * 62080
    if word >= 0 and word < 62080:
        value = pl.slice(weight, [1, 128], [word, c0])
        out = pl.assemble(out, value, [token, c0])
    else:
        out = pl.assemble(out, pl.full([1, 128], dtype=pl.BF16, value=0.0), [token, c0])
    return out


@pl.function(type=pl.FunctionType.Inline)
def embedding2(
    ids: pl.Tensor[[24], pl.INT64],
    weight: pl.Tensor[[62080, 5120], pl.BF16],
    out: pl.Tensor[[24, 5120], pl.BF16],
) -> pl.Tensor[[24, 5120], pl.BF16]:
    with pl.spmd(24 * (5120 // 128), allow_early_resolve=True):
        out = embedding2_kernel(ids, weight, out)
    return out


@pl.function(type=pl.FunctionType.InCore)
def embedding3_kernel(
    ids: pl.Tensor[[24], pl.INT64],
    weight: pl.Tensor[[62080, 5120], pl.BF16],
    out: pl.Tensor[[24, 5120], pl.BF16],
) -> pl.Tensor[[24, 5120], pl.BF16]:
    pl.func_attr({"static_block_mapping": True})
    job = pl.tile.get_block_idx()
    token = job // (5120 // 128)
    c0 = job % (5120 // 128) * 128
    word = pl.read(ids, [token]) - 3 * 62080
    if word >= 0 and word < 62080:
        value = pl.slice(weight, [1, 128], [word, c0])
        out = pl.assemble(out, value, [token, c0])
    else:
        out = pl.assemble(out, pl.full([1, 128], dtype=pl.BF16, value=0.0), [token, c0])
    return out


@pl.function(type=pl.FunctionType.Inline)
def embedding3(
    ids: pl.Tensor[[24], pl.INT64],
    weight: pl.Tensor[[62080, 5120], pl.BF16],
    out: pl.Tensor[[24, 5120], pl.BF16],
) -> pl.Tensor[[24, 5120], pl.BF16]:
    with pl.spmd(24 * (5120 // 128), allow_early_resolve=True):
        out = embedding3_kernel(ids, weight, out)
    return out
