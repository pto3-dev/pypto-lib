# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -------------------------------------------------------------------------------------------------------
"""Append rank-local K/V to the native BSND paged cache without layout conversion."""

from pathlib import Path

import pypto.language as pl
from simpler_setup import KernelCompiler

from .linear import KV_WIDTH
from .quant import TOKENS_DYN

BLOCKS_DYN = pl.dynamic("QWEN30_KV_BLOCKS")
BLOCK_SIZE = 128
CACHE_BLOCK_WIDTH = BLOCK_SIZE * KV_WIDTH


@pl.jit.inline
def append_kv_cache_impl(
    k: pl.Tensor[[TOKENS_DYN, KV_WIDTH], pl.BF16],
    v: pl.Tensor[[TOKENS_DYN, KV_WIDTH], pl.BF16],
    slots: pl.Tensor[[TOKENS_DYN, 1], pl.INT32],
    key_cache: pl.InOut[pl.Tensor[[BLOCKS_DYN, CACHE_BLOCK_WIDTH], pl.BF16]],
    value_cache: pl.InOut[pl.Tensor[[BLOCKS_DYN, CACHE_BLOCK_WIDTH], pl.BF16]],
) -> tuple[pl.Tensor, pl.Tensor]:
    tokens = pl.tensor.dim(k, 0)
    for token in pl.spmd(tokens, name_hint="append_kv"):
        slot = pl.cast(pl.read(slots, [token, 0]), pl.INDEX)
        if slot >= 0:
            block = slot // BLOCK_SIZE
            col = (slot % BLOCK_SIZE) * KV_WIDTH
            key_cache[block : block + 1, col : col + KV_WIDTH] = k[token : token + 1, :]
            value_cache[block : block + 1, col : col + KV_WIDTH] = v[token : token + 1, :]
    return key_cache, value_cache


@pl.jit
def append_kv_cache(
    k: pl.Tensor[[TOKENS_DYN, KV_WIDTH], pl.BF16],
    v: pl.Tensor[[TOKENS_DYN, KV_WIDTH], pl.BF16],
    slots: pl.Tensor[[TOKENS_DYN, 1], pl.INT32],
    key_cache: pl.InOut[pl.Tensor[[BLOCKS_DYN, CACHE_BLOCK_WIDTH], pl.BF16]],
    value_cache: pl.InOut[pl.Tensor[[BLOCKS_DYN, CACHE_BLOCK_WIDTH], pl.BF16]],
) -> tuple[pl.Tensor, pl.Tensor]:
    k.bind_dynamic(0, TOKENS_DYN)
    v.bind_dynamic(0, TOKENS_DYN)
    slots.bind_dynamic(0, TOKENS_DYN)
    key_cache.bind_dynamic(0, BLOCKS_DYN)
    value_cache.bind_dynamic(0, BLOCKS_DYN)
    return append_kv_cache_impl(k, v, slots, key_cache, value_cache)


def compile_a1_kv_append(pto_isa_root: str | Path) -> bytes:
    """Compile the single-token, four-head A1 BF16 cache update.

    Arguments are K/V `[4,128]`, key/value caches `[slots,512]`, an INT32
    slot carrier and a separate INT32 status carrier, each at least eight
    elements. Negative slots skip the update. Status element zero is 0 for
    success/skip, 1 for an out-of-range slot, or 2 for incompatible geometry.
    All tensor pointers and shapes stay fixed during graph capture/replay.
    """
    source = Path(__file__).with_name("a1_kv_append.cpp")
    return KernelCompiler("a1").compile_single_launch(str(source), str(pto_isa_root))


def compile_a1_kv_gather(pto_isa_root: str | Path) -> bytes:
    """Compile a four-block, device-indexed A1 paged-cache staging kernel.

    Inputs are cache tensors `[slots,512]`, a physical page table and INT32
    lengths. Outputs are head-major `[4*capacity,128]` K/V staging buffers and
    status `[4,16]`, one independent cache line per block. Capacity and slot
    count are multiples of 128. Inactive rows are zero-filled. Status is 0 on
    success, 1 for invalid length, 2 for invalid page, or 3 for invalid geometry.
    """
    source = Path(__file__).with_name("a1_kv_gather.cpp")
    return KernelCompiler("a1").compile_single_launch(str(source), str(pto_isa_root))
