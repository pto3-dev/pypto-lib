# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""B1 NZ QKV projection with native-rounding Q/K Norm and Neox RoPE epilogues."""

from typing import Any

import pypto.language as pl

from .config import HIDDEN
from .kv_cache import BLOCK_SIZE, CACHE_BLOCK_WIDTH
from .linear import HEAD_DIM, KV_WIDTH, Q_WIDTH, QKV_WIDTH
from .projection_output import K_TILE, N_TILE
from .qk_rope import EPS


def make_qkv_norm_rope(
    cache_positions: int, *, inline: bool = False, cache_blocks: int = 1, append_cache: bool = False,
) -> Any:
    """Fuse the B1 projection handoff while retaining its BF16 rounding point.

    Args:
        cache_positions: Number of native rotary-cache entries.
        inline: Inline into a composed Attention callable.
        cache_blocks: Number of physical paged-cache blocks.
        append_cache: Write rounded K/V into the slot-mapped cache.

    Returns:
        QKV projection with final head outputs and optional cache updates.
    """
    if cache_positions < 1:
        raise ValueError(f"RoPE cache capacity must be positive, got {cache_positions}")
    if cache_blocks < 1:
        raise ValueError(f"KV cache capacity must be positive, got {cache_blocks}")
    decorator = pl.jit.inline if inline else pl.jit

    @decorator
    def qkv_norm_rope(
        x: pl.Tensor[[1, HIDDEN], pl.INT8],
        input_scale: pl.Tensor[[1, 1], pl.FP32],
        weight: pl.Tensor[[1, HIDDEN, QKV_WIDTH], pl.INT8, pl.NZ],
        weight_scale: pl.Tensor[[1, QKV_WIDTH], pl.FP32],
        q_weight: pl.Tensor[[1, HEAD_DIM], pl.BF16],
        k_weight: pl.Tensor[[1, HEAD_DIM], pl.BF16],
        positions: pl.Tensor[[1, 1], pl.INT64],
        cos_sin_cache: pl.Tensor[[cache_positions, HEAD_DIM], pl.BF16],
        q: pl.Out[pl.Tensor[[16, HEAD_DIM], pl.BF16]],
        k: pl.Out[pl.Tensor[[2, HEAD_DIM], pl.BF16]],
        v: pl.Out[pl.Tensor[[2, HEAD_DIM], pl.BF16]],
        slots: pl.Tensor[[1, 1], pl.INT32],
        key_cache: pl.InOut[pl.Tensor[[cache_blocks, CACHE_BLOCK_WIDTH], pl.BF16]],
        value_cache: pl.InOut[pl.Tensor[[cache_blocks, CACHE_BLOCK_WIDTH], pl.BF16]],
    ) -> tuple[pl.Tensor, pl.Tensor, pl.Tensor, pl.Tensor, pl.Tensor]:
        for block in pl.spmd(QKV_WIDTH // N_TILE, name_hint="qkv_norm_rope", allow_early_resolve=True):
            pl.set_cache_policy(weight, pl.CachePolicy.BYPASS)
            for n0 in pl.unroll(0, QKV_WIDTH, N_TILE):
                if block == n0 // N_TILE:
                    acc = pl.create_tensor([1, N_TILE], dtype=pl.INT32)
                    for kb in pl.pipeline(HIDDEN // K_TILE, stage=2):
                        k0 = kb * K_TILE
                        lhs = pl.slice(x, [1, K_TILE], [0, k0])
                        rhs = pl.reshape(pl.slice(weight, [1, K_TILE, N_TILE], [0, k0, n0]), [K_TILE, N_TILE])
                        acc = pl.matmul_acc(acc, lhs, rhs, init_cond=(kb == 0))
                    value = pl.cast(acc, pl.FP32, mode="rint")
                    scale = pl.read(input_scale, [0, 0])
                    combined_scale = pl.mul(weight_scale[:, n0 : n0 + N_TILE], scale)
                    projected = pl.cast(pl.mul(value, combined_scale), pl.BF16, mode="rint")
                    if n0 < Q_WIDTH + KV_WIDTH:
                        if n0 < Q_WIDTH:
                            gamma = pl.cast(q_weight[:, :], pl.FP32)
                        else:
                            gamma = pl.cast(k_weight[:, :], pl.FP32)
                        pos = pl.cast(pl.read(positions, [0, 0]), pl.INDEX)
                        cos = pl.cast(cos_sin_cache[pos : pos + 1, :64], pl.FP32)
                        sin = pl.cast(cos_sin_cache[pos : pos + 1, 64:], pl.FP32)
                        for head in pl.unroll(N_TILE // HEAD_DIM):
                            full = pl.col_expand_mul(
                                pl.full([8, HEAD_DIM], dtype=pl.FP32, value=1.0),
                                pl.cast(projected[:, head * HEAD_DIM : (head + 1) * HEAD_DIM], pl.FP32),
                            )
                            variance = pl.mul(pl.row_sum(pl.mul(full, full)), 1.0 / HEAD_DIM)
                            inv = pl.rsqrt(pl.add(variance, EPS), high_precision=True)
                            left = full[:, :64]
                            right = full[:, 64:]
                            left_norm = pl.col_expand_mul(pl.row_expand_mul(left, inv), gamma[:, :64])
                            right_norm = pl.col_expand_mul(pl.row_expand_mul(right, inv), gamma[:, 64:])
                            left_out = pl.cast(
                                pl.sub(pl.col_expand_mul(left_norm, cos), pl.col_expand_mul(right_norm, sin)),
                                pl.BF16,
                                mode="rint",
                            )
                            right_out = pl.cast(
                                pl.add(pl.col_expand_mul(right_norm, cos), pl.col_expand_mul(left_norm, sin)),
                                pl.BF16,
                                mode="rint",
                            )
                            if n0 < Q_WIDTH:
                                output_head = n0 // HEAD_DIM + head
                                q[output_head : output_head + 1, :64] = left_out[:1, :]
                                q[output_head : output_head + 1, 64:] = right_out[:1, :]
                            else:
                                k[head : head + 1, :64] = left_out[:1, :]
                                k[head : head + 1, 64:] = right_out[:1, :]
                                if append_cache:
                                    slot = pl.cast(pl.read(slots, [0, 0]), pl.INDEX)
                                    if slot >= 0:
                                        cache_block = slot // BLOCK_SIZE
                                        cache_col = (slot % BLOCK_SIZE) * KV_WIDTH + head * HEAD_DIM
                                        key_cache[cache_block : cache_block + 1, cache_col : cache_col + 64] = left_out[:1, :]
                                        key_cache[cache_block : cache_block + 1, cache_col + 64 : cache_col + HEAD_DIM] = right_out[:1, :]
                    else:
                        for value_head in pl.unroll(N_TILE // HEAD_DIM):
                            v[value_head : value_head + 1, :] = projected[
                                :, value_head * HEAD_DIM : (value_head + 1) * HEAD_DIM
                            ]
                            if append_cache:
                                slot = pl.cast(pl.read(slots, [0, 0]), pl.INDEX)
                                if slot >= 0:
                                    cache_block = slot // BLOCK_SIZE
                                    cache_col = (slot % BLOCK_SIZE) * KV_WIDTH + value_head * HEAD_DIM
                                    value_tile = projected[:, value_head * HEAD_DIM : (value_head + 1) * HEAD_DIM]
                                    value_cache[cache_block : cache_block + 1, cache_col : cache_col + HEAD_DIM] = value_tile
        return q, k, v, key_cache, value_cache

    return qkv_norm_rope
