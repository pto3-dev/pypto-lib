# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under
# the terms and conditions of CANN Open Software License Agreement Version 2.0.
# Please refer to the License for details. You may not use this file except in
# compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the
# License.
# -----------------------------------------------------------------------------
"""Qwen3-14B per-head Q/K RMSNorm followed by NeoX-style RoPE."""

import pypto.language as pl


BATCH_PAD = 16
NUM_Q_HEADS = 40
NUM_KV_HEADS = 8
Q_PER_KV = NUM_Q_HEADS // NUM_KV_HEADS
Q_HEAD_PAD = 8
K_HEAD_PAD = 8
HEAD_DIM = 128
HALF_DIM = HEAD_DIM // 2
Q_SIZE = NUM_Q_HEADS * HEAD_DIM
KV_SIZE = NUM_KV_HEADS * HEAD_DIM
EPS = 1.0e-6
HEAD_DIM_INV = 1.0 / HEAD_DIM


@pl.jit(auto_scope=False)
def qwen3_qk_norm_rope_decode(
    q: pl.Tensor[[BATCH_PAD, Q_SIZE], pl.BF16],
    k: pl.Tensor[[BATCH_PAD, KV_SIZE], pl.BF16],
    q_norm_weight: pl.Tensor[[1, HEAD_DIM], pl.BF16],
    k_norm_weight: pl.Tensor[[1, HEAD_DIM], pl.BF16],
    rope_cos: pl.Tensor[[BATCH_PAD, HALF_DIM], pl.BF16],
    rope_sin: pl.Tensor[[BATCH_PAD, HALF_DIM], pl.BF16],
    q_out: pl.Out[pl.Tensor[[BATCH_PAD, Q_SIZE], pl.BF16]],
    k_out: pl.Out[pl.Tensor[[BATCH_PAD, KV_SIZE], pl.BF16]],
):
    """Normalize each head, then rotate its two 64-element halves."""
    for group in pl.parallel(0, BATCH_PAD * NUM_KV_HEADS, 1):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="qk_norm_rope"):
            batch_idx = group // NUM_KV_HEADS
            kv_head = group - batch_idx * NUM_KV_HEADS
            cos = pl.cast(rope_cos[batch_idx : batch_idx + 1, :], pl.FP32)
            q_gamma = pl.cast(q_norm_weight, pl.FP32)
            k_gamma = pl.cast(k_norm_weight, pl.FP32)
            sin = pl.cast(rope_sin[batch_idx : batch_idx + 1, :], pl.FP32)

            q_col = kv_head * Q_PER_KV * HEAD_DIM
            q_heads = pl.reshape(
                pl.concat(
                    pl.cast(
                        q[
                            batch_idx : batch_idx + 1,
                            q_col : q_col + Q_PER_KV * HEAD_DIM,
                        ],
                        pl.FP32,
                    ),
                    pl.full([1, (Q_HEAD_PAD - Q_PER_KV) * HEAD_DIM], dtype=pl.FP32, value=0.0),
                ),
                [Q_HEAD_PAD, HEAD_DIM],
            )
            q_ss = pl.row_sum(pl.mul(q_heads, q_heads))
            q_inv = pl.rsqrt(pl.add(pl.mul(q_ss, HEAD_DIM_INV), EPS))
            q_normed = pl.col_expand_mul(
                pl.row_expand_mul(q_heads, q_inv),
                q_gamma,
            )
            q_normed = pl.cast(pl.cast(q_normed, pl.BF16, mode="rint"), pl.FP32)
            q_lo = q_normed[:, 0:HALF_DIM]
            q_hi = q_normed[:, HALF_DIM:HEAD_DIM]
            q_rotated = pl.concat(
                pl.sub(
                    pl.col_expand_mul(q_lo, cos),
                    pl.col_expand_mul(q_hi, sin),
                ),
                pl.add(
                    pl.col_expand_mul(q_hi, cos),
                    pl.col_expand_mul(q_lo, sin),
                ),
            )
            q_out = pl.assemble(
                q_out,
                pl.reshape(
                    pl.cast(q_rotated[0:Q_PER_KV, :], pl.BF16, mode="rint"),
                    [1, Q_PER_KV * HEAD_DIM],
                ),
                [batch_idx, q_col],
            )

            k_col = kv_head * HEAD_DIM
            k_head = pl.reshape(
                pl.concat(
                    pl.cast(
                        k[
                            batch_idx : batch_idx + 1,
                            k_col : k_col + HEAD_DIM,
                        ],
                        pl.FP32,
                    ),
                    pl.full([1, (K_HEAD_PAD - 1) * HEAD_DIM], dtype=pl.FP32, value=0.0),
                ),
                [K_HEAD_PAD, HEAD_DIM],
            )
            k_ss = pl.row_sum(pl.mul(k_head, k_head))
            k_inv = pl.rsqrt(pl.add(pl.mul(k_ss, HEAD_DIM_INV), EPS))
            k_normed = pl.col_expand_mul(
                pl.row_expand_mul(k_head, k_inv),
                k_gamma,
            )
            k_normed = pl.cast(pl.cast(k_normed, pl.BF16, mode="rint"), pl.FP32)
            k_lo = k_normed[:, 0:HALF_DIM]
            k_hi = k_normed[:, HALF_DIM:HEAD_DIM]
            k_rotated = pl.concat(
                pl.sub(
                    pl.col_expand_mul(k_lo, cos),
                    pl.col_expand_mul(k_hi, sin),
                ),
                pl.add(
                    pl.col_expand_mul(k_hi, cos),
                    pl.col_expand_mul(k_lo, sin),
                ),
            )
            k_out = pl.assemble(
                k_out,
                pl.cast(k_rotated[0:1, :], pl.BF16, mode="rint"),
                [batch_idx, k_col],
            )
    return q_out, k_out
