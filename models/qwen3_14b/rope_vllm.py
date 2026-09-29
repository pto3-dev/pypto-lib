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
"""Qwen3-14B NeoX-style RoPE for normalized BF16 Q/K tensors."""

import pypto.language as pl


BATCH_PAD = 16
NUM_Q_HEADS = 40
NUM_KV_HEADS = 8
Q_PER_KV = NUM_Q_HEADS // NUM_KV_HEADS
HEAD_PAD = 8
HEAD_DIM = 128
HALF_DIM = HEAD_DIM // 2
Q_SIZE = NUM_Q_HEADS * HEAD_DIM
KV_SIZE = NUM_KV_HEADS * HEAD_DIM


@pl.jit(auto_scope=False)
def qwen3_rope_decode(
    q: pl.Tensor[[BATCH_PAD, Q_SIZE], pl.BF16],
    k: pl.Tensor[[BATCH_PAD, KV_SIZE], pl.BF16],
    rope_cos: pl.Tensor[[BATCH_PAD, HALF_DIM], pl.BF16],
    rope_sin: pl.Tensor[[BATCH_PAD, HALF_DIM], pl.BF16],
    q_out: pl.Out[pl.Tensor[[BATCH_PAD, Q_SIZE], pl.BF16]],
    k_out: pl.Out[pl.Tensor[[BATCH_PAD, KV_SIZE], pl.BF16]],
):
    """Rotate each Q/K head while preserving vLLM's packed layout."""
    for group in pl.parallel(0, BATCH_PAD * NUM_KV_HEADS, 1):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="rope"):
            batch_idx = group // NUM_KV_HEADS
            kv_head = group - batch_idx * NUM_KV_HEADS
            cos = pl.cast(rope_cos[batch_idx : batch_idx + 1, :], pl.FP32)
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
                    pl.full(
                        [1, (HEAD_PAD - Q_PER_KV) * HEAD_DIM],
                        dtype=pl.FP32,
                        value=0.0,
                    ),
                ),
                [HEAD_PAD, HEAD_DIM],
            )
            q_lo = q_heads[:, 0:HALF_DIM]
            q_hi = q_heads[:, HALF_DIM:HEAD_DIM]
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
                    pl.full(
                        [1, (HEAD_PAD - 1) * HEAD_DIM],
                        dtype=pl.FP32,
                        value=0.0,
                    ),
                ),
                [HEAD_PAD, HEAD_DIM],
            )
            k_lo = k_head[:, 0:HALF_DIM]
            k_hi = k_head[:, HALF_DIM:HEAD_DIM]
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
