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
# ------------------------------------------------------------------------------
"""Qwen3-14B one-page GQA Decode attention for the vLLM shadow bridge."""

import math

import pypto.language as pl

from paged_attention_pypto import (
    FFTS_WORKSPACE_ELEMENTS,
    STACK_TOKENS,
    TRANSFER_ROWS,
    paged_attention_pypto_swpipe,
)


BATCH_PAD = 16
NUM_Q_HEADS = 40
NUM_KV_HEADS = 8
Q_PER_KV = NUM_Q_HEADS // NUM_KV_HEADS
HEAD_PAD = 16
HEAD_DIM = 128
BLOCK_SIZE = 128
VALUE_TILE = 64
Q_SIZE = NUM_Q_HEADS * HEAD_DIM
KV_SIZE = NUM_KV_HEADS * HEAD_DIM
SCALE = 1.0 / math.sqrt(HEAD_DIM)
NUM_BLOCKS_DYN = pl.dynamic("QWEN3_VLLM_PA_NUM_BLOCKS_DYN")
FUSED_BATCH_DYN = pl.dynamic("QWEN3_VLLM_FUSED_PA_BATCH_DYN")
FUSED_CACHE_ROWS_DYN = pl.dynamic("QWEN3_VLLM_FUSED_PA_CACHE_ROWS_DYN")
FUSED_BLOCK_TABLE_DYN = pl.dynamic("QWEN3_VLLM_FUSED_PA_BLOCK_TABLE_DYN")
FUSED_ROPE_DYN = pl.dynamic("QWEN3_VLLM_FUSED_PA_ROPE_DYN")


@pl.jit(auto_scope=False)
def qwen3_paged_attention_one_page(
    q: pl.Tensor[[BATCH_PAD, Q_SIZE], pl.BF16],
    key_blocks: pl.Tensor[[BATCH_PAD, BLOCK_SIZE, KV_SIZE], pl.BF16],
    value_blocks: pl.Tensor[[BATCH_PAD, BLOCK_SIZE, KV_SIZE], pl.BF16],
    seq_lens: pl.Tensor[[BATCH_PAD], pl.INT32],
    out: pl.Out[pl.Tensor[[BATCH_PAD, Q_SIZE], pl.BF16]],
):
    """Compute one-page GQA attention from compact blocks gathered by vLLM."""
    key_2d = pl.reshape(key_blocks, [BATCH_PAD * BLOCK_SIZE, KV_SIZE])
    value_2d = pl.reshape(value_blocks, [BATCH_PAD * BLOCK_SIZE, KV_SIZE])

    for task in pl.parallel(0, BATCH_PAD * NUM_KV_HEADS, 1):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="paged_attention"):
            batch_idx = task // NUM_KV_HEADS
            kv_head = task - batch_idx * NUM_KV_HEADS
            q_col = kv_head * Q_PER_KV * HEAD_DIM
            kv_col = kv_head * HEAD_DIM
            seq_len = pl.cast(pl.read(seq_lens, [batch_idx]), pl.INDEX)

            q_heads = pl.reshape(
                pl.concat(
                    q[
                        batch_idx : batch_idx + 1,
                        q_col : q_col + Q_PER_KV * HEAD_DIM,
                    ],
                    pl.full(
                        [1, (HEAD_PAD - Q_PER_KV) * HEAD_DIM],
                        dtype=pl.BF16,
                        value=0.0,
                    ),
                ),
                [HEAD_PAD, HEAD_DIM],
            )
            key_tile = pl.slice(
                key_2d,
                [BLOCK_SIZE, HEAD_DIM],
                [batch_idx * BLOCK_SIZE, kv_col],
            )
            scores = pl.matmul(
                q_heads,
                key_tile,
                b_trans=True,
                out_dtype=pl.FP32,
            )
            scores = pl.mul(scores, SCALE)
            scores = pl.fillpad(
                pl.set_validshape(scores, Q_PER_KV, seq_len),
                pad_value=pl.PadValue.min,
            )
            row_max = pl.row_max(scores)
            probabilities = pl.exp(pl.row_expand_sub(scores, row_max))
            probability_sum = pl.row_sum(probabilities)
            probabilities = pl.row_expand_div(probabilities, probability_sum)

            for value_block in pl.range(HEAD_DIM // VALUE_TILE):
                value_col = kv_col + value_block * VALUE_TILE
                value_tile = pl.slice(
                    value_2d,
                    [BLOCK_SIZE, VALUE_TILE],
                    [batch_idx * BLOCK_SIZE, value_col],
                )
                value_transposed = pl.cast(
                    pl.transpose(value_tile, axis1=0, axis2=1),
                    target_type=pl.FP32,
                )
                for q_head in pl.range(Q_PER_KV):
                    weighted_values = pl.col_expand_mul(
                        value_transposed,
                        probabilities[q_head : q_head + 1, :],
                    )
                    dim_start = value_block * VALUE_TILE
                    row_context = pl.reshape(
                        pl.row_sum(weighted_values),
                        [1, VALUE_TILE],
                    )
                    row_context = pl.cast(
                        row_context,
                        target_type=pl.BF16,
                        mode="rint",
                    )
                    out_col = q_col + q_head * HEAD_DIM + dim_start
                    out = pl.assemble(out, row_context, [batch_idx, out_col])
    return out


@pl.jit(auto_scope=False)
def qwen3_paged_attention_one_page_direct(
    q: pl.Tensor[[BATCH_PAD, Q_SIZE], pl.BF16],
    key_cache: pl.Tensor[[NUM_BLOCKS_DYN, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM], pl.BF16],
    value_cache: pl.Tensor[[NUM_BLOCKS_DYN, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM], pl.BF16],
    physical_blocks: pl.Tensor[[BATCH_PAD], pl.INT32],
    seq_lens: pl.Tensor[[BATCH_PAD], pl.INT32],
    out: pl.Out[pl.Tensor[[BATCH_PAD, Q_SIZE], pl.BF16]],
):
    """Compute one-page GQA attention directly from vLLM's paged KV cache."""
    key_cache.bind_dynamic(0, NUM_BLOCKS_DYN)
    value_cache.bind_dynamic(0, NUM_BLOCKS_DYN)
    cache_rows = pl.tensor.dim(key_cache, 0) * BLOCK_SIZE
    key_2d = pl.reshape(key_cache, [cache_rows, KV_SIZE])
    value_2d = pl.reshape(value_cache, [cache_rows, KV_SIZE])

    for task in pl.parallel(0, BATCH_PAD * NUM_KV_HEADS, 1):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="paged_attention_direct"):
            batch_idx = task // NUM_KV_HEADS
            kv_head = task - batch_idx * NUM_KV_HEADS
            q_col = kv_head * Q_PER_KV * HEAD_DIM
            kv_col = kv_head * HEAD_DIM
            physical_block = pl.cast(
                pl.read(physical_blocks, [batch_idx]),
                pl.INDEX,
            )
            cache_row = physical_block * BLOCK_SIZE
            seq_len = pl.cast(pl.read(seq_lens, [batch_idx]), pl.INDEX)

            q_heads = pl.reshape(
                pl.concat(
                    q[
                        batch_idx : batch_idx + 1,
                        q_col : q_col + Q_PER_KV * HEAD_DIM,
                    ],
                    pl.full(
                        [1, (HEAD_PAD - Q_PER_KV) * HEAD_DIM],
                        dtype=pl.BF16,
                        value=0.0,
                    ),
                ),
                [HEAD_PAD, HEAD_DIM],
            )
            key_tile = pl.slice(
                key_2d,
                [BLOCK_SIZE, HEAD_DIM],
                [cache_row, kv_col],
            )
            scores = pl.matmul(
                q_heads,
                key_tile,
                b_trans=True,
                out_dtype=pl.FP32,
            )
            scores = pl.mul(scores, SCALE)
            scores = pl.fillpad(
                pl.set_validshape(scores, Q_PER_KV, seq_len),
                pad_value=pl.PadValue.min,
            )
            row_max = pl.row_max(scores)
            probabilities = pl.exp(pl.row_expand_sub(scores, row_max))
            probability_sum = pl.row_sum(probabilities)
            probabilities = pl.row_expand_div(probabilities, probability_sum)

            for value_block in pl.range(HEAD_DIM // VALUE_TILE):
                value_col = kv_col + value_block * VALUE_TILE
                value_tile = pl.slice(
                    value_2d,
                    [BLOCK_SIZE, VALUE_TILE],
                    [cache_row, value_col],
                )
                value_transposed = pl.cast(
                    pl.transpose(value_tile, axis1=0, axis2=1),
                    target_type=pl.FP32,
                )
                for q_head in pl.range(Q_PER_KV):
                    weighted_values = pl.col_expand_mul(
                        value_transposed,
                        probabilities[q_head : q_head + 1, :],
                    )
                    dim_start = value_block * VALUE_TILE
                    row_context = pl.reshape(
                        pl.row_sum(weighted_values),
                        [1, VALUE_TILE],
                    )
                    row_context = pl.cast(
                        row_context,
                        target_type=pl.BF16,
                        mode="rint",
                    )
                    out_col = q_col + q_head * HEAD_DIM + dim_start
                    out = pl.assemble(out, row_context, [batch_idx, out_col])
    return out


@pl.jit(auto_scope=False)
def qwen3_vllm_fused_attention(
    qkv: pl.Tensor[[FUSED_BATCH_DYN, Q_SIZE + 2 * KV_SIZE], pl.BF16],
    key_cache: pl.InOut[pl.Tensor[[FUSED_CACHE_ROWS_DYN, KV_SIZE], pl.BF16]],
    value_cache: pl.InOut[pl.Tensor[[FUSED_CACHE_ROWS_DYN, KV_SIZE], pl.BF16]],
    block_table: pl.Tensor[[FUSED_BLOCK_TABLE_DYN], pl.INT32],
    seq_lens: pl.Tensor[[FUSED_BATCH_DYN], pl.INT32],
    slot_mapping: pl.Tensor[[FUSED_BATCH_DYN], pl.INT32],
    rope_cos: pl.Tensor[[FUSED_ROPE_DYN, HEAD_DIM], pl.FP32],
    rope_sin: pl.Tensor[[FUSED_ROPE_DYN, HEAD_DIM], pl.FP32],
    q_norm_w: pl.Tensor[[1, HEAD_DIM], pl.FP32],
    k_norm_w: pl.Tensor[[1, HEAD_DIM], pl.FP32],
    out: pl.Out[pl.Tensor[[FUSED_BATCH_DYN, Q_SIZE], pl.BF16]],
) -> pl.Tensor[[FUSED_BATCH_DYN, Q_SIZE], pl.BF16]:
    """Adapt vLLM's post-input-RMS BF16 QKV to the fused PyPTO PA body."""
    qkv.bind_dynamic(0, FUSED_BATCH_DYN)
    key_cache.bind_dynamic(0, FUSED_CACHE_ROWS_DYN)
    value_cache.bind_dynamic(0, FUSED_CACHE_ROWS_DYN)
    block_table.bind_dynamic(0, FUSED_BLOCK_TABLE_DYN)
    seq_lens.bind_dynamic(0, FUSED_BATCH_DYN)
    slot_mapping.bind_dynamic(0, FUSED_BATCH_DYN)
    rope_cos.bind_dynamic(0, FUSED_ROPE_DYN)
    rope_sin.bind_dynamic(0, FUSED_ROPE_DYN)
    out.bind_dynamic(0, FUSED_BATCH_DYN)

    active_batch = pl.tensor.dim(seq_lens, 0)
    q_proj = pl.create_tensor([active_batch, Q_SIZE], dtype=pl.FP32)
    k_proj = pl.create_tensor([active_batch, KV_SIZE], dtype=pl.FP32)
    v_proj = pl.create_tensor([active_batch, KV_SIZE], dtype=pl.FP32)
    inv_rms_states = pl.create_tensor([active_batch, 1], dtype=pl.FP32)

    # vLLM has already applied the decoder input RMSNorm before QKV projection.
    # Promote the BF16 projection result and use an identity deferred RMS factor.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="vllm_q_promote") as q_proj_tid:
        for batch_idx in pl.range(BATCH_PAD):
            if batch_idx < active_batch:
                q_proj = pl.assemble(
                    q_proj,
                    pl.cast(qkv[batch_idx : batch_idx + 1, 0:Q_SIZE], target_type=pl.FP32),
                    [batch_idx, 0],
                )
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="vllm_k_promote") as k_proj_tid:
        for batch_idx in pl.range(BATCH_PAD):
            if batch_idx < active_batch:
                k_proj = pl.assemble(
                    k_proj,
                    pl.cast(
                        qkv[batch_idx : batch_idx + 1, Q_SIZE : Q_SIZE + KV_SIZE],
                        target_type=pl.FP32,
                    ),
                    [batch_idx, 0],
                )
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="vllm_v_promote") as v_proj_tid:
        for batch_idx in pl.range(BATCH_PAD):
            if batch_idx < active_batch:
                v_proj = pl.assemble(
                    v_proj,
                    pl.cast(qkv[batch_idx : batch_idx + 1, Q_SIZE + KV_SIZE :], target_type=pl.FP32),
                    [batch_idx, 0],
                )
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="vllm_identity_rms") as rms_tid:
        for batch_idx in pl.range(BATCH_PAD):
            if batch_idx < active_batch:
                pl.write(inv_rms_states, [batch_idx, 0], pl.cast(1.0, pl.FP32))

    q_tnd_flat = pl.create_tensor([active_batch * NUM_Q_HEADS, HEAD_DIM], dtype=pl.BF16)
    score_transfer = pl.create_tensor([TRANSFER_ROWS, STACK_TOKENS], dtype=pl.FP32)
    probability_transfer = pl.create_tensor([TRANSFER_ROWS, STACK_TOKENS], dtype=pl.BF16)
    pv_transfer = pl.create_tensor([TRANSFER_ROWS, HEAD_DIM], dtype=pl.FP32)
    ffts_workspace = pl.create_tensor([FFTS_WORKSPACE_ELEMENTS], dtype=pl.INT64)
    seed_tid = pl.system.task_dummy(deps=[])
    paged_attention_pypto_swpipe(
        q_tnd_flat,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        inv_rms_states,
        slot_mapping,
        rope_cos,
        rope_sin,
        q_proj,
        k_proj,
        v_proj,
        q_norm_w,
        k_norm_w,
        0,
        out,
        score_transfer,
        probability_transfer,
        pv_transfer,
        ffts_workspace,
        q_proj_tid,
        k_proj_tid,
        v_proj_tid,
        rms_tid,
        seed_tid,
        seed_tid,
        seed_tid,
    )
    return out
