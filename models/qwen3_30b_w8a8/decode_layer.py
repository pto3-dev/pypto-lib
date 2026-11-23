# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""One Qwen30 Decode layer with input TP/Norm, Attention and post-Norm MoE."""

from typing import Any

import pypto.language as pl
import pypto.language.distributed as pld

from .attention_decode import make_attention_decode
from .config import EXPERTS, HIDDEN, LOCAL_INTERMEDIATE, W13_WIDTH
from .kv_cache import CACHE_BLOCK_WIDTH
from .linear import HEAD_DIM, QKV_WIDTH
from .post_norm_moe import make_post_norm_moe
from .tp_norm import SIGNAL_TILE, TP_OPERATIONS


def make_decode_layer(
    cache_blocks: int, *, first: bool = False, last: bool = False,
    fuse_qkv_norm_rope: bool = False, fuse_page_qk: bool = False,
    page_qk_pages_per_block: int = 4, attention_fusion: str = "none", moe_fusion: str = "none",
) -> Any:
    """Compose the retained B1/TP2 operators without changing their arithmetic."""
    attention = make_attention_decode(
        1, cache_blocks, with_residual=not first, native_fia=True,
        fuse_qkv_norm_rope=fuse_qkv_norm_rope, fuse_page_qk=fuse_page_qk,
        page_qk_pages_per_block=page_qk_pages_per_block, attention_fusion=attention_fusion,
        input_tp_reduction=not first, inline=True, tp_arguments=True,
    )
    moe = make_post_norm_moe(1, tp_reduction=True, expert_tp_reduction=last, moe_fusion=moe_fusion, inline=True)

    @pl.jit
    def decode_layer(
        x: pl.Tensor[[1, HIDDEN], pl.BF16],
        residual: pl.Tensor[[1, HIDDEN], pl.BF16],
        norm_weight: pl.Tensor[[1, HIDDEN], pl.BF16],
        qkv_weight: pl.Tensor[[1, HIDDEN, QKV_WIDTH], pl.INT8, pl.NZ],
        qkv_scale: pl.Tensor[[1, QKV_WIDTH], pl.FP32],
        q_weight: pl.Tensor[[1, HEAD_DIM], pl.BF16],
        k_weight: pl.Tensor[[1, HEAD_DIM], pl.BF16],
        positions: pl.Tensor[[1, 1], pl.INT64],
        cos_sin_cache: pl.Tensor[[512, HEAD_DIM], pl.BF16],
        slots: pl.Tensor[[1, 1], pl.INT32],
        block_table: pl.Tensor[[1, 4], pl.INT32],
        lengths: pl.Tensor[[1, 1], pl.INT32],
        key_cache: pl.InOut[pl.Tensor[[cache_blocks, CACHE_BLOCK_WIDTH], pl.BF16]],
        value_cache: pl.InOut[pl.Tensor[[cache_blocks, CACHE_BLOCK_WIDTH], pl.BF16]],
        o_weight: pl.Tensor[[1, HIDDEN, HIDDEN], pl.INT8, pl.NZ],
        o_scale: pl.Tensor[[1, HIDDEN], pl.FP32],
        attention_y: pl.Out[pl.Tensor[[1, HIDDEN], pl.BF16]],
        attention_residual_out: pl.Out[pl.Tensor[[1, HIDDEN], pl.BF16]],
        post_norm_weight: pl.Tensor[[1, HIDDEN], pl.BF16],
        gate: pl.Tensor[[EXPERTS, HIDDEN], pl.BF16],
        w13: pl.Tensor[[EXPERTS, HIDDEN, W13_WIDTH], pl.INT8, pl.NZ],
        w13_scale: pl.Tensor[[EXPERTS, W13_WIDTH], pl.FP32],
        w2: pl.Tensor[[EXPERTS, LOCAL_INTERMEDIATE, HIDDEN], pl.INT8, pl.NZ],
        w2_scale: pl.Tensor[[EXPERTS, HIDDEN], pl.FP32],
        y: pl.Out[pl.Tensor[[1, HIDDEN], pl.BF16]],
        residual_out: pl.Out[pl.Tensor[[1, HIDDEN], pl.BF16]],
        data: pl.InOut[pld.DistributedTensor[[TP_OPERATIONS, HIDDEN], pl.BF16]],
        signal: pl.InOut[pld.DistributedTensor[[TP_OPERATIONS * 4, SIGNAL_TILE], pl.INT32]],
        epochs: pl.InOut[pl.Tensor[[TP_OPERATIONS], pl.INT32]],
        operation: pl.Scalar[pl.INT32],
    ) -> tuple[pl.Tensor, pl.Tensor, pl.Tensor, pl.Tensor, pl.Tensor, pl.Tensor]:
        input_operation = operation - 1
        att_after, skip_after, key_after, value_after = attention(
            x, residual, norm_weight, qkv_weight, qkv_scale, q_weight, k_weight,
            positions, cos_sin_cache, slots, block_table, lengths, key_cache, value_cache,
            o_weight, o_scale, attention_y, attention_residual_out,
            data, signal, epochs, input_operation,
        )
        y_after, residual_after = moe(
            attention_y, attention_residual_out, post_norm_weight, gate, w13, w13_scale, w2, w2_scale, y, residual_out,
            data, signal, epochs, operation,
        )
        return y_after, residual_after, att_after, skip_after, key_after, value_after

    return decode_layer
