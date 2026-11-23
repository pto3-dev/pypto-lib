# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -------------------------------------------------------------------------------------------------------
"""Rank-local Decode attention in one callable with fixed context-capacity buckets.

Consumes borrowed native NZ weights and BSND cache views. Native TP reduction
remains outside. Input/output residual buffers must not alias: normalization
rereads its original inputs after emitting the residual output.
"""

from typing import Any

import pypto.language as pl
import pypto.language.distributed as pld

from .config import HIDDEN
from .kv_cache import CACHE_BLOCK_WIDTH, append_kv_cache_impl
from .linear import HEAD_DIM, KV_WIDTH, Q_WIDTH, QKV_WIDTH
from .norm import make_rms_norm_hidden_impl
from .norm_quant_bf16 import make_bf16_add_rms_norm_quant
from .norm_quant import add_rms_quant_hidden_impl, add_rms_quant_retained_impl, add_rms_quant_parallel_impl
from .paged_attention import make_paged_attention
from .paged_attention_draft import make_paged_attention as make_fixed_attention
from .paged_attention_draft import make_paged_attention_quant, make_page_qk
from .projection_output import make_projection_output
from .qk_rope import make_qk_norm_rope
from .qkv_norm_rope import make_qkv_norm_rope
from .quant import dynamic_quantize_hidden, make_dynamic_quantize_hidden_retained, dynamic_quantize_hidden_tiled
from .tp_norm import SIGNAL_TILE, TP_OPERATIONS, tp_add_rms_quant_hidden_impl


def make_attention_decode(
    rows: int,
    cache_blocks: int,
    *,
    with_residual: bool,
    cache_positions: int = 512,
    max_context: int = 512,
    direct_pages: bool | None = None,
    active_chunks: bool | None = None,
    pv_cols: int = 128,
    retain_normalized: bool = True,
    retain_quantized_input: bool = True,
    row_parallel_norm: bool | None = None,
    tiled_quant: bool | None = None,
    vector_projection_epilogue: bool | None = None,
    native_fia: bool = False,
    fuse_fia_pv_normalize: bool | None = None,
    fuse_qkv_norm_rope: bool = False,
    fuse_kv_append: bool | None = None,
    fuse_page_qk: bool = False,
    page_qk_pages_per_block: int = 4,
    attention_fusion: str = "none",
    input_tp_reduction: bool = False,
    inline: bool = False,
    tp_arguments: bool = False,
) -> Any:
    """Compose a local partial; opt-in native FIA currently supports B1/512 only."""
    if not 1 <= rows <= 16:
        raise ValueError(f"Unsupported attention bucket: {rows}")
    if input_tp_reduction and (rows != 1 or not with_residual or not native_fia):
        raise ValueError("Input TP reduction requires rows=1, with_residual=True and native_fia=True")
    if tp_arguments and (rows != 1 or not native_fia):
        raise ValueError("Communication arguments require rows=1 and native_fia=True")
    if cache_blocks < 1 or cache_positions < max_context:
        raise ValueError("Attention requires positive cache blocks and enough RoPE positions for its context")
    if native_fia and (rows != 1 or max_context != 512):
        raise ValueError("Native FIA composition currently requires rows=1 and max_context=512")
    if fuse_qkv_norm_rope and rows != 1:
        raise ValueError(f"Fused QKV Norm/RoPE currently requires rows=1, got {rows}")
    if fuse_kv_append is None:
        fuse_kv_append = fuse_qkv_norm_rope
    if fuse_kv_append and not fuse_qkv_norm_rope:
        raise ValueError("Fused KV append requires fused QKV Norm/RoPE")
    if fuse_page_qk and not native_fia:
        raise ValueError("Fused page QK requires native_fia=True")
    if attention_fusion not in ("none", "softmax_pv", "qk_softmax_pv", "softmax_pv_quant"):
        raise ValueError(f"Unsupported Attention fusion mode: {attention_fusion}")
    if attention_fusion != "none" and not native_fia:
        raise ValueError("Attention fusion requires native_fia=True")
    if fuse_fia_pv_normalize is None:
        fuse_fia_pv_normalize = native_fia and pv_cols == 128
    if fuse_fia_pv_normalize and not native_fia:
        raise ValueError("Fused FIA PV normalization requires native_fia=True")
    # Extra scale packing pays off at the qualified larger composed buckets.
    # Leave smaller and unmeasured buckets on the retained serial epilogue.
    if row_parallel_norm is None:
        row_parallel_norm = retain_normalized and rows in (8, 16)
    if row_parallel_norm and not retain_normalized:
        raise ValueError("Row-parallel norm requires the retained epilogue")
    qkv_project = make_projection_output(
        QKV_WIDTH, rows, inline=True, vector_epilogue=vector_projection_epilogue
    )
    output_project = make_projection_output(
        HIDDEN, rows, inline=True, vector_epilogue=vector_projection_epilogue
    )
    norm_rope = make_qk_norm_rope(fused=True, inline=True, diagnostics=False)
    fused_projection = make_qkv_norm_rope(
        cache_positions, inline=True, cache_blocks=cache_blocks, append_cache=fuse_kv_append,
    )
    page_qk_impl = make_page_qk(1, pages_per_block=page_qk_pages_per_block)
    if native_fia:
        adaptive_attention = make_fixed_attention(
            rows, chunk=512, sequential_sum=True, inline=True, pv_cols=pv_cols,
            fuse_pv_normalize=fuse_fia_pv_normalize,
            fuse_page_qk=fuse_page_qk,
            page_qk_pages_per_block=page_qk_pages_per_block,
            fuse_softmax_pv=attention_fusion == "softmax_pv",
            fuse_qk_softmax_pv=attention_fusion == "qk_softmax_pv",
            page_qk_impl=page_qk_impl,
        )
    else:
        adaptive_attention = make_paged_attention(
            rows,
            fold_sum=True,
            inline=True,
            pv_cols=pv_cols,
            max_context=max_context,
            direct_pages=direct_pages,
            active_chunks=active_chunks,
        )
    fuse_attention_quant = attention_fusion == "softmax_pv_quant"
    attention_quant = make_paged_attention_quant(1, page_qk_impl=page_qk_impl)
    norm_quant = add_rms_quant_retained_impl if retain_normalized else add_rms_quant_hidden_impl
    if row_parallel_norm:
        norm_quant = add_rms_quant_parallel_impl
    normalize = make_rms_norm_hidden_impl(early=rows == 1)
    quantize = make_dynamic_quantize_hidden_retained(early=rows == 1)
    if not retain_quantized_input:
        quantize = dynamic_quantize_hidden
    if tiled_quant is None:
        tiled_quant = retain_quantized_input and rows == 16
    if tiled_quant:
        if not retain_quantized_input:
            raise ValueError("Tiled quant requires retained quantized input")
        quantize = dynamic_quantize_hidden_tiled

    if with_residual and native_fia:
        bf16_norm_quant = make_bf16_add_rms_norm_quant(rows, fused=True, inline=True, early=rows == 1)

        @pl.jit.inline
        def input_norm_quant(
            x: pl.Tensor[[rows, HIDDEN], pl.BF16],
            residual: pl.Tensor[[rows, HIDDEN], pl.BF16],
            weight: pl.Tensor[[1, HIDDEN], pl.BF16],
            quant: pl.Tensor[[rows, HIDDEN], pl.INT8],
            scale: pl.Tensor[[rows, 1], pl.FP32],
            skip: pl.Tensor[[rows, HIDDEN], pl.BF16],
        ) -> tuple[pl.Tensor, pl.Tensor, pl.Tensor]:
            normalized = pl.create_tensor([rows, HIDDEN], dtype=pl.BF16)
            _norm_after, skip_after, quant_after, scale_after = bf16_norm_quant(
                x, residual, weight, normalized, skip, quant, scale
            )
            return quant_after, scale_after, skip_after
    elif with_residual:

        @pl.jit.inline
        def input_norm_quant(
            x: pl.Tensor[[rows, HIDDEN], pl.BF16],
            residual: pl.Tensor[[rows, HIDDEN], pl.BF16],
            weight: pl.Tensor[[1, HIDDEN], pl.BF16],
            quant: pl.Tensor[[rows, HIDDEN], pl.INT8],
            scale: pl.Tensor[[rows, 1], pl.FP32],
            skip: pl.Tensor[[rows, HIDDEN], pl.BF16],
        ) -> tuple[pl.Tensor, pl.Tensor, pl.Tensor]:
            quant_after, scale_after, skip_after = norm_quant(x, residual, weight, quant, scale, skip)
            return quant_after, scale_after, skip_after
    else:

        @pl.jit.inline
        def input_norm_quant(
            x: pl.Tensor[[rows, HIDDEN], pl.BF16],
            residual: pl.Tensor[[rows, HIDDEN], pl.BF16],
            weight: pl.Tensor[[1, HIDDEN], pl.BF16],
            quant: pl.Tensor[[rows, HIDDEN], pl.INT8],
            scale: pl.Tensor[[rows, 1], pl.FP32],
            skip: pl.Tensor[[rows, HIDDEN], pl.BF16],
        ) -> tuple[pl.Tensor, pl.Tensor, pl.Tensor]:
            normalized = pl.create_tensor([rows, HIDDEN], dtype=pl.BF16)
            norm_after = normalize(x, weight, normalized)
            quant_after, scale_after = quantize(norm_after, quant, scale)
            for _copy in pl.spmd(1, name_hint="first_layer_residual"):
                skip = pl.assemble(skip, pl.slice(x, [rows, HIDDEN], [0, 0]), [0, 0])
            return quant_after, scale_after, skip

    @pl.jit.inline
    def attention_after_input(
        quant_x: pl.Tensor[[rows, HIDDEN], pl.INT8],
        scale_x: pl.Tensor[[rows, 1], pl.FP32],
        qkv_weight: pl.Tensor[[1, HIDDEN, QKV_WIDTH], pl.INT8, pl.NZ],
        qkv_scale: pl.Tensor[[1, QKV_WIDTH], pl.FP32],
        q_weight: pl.Tensor[[1, HEAD_DIM], pl.BF16],
        k_weight: pl.Tensor[[1, HEAD_DIM], pl.BF16],
        positions: pl.Tensor[[rows, 1], pl.INT64],
        cos_sin_cache: pl.Tensor[[cache_positions, HEAD_DIM], pl.BF16],
        slots: pl.Tensor[[rows, 1], pl.INT32],
        block_table: pl.Tensor[[rows, max_context // 128], pl.INT32],
        lengths: pl.Tensor[[rows, 1], pl.INT32],
        key_cache: pl.Tensor[[cache_blocks, CACHE_BLOCK_WIDTH], pl.BF16],
        value_cache: pl.Tensor[[cache_blocks, CACHE_BLOCK_WIDTH], pl.BF16],
        o_weight: pl.Tensor[[1, HIDDEN, HIDDEN], pl.INT8, pl.NZ],
        o_scale: pl.Tensor[[1, HIDDEN], pl.FP32],
        y: pl.Tensor[[rows, HIDDEN], pl.BF16],
        residual_out: pl.Tensor[[rows, HIDDEN], pl.BF16],
    ) -> tuple[pl.Tensor, pl.Tensor, pl.Tensor, pl.Tensor]:
        q = pl.create_tensor([rows * 16, HEAD_DIM], dtype=pl.BF16)
        k = pl.create_tensor([rows * 2, HEAD_DIM], dtype=pl.BF16)
        v = pl.create_tensor([rows * 2, HEAD_DIM], dtype=pl.BF16)
        if fuse_qkv_norm_rope:
            _q_after, _k_after, _v_after, _key_after, _value_after = fused_projection(
                quant_x, scale_x, qkv_weight, qkv_scale, q_weight, k_weight, positions, cos_sin_cache,
                q, k, v, slots, key_cache, value_cache,
            )
        else:
            qkv = pl.create_tensor([rows, QKV_WIDTH], dtype=pl.BF16)
            _qkv_after = qkv_project(quant_x, scale_x, qkv_weight, qkv_scale, qkv)
            qkv_heads = pl.reshape(qkv, [rows * 20, HEAD_DIM])
            q_norm = pl.create_tensor([rows * 16, HEAD_DIM], dtype=pl.BF16)
            k_norm = pl.create_tensor([rows * 2, HEAD_DIM], dtype=pl.BF16)
            _q_norm_after, _k_norm_after, _q_after, _k_after, _v_after = norm_rope(
                qkv_heads, q_weight, k_weight, positions, cos_sin_cache, q_norm, k_norm, q, k, v
            )
        if not fuse_kv_append:
            k_tokens = pl.reshape(k, [rows, KV_WIDTH])
            v_tokens = pl.reshape(v, [rows, KV_WIDTH])
            _key_after, _value_after = append_kv_cache_impl(k_tokens, v_tokens, slots, key_cache, value_cache)
        key_tokens = pl.reshape(key_cache, [cache_blocks * 128, KV_WIDTH])
        value_tokens = pl.reshape(value_cache, [cache_blocks * 128, KV_WIDTH])
        context = pl.create_tensor([rows * 16, HEAD_DIM], dtype=pl.BF16)
        quant_o = pl.create_tensor([rows, HIDDEN], dtype=pl.INT8)
        scale_o = pl.create_tensor([rows, 1], dtype=pl.FP32)
        if fuse_attention_quant:
            _quant_o_after, _scale_o_after = attention_quant(
                q, key_tokens, value_tokens, block_table, lengths, quant_o, scale_o
            )
        else:
            _context_after = adaptive_attention(q, key_tokens, value_tokens, block_table, lengths, context)
            context_tokens = pl.reshape(context, [rows, Q_WIDTH])
            _quant_o_after, _scale_o_after = quantize(context_tokens, quant_o, scale_o)
        y_after = output_project(quant_o, scale_o, o_weight, o_scale, y)
        return y_after, residual_out, key_cache, value_cache

    decorator = pl.jit.inline if inline else pl.jit
    if input_tp_reduction or tp_arguments:

        @decorator
        def tp_attention_decode(
            x: pl.Tensor[[rows, HIDDEN], pl.BF16],
            residual: pl.Tensor[[rows, HIDDEN], pl.BF16],
            norm_weight: pl.Tensor[[1, HIDDEN], pl.BF16],
            qkv_weight: pl.Tensor[[1, HIDDEN, QKV_WIDTH], pl.INT8, pl.NZ],
            qkv_scale: pl.Tensor[[1, QKV_WIDTH], pl.FP32],
            q_weight: pl.Tensor[[1, HEAD_DIM], pl.BF16],
            k_weight: pl.Tensor[[1, HEAD_DIM], pl.BF16],
            positions: pl.Tensor[[rows, 1], pl.INT64],
            cos_sin_cache: pl.Tensor[[cache_positions, HEAD_DIM], pl.BF16],
            slots: pl.Tensor[[rows, 1], pl.INT32],
            block_table: pl.Tensor[[rows, max_context // 128], pl.INT32],
            lengths: pl.Tensor[[rows, 1], pl.INT32],
            key_cache: pl.InOut[pl.Tensor[[cache_blocks, CACHE_BLOCK_WIDTH], pl.BF16]],
            value_cache: pl.InOut[pl.Tensor[[cache_blocks, CACHE_BLOCK_WIDTH], pl.BF16]],
            o_weight: pl.Tensor[[1, HIDDEN, HIDDEN], pl.INT8, pl.NZ],
            o_scale: pl.Tensor[[1, HIDDEN], pl.FP32],
            y: pl.Out[pl.Tensor[[rows, HIDDEN], pl.BF16]],
            residual_out: pl.Out[pl.Tensor[[rows, HIDDEN], pl.BF16]],
            data: pl.InOut[pld.DistributedTensor[[TP_OPERATIONS, HIDDEN], pl.BF16]],
            signal: pl.InOut[pld.DistributedTensor[[TP_OPERATIONS * 4, SIGNAL_TILE], pl.INT32]],
            epochs: pl.InOut[pl.Tensor[[TP_OPERATIONS], pl.INT32]],
            operation: pl.Scalar[pl.INT32],
        ) -> tuple[pl.Tensor, pl.Tensor, pl.Tensor, pl.Tensor]:
            reduced = pl.create_tensor([rows, HIDDEN], dtype=pl.BF16)
            normalized = pl.create_tensor([rows, HIDDEN], dtype=pl.BF16)
            quant_x = pl.create_tensor([rows, HIDDEN], dtype=pl.INT8)
            scale_x = pl.create_tensor([rows, 1], dtype=pl.FP32)
            if input_tp_reduction:
                _reduced, _normalized, skip_after, quant_after, scale_after = tp_add_rms_quant_hidden_impl(
                    x, residual, norm_weight, reduced, normalized, residual_out, quant_x, scale_x,
                    data, signal, epochs, operation,
                )
            else:
                quant_after, scale_after, skip_after = input_norm_quant(
                    x, residual, norm_weight, quant_x, scale_x, residual_out,
                )
            return attention_after_input(
                quant_after, scale_after, qkv_weight, qkv_scale, q_weight, k_weight,
                positions, cos_sin_cache, slots, block_table, lengths, key_cache, value_cache,
                o_weight, o_scale, y, skip_after,
            )

        return tp_attention_decode

    @decorator
    def attention_decode(
        x: pl.Tensor[[rows, HIDDEN], pl.BF16],
        residual: pl.Tensor[[rows, HIDDEN], pl.BF16],
        norm_weight: pl.Tensor[[1, HIDDEN], pl.BF16],
        qkv_weight: pl.Tensor[[1, HIDDEN, QKV_WIDTH], pl.INT8, pl.NZ],
        qkv_scale: pl.Tensor[[1, QKV_WIDTH], pl.FP32],
        q_weight: pl.Tensor[[1, HEAD_DIM], pl.BF16],
        k_weight: pl.Tensor[[1, HEAD_DIM], pl.BF16],
        positions: pl.Tensor[[rows, 1], pl.INT64],
        cos_sin_cache: pl.Tensor[[cache_positions, HEAD_DIM], pl.BF16],
        slots: pl.Tensor[[rows, 1], pl.INT32],
        block_table: pl.Tensor[[rows, max_context // 128], pl.INT32],
        lengths: pl.Tensor[[rows, 1], pl.INT32],
        key_cache: pl.InOut[pl.Tensor[[cache_blocks, CACHE_BLOCK_WIDTH], pl.BF16]],
        value_cache: pl.InOut[pl.Tensor[[cache_blocks, CACHE_BLOCK_WIDTH], pl.BF16]],
        o_weight: pl.Tensor[[1, HIDDEN, HIDDEN], pl.INT8, pl.NZ],
        o_scale: pl.Tensor[[1, HIDDEN], pl.FP32],
        y: pl.Out[pl.Tensor[[rows, HIDDEN], pl.BF16]],
        residual_out: pl.Out[pl.Tensor[[rows, HIDDEN], pl.BF16]],
    ) -> tuple[pl.Tensor, pl.Tensor, pl.Tensor, pl.Tensor]:
        quant_x = pl.create_tensor([rows, HIDDEN], dtype=pl.INT8)
        scale_x = pl.create_tensor([rows, 1], dtype=pl.FP32)
        quant_after, scale_after, skip_after = input_norm_quant(
            x, residual, norm_weight, quant_x, scale_x, residual_out
        )
        return attention_after_input(
            quant_after, scale_after, qkv_weight, qkv_scale, q_weight, k_weight,
            positions, cos_sin_cache, slots, block_table, lengths, key_cache, value_cache,
            o_weight, o_scale, y, skip_after,
        )

    return attention_decode
