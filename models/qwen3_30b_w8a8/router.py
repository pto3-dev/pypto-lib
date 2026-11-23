# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -------------------------------------------------------------------------------------------------------
"""Unquantized router projection and top-8 with BF16 consumer weight semantics."""

import pypto.language as pl
from typing import Any
from pathlib import Path

from .config import EXPERTS, HIDDEN, TOP_K
from .quant import TOKENS_DYN


def make_router_projection(rows: int, k_tile: int = 32) -> Any:
    """Specialize a graph bucket without padded-row metadata."""
    if not 1 <= rows <= 16:
        raise ValueError(f"Unsupported router bucket: {rows}")
    if k_tile in (32, 64):
        return make_compensated_router(rows, partial_k=k_tile)
    if k_tile not in (128, 256, 512, 1024):
        raise ValueError(f"Unsupported router K tile: {k_tile}")
    stages = 1 if k_tile == 1024 else 2

    @pl.jit
    def projection(
        x: pl.Tensor[[rows, HIDDEN], pl.BF16],
        weight: pl.Tensor[[EXPERTS, HIDDEN], pl.BF16],
        logits: pl.Out[pl.Tensor[[rows, EXPERTS], pl.BF16]],
    ) -> pl.Tensor:
        for _block in pl.spmd(1, name_hint="router_projection"):
            acc = pl.create_tensor([rows, EXPERTS], dtype=pl.FP32)
            for kb in pl.pipeline(HIDDEN // k_tile, stage=stages):
                k0 = kb * k_tile
                acc = pl.matmul_acc(
                    acc,
                    x[:, k0 : k0 + k_tile],
                    weight[:, k0 : k0 + k_tile],
                    b_trans=True,
                    init_cond=(kb == 0),
                )
            logits[:, :] = pl.cast(acc, pl.BF16, mode="rint")
        return logits

    return projection


def _make_native_b1_router(*, inline: bool) -> Any:
    """Compute the CANN 8.5.1 / 910B4 BF16 B1 Router projection."""
    decorator = pl.jit.inline if inline else pl.jit

    @decorator
    def router_native_projection(
        x: pl.Tensor[[1, HIDDEN], pl.BF16],
        weight: pl.Tensor[[EXPERTS, HIDDEN], pl.BF16],
        logits: pl.Out[pl.Tensor[[1, EXPERTS], pl.BF16]],
    ) -> pl.Tensor:
        # MatMulV2 tiling98513: N16, L0 K128, L1 K512, eight output partitions.
        for block in pl.spmd(EXPERTS // 16, name_hint="router_native_projection", allow_early_resolve=True):
            col0 = block * 16
            acc = pl.create_tensor([1, 16], dtype=pl.FP32)
            for kb in pl.pipeline(HIDDEN // 128, stage=2):
                group = kb // 4
                if block % 2 == 1:
                    group = HIDDEN // 512 - 1 - group
                group = (group + block // 2) % (HIDDEN // 512)
                k0 = group * 512 + (kb % 4) * 128
                lhs = x[:, k0 : k0 + 128]
                rhs = weight[col0 : col0 + 16, k0 : k0 + 128]
                acc = pl.matmul_acc(acc, lhs, rhs, b_trans=True, init_cond=(kb == 0))
            logits[:, col0 : col0 + 16] = acc
        return logits

    return router_native_projection


def make_compensated_router(
    rows: int, *, inline: bool = False, partial_k: int = 32, sum_rows: int | None = None,
    sum_stages: int = 1, partials_per_block: int = 1,
) -> Any:
    """Use native B1/K32 arithmetic, or compensated partials for other buckets.

    The factory name and tuning arguments remain compatible with existing
    callers. Sum partition/stage/group controls apply only to the compensated
    branch; B1/K32 follows the pinned native cube traversal instead.
    """
    if not 1 <= rows <= 16:
        raise ValueError(f"Unsupported router bucket: {rows}")
    if partial_k not in (32, 64):
        raise ValueError(f"Unsupported compensated router K tile: {partial_k}")
    if sum_stages not in (1, 2):
        raise ValueError(f"Unsupported router sum stages: {sum_stages}")
    if partials_per_block not in (1, 2, 4, 8, 16):
        raise ValueError(f"Unsupported router partials per block: {partials_per_block}")
    if sum_rows is not None and (sum_rows not in (1, 2, 4, 8, 16) or rows % min(sum_rows, rows)):
        raise ValueError(f"Unsupported router sum rows {sum_rows} for bucket {rows}")
    # Only the B16/K32 composed chain has passed the partition performance gate.
    if sum_rows is None:
        sum_rows = 4 if rows == 16 and partial_k == 32 else rows
    else:
        sum_rows = min(rows, sum_rows)
    groups = HIDDEN // partial_k
    early = rows == 1 and partial_k == 32
    if early:
        return _make_native_b1_router(inline=inline)

    decorator = pl.jit.inline if inline else pl.jit

    @decorator
    def compensated(
        x: pl.Tensor[[rows, HIDDEN], pl.BF16],
        weight: pl.Tensor[[EXPERTS, HIDDEN], pl.BF16],
        logits: pl.Out[pl.Tensor[[rows, EXPERTS], pl.BF16]],
    ) -> pl.Tensor:
        partials = pl.create_tensor([rows, groups * EXPERTS], dtype=pl.FP32)
        if partials_per_block == 1:
            for block in pl.spmd(groups, name_hint="router_partials", allow_early_resolve=early):
                k0 = block * partial_k
                partials[:, block * EXPERTS : (block + 1) * EXPERTS] = pl.matmul(
                    pl.slice(x, [rows, partial_k], [0, k0]),
                    pl.slice(weight, [EXPERTS, partial_k], [0, k0]),
                    b_trans=True,
                    out_dtype=pl.FP32,
                )
        else:
            for block in pl.spmd(groups // partials_per_block, name_hint="router_partials_grouped", allow_early_resolve=early):
                for group in pl.range(block * partials_per_block, (block + 1) * partials_per_block):
                    k0 = group * partial_k
                    partials[:, group * EXPERTS : (group + 1) * EXPERTS] = pl.matmul(
                        pl.slice(x, [rows, partial_k], [0, k0]),
                        pl.slice(weight, [EXPERTS, partial_k], [0, k0]),
                        b_trans=True,
                        out_dtype=pl.FP32,
                    )
        for row_block in pl.spmd(rows // sum_rows, name_hint="router_twosum", allow_early_resolve=early):
            row0 = row_block * sum_rows
            total = pl.full([sum_rows, EXPERTS], dtype=pl.FP32, value=0.0)
            low = pl.full([sum_rows, EXPERTS], dtype=pl.FP32, value=0.0)
            if sum_stages == 1:
                # Keep the original control's loop and reduction order unchanged.
                for group in pl.range(groups):
                    part = pl.slice(partials, [sum_rows, EXPERTS], [row0, group * EXPERTS])
                    next_total = pl.add(total, part)
                    part_virtual = pl.sub(next_total, total)
                    total_virtual = pl.sub(next_total, part_virtual)
                    error_total = pl.sub(total, total_virtual)
                    error_part = pl.sub(part, part_virtual)
                    low = pl.add(low, pl.add(error_total, error_part))
                    total = next_total
            else:
                # Pipeline independent loads, never reassociate the TwoSum recurrence.
                for group in pl.pipeline(groups, stage=sum_stages):
                    part = pl.slice(partials, [sum_rows, EXPERTS], [row0, group * EXPERTS])
                    next_total = pl.add(total, part)
                    part_virtual = pl.sub(next_total, total)
                    total_virtual = pl.sub(next_total, part_virtual)
                    error_total = pl.sub(total, total_virtual)
                    error_part = pl.sub(part, part_virtual)
                    low = pl.add(low, pl.add(error_total, error_part))
                    total = next_total
            logits[row0 : row0 + sum_rows, :] = pl.cast(pl.add(total, low), pl.BF16, mode="rint")
        return logits

    return compensated


router_projection = make_router_projection(16)


def make_router_topk_impl(*, early: bool = False) -> Any:
    """Specialize the producer-side scheduling policy."""
    @pl.jit.inline
    def router_topk_impl(
        logits: pl.Tensor[[TOKENS_DYN, EXPERTS], pl.BF16],
        ids: pl.Tensor[[TOKENS_DYN, TOP_K], pl.INT32],
        weights: pl.Tensor[[TOKENS_DYN, TOP_K], pl.FP32],
    ) -> tuple[pl.Tensor, pl.Tensor]:
        tokens = pl.tensor.dim(logits, 0)
        for token in pl.spmd(tokens, name_hint="router_topk", allow_early_resolve=early):
            row = pl.cast(logits[token : token + 1, :], pl.FP32)
            rows = pl.col_expand_mul(pl.full([8, EXPERTS], dtype=pl.FP32, value=1.0), row)
            shifted = pl.row_expand_sub(rows, pl.row_max(rows))
            exp = pl.exp(shifted)
            probabilities = pl.row_expand_div(exp, pl.row_sum(exp))
            ramp = pl.arange(0, [1, EXPERTS], dtype=pl.UINT32)
            sorted_pairs = pl.mrgsort(pl.sort32(probabilities[:1, :], ramp), block_len=64)
            pairs = sorted_pairs[:, :16]
            selected_ids = pl.gather(pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
            selected = pl.gather(pairs, mask_pattern=pl.tile.MaskPattern.P0101, output_dtype=pl.FP32)
            selected_rows = pl.col_expand_mul(pl.full([8, TOP_K], dtype=pl.FP32, value=1.0), selected)
            normalized = pl.row_expand_div(selected_rows, pl.row_sum(selected_rows))
            padded = pl.full([1, 16], dtype=pl.FP32, value=0.0)
            for column in pl.range(TOP_K):
                pl.write(padded, [0, column], pl.read(normalized, [0, column]))
            rounded = pl.cast(pl.cast(padded, pl.BF16, mode="rint"), pl.FP32)
            ids[token : token + 1, :] = selected_ids
            weights = pl.assemble(weights, pl.set_validshape(rounded, 1, TOP_K), [token, 0])
        return ids, weights

    return router_topk_impl


router_topk_impl = make_router_topk_impl()


@pl.jit
def router_topk(
    logits: pl.Tensor[[TOKENS_DYN, EXPERTS], pl.BF16],
    ids: pl.Out[pl.Tensor[[TOKENS_DYN, TOP_K], pl.INT32]],
    weights: pl.Out[pl.Tensor[[TOKENS_DYN, TOP_K], pl.FP32]],
) -> tuple[pl.Tensor, pl.Tensor]:
    logits.bind_dynamic(0, TOKENS_DYN)
    ids.bind_dynamic(0, TOKENS_DYN)
    weights.bind_dynamic(0, TOKENS_DYN)
    return router_topk_impl(logits, ids, weights)


def compile_a1_router_projection(pto_isa_root: str | Path) -> bytes:
    """Compile B1 BF16 Router with native partition traversal and FP32 accumulation.

    Prepare eight blocks with BF16 x[1,2048], weight[128,2048], logits[1,128]
    and INT32 status[8,16]. Each block owns one status cache line. Status zero
    indicates completion; one indicates incompatible tensor geometry.
    """
    from simpler_setup import KernelCompiler

    source = Path(__file__).with_name("a1_router.cpp")
    return KernelCompiler("a1").compile_single_launch(str(source), str(pto_isa_root))


def compile_a1_router_topk(pto_isa_root: str | Path) -> bytes:
    """Compile B1 top-8 with BF16 consumer weights and stable equal-logit IDs.

    ABI: BF16 logits[1,128], INT32 ids[1,8], FP32 weights[1,8], INT32
    status[1,16]. The active consumer weights are rounded through BF16, as in
    the native expert reduction. One block; no CPU route selection on replay.
    Native-device tie ordering needs a separate provenance qualification.
    """
    from simpler_setup import KernelCompiler

    source = Path(__file__).with_name("a1_router_topk.cpp")
    return KernelCompiler("a1").compile_single_launch(str(source), str(pto_isa_root))
