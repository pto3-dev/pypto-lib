# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""projection_bf16 operators. Explicit PyPTO functions; no source generator at build time.

Migrated from the validated L1K2560 graph; tuning is audited separately.
"""
import pypto.language as pl
from qwen36.tuning import GDN_N,GDN_L1_K,GDN_L0_K,Z_N,Z_L1_K,Z_L0_K,HEAD_N,HEAD_L1_K,HEAD_L0_K
from qwen36.native_gemm_order import N_GROUP_SPAN,HALF_GROUPS,OUTER,OFFSET_SCALE,NO_SHIFT_BEGIN
from qwen36.native_head_order import HEAD_GROUP_SPAN,HEAD_OUTER,HEAD_HALF_GROUPS,HEAD_OFFSET_SCALE,HEAD_NO_SHIFT_BEGIN
from qwen36.native_head_small_order import SMALL_GROUP_SPAN,SMALL_OUTER,SMALL_HALF_GROUPS,SMALL_OFFSET_SCALE,SMALL_NO_SHIFT_BEGIN,SMALL_L1_K,SMALL_L0_K
from qwen36.native_gdn_b2_order import B2_N,B2_GROUP_SPAN,B2_HALF_GROUPS,B2_NO_SHIFT_BEGIN,B2_OUTER,B2_OFFSET_SCALE,B2_L1_K,B2_L0_K

@pl.function(type=pl.FunctionType.InCore)
def greedy_projection_spmd_kernel(x__ssa_v0: pl.Tensor[[24, 1536], pl.BF16], w__ssa_v0: pl.Tensor[[16, 5120, 1536], pl.BF16, pl.NZ], out__ssa_v0: pl.Out[pl.Tensor[[24, 5120], pl.FP32]], weight_group: pl.Scalar[pl.INT64]) -> pl.Tensor[[24, 5120], pl.FP32]:
    pl.func_attr({'static_block_mapping': True})
    for m0 in pl.range(0, 24, 32):
        vm = pl.min(32, 24 - m0)
        a0__ssa_v0 = pl.tile.load(x__ssa_v0, [m0, 0], [32, 1536], [vm, 1536], target_memory=pl.Mem.Mat)
        for group_index in pl.range(pl.tile.get_block_idx(), 20, pl.tile.get_block_num()):
            column_base = group_index * 256
            b0__ssa_v0 = pl.tile.reshape(pl.tile.load(w__ssa_v0, [weight_group, column_base + 0, 0], [1, 64, 1536], [1, 64, 1536], target_memory=pl.Mem.Mat, cache=1), [64, 1536])
            bt0__ssa_v0 = pl.tile.transpose_view(b0__ssa_v0)
            c0__ssa_v0 = pl.tile.matmul(a0__ssa_v0, bt0__ssa_v0)
            v0__ssa_v0 = pl.tile.set_validshape(c0__ssa_v0, vm, 64)
            out__ssa_v1 = pl.tile.store(v0__ssa_v0, [m0, column_base + 0], out__ssa_v0)
            b1__ssa_v0 = pl.tile.reshape(pl.tile.load(w__ssa_v0, [weight_group, column_base + 64, 0], [1, 64, 1536], [1, 64, 1536], target_memory=pl.Mem.Mat, cache=1), [64, 1536])
            bt1__ssa_v0 = pl.tile.transpose_view(b1__ssa_v0)
            c1__ssa_v0 = pl.tile.matmul(a0__ssa_v0, bt1__ssa_v0)
            v1__ssa_v0 = pl.tile.set_validshape(c1__ssa_v0, vm, 64)
            out__ssa_v2 = pl.tile.store(v1__ssa_v0, [m0, column_base + 64], out__ssa_v0)
            b2__ssa_v0 = pl.tile.reshape(pl.tile.load(w__ssa_v0, [weight_group, column_base + 128, 0], [1, 64, 1536], [1, 64, 1536], target_memory=pl.Mem.Mat, cache=1), [64, 1536])
            bt2__ssa_v0 = pl.tile.transpose_view(b2__ssa_v0)
            c2__ssa_v0 = pl.tile.matmul(a0__ssa_v0, bt2__ssa_v0)
            v2__ssa_v0 = pl.tile.set_validshape(c2__ssa_v0, vm, 64)
            out__ssa_v3 = pl.tile.store(v2__ssa_v0, [m0, column_base + 128], out__ssa_v0)
            b3__ssa_v0 = pl.tile.reshape(pl.tile.load(w__ssa_v0, [weight_group, column_base + 192, 0], [1, 64, 1536], [1, 64, 1536], target_memory=pl.Mem.Mat, cache=1), [64, 1536])
            bt3__ssa_v0 = pl.tile.transpose_view(b3__ssa_v0)
            c3__ssa_v0 = pl.tile.matmul(a0__ssa_v0, bt3__ssa_v0)
            v3__ssa_v0 = pl.tile.set_validshape(c3__ssa_v0, vm, 64)
            out__ssa_v4 = pl.tile.store(v3__ssa_v0, [m0, column_base + 192], out__ssa_v0)
            out__ssa_v0 = out__ssa_v4
    return out__ssa_v0

@pl.function(type=pl.FunctionType.Inline)
def project_attention_out(x: pl.Tensor[[24, 1536], pl.BF16], w: pl.Tensor[[16, 5120, 1536], pl.BF16, pl.NZ], out: pl.Out[pl.Tensor[[24, 5120], pl.FP32]], weight_group: pl.Scalar[pl.INT64]) -> pl.Tensor[[24, 5120], pl.FP32]:
    available = pl.system.available_cluster_count()
    width = pl.min(available, 20)
    with pl.spmd(width, allow_early_resolve=True):
        out = greedy_projection_spmd_kernel(x, w, out, weight_group)
    return out

@pl.function(type=pl.FunctionType.InCore)
def project_gdn_kernel(x: pl.Tensor[[24, 5120], pl.BF16], weight: pl.Tensor[[16, 4096, 5120], pl.BF16, pl.NZ], out: pl.Tensor[[24, 2560], pl.BF16], weight_group: pl.Scalar[pl.INT64], row_ends: pl.Tensor[[4], pl.INT32]) -> pl.Tensor[[24, 2560], pl.BF16]:
    pl.func_attr({'static_block_mapping': True})
    rows = 24
    core = pl.tile.get_block_idx()
    grid = pl.tile.get_block_num()
    jobs = 2560 // GDN_N
    native_rows = pl.read(row_ends, [0]) + pl.read(row_ends, [1]) + pl.read(row_ends, [2]) + pl.read(row_ends, [3]) - 36
    for m0 in pl.range(0, rows, 32):
        vm = pl.min(32, rows - m0)
        activation = pl.tile.load(x, [m0, 0], [32, 5120], valid_shape=[vm, 5120], target_memory=pl.Mem.Mat)
        if native_rows < 16:
            for small_job in pl.range(core, 2560 // B2_N, grid):
                small_n0 = small_job * B2_N
                small_group = small_n0 // B2_GROUP_SPAN
                small_enabled = (1 + small_group // B2_NO_SHIFT_BEGIN) % 2
                small_reverse = (small_group % 2) * small_enabled
                small_offset = B2_OFFSET_SCALE * ((small_group // 2) % B2_HALF_GROUPS) * small_enabled
                small_acc = pl.tile.create([32, B2_N], pl.FP32, target_memory=pl.Mem.Acc, compact=True)
                for small_step in pl.pipeline(0, B2_OUTER, 1, stage=2):
                    small_outer = ((1 + (B2_OUTER - 2) * small_reverse) * small_step + (B2_OUTER - 1) * small_reverse + small_offset) % B2_OUTER
                    small_k0 = small_outer * B2_L1_K
                    small_weight = pl.tile.reshape(pl.tile.load(weight, [weight_group, small_n0, small_k0], [1, B2_N, B2_L1_K], target_memory=pl.Mem.Mat, cache=pl.CachePolicy.BYPASS), [B2_N, B2_L1_K])
                    small_transposed = pl.tile.transpose_view(small_weight)
                    for small_kk in pl.pipeline(0, B2_L1_K, B2_L0_K, stage=2):
                        small_left = pl.tile.extract(activation, 0, small_k0 + small_kk, [32, B2_L0_K], target_memory=pl.Mem.Left)
                        small_right = pl.tile.extract(small_transposed, small_kk, 0, [B2_L0_K, B2_N], target_memory=pl.Mem.Right)
                        small_acc = pl.tile.matmul_acc(small_acc, small_left, small_right, init_cond=small_step == 0 and small_kk == 0)
                out = pl.tile.store(pl.tile.set_validshape(small_acc, vm, B2_N), [m0, small_n0], out)
        elif native_rows == 16:
            # Actual M16 key230083 retains K256/K64 but disables shifting.
            for aligned_job in pl.range(core, 2560 // B2_N, grid):
                aligned_n0 = aligned_job * B2_N
                aligned_acc = pl.tile.create([32, B2_N], pl.FP32, target_memory=pl.Mem.Acc, compact=True)
                for aligned_step in pl.pipeline(0, B2_OUTER, 1, stage=2):
                    aligned_k0 = aligned_step * B2_L1_K
                    aligned_weight = pl.tile.reshape(pl.tile.load(weight, [weight_group, aligned_n0, aligned_k0], [1, B2_N, B2_L1_K], target_memory=pl.Mem.Mat, cache=pl.CachePolicy.BYPASS), [B2_N, B2_L1_K])
                    aligned_transposed = pl.tile.transpose_view(aligned_weight)
                    for aligned_kk in pl.pipeline(0, B2_L1_K, B2_L0_K, stage=2):
                        aligned_left = pl.tile.extract(activation, 0, aligned_k0 + aligned_kk, [32, B2_L0_K], target_memory=pl.Mem.Left)
                        aligned_right = pl.tile.extract(aligned_transposed, aligned_kk, 0, [B2_L0_K, B2_N], target_memory=pl.Mem.Right)
                        aligned_acc = pl.tile.matmul_acc(aligned_acc, aligned_left, aligned_right, init_cond=aligned_step == 0 and aligned_kk == 0)
                out = pl.tile.store(pl.tile.set_validshape(aligned_acc, vm, B2_N), [m0, aligned_n0], out)
        else:
            for job in pl.range(core, jobs, grid):
                n0 = job * GDN_N
                logical_group = n0 // N_GROUP_SPAN
                enabled = (1 + logical_group // NO_SHIFT_BEGIN) % 2
                reverse = (logical_group % 2) * enabled
                offset = OFFSET_SCALE * ((logical_group // 2) % HALF_GROUPS) * enabled
                acc = pl.tile.create([32, GDN_N], pl.FP32, target_memory=pl.Mem.Acc, compact=True)
                for step in pl.pipeline(0, OUTER, 1, stage=2):
                    outer = ((1 + (OUTER - 2) * reverse) * step + (OUTER - 1) * reverse + offset) % OUTER
                    k0 = outer * GDN_L1_K
                    weight_tile = pl.tile.reshape(pl.tile.load(weight, [weight_group, n0, k0], [1, GDN_N, GDN_L1_K], target_memory=pl.Mem.Mat, cache=pl.CachePolicy.BYPASS), [GDN_N, GDN_L1_K])
                    transposed = pl.tile.transpose_view(weight_tile)
                    for kk in pl.pipeline(0, GDN_L1_K, GDN_L0_K, stage=2):
                        left = pl.tile.extract(activation, 0, k0 + kk, [32, GDN_L0_K], target_memory=pl.Mem.Left)
                        right = pl.tile.extract(transposed, kk, 0, [GDN_L0_K, GDN_N], target_memory=pl.Mem.Right)
                        acc = pl.tile.matmul_acc(acc, left, right, init_cond=step == 0 and kk == 0)
                out = pl.tile.store(pl.tile.set_validshape(acc, vm, GDN_N), [m0, n0], out)
    return out

@pl.function(type=pl.FunctionType.Inline)
def project_gdn(x: pl.Tensor[[24, 5120], pl.BF16], weight: pl.Tensor[[16, 4096, 5120], pl.BF16, pl.NZ], out: pl.Tensor[[24, 2560], pl.BF16], weight_group: pl.Scalar[pl.INT64], row_ends: pl.Tensor[[4], pl.INT32]) -> pl.Tensor[[24, 2560], pl.BF16]:
    rows = 24
    with pl.spmd(pl.system.available_cluster_count(), allow_early_resolve=True):
        out = project_gdn_kernel(x, weight, out, weight_group, row_ends)
    return out

@pl.function(type=pl.FunctionType.InCore)
def project_z_kernel(x: pl.Tensor[[24, 5120], pl.BF16], weight: pl.Tensor[[16, 4096, 5120], pl.BF16, pl.NZ], out: pl.Tensor[[24, 1536], pl.BF16], weight_group: pl.Scalar[pl.INT64], row_ends: pl.Tensor[[4], pl.INT32]) -> pl.Tensor[[24, 1536], pl.BF16]:
    pl.func_attr({'static_block_mapping': True})
    rows = 24
    core = pl.tile.get_block_idx()
    grid = pl.tile.get_block_num()
    jobs = 1536 // Z_N
    native_rows = pl.read(row_ends, [0]) + pl.read(row_ends, [1]) + pl.read(row_ends, [2]) + pl.read(row_ends, [3]) - 36
    for m0 in pl.range(0, rows, 32):
        vm = pl.min(32, rows - m0)
        activation = pl.tile.load(x, [m0, 0], [32, 5120], valid_shape=[vm, 5120], target_memory=pl.Mem.Mat)
        if native_rows < 16:
            for small_job in pl.range(core, 1536 // B2_N, grid):
                small_n0 = small_job * B2_N
                small_group = (2560 + small_n0) // B2_GROUP_SPAN
                small_enabled = (1 + small_group // B2_NO_SHIFT_BEGIN) % 2
                small_reverse = (small_group % 2) * small_enabled
                small_offset = B2_OFFSET_SCALE * ((small_group // 2) % B2_HALF_GROUPS) * small_enabled
                small_acc = pl.tile.create([32, B2_N], pl.FP32, target_memory=pl.Mem.Acc, compact=True)
                for small_step in pl.pipeline(0, B2_OUTER, 1, stage=2):
                    small_outer = ((1 + (B2_OUTER - 2) * small_reverse) * small_step + (B2_OUTER - 1) * small_reverse + small_offset) % B2_OUTER
                    small_k0 = small_outer * B2_L1_K
                    small_weight = pl.tile.reshape(pl.tile.load(weight, [weight_group, 2560 + small_n0, small_k0], [1, B2_N, B2_L1_K], target_memory=pl.Mem.Mat, cache=pl.CachePolicy.BYPASS), [B2_N, B2_L1_K])
                    small_transposed = pl.tile.transpose_view(small_weight)
                    for small_kk in pl.pipeline(0, B2_L1_K, B2_L0_K, stage=2):
                        small_left = pl.tile.extract(activation, 0, small_k0 + small_kk, [32, B2_L0_K], target_memory=pl.Mem.Left)
                        small_right = pl.tile.extract(small_transposed, small_kk, 0, [B2_L0_K, B2_N], target_memory=pl.Mem.Right)
                        small_acc = pl.tile.matmul_acc(small_acc, small_left, small_right, init_cond=small_step == 0 and small_kk == 0)
                out = pl.tile.store(pl.tile.set_validshape(small_acc, vm, B2_N), [m0, small_n0], out)
        elif native_rows == 16:
            for aligned_job in pl.range(core, 1536 // B2_N, grid):
                aligned_n0 = aligned_job * B2_N
                aligned_acc = pl.tile.create([32, B2_N], pl.FP32, target_memory=pl.Mem.Acc, compact=True)
                for aligned_step in pl.pipeline(0, B2_OUTER, 1, stage=2):
                    aligned_k0 = aligned_step * B2_L1_K
                    aligned_weight = pl.tile.reshape(pl.tile.load(weight, [weight_group, 2560 + aligned_n0, aligned_k0], [1, B2_N, B2_L1_K], target_memory=pl.Mem.Mat, cache=pl.CachePolicy.BYPASS), [B2_N, B2_L1_K])
                    aligned_transposed = pl.tile.transpose_view(aligned_weight)
                    for aligned_kk in pl.pipeline(0, B2_L1_K, B2_L0_K, stage=2):
                        aligned_left = pl.tile.extract(activation, 0, aligned_k0 + aligned_kk, [32, B2_L0_K], target_memory=pl.Mem.Left)
                        aligned_right = pl.tile.extract(aligned_transposed, aligned_kk, 0, [B2_L0_K, B2_N], target_memory=pl.Mem.Right)
                        aligned_acc = pl.tile.matmul_acc(aligned_acc, aligned_left, aligned_right, init_cond=aligned_step == 0 and aligned_kk == 0)
                out = pl.tile.store(pl.tile.set_validshape(aligned_acc, vm, B2_N), [m0, aligned_n0], out)
        else:
            for job in pl.range(core, jobs, grid):
                n0 = job * Z_N
                logical_group = (2560 + n0) // N_GROUP_SPAN
                enabled = (1 + logical_group // NO_SHIFT_BEGIN) % 2
                reverse = (logical_group % 2) * enabled
                offset = OFFSET_SCALE * ((logical_group // 2) % HALF_GROUPS) * enabled
                acc = pl.tile.create([32, Z_N], pl.FP32, target_memory=pl.Mem.Acc, compact=True)
                for step in pl.pipeline(0, OUTER, 1, stage=2):
                    outer = ((1 + (OUTER - 2) * reverse) * step + (OUTER - 1) * reverse + offset) % OUTER
                    k0 = outer * Z_L1_K
                    weight_tile = pl.tile.reshape(pl.tile.load(weight, [weight_group, 2560 + n0, k0], [1, Z_N, Z_L1_K], target_memory=pl.Mem.Mat, cache=pl.CachePolicy.BYPASS), [Z_N, Z_L1_K])
                    transposed = pl.tile.transpose_view(weight_tile)
                    for kk in pl.pipeline(0, Z_L1_K, Z_L0_K, stage=2):
                        left = pl.tile.extract(activation, 0, k0 + kk, [32, Z_L0_K], target_memory=pl.Mem.Left)
                        right = pl.tile.extract(transposed, kk, 0, [Z_L0_K, Z_N], target_memory=pl.Mem.Right)
                        acc = pl.tile.matmul_acc(acc, left, right, init_cond=step == 0 and kk == 0)
                out = pl.tile.store(pl.tile.set_validshape(acc, vm, Z_N), [m0, n0], out)
    return out

@pl.function(type=pl.FunctionType.Inline)
def project_z(x: pl.Tensor[[24, 5120], pl.BF16], weight: pl.Tensor[[16, 4096, 5120], pl.BF16, pl.NZ], out: pl.Tensor[[24, 1536], pl.BF16], weight_group: pl.Scalar[pl.INT64], row_ends: pl.Tensor[[4], pl.INT32]) -> pl.Tensor[[24, 1536], pl.BF16]:
    rows = 24
    with pl.spmd(pl.system.available_cluster_count(), allow_early_resolve=True):
        out = project_z_kernel(x, weight, out, weight_group, row_ends)
    return out

@pl.function(type=pl.FunctionType.InCore)
def project_up_kernel(x: pl.Tensor[[24, 5120], pl.BF16], weight: pl.Tensor[[4352, 5120], pl.BF16], out: pl.Tensor[[24, 4352], pl.BF16]) -> pl.Tensor[[24, 4352], pl.BF16]:
    pl.func_attr({'static_block_mapping': True})
    rows = 24
    core = pl.tile.get_block_idx()
    grid = pl.tile.get_block_num()
    jobs = 4352 // 64
    count = jobs // grid
    remainder = jobs % grid
    first = core * count + pl.min(core, remainder)
    last = (core + 1) * count + pl.min(core + 1, remainder)
    for m0 in pl.range(0, rows, 32):
        vm = pl.min(32, rows - m0)
        activation = pl.tile.load(x, [m0, 0], [32, 5120], valid_shape=[vm, 5120], target_memory=pl.Mem.Mat)
        for job in pl.range(first, last):
            n0 = job * 64
            acc = pl.tile.create([32, 64], pl.FP32, target_memory=pl.Mem.Acc, compact=True)
            for k0 in pl.pipeline(0, 5120, 256, stage=2):
                weight_tile = pl.tile.load(weight, [n0, k0], [64, 256], target_memory=pl.Mem.Mat, cache=pl.CachePolicy.BYPASS)
                transposed = pl.tile.transpose_view(weight_tile)
                for kk in pl.pipeline(0, 256, 256, stage=2):
                    left = pl.tile.extract(activation, 0, k0 + kk, [32, 256], target_memory=pl.Mem.Left)
                    right = pl.tile.extract(transposed, kk, 0, [256, 64], target_memory=pl.Mem.Right)
                    acc = pl.tile.matmul_acc(acc, left, right, init_cond=k0 == 0 and kk == 0)
            out = pl.tile.store(pl.tile.set_validshape(acc, vm, 64), [m0, n0], out)
    return out

@pl.function(type=pl.FunctionType.Inline)
def project_up(x: pl.Tensor[[24, 5120], pl.BF16], weight: pl.Tensor[[4352, 5120], pl.BF16], out: pl.Tensor[[24, 4352], pl.BF16]) -> pl.Tensor[[24, 4352], pl.BF16]:
    rows = 24
    with pl.spmd(pl.system.available_cluster_count(), allow_early_resolve=True):
        out = project_up_kernel(x, weight, out)
    return out

@pl.function(type=pl.FunctionType.InCore)
def project_head_kernel(x: pl.Tensor[[24, 5120], pl.BF16], weight: pl.Tensor[[62080, 5120], pl.BF16, pl.NZ], out: pl.Tensor[[24, 62080], pl.BF16], row_ends: pl.Tensor[[4], pl.INT32]) -> pl.Tensor[[24, 62080], pl.BF16]:
    pl.func_attr({'static_block_mapping': True})
    rows = 24
    core = pl.tile.get_block_idx()
    grid = pl.tile.get_block_num()
    jobs = 62080 // HEAD_N
    # Ends are r*6+valid_count, including inactive requests. Read on AIC on
    # every replay, not once during prepared graph construction on AICPU.
    native_rows = pl.read(row_ends, [0]) + pl.read(row_ends, [1]) + pl.read(row_ends, [2]) + pl.read(row_ends, [3]) - 36
    for m0 in pl.range(0, rows, 32):
        vm = pl.min(32, rows - m0)
        activation = pl.tile.load(x, [m0, 0], [32, 5120], valid_shape=[vm, 5120], target_memory=pl.Mem.Mat)
        for job in pl.range(core, jobs, grid):
            n0 = job * HEAD_N
            acc = pl.tile.create([32, HEAD_N], pl.FP32, target_memory=pl.Mem.Acc, compact=True)
            if native_rows < 16:
                small_group = n0 // SMALL_GROUP_SPAN
                small_enabled = (1 + small_group // SMALL_NO_SHIFT_BEGIN) % 2
                small_reverse = (small_group % 2) * small_enabled
                small_offset = SMALL_OFFSET_SCALE * ((small_group // 2) % SMALL_HALF_GROUPS) * small_enabled
                for small_step in pl.pipeline(0, SMALL_OUTER, 1, stage=2):
                    small_outer = ((1 + (SMALL_OUTER - 2) * small_reverse) * small_step + (SMALL_OUTER - 1) * small_reverse + small_offset) % SMALL_OUTER
                    small_k0 = small_outer * SMALL_L1_K
                    small_weight = pl.tile.load(weight, [n0, small_k0], [HEAD_N, SMALL_L1_K], target_memory=pl.Mem.Mat, cache=pl.CachePolicy.BYPASS)
                    small_transposed = pl.tile.transpose_view(small_weight)
                    for small_kk in pl.pipeline(0, SMALL_L1_K, SMALL_L0_K, stage=2):
                        small_left = pl.tile.extract(activation, 0, small_k0 + small_kk, [32, SMALL_L0_K], target_memory=pl.Mem.Left)
                        small_right = pl.tile.extract(small_transposed, small_kk, 0, [SMALL_L0_K, HEAD_N], target_memory=pl.Mem.Right)
                        acc = pl.tile.matmul_acc(acc, small_left, small_right, init_cond=small_step == 0 and small_kk == 0)
            elif native_rows == 16:
                # Aligned-M16 key230083: same K256/K64 payload as the
                # unaligned small-row case, but no outer-K permutation.
                for aligned_step in pl.pipeline(0, SMALL_OUTER, 1, stage=2):
                    aligned_k0 = aligned_step * SMALL_L1_K
                    aligned_weight = pl.tile.load(weight, [n0, aligned_k0], [HEAD_N, SMALL_L1_K], target_memory=pl.Mem.Mat, cache=pl.CachePolicy.BYPASS)
                    aligned_transposed = pl.tile.transpose_view(aligned_weight)
                    for aligned_kk in pl.pipeline(0, SMALL_L1_K, SMALL_L0_K, stage=2):
                        aligned_left = pl.tile.extract(activation, 0, aligned_k0 + aligned_kk, [32, SMALL_L0_K], target_memory=pl.Mem.Left)
                        aligned_right = pl.tile.extract(aligned_transposed, aligned_kk, 0, [SMALL_L0_K, HEAD_N], target_memory=pl.Mem.Right)
                        acc = pl.tile.matmul_acc(acc, aligned_left, aligned_right, init_cond=aligned_step == 0 and aligned_kk == 0)
            else:
                logical_group = n0 // HEAD_GROUP_SPAN
                enabled = (1 + logical_group // HEAD_NO_SHIFT_BEGIN) % 2
                reverse = (logical_group % 2) * enabled
                offset = HEAD_OFFSET_SCALE * ((logical_group // 2) % HEAD_HALF_GROUPS) * enabled
                for step in pl.pipeline(0, HEAD_OUTER, 1, stage=2):
                    outer = ((1 + (HEAD_OUTER - 2) * reverse) * step + (HEAD_OUTER - 1) * reverse + offset) % HEAD_OUTER
                    k0 = outer * HEAD_L1_K
                    weight_tile = pl.tile.load(weight, [n0, k0], [HEAD_N, HEAD_L1_K], target_memory=pl.Mem.Mat, cache=pl.CachePolicy.BYPASS)
                    transposed = pl.tile.transpose_view(weight_tile)
                    for kk in pl.pipeline(0, HEAD_L1_K, HEAD_L0_K, stage=2):
                        left = pl.tile.extract(activation, 0, k0 + kk, [32, HEAD_L0_K], target_memory=pl.Mem.Left)
                        right = pl.tile.extract(transposed, kk, 0, [HEAD_L0_K, HEAD_N], target_memory=pl.Mem.Right)
                        acc = pl.tile.matmul_acc(acc, left, right, init_cond=step == 0 and kk == 0)
            out = pl.tile.store(pl.tile.set_validshape(acc, vm, HEAD_N), [m0, n0], out)
    return out

@pl.function(type=pl.FunctionType.Inline)
def project_head(x: pl.Tensor[[24, 5120], pl.BF16], weight: pl.Tensor[[62080, 5120], pl.BF16, pl.NZ], out: pl.Tensor[[24, 62080], pl.BF16], row_ends: pl.Tensor[[4], pl.INT32]) -> pl.Tensor[[24, 62080], pl.BF16]:
    rows = 24
    with pl.spmd(pl.system.available_cluster_count(), allow_early_resolve=True):
        out = project_head_kernel(x, weight, out, row_ends)
    return out
