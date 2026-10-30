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
from qwen36.tuning import GDN_N, GDN_L1_K, GDN_L0_K, Z_N, Z_L1_K, Z_L0_K, HEAD_N, HEAD_L1_K, HEAD_L0_K


@pl.function(type=pl.FunctionType.InCore)
def greedy_projection_spmd_kernel(
    x__ssa_v0: pl.Tensor[[24, 1536], pl.BF16],
    w__ssa_v0: pl.Tensor[[16, 5120, 1536], pl.BF16, pl.NZ],
    out__ssa_v0: pl.Out[pl.Tensor[[24, 5120], pl.FP32]],
    weight_group: pl.Scalar[pl.INT64],
) -> pl.Tensor[[24, 5120], pl.FP32]:
    pl.func_attr({"static_block_mapping": True})
    for m0 in pl.range(0, 24, 32):
        vm = pl.min(32, 24 - m0)
        a0__ssa_v0 = pl.tile.load(x__ssa_v0, [m0, 0], [32, 1536], [vm, 1536], target_memory=pl.Mem.Mat)
        for group_index in pl.range(pl.tile.get_block_idx(), 20, pl.tile.get_block_num()):
            column_base = group_index * 256
            b0__ssa_v0 = pl.tile.reshape(
                pl.tile.load(
                    w__ssa_v0,
                    [weight_group, column_base + 0, 0],
                    [1, 64, 1536],
                    [1, 64, 1536],
                    target_memory=pl.Mem.Mat,
                    cache=1,
                ),
                [64, 1536],
            )
            bt0__ssa_v0 = pl.tile.transpose_view(b0__ssa_v0)
            c0__ssa_v0 = pl.tile.matmul(a0__ssa_v0, bt0__ssa_v0)
            v0__ssa_v0 = pl.tile.set_validshape(c0__ssa_v0, vm, 64)
            out__ssa_v1 = pl.tile.store(v0__ssa_v0, [m0, column_base + 0], out__ssa_v0)
            b1__ssa_v0 = pl.tile.reshape(
                pl.tile.load(
                    w__ssa_v0,
                    [weight_group, column_base + 64, 0],
                    [1, 64, 1536],
                    [1, 64, 1536],
                    target_memory=pl.Mem.Mat,
                    cache=1,
                ),
                [64, 1536],
            )
            bt1__ssa_v0 = pl.tile.transpose_view(b1__ssa_v0)
            c1__ssa_v0 = pl.tile.matmul(a0__ssa_v0, bt1__ssa_v0)
            v1__ssa_v0 = pl.tile.set_validshape(c1__ssa_v0, vm, 64)
            out__ssa_v2 = pl.tile.store(v1__ssa_v0, [m0, column_base + 64], out__ssa_v0)
            b2__ssa_v0 = pl.tile.reshape(
                pl.tile.load(
                    w__ssa_v0,
                    [weight_group, column_base + 128, 0],
                    [1, 64, 1536],
                    [1, 64, 1536],
                    target_memory=pl.Mem.Mat,
                    cache=1,
                ),
                [64, 1536],
            )
            bt2__ssa_v0 = pl.tile.transpose_view(b2__ssa_v0)
            c2__ssa_v0 = pl.tile.matmul(a0__ssa_v0, bt2__ssa_v0)
            v2__ssa_v0 = pl.tile.set_validshape(c2__ssa_v0, vm, 64)
            out__ssa_v3 = pl.tile.store(v2__ssa_v0, [m0, column_base + 128], out__ssa_v0)
            b3__ssa_v0 = pl.tile.reshape(
                pl.tile.load(
                    w__ssa_v0,
                    [weight_group, column_base + 192, 0],
                    [1, 64, 1536],
                    [1, 64, 1536],
                    target_memory=pl.Mem.Mat,
                    cache=1,
                ),
                [64, 1536],
            )
            bt3__ssa_v0 = pl.tile.transpose_view(b3__ssa_v0)
            c3__ssa_v0 = pl.tile.matmul(a0__ssa_v0, bt3__ssa_v0)
            v3__ssa_v0 = pl.tile.set_validshape(c3__ssa_v0, vm, 64)
            out__ssa_v4 = pl.tile.store(v3__ssa_v0, [m0, column_base + 192], out__ssa_v0)
            out__ssa_v0 = out__ssa_v4
    return out__ssa_v0


@pl.function(type=pl.FunctionType.Inline)
def project_attention_out(
    x: pl.Tensor[[24, 1536], pl.BF16],
    w: pl.Tensor[[16, 5120, 1536], pl.BF16, pl.NZ],
    out: pl.Out[pl.Tensor[[24, 5120], pl.FP32]],
    weight_group: pl.Scalar[pl.INT64],
) -> pl.Tensor[[24, 5120], pl.FP32]:
    available = pl.system.available_cluster_count()
    width = pl.min(available, 20)
    with pl.spmd(width, allow_early_resolve=True):
        out = greedy_projection_spmd_kernel(x, w, out, weight_group)
    return out


@pl.function(type=pl.FunctionType.InCore)
def project_gdn_kernel(
    x: pl.Tensor[[24, 5120], pl.BF16],
    weight: pl.Tensor[[16, 4096, 5120], pl.BF16, pl.NZ],
    out: pl.Tensor[[24, 2560], pl.BF16],
    weight_group: pl.Scalar[pl.INT64],
) -> pl.Tensor[[24, 2560], pl.BF16]:
    pl.func_attr({"static_block_mapping": True})
    rows = 24
    core = pl.tile.get_block_idx()
    grid = pl.tile.get_block_num()
    jobs = 2560 // GDN_N
    for m0 in pl.range(0, rows, 32):
        vm = pl.min(32, rows - m0)
        activation = pl.tile.load(x, [m0, 0], [32, 5120], valid_shape=[vm, 5120], target_memory=pl.Mem.Mat)
        for job in pl.range(core, jobs, grid):
            n0 = job * GDN_N
            acc = pl.tile.create([32, GDN_N], pl.FP32, target_memory=pl.Mem.Acc, compact=True)
            for k0 in pl.pipeline(0, 5120, GDN_L1_K, stage=2):
                weight_tile = pl.tile.reshape(
                    pl.tile.load(
                        weight,
                        [weight_group, n0, k0],
                        [1, GDN_N, GDN_L1_K],
                        target_memory=pl.Mem.Mat,
                        cache=pl.CachePolicy.BYPASS,
                    ),
                    [GDN_N, GDN_L1_K],
                )
                transposed = pl.tile.transpose_view(weight_tile)
                for kk in pl.pipeline(0, GDN_L1_K, GDN_L0_K, stage=2):
                    left = pl.tile.extract(activation, 0, k0 + kk, [32, GDN_L0_K], target_memory=pl.Mem.Left)
                    right = pl.tile.extract(transposed, kk, 0, [GDN_L0_K, GDN_N], target_memory=pl.Mem.Right)
                    acc = pl.tile.matmul_acc(acc, left, right, init_cond=k0 == 0 and kk == 0)
            out = pl.tile.store(pl.tile.set_validshape(acc, vm, GDN_N), [m0, n0], out)
    return out


@pl.function(type=pl.FunctionType.Inline)
def project_gdn(
    x: pl.Tensor[[24, 5120], pl.BF16],
    weight: pl.Tensor[[16, 4096, 5120], pl.BF16, pl.NZ],
    out: pl.Tensor[[24, 2560], pl.BF16],
    weight_group: pl.Scalar[pl.INT64],
) -> pl.Tensor[[24, 2560], pl.BF16]:
    rows = 24
    with pl.spmd(pl.system.available_cluster_count(), allow_early_resolve=True):
        out = project_gdn_kernel(x, weight, out, weight_group)
    return out


@pl.function(type=pl.FunctionType.InCore)
def project_z_kernel(
    x: pl.Tensor[[24, 5120], pl.BF16],
    weight: pl.Tensor[[16, 4096, 5120], pl.BF16, pl.NZ],
    out: pl.Tensor[[24, 1536], pl.BF16],
    weight_group: pl.Scalar[pl.INT64],
) -> pl.Tensor[[24, 1536], pl.BF16]:
    pl.func_attr({"static_block_mapping": True})
    rows = 24
    core = pl.tile.get_block_idx()
    grid = pl.tile.get_block_num()
    jobs = 1536 // Z_N
    for m0 in pl.range(0, rows, 32):
        vm = pl.min(32, rows - m0)
        activation = pl.tile.load(x, [m0, 0], [32, 5120], valid_shape=[vm, 5120], target_memory=pl.Mem.Mat)
        for job in pl.range(core, jobs, grid):
            n0 = job * Z_N
            acc = pl.tile.create([32, Z_N], pl.FP32, target_memory=pl.Mem.Acc, compact=True)
            for k0 in pl.pipeline(0, 5120, Z_L1_K, stage=2):
                weight_tile = pl.tile.reshape(
                    pl.tile.load(
                        weight,
                        [weight_group, 2560 + n0, k0],
                        [1, Z_N, Z_L1_K],
                        target_memory=pl.Mem.Mat,
                        cache=pl.CachePolicy.BYPASS,
                    ),
                    [Z_N, Z_L1_K],
                )
                transposed = pl.tile.transpose_view(weight_tile)
                for kk in pl.pipeline(0, Z_L1_K, Z_L0_K, stage=2):
                    left = pl.tile.extract(activation, 0, k0 + kk, [32, Z_L0_K], target_memory=pl.Mem.Left)
                    right = pl.tile.extract(transposed, kk, 0, [Z_L0_K, Z_N], target_memory=pl.Mem.Right)
                    acc = pl.tile.matmul_acc(acc, left, right, init_cond=k0 == 0 and kk == 0)
            out = pl.tile.store(pl.tile.set_validshape(acc, vm, Z_N), [m0, n0], out)
    return out


@pl.function(type=pl.FunctionType.Inline)
def project_z(
    x: pl.Tensor[[24, 5120], pl.BF16],
    weight: pl.Tensor[[16, 4096, 5120], pl.BF16, pl.NZ],
    out: pl.Tensor[[24, 1536], pl.BF16],
    weight_group: pl.Scalar[pl.INT64],
) -> pl.Tensor[[24, 1536], pl.BF16]:
    rows = 24
    with pl.spmd(pl.system.available_cluster_count(), allow_early_resolve=True):
        out = project_z_kernel(x, weight, out, weight_group)
    return out


@pl.function(type=pl.FunctionType.InCore)
def project_up_kernel(
    x: pl.Tensor[[24, 5120], pl.BF16],
    weight: pl.Tensor[[4352, 5120], pl.BF16],
    out: pl.Tensor[[24, 4352], pl.BF16],
) -> pl.Tensor[[24, 4352], pl.BF16]:
    pl.func_attr({"static_block_mapping": True})
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
                weight_tile = pl.tile.load(
                    weight, [n0, k0], [64, 256], target_memory=pl.Mem.Mat, cache=pl.CachePolicy.BYPASS
                )
                transposed = pl.tile.transpose_view(weight_tile)
                for kk in pl.pipeline(0, 256, 256, stage=2):
                    left = pl.tile.extract(activation, 0, k0 + kk, [32, 256], target_memory=pl.Mem.Left)
                    right = pl.tile.extract(transposed, kk, 0, [256, 64], target_memory=pl.Mem.Right)
                    acc = pl.tile.matmul_acc(acc, left, right, init_cond=k0 == 0 and kk == 0)
            out = pl.tile.store(pl.tile.set_validshape(acc, vm, 64), [m0, n0], out)
    return out


@pl.function(type=pl.FunctionType.Inline)
def project_up(
    x: pl.Tensor[[24, 5120], pl.BF16],
    weight: pl.Tensor[[4352, 5120], pl.BF16],
    out: pl.Tensor[[24, 4352], pl.BF16],
) -> pl.Tensor[[24, 4352], pl.BF16]:
    rows = 24
    with pl.spmd(pl.system.available_cluster_count(), allow_early_resolve=True):
        out = project_up_kernel(x, weight, out)
    return out


@pl.function(type=pl.FunctionType.InCore)
def project_head_kernel(
    x: pl.Tensor[[24, 5120], pl.BF16],
    weight: pl.Tensor[[62080, 5120], pl.BF16, pl.NZ],
    out: pl.Tensor[[24, 62080], pl.BF16],
) -> pl.Tensor[[24, 62080], pl.BF16]:
    pl.func_attr({"static_block_mapping": True})
    rows = 24
    core = pl.tile.get_block_idx()
    grid = pl.tile.get_block_num()
    jobs = 62080 // HEAD_N
    for m0 in pl.range(0, rows, 32):
        vm = pl.min(32, rows - m0)
        activation = pl.tile.load(x, [m0, 0], [32, 5120], valid_shape=[vm, 5120], target_memory=pl.Mem.Mat)
        for job in pl.range(core, jobs, grid):
            n0 = job * HEAD_N
            acc = pl.tile.create([32, HEAD_N], pl.FP32, target_memory=pl.Mem.Acc, compact=True)
            for k0 in pl.pipeline(0, 5120, HEAD_L1_K, stage=2):
                weight_tile = pl.tile.load(
                    weight,
                    [n0, k0],
                    [HEAD_N, HEAD_L1_K],
                    target_memory=pl.Mem.Mat,
                    cache=pl.CachePolicy.BYPASS,
                )
                transposed = pl.tile.transpose_view(weight_tile)
                for kk in pl.pipeline(0, HEAD_L1_K, HEAD_L0_K, stage=2):
                    left = pl.tile.extract(activation, 0, k0 + kk, [32, HEAD_L0_K], target_memory=pl.Mem.Left)
                    right = pl.tile.extract(
                        transposed, kk, 0, [HEAD_L0_K, HEAD_N], target_memory=pl.Mem.Right
                    )
                    acc = pl.tile.matmul_acc(acc, left, right, init_cond=k0 == 0 and kk == 0)
            out = pl.tile.store(pl.tile.set_validshape(acc, vm, HEAD_N), [m0, n0], out)
    return out


@pl.function(type=pl.FunctionType.Inline)
def project_head(
    x: pl.Tensor[[24, 5120], pl.BF16],
    weight: pl.Tensor[[62080, 5120], pl.BF16, pl.NZ],
    out: pl.Tensor[[24, 62080], pl.BF16],
) -> pl.Tensor[[24, 62080], pl.BF16]:
    rows = 24
    with pl.spmd(pl.system.available_cluster_count(), allow_early_resolve=True):
        out = project_head_kernel(x, weight, out)
    return out
