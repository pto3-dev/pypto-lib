# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""quantized operators. Explicit PyPTO functions; no source generator at build time.

Migrated from the validated L1K2560 graph; tuning is audited separately.
"""

import pypto.language as pl
from qwen36.tuning import (
    GATE_UP_N,
    GATE_UP_L1_K,
    GATE_UP_L0_K,
    GATE_UP_JOBS,
    DOWN_N,
    DOWN_L1_K,
    DOWN_L0_K,
    DOWN_JOBS,
)


@pl.function(type=pl.FunctionType.InCore)
def mlp_fused_kernel(
    x: pl.Tensor[[24, 5120], pl.INT8],
    weight: pl.Tensor[[16, 5120, 8704], pl.INT8, pl.NZ],
    out: pl.Tensor[[24, 4352], pl.BF16],
    weight_group: pl.Scalar[pl.INT64],
    weight_scale: pl.Tensor[[16, 8704], pl.FP32],
    activation_scale: pl.Tensor[[24, 1], pl.FP32],
) -> pl.Tensor[[24, 4352], pl.BF16]:
    pl.func_attr({"static_block_mapping": True})
    core = pl.tile.get_block_idx()
    grid = pl.tile.get_block_num()
    activation = pl.tile.load(x, [0, 0], [32, 5120], valid_shape=[24, 5120], target_memory=pl.Mem.Mat)
    xs = pl.tile.load(activation_scale, [0, 0], [32, 1], valid_shape=[24, 1], target_memory=pl.Mem.Vec)
    for job in pl.range(core, GATE_UP_JOBS, grid):
        n0 = job * GATE_UP_N
        value_acc = pl.tile.create([32, GATE_UP_N], pl.INT32, target_memory=pl.Mem.Acc, compact=True)
        for k1 in pl.pipeline(0, 5120, GATE_UP_L1_K, stage=2):
            wt_l1 = pl.tile.reshape(
                pl.tile.load(
                    weight,
                    [weight_group, k1, n0],
                    [1, GATE_UP_L1_K, GATE_UP_N],
                    target_memory=pl.Mem.Mat,
                    cache=pl.CachePolicy.BYPASS,
                ),
                [GATE_UP_L1_K, GATE_UP_N],
            )
            for k0 in pl.pipeline(0, GATE_UP_L1_K, GATE_UP_L0_K, stage=2):
                left = pl.tile.extract(activation, 0, k1 + k0, [32, GATE_UP_L0_K], target_memory=pl.Mem.Left)
                right = pl.tile.extract(wt_l1, k0, 0, [GATE_UP_L0_K, GATE_UP_N], target_memory=pl.Mem.Right)
                value_acc = pl.tile.matmul_acc(value_acc, left, right, init_cond=k1 == 0 and k0 == 0)
        ws = pl.tile.load(weight_scale, [weight_group, n0], [1, GATE_UP_N], target_memory=pl.Mem.Vec)
        fp = pl.tile.cast(pl.tile.set_validshape(value_acc, 24, GATE_UP_N), pl.FP32)
        ones = pl.tile.set_validshape(pl.tile.full([32, GATE_UP_N], pl.FP32, 1.0), 24, GATE_UP_N)
        combined_scale = pl.tile.row_expand_mul(pl.tile.col_expand_mul(ones, ws), xs)
        dequant = pl.tile.mul(fp, combined_scale)
        value = pl.tile.cast(dequant, pl.BF16, mode="rint")
        up_acc = pl.tile.create([32, GATE_UP_N], pl.INT32, target_memory=pl.Mem.Acc, compact=True)
        for k1 in pl.pipeline(0, 5120, GATE_UP_L1_K, stage=2):
            wt_l1 = pl.tile.reshape(
                pl.tile.load(
                    weight,
                    [weight_group, k1, 4352 + n0],
                    [1, GATE_UP_L1_K, GATE_UP_N],
                    target_memory=pl.Mem.Mat,
                    cache=pl.CachePolicy.BYPASS,
                ),
                [GATE_UP_L1_K, GATE_UP_N],
            )
            for k0 in pl.pipeline(0, GATE_UP_L1_K, GATE_UP_L0_K, stage=2):
                left = pl.tile.extract(activation, 0, k1 + k0, [32, GATE_UP_L0_K], target_memory=pl.Mem.Left)
                right = pl.tile.extract(wt_l1, k0, 0, [GATE_UP_L0_K, GATE_UP_N], target_memory=pl.Mem.Right)
                up_acc = pl.tile.matmul_acc(up_acc, left, right, init_cond=k1 == 0 and k0 == 0)
        ws = pl.tile.load(weight_scale, [weight_group, 4352 + n0], [1, GATE_UP_N], target_memory=pl.Mem.Vec)
        fp = pl.tile.cast(pl.tile.set_validshape(up_acc, 24, GATE_UP_N), pl.FP32)
        ones = pl.tile.set_validshape(pl.tile.full([32, GATE_UP_N], pl.FP32, 1.0), 24, GATE_UP_N)
        combined_scale = pl.tile.row_expand_mul(pl.tile.col_expand_mul(ones, ws), xs)
        dequant = pl.tile.mul(fp, combined_scale)
        up = pl.tile.cast(dequant, pl.BF16, mode="rint")
        gate = pl.tile.cast(value, pl.FP32)
        up_fp = pl.tile.cast(up, pl.FP32)
        sigmoid = pl.tile.recip(pl.tile.adds(pl.tile.exp(pl.tile.neg(gate)), 1.0))
        value = pl.tile.cast(pl.tile.mul(pl.tile.mul(sigmoid, gate), up_fp), pl.BF16, mode="rint")
        out = pl.tile.store(value, [0, n0], out)
    return out


@pl.function(type=pl.FunctionType.Inline)
def mlp_from_quant(
    q: pl.Tensor[[24, 5120], pl.INT8],
    weight: pl.Tensor[[16, 5120, 8704], pl.INT8, pl.NZ],
    out: pl.Tensor[[24, 4352], pl.BF16],
    weight_group: pl.Scalar[pl.INT64],
    weight_scale: pl.Tensor[[16, 8704], pl.FP32],
    xs: pl.Tensor[[24, 1], pl.FP32],
) -> pl.Tensor[[24, 4352], pl.BF16]:
    available_cube = pl.system.available_cluster_count()
    cube_width = pl.min(available_cube, GATE_UP_JOBS)
    with pl.spmd(cube_width, allow_early_resolve=True):
        out = mlp_fused_kernel(q, weight, out, weight_group, weight_scale, xs)
    return out


@pl.function(type=pl.FunctionType.InCore)
def quant_5120(
    x: pl.Tensor[[24, 5120], pl.BF16],
    q: pl.Out[pl.Tensor[[24, 5120], pl.INT8]],
    scales: pl.Out[pl.Tensor[[24, 1], pl.FP32]],
) -> tuple[pl.Tensor[[24, 5120], pl.INT8], pl.Tensor[[24, 1], pl.FP32]]:
    pl.func_attr({"static_block_mapping": True})
    lane = pl.tile.get_block_idx()
    width = pl.tile.get_block_num()
    for row in pl.range(lane, 24, width):
        value = pl.cast(pl.load(x, [row, 0], [1, 5120]), pl.FP32)
        chunks = pl.tile.reshape(pl.abs(value), [8, 640])
        tmp = pl.tile.create([8, 640], pl.FP32)
        maximum = pl.row_max(chunks, tmp)
        magnitude = pl.read(maximum, [0, 0])
        for part in pl.range(1, 8):
            partial = pl.read(maximum, [part, 0])
            if partial > magnitude:
                magnitude = partial
        if magnitude == 0.0:
            normalized = pl.mul(value, 0.0)
        else:
            denominator = pl.mul(pl.tile.full([1, 8], pl.FP32, 1.0), magnitude)
            numerator = pl.tile.full([1, 8], pl.FP32, 127.0)
            ratio_tile = pl.div(numerator, denominator)
            ratio = pl.read(ratio_tile, [0, 0])
            normalized = pl.mul(value, ratio)
        rounded = pl.cast(normalized, pl.INT32, mode="rint")
        half = pl.cast(rounded, pl.FP16, mode="rint")
        quantized = pl.cast(half, pl.INT8, mode="trunc")
        q = pl.store(quantized, [row, 0], q)
        scale_tile = pl.mul(pl.tile.full([1, 8], pl.FP32, 1.0), magnitude * (1.0 / 127.0))
        scales = pl.store(pl.tile.set_validshape(scale_tile, 1, 1), [row, 0], scales)
    return (q, scales)


@pl.function(type=pl.FunctionType.InCore)
def quant_4352(
    x: pl.Tensor[[24, 4352], pl.BF16],
    q: pl.Out[pl.Tensor[[24, 4352], pl.INT8]],
    scales: pl.Out[pl.Tensor[[24, 1], pl.FP32]],
) -> tuple[pl.Tensor[[24, 4352], pl.INT8], pl.Tensor[[24, 1], pl.FP32]]:
    pl.func_attr({"static_block_mapping": True})
    lane = pl.tile.get_block_idx()
    width = pl.tile.get_block_num()
    for row in pl.range(lane, 24, width):
        value = pl.cast(pl.load(x, [row, 0], [1, 4352]), pl.FP32)
        chunks = pl.tile.reshape(pl.abs(value), [8, 544])
        tmp = pl.tile.create([8, 544], pl.FP32)
        maximum = pl.row_max(chunks, tmp)
        magnitude = pl.read(maximum, [0, 0])
        for part in pl.range(1, 8):
            partial = pl.read(maximum, [part, 0])
            if partial > magnitude:
                magnitude = partial
        if magnitude == 0.0:
            normalized = pl.mul(value, 0.0)
        else:
            denominator = pl.mul(pl.tile.full([1, 8], pl.FP32, 1.0), magnitude)
            numerator = pl.tile.full([1, 8], pl.FP32, 127.0)
            ratio_tile = pl.div(numerator, denominator)
            ratio = pl.read(ratio_tile, [0, 0])
            normalized = pl.mul(value, ratio)
        rounded = pl.cast(normalized, pl.INT32, mode="rint")
        half = pl.cast(rounded, pl.FP16, mode="rint")
        quantized = pl.cast(half, pl.INT8, mode="trunc")
        q = pl.store(quantized, [row, 0], q)
        scale_tile = pl.mul(pl.tile.full([1, 8], pl.FP32, 1.0), magnitude * (1.0 / 127.0))
        scales = pl.store(pl.tile.set_validshape(scale_tile, 1, 1), [row, 0], scales)
    return (q, scales)


@pl.function(type=pl.FunctionType.InCore)
def quant_1536(
    x: pl.Tensor[[24, 1536], pl.BF16],
    q: pl.Out[pl.Tensor[[24, 1536], pl.INT8]],
    scales: pl.Out[pl.Tensor[[24, 1], pl.FP32]],
) -> tuple[pl.Tensor[[24, 1536], pl.INT8], pl.Tensor[[24, 1], pl.FP32]]:
    pl.func_attr({"static_block_mapping": True})
    lane = pl.tile.get_block_idx()
    width = pl.tile.get_block_num()
    for row in pl.range(lane, 24, width):
        value = pl.cast(pl.load(x, [row, 0], [1, 1536]), pl.FP32)
        chunks = pl.tile.reshape(pl.abs(value), [8, 192])
        tmp = pl.tile.create([8, 192], pl.FP32)
        maximum = pl.row_max(chunks, tmp)
        magnitude = pl.read(maximum, [0, 0])
        for part in pl.range(1, 8):
            partial = pl.read(maximum, [part, 0])
            if partial > magnitude:
                magnitude = partial
        if magnitude == 0.0:
            normalized = pl.mul(value, 0.0)
        else:
            denominator = pl.mul(pl.tile.full([1, 8], pl.FP32, 1.0), magnitude)
            numerator = pl.tile.full([1, 8], pl.FP32, 127.0)
            ratio_tile = pl.div(numerator, denominator)
            ratio = pl.read(ratio_tile, [0, 0])
            normalized = pl.mul(value, ratio)
        rounded = pl.cast(normalized, pl.INT32, mode="rint")
        half = pl.cast(rounded, pl.FP16, mode="rint")
        quantized = pl.cast(half, pl.INT8, mode="trunc")
        q = pl.store(quantized, [row, 0], q)
        scale_tile = pl.mul(pl.tile.full([1, 8], pl.FP32, 1.0), magnitude * (1.0 / 127.0))
        scales = pl.store(pl.tile.set_validshape(scale_tile, 1, 1), [row, 0], scales)
    return (q, scales)


@pl.function(type=pl.FunctionType.Inline)
def mlp_fused(
    x: pl.Tensor[[24, 5120], pl.BF16],
    weight: pl.Tensor[[16, 5120, 8704], pl.INT8, pl.NZ],
    out: pl.Tensor[[24, 4352], pl.BF16],
    weight_group: pl.Scalar[pl.INT64],
    weight_scale: pl.Tensor[[16, 8704], pl.FP32],
) -> pl.Tensor[[24, 4352], pl.BF16]:
    q = pl.create_tensor([24, 5120], pl.INT8)
    xs = pl.create_tensor([24, 1], pl.FP32)
    available_aiv = pl.system.available_aiv_count()
    vector_width = pl.min(available_aiv, 24)
    with pl.spmd(vector_width, allow_early_resolve=True):
        q, xs = quant_5120(x, q, xs)
    available_cube = pl.system.available_cluster_count()
    cube_width = pl.min(available_cube, GATE_UP_JOBS)
    with pl.spmd(cube_width, allow_early_resolve=True):
        out = mlp_fused_kernel(q, weight, out, weight_group, weight_scale, xs)
    return out


@pl.function(type=pl.FunctionType.InCore)
def project_down_kernel(
    x: pl.Tensor[[24, 4352], pl.INT8],
    weight: pl.Tensor[[16, 4352, 5120], pl.INT8, pl.NZ],
    out: pl.Tensor[[24, 5120], pl.FP32],
    weight_group: pl.Scalar[pl.INT64],
    weight_scale: pl.Tensor[[16, 5120], pl.FP32],
    activation_scale: pl.Tensor[[24, 1], pl.FP32],
) -> pl.Tensor[[24, 5120], pl.FP32]:
    pl.func_attr({"static_block_mapping": True})
    core = pl.tile.get_block_idx()
    grid = pl.tile.get_block_num()
    activation = pl.tile.load(x, [0, 0], [32, 4352], valid_shape=[24, 4352], target_memory=pl.Mem.Mat)
    xs = pl.tile.load(activation_scale, [0, 0], [32, 1], valid_shape=[24, 1], target_memory=pl.Mem.Vec)
    for job in pl.range(core, DOWN_JOBS, grid):
        n0 = job * DOWN_N
        value_acc = pl.tile.create([32, DOWN_N], pl.INT32, target_memory=pl.Mem.Acc, compact=True)
        for k0 in pl.pipeline(0, 4352, DOWN_L1_K, stage=2):
            wt = pl.tile.reshape(
                pl.tile.load(
                    weight,
                    [weight_group, k0, n0],
                    [1, DOWN_L1_K, DOWN_N],
                    target_memory=pl.Mem.Mat,
                    cache=pl.CachePolicy.BYPASS,
                ),
                [DOWN_L1_K, DOWN_N],
            )
            left = pl.tile.extract(activation, 0, k0, [32, DOWN_L0_K], target_memory=pl.Mem.Left)
            right = pl.tile.extract(wt, 0, 0, [DOWN_L0_K, DOWN_N], target_memory=pl.Mem.Right)
            value_acc = pl.tile.matmul_acc(value_acc, left, right, init_cond=k0 == 0)
        ws = pl.tile.load(weight_scale, [weight_group, n0], [1, DOWN_N], target_memory=pl.Mem.Vec)
        fp = pl.tile.cast(pl.tile.set_validshape(value_acc, 24, DOWN_N), pl.FP32)
        ones = pl.tile.set_validshape(pl.tile.full([32, DOWN_N], pl.FP32, 1.0), 24, DOWN_N)
        combined_scale = pl.tile.row_expand_mul(pl.tile.col_expand_mul(ones, ws), xs)
        dequant = pl.tile.mul(fp, combined_scale)
        value = pl.tile.cast(dequant, pl.BF16, mode="rint")
        output_value = pl.tile.cast(value, pl.FP32)
        out = pl.tile.store(output_value, [0, n0], out)
    return out


@pl.function(type=pl.FunctionType.Inline)
def project_down(
    x: pl.Tensor[[24, 4352], pl.BF16],
    weight: pl.Tensor[[16, 4352, 5120], pl.INT8, pl.NZ],
    out: pl.Tensor[[24, 5120], pl.FP32],
    weight_group: pl.Scalar[pl.INT64],
    weight_scale: pl.Tensor[[16, 5120], pl.FP32],
) -> pl.Tensor[[24, 5120], pl.FP32]:
    q = pl.create_tensor([24, 4352], pl.INT8)
    xs = pl.create_tensor([24, 1], pl.FP32)
    available_aiv = pl.system.available_aiv_count()
    vector_width = pl.min(available_aiv, 24)
    with pl.spmd(vector_width, allow_early_resolve=True):
        q, xs = quant_4352(x, q, xs)
    available_cube = pl.system.available_cluster_count()
    cube_width = pl.min(available_cube, DOWN_JOBS)
    with pl.spmd(cube_width, allow_early_resolve=True):
        out = project_down_kernel(q, weight, out, weight_group, weight_scale, xs)
    return out


@pl.function(type=pl.FunctionType.InCore)
def project_qkv_kernel(
    x: pl.Tensor[[24, 5120], pl.INT8],
    weight: pl.Tensor[[16, 5120, 3584], pl.INT8, pl.NZ],
    mixed: pl.Tensor[[24, 3072], pl.BF16],
    z: pl.Tensor[[24, 256], pl.BF16],
    v: pl.Tensor[[24, 256], pl.BF16],
    weight_group: pl.Scalar[pl.INT64],
    weight_scale: pl.Tensor[[16, 3584], pl.FP32],
    activation_scale: pl.Tensor[[24, 1], pl.FP32],
) -> tuple[pl.Tensor[[24, 3072], pl.BF16], pl.Tensor[[24, 256], pl.BF16], pl.Tensor[[24, 256], pl.BF16]]:
    pl.func_attr({"static_block_mapping": True})
    core = pl.tile.get_block_idx()
    grid = pl.tile.get_block_num()
    activation = pl.tile.load(x, [0, 0], [32, 5120], valid_shape=[24, 5120], target_memory=pl.Mem.Mat)
    xs = pl.tile.load(activation_scale, [0, 0], [32, 1], valid_shape=[24, 1], target_memory=pl.Mem.Vec)
    for job in pl.range(core, 56, grid):
        n0 = job * 64
        value_acc = pl.tile.create([32, 64], pl.INT32, target_memory=pl.Mem.Acc, compact=True)
        for k0 in pl.pipeline(0, 5120, 256, stage=2):
            wt = pl.tile.reshape(
                pl.tile.load(
                    weight,
                    [weight_group, k0, n0],
                    [1, 256, 64],
                    target_memory=pl.Mem.Mat,
                    cache=pl.CachePolicy.BYPASS,
                ),
                [256, 64],
            )
            left = pl.tile.extract(activation, 0, k0, [32, 256], target_memory=pl.Mem.Left)
            right = pl.tile.extract(wt, 0, 0, [256, 64], target_memory=pl.Mem.Right)
            value_acc = pl.tile.matmul_acc(value_acc, left, right, init_cond=k0 == 0)
        ws = pl.tile.load(weight_scale, [weight_group, n0], [1, 64], target_memory=pl.Mem.Vec)
        fp = pl.tile.cast(pl.tile.set_validshape(value_acc, 24, 64), pl.FP32)
        ones = pl.tile.set_validshape(pl.tile.full([32, 64], pl.FP32, 1.0), 24, 64)
        combined_scale = pl.tile.row_expand_mul(pl.tile.col_expand_mul(ones, ws), xs)
        dequant = pl.tile.mul(fp, combined_scale)
        value = pl.tile.cast(dequant, pl.BF16, mode="rint")
        if n0 < 3072:
            mixed = pl.tile.store(value, [0, n0], mixed)
        elif n0 < 3328:
            z = pl.tile.store(value, [0, n0 - 3072], z)
        else:
            v = pl.tile.store(value, [0, n0 - 3328], v)
    return (mixed, z, v)


@pl.function(type=pl.FunctionType.Inline)
def project_qkv(
    x: pl.Tensor[[24, 5120], pl.BF16],
    weight: pl.Tensor[[16, 5120, 3584], pl.INT8, pl.NZ],
    mixed: pl.Tensor[[24, 3072], pl.BF16],
    z: pl.Tensor[[24, 256], pl.BF16],
    v: pl.Tensor[[24, 256], pl.BF16],
    weight_group: pl.Scalar[pl.INT64],
    weight_scale: pl.Tensor[[16, 3584], pl.FP32],
) -> tuple[pl.Tensor[[24, 3072], pl.BF16], pl.Tensor[[24, 256], pl.BF16], pl.Tensor[[24, 256], pl.BF16]]:
    q = pl.create_tensor([24, 5120], pl.INT8)
    xs = pl.create_tensor([24, 1], pl.FP32)
    available_aiv = pl.system.available_aiv_count()
    vector_width = pl.min(available_aiv, 24)
    with pl.spmd(vector_width, allow_early_resolve=True):
        q, xs = quant_5120(x, q, xs)
    available_cube = pl.system.available_cluster_count()
    cube_width = pl.min(available_cube, 56)
    with pl.spmd(cube_width, allow_early_resolve=True):
        mixed, z, v = project_qkv_kernel(q, weight, mixed, z, v, weight_group, weight_scale, xs)
    return (mixed, z, v)


@pl.function(type=pl.FunctionType.InCore)
def project_attention_out_w8a8_kernel(
    x: pl.Tensor[[24, 1536], pl.INT8],
    weight: pl.Tensor[[16, 1536, 5120], pl.INT8, pl.NZ],
    out: pl.Tensor[[24, 5120], pl.FP32],
    weight_group: pl.Scalar[pl.INT64],
    weight_scale: pl.Tensor[[16, 5120], pl.FP32],
    activation_scale: pl.Tensor[[24, 1], pl.FP32],
) -> pl.Tensor[[24, 5120], pl.FP32]:
    pl.func_attr({"static_block_mapping": True})
    core = pl.tile.get_block_idx()
    grid = pl.tile.get_block_num()
    activation = pl.tile.load(x, [0, 0], [32, 1536], valid_shape=[24, 1536], target_memory=pl.Mem.Mat)
    xs = pl.tile.load(activation_scale, [0, 0], [32, 1], valid_shape=[24, 1], target_memory=pl.Mem.Vec)
    for job in pl.range(core, 80, grid):
        n0 = job * 64
        value_acc = pl.tile.create([32, 64], pl.INT32, target_memory=pl.Mem.Acc, compact=True)
        for k0 in pl.pipeline(0, 1536, 256, stage=2):
            wt = pl.tile.reshape(
                pl.tile.load(
                    weight,
                    [weight_group, k0, n0],
                    [1, 256, 64],
                    target_memory=pl.Mem.Mat,
                    cache=pl.CachePolicy.BYPASS,
                ),
                [256, 64],
            )
            left = pl.tile.extract(activation, 0, k0, [32, 256], target_memory=pl.Mem.Left)
            right = pl.tile.extract(wt, 0, 0, [256, 64], target_memory=pl.Mem.Right)
            value_acc = pl.tile.matmul_acc(value_acc, left, right, init_cond=k0 == 0)
        ws = pl.tile.load(weight_scale, [weight_group, n0], [1, 64], target_memory=pl.Mem.Vec)
        fp = pl.tile.cast(pl.tile.set_validshape(value_acc, 24, 64), pl.FP32)
        ones = pl.tile.set_validshape(pl.tile.full([32, 64], pl.FP32, 1.0), 24, 64)
        combined_scale = pl.tile.row_expand_mul(pl.tile.col_expand_mul(ones, ws), xs)
        dequant = pl.tile.mul(fp, combined_scale)
        value = pl.tile.cast(dequant, pl.BF16, mode="rint")
        output_value = pl.tile.cast(value, pl.FP32)
        out = pl.tile.store(output_value, [0, n0], out)
    return out


@pl.function(type=pl.FunctionType.Inline)
def project_attention_out_w8a8(
    x: pl.Tensor[[24, 1536], pl.BF16],
    weight: pl.Tensor[[16, 1536, 5120], pl.INT8, pl.NZ],
    out: pl.Tensor[[24, 5120], pl.FP32],
    weight_group: pl.Scalar[pl.INT64],
    weight_scale: pl.Tensor[[16, 5120], pl.FP32],
) -> pl.Tensor[[24, 5120], pl.FP32]:
    q = pl.create_tensor([24, 1536], pl.INT8)
    xs = pl.create_tensor([24, 1], pl.FP32)
    available_aiv = pl.system.available_aiv_count()
    vector_width = pl.min(available_aiv, 24)
    with pl.spmd(vector_width, allow_early_resolve=True):
        q, xs = quant_1536(x, q, xs)
    available_cube = pl.system.available_cluster_count()
    cube_width = pl.min(available_cube, DOWN_JOBS)
    with pl.spmd(cube_width, allow_early_resolve=True):
        out = project_attention_out_w8a8_kernel(q, weight, out, weight_group, weight_scale, xs)
    return out
