# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""PTO BF16 projections, RMS normalization, and pointwise activations."""

import pypto.language as pl

ROWS = pl.dynamic("rows")


def make_gemm(
    input_dim,
    output_dim,
    inline=False,
    out_fp32=False,
    n_tile=128,
    k_tile=256,
    cube_bf16_store=False,
    weight_cache_bypass=False,
):
    weight_cache_policy = pl.CachePolicy.BYPASS if weight_cache_bypass else pl.CachePolicy.DEFAULT
    output_dtype = pl.FP32 if out_fp32 else pl.BF16
    decorate = pl.jit.inline if inline else pl.jit

    @decorate
    def gemm(
        x: pl.Tensor[[ROWS, input_dim], pl.BF16],
        weight: pl.Tensor[[output_dim, input_dim], pl.BF16],
        out: pl.Out[pl.Tensor[[ROWS, output_dim], output_dtype]],
    ):
        rows = pl.tensor.dim(x, 0)
        # Query launch geometry on the owning worker; use that launch width in-core.
        for core in pl.spmd(pl.system.available_cluster_count(), name_hint="projection"):
            pl.set_cache_policy(weight, weight_cache_policy)
            grid = pl.tile.get_block_num()
            jobs = ((rows + 15) // 16) * ((output_dim + (n_tile - 1)) // n_tile)
            per_core = jobs // grid
            remainder = jobs % grid
            first = core * per_core + pl.min(core, remainder)
            last = (core + 1) * per_core + pl.min(core + 1, remainder)
            for job in pl.range(first, last):
                m0 = job // ((output_dim + (n_tile - 1)) // n_tile) * 16
                n0 = job % ((output_dim + (n_tile - 1)) // n_tile) * n_tile
                vm = pl.min(16, rows - m0)
                vn = pl.min(n_tile, output_dim - n0)
                acc = pl.create_tensor([16, n_tile], pl.FP32)
                for kb in pl.pipeline(0, (input_dim + (k_tile - 1)) // k_tile, stage=2):
                    k0 = kb * k_tile
                    vk = pl.min(k_tile, input_dim - k0)
                    xt = pl.slice(x, [16, k_tile], [m0, k0], valid_shape=[vm, vk], pad_value=0, clamp=True)
                    wt = pl.slice(
                        weight, [n_tile, k_tile], [n0, k0], valid_shape=[vn, vk], pad_value=0, clamp=True
                    )
                    acc = pl.matmul_acc(acc, xt, wt, b_trans=True, init_cond=(kb == 0))
                if out_fp32:
                    fp32_result = pl.tensor.set_validshape(acc, vm, vn)
                    out = pl.assemble(out, fp32_result, [m0, n0])
                elif cube_bf16_store:
                    # FIXPIPE can narrow an FP32 accumulator directly into
                    # BF16 GM with round-to-nearest-even, keeping the epilogue
                    # on Cube and avoiding a separate vector worker per tile.
                    narrowed_valid = pl.tensor.set_validshape(acc, vm, vn)
                    out = pl.assemble(out, narrowed_valid, [m0, n0])
                else:
                    bf16_result = pl.cast(acc, pl.BF16, mode="rint")
                    bf16_valid = pl.tensor.set_validshape(bf16_result, vm, vn)
                    out = pl.assemble(out, bf16_valid, [m0, n0])
        return out

    return gemm


def make_norm(
    dim, weight_offset=1.0, gated=False, inline=False, native_round=False, tile=128, output_tile=None
):
    output_tile = tile if output_tile is None else output_tile
    if tile % 128 or dim % tile or output_tile % 128 or dim % output_tile:
        raise ValueError("RMSNorm tile must divide the dimension and be a multiple of 128.")
    decorate = pl.jit.inline if inline else pl.jit

    @decorate
    def norm(
        x: pl.Tensor[[ROWS, dim], pl.BF16],
        weight: pl.Tensor[[1, dim], pl.BF16],
        z: pl.Tensor[[ROWS, dim], pl.BF16],
        out: pl.Out[pl.Tensor[[ROWS, dim], pl.BF16]],
    ):
        rows = pl.tensor.dim(x, 0)
        for job in pl.spmd((rows + 7) // 8, name_hint="rmsnorm"):
            r0 = job * 8
            valid = pl.min(8, rows - r0)
            zero_tile = pl.full([8, 128], dtype=pl.FP32, value=0.0)
            sq = pl.row_sum(zero_tile)
            for c0 in pl.range(0, dim, tile):
                xt = pl.cast(pl.slice(x, [8, tile], [r0, c0], [valid, tile], clamp=True), pl.FP32)
                sq = pl.add(sq, pl.row_sum(pl.mul(xt, xt)))
            variance = pl.add(pl.mul(sq, 1.0 / dim), 1e-6)
            inv = pl.rsqrt(variance)
            # Refine the hardware estimate before narrowing. Small input
            # variances otherwise amplify its error through the MLP chain.
            correction = pl.add(pl.mul(pl.mul(variance, pl.mul(inv, inv)), -0.5), 1.5)
            inv = pl.mul(inv, correction)
            for c0 in pl.range(0, dim, output_tile):
                norm_x = pl.cast(
                    pl.slice(x, [8, output_tile], [r0, c0], [valid, output_tile], clamp=True), pl.FP32
                )
                wt = pl.cast(pl.slice(weight, [1, output_tile], [0, c0]), pl.FP32)
                wt = pl.add(wt, weight_offset)
                normalized = pl.row_expand_mul(norm_x, inv)
                if native_round:
                    # Ascend BF16 residual RMSNorm narrows both normalized
                    # activation and Gemma gamma before their multiplication.
                    normalized = pl.cast(pl.cast(normalized, pl.BF16, mode="rint"), pl.FP32)
                    wt = pl.cast(pl.cast(wt, pl.BF16, mode="rint"), pl.FP32)
                result = pl.col_expand_mul(normalized, wt)
                if gated:
                    zt = pl.cast(
                        pl.slice(z, [8, output_tile], [r0, c0], [valid, output_tile], clamp=True), pl.FP32
                    )
                    silu = pl.mul(zt, pl.recip(pl.add(pl.exp(pl.neg(zt)), 1.0)))
                    result = pl.mul(result, silu)
                out = pl.assemble(out, pl.cast(result, pl.BF16, mode="rint"), [r0, c0])
        return out

    return norm


def make_pointwise(dim, mode, inline=False, tile=128):
    if mode not in ("add", "silu_mul", "sigmoid_mul"):
        raise ValueError(f"Unsupported pointwise operation: {mode}")
    mode_code = {"add": 0, "silu_mul": 1, "sigmoid_mul": 2}[mode]

    decorate = pl.jit.inline if inline else pl.jit

    @decorate
    def pointwise(
        x: pl.Tensor[[ROWS, dim], pl.BF16],
        y: pl.Tensor[[ROWS, dim], pl.BF16],
        out: pl.Out[pl.Tensor[[ROWS, dim], pl.BF16]],
    ):
        rows = pl.tensor.dim(x, 0)
        for job in pl.spmd(rows * ((dim + tile - 1) // tile), name_hint="pointwise"):
            row = job // ((dim + tile - 1) // tile)
            c0 = job % ((dim + tile - 1) // tile) * tile
            valid = pl.min(tile, dim - c0)
            xt = pl.cast(pl.slice(x, [1, tile], [row, c0], [1, valid], clamp=True), pl.FP32)
            yt = pl.cast(pl.slice(y, [1, tile], [row, c0], [1, valid], clamp=True), pl.FP32)
            if mode_code == 0:
                result = pl.add(xt, yt)
            else:
                sigmoid = pl.recip(pl.add(pl.exp(pl.neg(xt)), 1.0))
                if mode_code == 1:
                    sigmoid = pl.mul(sigmoid, xt)
                else:
                    sigmoid = pl.cast(pl.cast(sigmoid, pl.BF16, mode="rint"), pl.FP32)
                result = pl.mul(sigmoid, yt)
            out = pl.assemble(out, pl.cast(result, pl.BF16, mode="rint"), [row, c0])
        return out

    return pointwise
