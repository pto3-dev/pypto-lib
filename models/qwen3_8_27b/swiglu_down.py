# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Experimental mixed SwiGLU/down projection with the existing BF16 boundary."""

import pypto.language as pl

ROWS = pl.dynamic("rows")


def make_swiglu_down(k=8704, n=5120, nt=128, kt=256, stage=2, inline=False):
    decorate = pl.jit.inline if inline else pl.jit

    @decorate
    def kernel(
        x: pl.Tensor[[ROWS, k * 2], pl.BF16],
        weight: pl.Tensor[[n // nt, k // kt, nt, kt], pl.BF16],
        out: pl.Out[pl.Tensor[[ROWS, n], pl.FP32]],
    ):
        rows = pl.tensor.dim(x, 0)
        for core in pl.spmd(pl.system.available_cluster_count(), name_hint="swiglu_down"):
            for job in pl.range(core, ((rows + 15) // 16) * (n // nt), pl.tile.get_block_num()):
                m0 = job // (n // nt) * 16
                column = job % (n // nt)
                vm = pl.min(16, rows - m0)
                acc = pl.create_tensor([16, nt], pl.FP32)
                for kb in pl.pipeline(0, k // kt, stage=stage):
                    k0 = kb * kt
                    gate = pl.cast(
                        pl.slice(x, [16, kt], [m0, k0], [vm, kt], pad_value=0, clamp=True), pl.FP32
                    )
                    up = pl.cast(
                        pl.slice(x, [16, kt], [m0, k + k0], [vm, kt], pad_value=0, clamp=True), pl.FP32
                    )
                    sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate)), 1.0))
                    activated = pl.cast(pl.mul(pl.mul(sigmoid, gate), up), pl.BF16, mode="rint")
                    tile = pl.reshape(pl.slice(weight, [1, 1, nt, kt], [column, kb, 0, 0]), [nt, kt])
                    acc = pl.matmul_acc(acc, activated, tile, b_trans=True, init_cond=(kb == 0))
                out = pl.assemble(out, pl.tensor.set_validshape(acc, vm, nt), [m0, column * nt])
        return out

    return kernel


# Shared native ND weights use the same BF16 activation boundary.
# Cube performs its regular ND-to-NZ load conversion for each weight tile.
def make_nd_swiglu_down(k=8704, n=5120, nt=128, kt=256, stage=2, inline=False):
    decorate = pl.jit.inline if inline else pl.jit

    @decorate
    def kernel(
        x: pl.Tensor[[ROWS, k * 2], pl.BF16],
        weight: pl.Tensor[[n, k], pl.BF16],
        out: pl.Out[pl.Tensor[[ROWS, n], pl.FP32]],
    ):
        rows = pl.tensor.dim(x, 0)
        for core in pl.spmd(pl.system.available_cluster_count(), name_hint="swiglu_down"):
            for job in pl.range(core, ((rows + 15) // 16) * (n // nt), pl.tile.get_block_num()):
                m0 = job // (n // nt) * 16
                column = job % (n // nt)
                vm = pl.min(16, rows - m0)
                acc = pl.create_tensor([16, nt], pl.FP32)
                for kb in pl.pipeline(0, k // kt, stage=stage):
                    k0 = kb * kt
                    gate = pl.cast(
                        pl.slice(x, [16, kt], [m0, k0], [vm, kt], pad_value=0, clamp=True), pl.FP32
                    )
                    up = pl.cast(
                        pl.slice(x, [16, kt], [m0, k + k0], [vm, kt], pad_value=0, clamp=True), pl.FP32
                    )
                    sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate)), 1.0))
                    activated = pl.cast(pl.mul(pl.mul(sigmoid, gate), up), pl.BF16, mode="rint")
                    tile = pl.slice(weight, [nt, kt], [column * nt, k0])
                    acc = pl.matmul_acc(acc, activated, tile, b_trans=True, init_cond=(kb == 0))
                out = pl.assemble(out, pl.tensor.set_validshape(acc, vm, nt), [m0, column * nt])
        return out

    return kernel
