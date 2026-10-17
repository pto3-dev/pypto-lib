# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Head-aligned BA GEMV + gates; preserve BF16 projection rounding."""

import pypto.language as pl

ROWS = pl.dynamic("rows")


def make_ba_gates(inline=False):
    decorate = pl.jit.inline if inline else pl.jit

    @decorate
    def fused(
        x: pl.Tensor[[ROWS, 5120], pl.BF16],
        weight: pl.Tensor[[48, 5120], pl.BF16],
        a_log: pl.Tensor[[1, 24], pl.FP32],
        dt_bias: pl.Tensor[[1, 24], pl.FP32],
        alpha: pl.Out[pl.Tensor[[ROWS, 24], pl.FP32]],
        beta: pl.Out[pl.Tensor[[ROWS, 24], pl.FP32]],
    ):
        rows = pl.tensor.dim(x, 0)
        available = pl.system.available_aiv_count()
        for core in pl.spmd(pl.min(available, rows * 3), name_hint="ba_gates_fused"):
            for job in pl.range(core, rows * 3, pl.tile.get_block_num()):
                r = job // 3
                h0 = (job % 3) * 8
                seed = pl.full([16, 128], dtype=pl.FP32, value=0.0)
                acc_b = pl.row_sum(seed)
                acc_a = pl.row_sum(seed)
                for kb in pl.pipeline(0, 5, stage=1):
                    xt = pl.cast(pl.slice(x, [1, 1024], [r, kb * 1024]), pl.FP32)
                    wb = pl.cast(
                        pl.slice(
                            weight,
                            [16, 1024],
                            [h0, kb * 1024],
                            valid_shape=[8, 1024],
                            pad_value=0,
                            clamp=True,
                        ),
                        pl.FP32,
                    )
                    acc_b = pl.add(acc_b, pl.row_sum(pl.col_expand_mul(wb, xt)))
                    wa = pl.cast(
                        pl.slice(
                            weight,
                            [16, 1024],
                            [24 + h0, kb * 1024],
                            valid_shape=[8, 1024],
                            pad_value=0,
                            clamp=True,
                        ),
                        pl.FP32,
                    )
                    acc_a = pl.add(acc_a, pl.row_sum(pl.col_expand_mul(wa, xt)))
                bt = pl.cast(pl.cast(pl.reshape(acc_b, [1, 16]), pl.BF16, mode="rint"), pl.FP32)
                at = pl.cast(pl.cast(pl.reshape(acc_a, [1, 16]), pl.BF16, mode="rint"), pl.FP32)
                bias = pl.slice(dt_bias, [1, 16], [0, h0], valid_shape=[1, 8], pad_value=0, clamp=True)
                log_decay = pl.slice(a_log, [1, 16], [0, h0], valid_shape=[1, 8], pad_value=0, clamp=True)
                a_bias = pl.add(at, bias)
                softplus = pl.add(
                    pl.maximum(a_bias, 0.0), pl.log(pl.add(pl.exp(pl.neg(pl.abs(a_bias))), 1.0))
                )
                decay = pl.neg(pl.mul(pl.exp(log_decay), softplus))
                alpha = pl.assemble(alpha, pl.tensor.set_validshape(pl.exp(decay), 1, 8), [r, h0])
                beta_value = pl.recip(pl.add(pl.exp(pl.neg(bt)), 1.0))
                beta = pl.assemble(beta, pl.tensor.set_validshape(beta_value, 1, 8), [r, h0])
        return alpha, beta

    return fused
