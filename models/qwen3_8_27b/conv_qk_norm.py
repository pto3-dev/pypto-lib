# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Decode convolution with head-local Q/K normalization on the producing AIV."""

import pypto.language as pl

TOKENS = pl.dynamic("tokens")
SLOTS = pl.dynamic("slots")
REQUESTS = pl.dynamic("requests")
BOUNDARIES = pl.dynamic("boundaries")


def make_conv_qk_norm(inline=False, tokens=TOKENS):
    decorate = pl.jit.inline if inline else pl.jit

    @decorate
    def causal_conv(
        x: pl.Tensor[[tokens, 5120], pl.BF16],
        weight: pl.Tensor[[4, 5120], pl.BF16],
        state: pl.InOut[pl.Tensor[[SLOTS, 3, 5120], pl.BF16]],
        starts: pl.Tensor[[BOUNDARIES], pl.INT32],
        indices: pl.Tensor[[REQUESTS], pl.INT32],
        initial: pl.Tensor[[REQUESTS], pl.INT32],
        out: pl.Out[pl.Tensor[[tokens, 5120], pl.BF16]],
        q_out: pl.Out[pl.Tensor[[tokens, 8, 128], pl.FP32]],
        k_out: pl.Out[pl.Tensor[[tokens, 8, 128], pl.FP32]],
    ):
        flat = pl.reshape(state, [pl.tensor.dim(state, 0) * 3, 5120])
        qflat = pl.reshape(q_out, [tokens * 8, 128])
        kflat = pl.reshape(k_out, [tokens * 8, 128])
        requests = pl.tensor.dim(indices, 0)
        for job in pl.spmd(requests * 40, name_hint="causal_conv_qk_norm"):
            request = job // 40
            c0 = job % 40 * 128
            slot = pl.read(indices, [request])
            begin = pl.read(starts, [request])
            end = pl.read(starts, [request + 1])
            if slot >= 0:
                h0 = pl.full([1, 128], dtype=pl.BF16, value=0.0)
                h1 = pl.full([1, 128], dtype=pl.BF16, value=0.0)
                h2 = pl.full([1, 128], dtype=pl.BF16, value=0.0)
                if pl.read(initial, [request]) != 0:
                    h0 = pl.slice(flat, [1, 128], [slot * 3, c0])
                    h1 = pl.slice(flat, [1, 128], [slot * 3 + 1, c0])
                    h2 = pl.slice(flat, [1, 128], [slot * 3 + 2, c0])
                w0 = pl.cast(pl.slice(weight, [1, 128], [0, c0]), pl.FP32)
                w1 = pl.cast(pl.slice(weight, [1, 128], [1, c0]), pl.FP32)
                w2 = pl.cast(pl.slice(weight, [1, 128], [2, c0]), pl.FP32)
                w3 = pl.cast(pl.slice(weight, [1, 128], [3, c0]), pl.FP32)
                for t in pl.range(begin, end):
                    xt = pl.slice(x, [1, 128], [t, c0])
                    acc = pl.mul(pl.cast(h0, pl.FP32), w0)
                    acc = pl.add(acc, pl.mul(pl.cast(h1, pl.FP32), w1))
                    acc = pl.add(acc, pl.mul(pl.cast(h2, pl.FP32), w2))
                    acc = pl.add(acc, pl.mul(pl.cast(xt, pl.FP32), w3))
                    denom = pl.add(pl.exp(pl.neg(acc)), 1.0)
                    result = pl.mul(acc, pl.recip(denom))
                    rounded = pl.cast(result, pl.BF16, mode="rint")
                    out = pl.assemble(out, rounded, [t, c0])
                    if c0 < 2048:
                        value = pl.cast(rounded, pl.FP32)
                        padded = pl.reshape(
                            pl.concat(value, pl.full([1, 896], dtype=pl.FP32, value=0.0)), [8, 128]
                        )
                        variance = pl.add(pl.row_sum(pl.mul(padded, padded)), 1e-6)
                        inv = pl.rsqrt(variance)
                        inv = pl.mul(inv, pl.add(pl.mul(pl.mul(variance, pl.mul(inv, inv)), -0.5), 1.5))
                        inv = pl.mul(inv, pl.add(pl.mul(pl.mul(variance, pl.mul(inv, inv)), -0.5), 1.5))
                        scale = pl.read(inv, [0, 0])
                        if c0 < 1024:
                            qn = pl.mul(value, pl.mul(scale, 0.08838834764831845))
                            qflat = pl.assemble(qflat, qn, [t * 8 + c0 // 128, 0])
                        else:
                            kn = pl.mul(value, scale)
                            kflat = pl.assemble(kflat, kn, [t * 8 + (c0 - 1024) // 128, 0])
                    h0 = h1
                    h1 = h2
                    h2 = xt
                flat = pl.assemble(flat, h0, [slot * 3, c0])
                flat = pl.assemble(flat, h1, [slot * 3 + 1, c0])
                flat = pl.assemble(flat, h2, [slot * 3 + 2, c0])
            else:
                for t in pl.range(begin, end):
                    zero = pl.full([1, 128], dtype=pl.BF16, value=0.0)
                    out = pl.assemble(out, zero, [t, c0])
                    if c0 < 1024:
                        qflat = pl.assemble(
                            qflat, pl.full([1, 128], dtype=pl.FP32, value=0.0), [t * 8 + c0 // 128, 0]
                        )
                    elif c0 < 2048:
                        kflat = pl.assemble(
                            kflat,
                            pl.full([1, 128], dtype=pl.FP32, value=0.0),
                            [t * 8 + (c0 - 1024) // 128, 0],
                        )
        return out, q_out, k_out, state

    return causal_conv
