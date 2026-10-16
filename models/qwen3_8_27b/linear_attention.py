# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""PTO causal convolution, QK normalization, gates, and recurrent GDN."""

import pypto.language as pl

TOKENS = pl.dynamic("tokens")
SLOTS = pl.dynamic("slots")
REQUESTS = pl.dynamic("requests")
BOUNDARIES = pl.dynamic("boundaries")


def make_conv(channels, inline=False, tokens=TOKENS):
    decorate = pl.jit.inline if inline else pl.jit

    @decorate
    def causal_conv(
        x: pl.Tensor[[tokens, channels], pl.BF16],
        weight: pl.Tensor[[4, channels], pl.BF16],
        state: pl.InOut[pl.Tensor[[SLOTS, 3, channels], pl.BF16]],
        starts: pl.Tensor[[BOUNDARIES], pl.INT32],
        indices: pl.Tensor[[REQUESTS], pl.INT32],
        initial: pl.Tensor[[REQUESTS], pl.INT32],
        out: pl.Out[pl.Tensor[[tokens, channels], pl.BF16]],
    ):
        flat = pl.reshape(state, [pl.tensor.dim(state, 0) * 3, channels])
        requests = pl.tensor.dim(indices, 0)
        for job in pl.spmd(requests * (channels // 128), name_hint="causal_conv"):
            request = job // (channels // 128)
            c0 = job % (channels // 128) * 128
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
                    out = pl.assemble(out, pl.cast(result, pl.BF16, mode="rint"), [t, c0])
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
        return out, state

    return causal_conv


def make_qk_norm(heads, inline=False, tokens=TOKENS, refine=True):
    decorate = pl.jit.inline if inline else pl.jit

    @decorate
    def qk_norm(
        q: pl.Tensor[[tokens, heads, 128], pl.BF16],
        k: pl.Tensor[[tokens, heads, 128], pl.BF16],
        q_out: pl.Out[pl.Tensor[[tokens, heads, 128], pl.FP32]],
        k_out: pl.Out[pl.Tensor[[tokens, heads, 128], pl.FP32]],
    ):
        rows = pl.tensor.dim(q, 0) * heads
        q_flat = pl.reshape(q, [rows, 128])
        k_flat = pl.reshape(k, [rows, 128])
        qo = pl.reshape(q_out, [rows, 128])
        ko = pl.reshape(k_out, [rows, 128])
        for job in pl.spmd((rows + 7) // 8, name_hint="gdn_qk_norm"):
            row = job * 8
            valid = pl.min(8, rows - row)
            qt = pl.cast(pl.slice(q_flat, [8, 128], [row, 0], [valid, 128], clamp=True), pl.FP32)
            kt = pl.cast(pl.slice(k_flat, [8, 128], [row, 0], [valid, 128], clamp=True), pl.FP32)
            q_variance = pl.add(pl.row_sum(pl.mul(qt, qt)), 1e-6)
            k_variance = pl.add(pl.row_sum(pl.mul(kt, kt)), 1e-6)
            qi = pl.rsqrt(q_variance)
            ki = pl.rsqrt(k_variance)
            if refine:
                qi = pl.mul(qi, pl.add(pl.mul(pl.mul(q_variance, pl.mul(qi, qi)), -0.5), 1.5))
                ki = pl.mul(ki, pl.add(pl.mul(pl.mul(k_variance, pl.mul(ki, ki)), -0.5), 1.5))
                qi = pl.mul(qi, pl.add(pl.mul(pl.mul(q_variance, pl.mul(qi, qi)), -0.5), 1.5))
                ki = pl.mul(ki, pl.add(pl.mul(pl.mul(k_variance, pl.mul(ki, ki)), -0.5), 1.5))
            qn = pl.row_expand_mul(qt, pl.mul(qi, 0.08838834764831845))
            kn = pl.row_expand_mul(kt, ki)
            qo = pl.assemble(qo, qn, [row, 0])
            ko = pl.assemble(ko, kn, [row, 0])
        return q_out, k_out

    return qk_norm


def make_gates(heads, inline=False, tokens=TOKENS):
    decorate = pl.jit.inline if inline else pl.jit

    @decorate
    def gates(
        a: pl.Tensor[[tokens, heads], pl.BF16],
        b: pl.Tensor[[tokens, heads], pl.BF16],
        a_log: pl.Tensor[[1, heads], pl.FP32],
        dt_bias: pl.Tensor[[1, heads], pl.FP32],
        alpha: pl.Out[pl.Tensor[[tokens, heads], pl.FP32]],
        beta: pl.Out[pl.Tensor[[tokens, heads], pl.FP32]],
    ):
        tokens = pl.tensor.dim(a, 0)
        for t in pl.spmd(tokens, name_hint="gdn_gates"):
            at = pl.cast(pl.slice(a, [1, heads], [t, 0]), pl.FP32)
            bt = pl.cast(pl.slice(b, [1, heads], [t, 0]), pl.FP32)
            x = pl.add(at, pl.slice(dt_bias, [1, heads], [0, 0]))
            softplus = pl.add(pl.maximum(x, 0.0), pl.log(pl.add(pl.exp(pl.neg(pl.abs(x))), 1.0)))
            decay = pl.neg(pl.mul(pl.exp(pl.slice(a_log, [1, heads], [0, 0])), softplus))
            alpha = pl.assemble(alpha, pl.exp(decay), [t, 0])
            beta_t = pl.recip(pl.add(pl.exp(pl.neg(bt)), 1.0))
            beta = pl.assemble(beta, beta_t, [t, 0])
        return alpha, beta

    return gates


def make_recurrent(key_heads, value_heads, state_layout="VK", inline=False, tokens=TOKENS):
    if state_layout not in ("VK", "KV"):
        raise ValueError("GDN state layout must be VK (upstream vLLM) or KV (vLLM-Ascend).")
    transpose_state = int(state_layout == "KV")

    decorate = pl.jit.inline if inline else pl.jit

    @decorate
    def recurrent(
        q: pl.Tensor[[tokens, key_heads, 128], pl.FP32],
        k: pl.Tensor[[tokens, key_heads, 128], pl.FP32],
        v: pl.Tensor[[tokens, value_heads, 128], pl.BF16],
        alpha: pl.Tensor[[tokens, value_heads], pl.FP32],
        beta: pl.Tensor[[tokens, value_heads], pl.FP32],
        state: pl.InOut[pl.Tensor[[SLOTS, value_heads, 128, 128], pl.FP32]],
        starts: pl.Tensor[[BOUNDARIES], pl.INT32],
        indices: pl.Tensor[[REQUESTS], pl.INT32],
        initial: pl.Tensor[[REQUESTS], pl.INT32],
        out: pl.Out[pl.Tensor[[tokens, value_heads, 128], pl.BF16]],
    ):
        s = pl.reshape(state, [pl.tensor.dim(state, 0) * value_heads * 128, 128])
        qf = pl.reshape(q, [pl.tensor.dim(q, 0) * key_heads, 128])
        kf = pl.reshape(k, [pl.tensor.dim(k, 0) * key_heads, 128])
        vf = pl.reshape(v, [pl.tensor.dim(v, 0), value_heads * 128])
        of = pl.reshape(out, [pl.tensor.dim(out, 0), value_heads * 128])
        requests = pl.tensor.dim(indices, 0)
        for job in pl.spmd(requests * value_heads * 4, name_hint="gdn_recurrent"):
            request = job // (value_heads * 4)
            head = job // 4 % value_heads
            v0 = job % 4 * 32
            key_head = head // (value_heads // key_heads)
            slot = pl.read(indices, [request])
            begin = pl.read(starts, [request])
            end = pl.read(starts, [request + 1])
            if slot >= 0:
                srow = (slot * value_heads + head) * 128 + v0
                st = pl.full([32, 128], dtype=pl.FP32, value=0.0)
                if pl.read(initial, [request]) != 0:
                    if transpose_state == 0:
                        st = pl.slice(s, [32, 128], [srow, 0])
                    else:
                        native = pl.slice(s, [128, 32], [(slot * value_heads + head) * 128, v0])
                        transposed = pl.transpose(native, 0, 1)
                        st = pl.add(transposed, 0.0)
                for t in pl.range(begin, end):
                    qt = pl.slice(qf, [1, 128], [t * key_heads + key_head, 0])
                    kt = pl.slice(kf, [1, 128], [t * key_heads + key_head, 0])
                    vt = pl.cast(pl.slice(vf, [1, 32], [t, head * 128 + v0]), pl.FP32)
                    st = pl.mul(st, pl.read(alpha, [t, head]))
                    prediction = pl.row_sum(pl.col_expand_mul(st, kt))
                    error = pl.sub(pl.reshape(vt, [32, 1]), prediction)
                    error = pl.mul(error, pl.read(beta, [t, head]))
                    update = pl.row_expand_mul(
                        pl.col_expand_mul(pl.full([32, 128], dtype=pl.FP32, value=1.0), kt), error
                    )
                    st = pl.add(st, update)
                    result = pl.reshape(pl.row_sum(pl.col_expand_mul(st, qt)), [1, 32])
                    of = pl.assemble(of, pl.cast(result, pl.BF16, mode="rint"), [t, head * 128 + v0])
                if transpose_state == 0:
                    s = pl.assemble(s, st, [srow, 0])
                else:
                    native_out = pl.transpose(st, 0, 1)
                    s = pl.assemble(s, native_out, [(slot * value_heads + head) * 128, v0])
            else:
                for t in pl.range(begin, end):
                    of = pl.assemble(of, pl.full([1, 32], dtype=pl.BF16, value=0.0), [t, head * 128 + v0])
        return out, state

    return recurrent
