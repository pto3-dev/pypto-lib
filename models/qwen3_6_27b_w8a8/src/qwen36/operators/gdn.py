# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""gdn operators. Explicit PyPTO functions; no source generator at build time.

Migrated from the validated L1K2560 graph; tuning is audited separately.
"""

import pypto.language as pl


@pl.function(type=pl.FunctionType.Inline)
def convolve(
    x: pl.Tensor[[24, 2560], pl.BF16],
    weight: pl.Tensor[[4, 2560], pl.BF16],
    state: pl.Tensor[[192, 8, 2560], pl.BF16],
    starts: pl.Tensor[[25], pl.INT32],
    indices: pl.Tensor[[24], pl.INT32],
    initial: pl.Tensor[[24], pl.INT32],
    accepted: pl.Tensor[[24], pl.INT32],
    out: pl.Tensor[[24, 2560], pl.BF16],
    q_out: pl.Tensor[[24, 4, 128], pl.FP32],
    k_out: pl.Tensor[[24, 4, 128], pl.FP32],
    v_out: pl.Tensor[[24, 12, 128], pl.BF16],
) -> tuple[
    pl.Tensor[[24, 2560], pl.BF16],
    pl.Tensor[[24, 4, 128], pl.FP32],
    pl.Tensor[[24, 4, 128], pl.FP32],
    pl.Tensor[[24, 12, 128], pl.BF16],
    pl.Tensor[[192, 8, 2560], pl.BF16],
]:
    flat = pl.reshape(state, [192 * 8, 2560])
    qflat = pl.reshape(q_out, [24 * 4, 128])
    kflat = pl.reshape(k_out, [24 * 4, 128])
    vflat = pl.reshape(v_out, [24 * 12, 128])
    requests = 4
    available = pl.system.available_aiv_count()
    width = pl.min(available, requests * 20)
    for core in pl.spmd(width, name_hint="causal_conv_qk_norm", allow_early_resolve=True):
        for job in pl.range(core, requests * 20, pl.tile.get_block_num()):
            request = job // 20
            c0 = job % 20 * 128
            slot = pl.read(initial, [request + 4])
            history_offset = pl.read(accepted, [request]) - 1
            begin = pl.read(starts, [request])
            end = pl.read(starts, [request + 1])
            if slot >= 0:
                h0 = pl.full([1, 128], dtype=pl.BF16, value=0.0)
                h1 = pl.full([1, 128], dtype=pl.BF16, value=0.0)
                h2 = pl.full([1, 128], dtype=pl.BF16, value=0.0)
                if slot >= 0:
                    h0 = pl.slice(flat, [1, 128], [slot * 8 + history_offset, c0])
                    h1 = pl.slice(flat, [1, 128], [slot * 8 + history_offset + 1, c0])
                    h2 = pl.slice(flat, [1, 128], [slot * 8 + history_offset + 2, c0])
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
                    if c0 < 2 * 512:
                        value = pl.cast(rounded, pl.FP32)
                        padded = pl.reshape(
                            pl.concat(value, pl.full([1, 896], dtype=pl.FP32, value=0.0)), [8, 128]
                        )
                        variance = pl.add(pl.row_sum(pl.mul(padded, padded)), 1e-06)
                        inv = pl.rsqrt(variance)
                        inv = pl.mul(inv, pl.add(pl.mul(pl.mul(variance, pl.mul(inv, inv)), -0.5), 1.5))
                        inv = pl.mul(inv, pl.add(pl.mul(pl.mul(variance, pl.mul(inv, inv)), -0.5), 1.5))
                        scale = pl.read(inv, [0, 0])
                        if c0 < 512:
                            qn = pl.mul(value, pl.mul(scale, 0.08838834764831845))
                            qflat = pl.assemble(qflat, qn, [t * 4 + c0 // 128, 0])
                        else:
                            kn = pl.mul(value, scale)
                            kflat = pl.assemble(kflat, kn, [t * 4 + (c0 - 512) // 128, 0])
                    else:
                        vflat = pl.assemble(vflat, rounded, [t * 12 + (c0 - 2 * 512) // 128, 0])
                    h0 = h1
                    h1 = h2
                    h2 = xt
                    if t == begin:
                        flat = pl.assemble(flat, h0, [slot * 8, c0])
                        flat = pl.assemble(flat, h1, [slot * 8 + 1, c0])
                        flat = pl.assemble(flat, h2, [slot * 8 + 2, c0])
                    else:
                        flat = pl.assemble(flat, h2, [slot * 8 + 2 + t - begin, c0])
            else:
                for t in pl.range(begin, end):
                    zero = pl.full([1, 128], dtype=pl.BF16, value=0.0)
                    out = pl.assemble(out, zero, [t, c0])
                    if c0 < 512:
                        qflat = pl.assemble(
                            qflat, pl.full([1, 128], dtype=pl.FP32, value=0.0), [t * 4 + c0 // 128, 0]
                        )
                    elif c0 < 2 * 512:
                        kflat = pl.assemble(
                            kflat, pl.full([1, 128], dtype=pl.FP32, value=0.0), [t * 4 + (c0 - 512) // 128, 0]
                        )
                    else:
                        vflat = pl.assemble(vflat, zero, [t * 12 + (c0 - 2 * 512) // 128, 0])
    return (out, q_out, k_out, v_out, state)


@pl.function(type=pl.FunctionType.InCore)
def recurrent_kernel(
    q: pl.Tensor[[24, 4, 128], pl.FP32],
    k: pl.Tensor[[24, 4, 128], pl.FP32],
    v: pl.Tensor[[24, 12, 128], pl.BF16],
    alpha: pl.Tensor[[24, 12], pl.FP32],
    beta: pl.Tensor[[24, 12], pl.FP32],
    state: pl.Tensor[[192, 12, 128, 128], pl.FP32],
    starts: pl.Tensor[[25], pl.INT32],
    indices: pl.Tensor[[24], pl.INT32],
    initial: pl.Tensor[[24], pl.INT32],
    out: pl.Tensor[[24, 12, 128], pl.BF16],
    s: pl.Tensor[[294912, 128], pl.FP32],
    qf: pl.Tensor[[96, 128], pl.FP32],
    kf: pl.Tensor[[96, 128], pl.FP32],
    vf: pl.Tensor[[24, 1536], pl.BF16],
    of: pl.Tensor[[24, 1536], pl.BF16],
) -> tuple[pl.Tensor[[24, 12, 128], pl.BF16], pl.Tensor[[192, 12, 128, 128], pl.FP32]]:
    pl.func_attr({"static_block_mapping": True})
    requests = 4
    vector_core = pl.tile.get_block_idx()
    for job in pl.range(vector_core, requests * 12 * 4, pl.tile.get_block_num()):
        request = job // (12 * 4)
        head = job // 4 % 12
        v0 = job % 4 * 32
        key_head = head // (12 // 4)
        slot = pl.read(indices, [request])
        begin = pl.read(starts, [request])
        end = pl.read(starts, [request + 1])
        if slot >= 0:
            srow = (slot * 12 + head) * 128 + v0
            st = pl.full([32, 128], dtype=pl.FP32, value=0.0)
            if slot >= 0:
                if 1 == 0:
                    st = pl.slice(s, [32, 128], [srow, 0])
                else:
                    native = pl.slice(s, [128, 32], [(slot * 12 + head) * 128, v0])
                    transposed = pl.transpose(native, 0, 1)
                    st = pl.add(transposed, 0.0)
            for t in pl.range(begin, end):
                qt = pl.slice(qf, [1, 128], [t * 4 + key_head, 0])
                kt = pl.slice(kf, [1, 128], [t * 4 + key_head, 0])
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
                destination = pl.read(initial, [request])
                if t != begin:
                    destination = pl.read(indices, [request + (t - begin) * 4])
                if 1 == 0:
                    s = pl.assemble(s, st, [(destination * 12 + head) * 128 + v0, 0])
                else:
                    native_out = pl.transpose(st, 0, 1)
                    s = pl.assemble(s, native_out, [(destination * 12 + head) * 128, v0])
        else:
            for t in pl.range(begin, end):
                of = pl.assemble(of, pl.full([1, 32], dtype=pl.BF16, value=0.0), [t, head * 128 + v0])
    return (out, state)


@pl.function(type=pl.FunctionType.Inline)
def recurrent(
    q: pl.Tensor[[24, 4, 128], pl.FP32],
    k: pl.Tensor[[24, 4, 128], pl.FP32],
    v: pl.Tensor[[24, 12, 128], pl.BF16],
    alpha: pl.Tensor[[24, 12], pl.FP32],
    beta: pl.Tensor[[24, 12], pl.FP32],
    state: pl.Tensor[[192, 12, 128, 128], pl.FP32],
    starts: pl.Tensor[[25], pl.INT32],
    indices: pl.Tensor[[24], pl.INT32],
    initial: pl.Tensor[[24], pl.INT32],
    out: pl.Tensor[[24, 12, 128], pl.BF16],
) -> tuple[pl.Tensor[[24, 12, 128], pl.BF16], pl.Tensor[[192, 12, 128, 128], pl.FP32]]:
    s = pl.reshape(state, [192 * 12 * 128, 128])
    qf = pl.reshape(q, [24 * 4, 128])
    kf = pl.reshape(k, [24 * 4, 128])
    vf = pl.reshape(v, [24, 12 * 128])
    of = pl.reshape(out, [24, 12 * 128])
    requests = 4
    available = pl.system.available_aiv_count()
    width = pl.min(available, requests * 12 * 4)
    with pl.spmd(width, allow_early_resolve=True):
        out, state = recurrent_kernel(
            q, k, v, alpha, beta, state, starts, indices, initial, out, s, qf, kf, vf, of
        )
    return (out, state)


@pl.function(type=pl.FunctionType.Inline)
def gates(
    x: pl.Tensor[[24, 5120], pl.BF16],
    weight: pl.Tensor[[24, 5120], pl.BF16],
    a_log: pl.Tensor[[1, 12], pl.FP32],
    dt_bias: pl.Tensor[[1, 12], pl.FP32],
    alpha: pl.Tensor[[24, 12], pl.FP32],
    beta: pl.Tensor[[24, 12], pl.FP32],
) -> tuple[pl.Tensor[[24, 12], pl.FP32], pl.Tensor[[24, 12], pl.FP32]]:
    rows = 24
    available = pl.system.available_aiv_count()
    for core in pl.spmd(pl.min(available, rows * 2), name_hint="ba_gates_fused", allow_early_resolve=True):
        for job in pl.range(core, rows * 2, pl.tile.get_block_num()):
            r = job // 2
            h0 = job % 2 * 8
            valid = pl.min(8, 12 - h0)
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
                        valid_shape=[valid, 1024],
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
                        [12 + h0, kb * 1024],
                        valid_shape=[valid, 1024],
                        pad_value=0,
                        clamp=True,
                    ),
                    pl.FP32,
                )
                acc_a = pl.add(acc_a, pl.row_sum(pl.col_expand_mul(wa, xt)))
            bt = pl.cast(pl.cast(pl.reshape(acc_b, [1, 16]), pl.BF16, mode="rint"), pl.FP32)
            at = pl.cast(pl.cast(pl.reshape(acc_a, [1, 16]), pl.BF16, mode="rint"), pl.FP32)
            bias = pl.slice(dt_bias, [1, 16], [0, h0], valid_shape=[1, valid], pad_value=0, clamp=True)
            log_decay = pl.slice(a_log, [1, 16], [0, h0], valid_shape=[1, valid], pad_value=0, clamp=True)
            a_bias = pl.add(at, bias)
            softplus = pl.add(pl.maximum(a_bias, 0.0), pl.log(pl.add(pl.exp(pl.neg(pl.abs(a_bias))), 1.0)))
            decay = pl.neg(pl.mul(pl.exp(log_decay), softplus))
            alpha = pl.assemble(alpha, pl.tensor.set_validshape(pl.exp(decay), 1, valid), [r, h0])
            beta_value = pl.recip(pl.add(pl.exp(pl.neg(bt)), 1.0))
            beta = pl.assemble(beta, pl.tensor.set_validshape(beta_value, 1, valid), [r, h0])
    return (alpha, beta)
