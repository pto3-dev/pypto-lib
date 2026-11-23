# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from ..sequence_config import CACHE_ROWS,STATE_SLOTS
"""gdn operators. Explicit PyPTO functions; no source generator at build time.

Migrated from the validated L1K2560 graph; tuning is audited separately.
"""
import pypto.language as pl

@pl.function(type=pl.FunctionType.Inline)
def convolve(x: pl.Tensor[[24, 2560], pl.BF16], weight: pl.Tensor[[4, 2560], pl.BF16], state: pl.Tensor[[STATE_SLOTS, 8, 2560], pl.BF16], starts: pl.Tensor[[25], pl.INT32], indices: pl.Tensor[[24], pl.INT32], initial: pl.Tensor[[24], pl.INT32], accepted: pl.Tensor[[24], pl.INT32], out: pl.Tensor[[24, 2560], pl.BF16], q_out: pl.Tensor[[24, 4, 128], pl.FP32], k_out: pl.Tensor[[24, 4, 128], pl.FP32], v_out: pl.Tensor[[24, 12, 128], pl.BF16]) -> tuple[pl.Tensor[[24, 2560], pl.BF16], pl.Tensor[[24, 4, 128], pl.FP32], pl.Tensor[[24, 4, 128], pl.FP32], pl.Tensor[[24, 12, 128], pl.BF16], pl.Tensor[[STATE_SLOTS, 8, 2560], pl.BF16]]:
    flat = pl.reshape(state, [STATE_SLOTS * 8, 2560])
    qflat = pl.reshape(q_out, [24 * 4, 128])
    kflat = pl.reshape(k_out, [24 * 4, 128])
    vflat = pl.reshape(v_out, [24 * 12, 128])
    requests = 4
    available = pl.system.available_aiv_count()
    width = pl.min(available, requests * 20)
    for core in pl.spmd(width, name_hint='causal_conv_qk_norm', allow_early_resolve=True):
        for job in pl.range(core, requests * 20, pl.tile.get_block_num()):
            request = job // 20
            c0 = job % 20 * 128
            slot = pl.read(initial, [request + 4])
            history_offset = pl.read(accepted, [request]) - 1
            begin = pl.read(starts, [request])
            end = pl.read(starts, [request + 5])
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
                    # Preserve native SiLU's division before BF16 storage.
                    result = pl.div(acc, denom)
                    rounded = pl.cast(result, pl.BF16, mode='rint')
                    out = pl.assemble(out, rounded, [t, c0])
                    if c0 < 2 * 512:
                        value = pl.cast(rounded, pl.FP32)
                        padded = pl.reshape(pl.concat(value, pl.full([1, 896], dtype=pl.FP32, value=0.0)), [8, 128])
                        variance = pl.add(pl.row_sum(pl.mul(padded, padded)), 1e-06)
                        # Real native l2norm calibration: preserve sqrt/recip
                        # rounding before the normalized BF16 intermediate.
                        inv = pl.recip(pl.sqrt(variance))
                        scale = pl.read(inv, [0, 0])
                        # Native l2norm_fwd stores its BF16 input dtype before
                        # recurrent_gated_delta_rule casts Q/K back to FP32.
                        normalized = pl.cast(pl.cast(pl.mul(value, scale), pl.BF16, mode='rint'), pl.FP32)
                        if c0 < 512:
                            qn = pl.mul(normalized, 0.08838834764831845)
                            qflat = pl.assemble(qflat, qn, [t * 4 + c0 // 128, 0])
                        else:
                            kn = normalized
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
                        qflat = pl.assemble(qflat, pl.full([1, 128], dtype=pl.FP32, value=0.0), [t * 4 + c0 // 128, 0])
                    elif c0 < 2 * 512:
                        kflat = pl.assemble(kflat, pl.full([1, 128], dtype=pl.FP32, value=0.0), [t * 4 + (c0 - 512) // 128, 0])
                    else:
                        vflat = pl.assemble(vflat, zero, [t * 12 + (c0 - 2 * 512) // 128, 0])
    return (out, q_out, k_out, v_out, state)

@pl.function(type=pl.FunctionType.InCore)
def recurrent_kernel(q: pl.Tensor[[24, 4, 128], pl.FP32], k: pl.Tensor[[24, 4, 128], pl.FP32], v: pl.Tensor[[24, 12, 128], pl.BF16], alpha: pl.Tensor[[24, 12], pl.FP32], beta: pl.Tensor[[24, 12], pl.FP32], state: pl.Tensor[[STATE_SLOTS, 12, 128, 128], pl.FP32], starts: pl.Tensor[[25], pl.INT32], indices: pl.Tensor[[24], pl.INT32], initial: pl.Tensor[[24], pl.INT32], out: pl.Tensor[[24, 12, 128], pl.BF16], s: pl.Tensor[[CACHE_ROWS, 128], pl.FP32], qf: pl.Tensor[[96, 128], pl.FP32], kf: pl.Tensor[[96, 128], pl.FP32], vf: pl.Tensor[[24, 1536], pl.BF16], of: pl.Tensor[[24, 1536], pl.BF16]) -> tuple[pl.Tensor[[24, 12, 128], pl.BF16], pl.Tensor[[STATE_SLOTS, 12, 128, 128], pl.FP32]]:
    pl.func_attr({'static_block_mapping': True})
    requests = 4
    vector_core = pl.tile.get_block_idx()
    for job in pl.range(vector_core, requests * 12 * 2, pl.tile.get_block_num()):
        request = job // (12 * 2)
        head = job // 2 % 12
        v0 = job % 2 * 64
        key_head = head // (12 // 4)
        slot = pl.read(indices, [request])
        begin = pl.read(starts, [request])
        end = pl.read(starts, [request + 5])
        if slot >= 0:
            srow = (slot * 12 + head) * 128 + v0
            st = pl.full([64, 128], dtype=pl.FP32, value=0.0)
            # Native 0.23 stores [slot, head, V, K]; no state mirror or transpose.
            st = pl.slice(s, [64, 128], [srow, 0])
            for t in pl.range(begin, end):
                qt = pl.slice(qf, [1, 128], [t * 4 + key_head, 0])
                kt = pl.slice(kf, [1, 128], [t * 4 + key_head, 0])
                vt = pl.cast(pl.slice(vf, [1, 64], [t, head * 128 + v0]), pl.FP32)
                st = pl.mul(st, pl.read(alpha, [t, head]))
                prediction = pl.row_sum(pl.col_expand_mul(st, kt))
                error = pl.sub(pl.reshape(vt, [64, 1]), prediction)
                error = pl.mul(error, pl.read(beta, [t, head]))
                update = pl.row_expand_mul(pl.col_expand(st, kt), error)
                st = pl.add(st, update)
                result = pl.reshape(pl.row_sum(pl.col_expand_mul(st, qt)), [1, 64])
                of = pl.assemble(of, pl.cast(result, pl.BF16, mode='rint'), [t, head * 128 + v0])
                destination = pl.read(initial, [request])
                if t != begin:
                    destination = pl.read(indices, [request + (t - begin) * 4])
                s = pl.assemble(s, st, [(destination * 12 + head) * 128 + v0, 0])
        else:
            for t in pl.range(begin, end):
                of = pl.assemble(of, pl.full([1, 64], dtype=pl.BF16, value=0.0), [t, head * 128 + v0])
    return (out, state)

@pl.function(type=pl.FunctionType.Inline)
def recurrent(q: pl.Tensor[[24, 4, 128], pl.FP32], k: pl.Tensor[[24, 4, 128], pl.FP32], v: pl.Tensor[[24, 12, 128], pl.BF16], alpha: pl.Tensor[[24, 12], pl.FP32], beta: pl.Tensor[[24, 12], pl.FP32], state: pl.Tensor[[STATE_SLOTS, 12, 128, 128], pl.FP32], starts: pl.Tensor[[25], pl.INT32], indices: pl.Tensor[[24], pl.INT32], initial: pl.Tensor[[24], pl.INT32], out: pl.Tensor[[24, 12, 128], pl.BF16]) -> tuple[pl.Tensor[[24, 12, 128], pl.BF16], pl.Tensor[[STATE_SLOTS, 12, 128, 128], pl.FP32]]:
    s = pl.reshape(state, [STATE_SLOTS * 12 * 128, 128])
    qf = pl.reshape(q, [24 * 4, 128])
    kf = pl.reshape(k, [24 * 4, 128])
    vf = pl.reshape(v, [24, 12 * 128])
    of = pl.reshape(out, [24, 12 * 128])
    requests = 4
    available = pl.system.available_aiv_count()
    width = pl.min(available, requests * 12 * 2)
    with pl.spmd(width, allow_early_resolve=True):
        out, state = recurrent_kernel(q, k, v, alpha, beta, state, starts, indices, initial, out, s, qf, kf, vf, of)
    return (out, state)

@pl.function(type=pl.FunctionType.InCore, attrs={'split':pl.SplitMode.UP_DOWN})
def gates_kernel(x:pl.Tensor[[24,5120],pl.BF16],weight:pl.Tensor[[24,5120],pl.BF16],
        a_log:pl.Tensor[[1,12],pl.FP32],dt_bias:pl.Tensor[[1,12],pl.FP32],
        alpha:pl.Out[pl.Tensor[[24,12],pl.FP32]],beta:pl.Out[pl.Tensor[[24,12],pl.FP32]]):
    # Real native payload98513: K512 L1 / K128 L0, ascending K, no split-K.
    # One complete M tile shares A for B/A; both AIV lanes own complete heads.
    # The C->V moves are normal DSL boundaries, lowered by the mixed-kernel pass.
    activation=pl.tile.load(x,[0,0],[32,5120],valid_shape=[24,5120],target_memory=pl.Mem.Mat)
    acc_b=pl.tile.create([32,16],pl.FP32,target_memory=pl.Mem.Acc,compact=True)
    acc_a=pl.tile.create([32,16],pl.FP32,target_memory=pl.Mem.Acc,compact=True)
    for k1 in pl.pipeline(0,5120,512,stage=2):
        wb=pl.tile.load(weight,[0,k1],[16,512],valid_shape=[12,512],target_memory=pl.Mem.Mat,cache=pl.CachePolicy.BYPASS)
        wa=pl.tile.load(weight,[12,k1],[16,512],valid_shape=[12,512],target_memory=pl.Mem.Mat,cache=pl.CachePolicy.BYPASS)
        bt=pl.tile.transpose_view(wb)
        at=pl.tile.transpose_view(wa)
        for k0 in pl.pipeline(0,512,128,stage=2):
            left=pl.tile.extract(activation,0,k1+k0,[32,128],target_memory=pl.Mem.Left)
            rb=pl.tile.extract(bt,k0,0,[128,16],target_memory=pl.Mem.Right)
            acc_b=pl.tile.matmul_acc(acc_b,left,rb,init_cond=k1==0 and k0==0)
            ra=pl.tile.extract(at,k0,0,[128,16],target_memory=pl.Mem.Right)
            acc_a=pl.tile.matmul_acc(acc_a,left,ra,init_cond=k1==0 and k0==0)
    b_transport=pl.tile.move(pl.tile.set_validshape(acc_b,32,16),target_memory=pl.Mem.Vec,blayout=pl.TileLayout.row_major,slayout=pl.TileLayout.none_box)
    a_transport=pl.tile.move(pl.tile.set_validshape(acc_a,32,16),target_memory=pl.Mem.Vec,blayout=pl.TileLayout.row_major,slayout=pl.TileLayout.none_box)
    # Carry the full physical box across C2V, then narrow in Vec.
    # Padding rows/columns are never stored or reduced into real outputs.
    # AutoVectorSplit localizes the 24-row extent to 16/8 rows per lane.
    b_vec=pl.tile.slice(b_transport,[32,16],[0,0],valid_shape=[24,12])
    a_vec=pl.tile.slice(a_transport,[32,16],[0,0],valid_shape=[24,12])
    # Preserve the native BF16 projection result before FP32 gating arithmetic.
    bv=pl.tile.cast(pl.tile.cast(b_vec,pl.BF16,mode='rint'),pl.FP32)
    av=pl.tile.cast(pl.tile.cast(a_vec,pl.BF16,mode='rint'),pl.FP32)
    bias=pl.tile.load(dt_bias,[0,0],[1,16],valid_shape=[1,12],target_memory=pl.Mem.Vec)
    log_decay=pl.tile.load(a_log,[0,0],[1,16],valid_shape=[1,12],target_memory=pl.Mem.Vec)
    a_bias=pl.tile.col_expand_add(av,bias)
    softplus=pl.tile.log(pl.tile.adds(pl.tile.exp(pl.minimum(a_bias,20.0)),1.0))
    softplus=pl.tile.select(pl.tile.cmps(a_bias,20.0,cmp_type=3),softplus,a_bias)
    decay=pl.tile.col_expand_mul(softplus,pl.tile.neg(pl.tile.exp(log_decay)))
    alpha=pl.tile.store(pl.tile.exp(decay),[0,0],alpha)
    beta_value=pl.tile.recip(pl.tile.adds(pl.tile.exp(pl.tile.neg(bv)),1.0))
    beta_value=pl.tile.cast(pl.tile.cast(beta_value,pl.BF16,mode='rint'),pl.FP32)
    beta=pl.tile.store(beta_value,[0,0],beta)
    return alpha,beta


@pl.function(type=pl.FunctionType.Inline)
def gates(x:pl.Tensor[[24,5120],pl.BF16],weight:pl.Tensor[[24,5120],pl.BF16],
        a_log:pl.Tensor[[1,12],pl.FP32],dt_bias:pl.Tensor[[1,12],pl.FP32],
        alpha:pl.Tensor[[24,12],pl.FP32],beta:pl.Tensor[[24,12],pl.FP32]):
    # One mixed task; query the real backend instead of assuming physical cores.
    available=pl.system.available_cluster_count()
    with pl.spmd(pl.min(available,1),allow_early_resolve=True):
        alpha,beta=gates_kernel(x,weight,a_log,dt_bias,alpha,beta)
    return alpha,beta
