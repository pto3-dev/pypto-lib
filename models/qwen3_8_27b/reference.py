# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent FP32 reference equations and native cache addressing for tests."""

import torch
import torch.nn.functional as F


def gating(a, b, a_log, dt_bias):
    g = -a_log.float().exp() * F.softplus(a.float() + dt_bias.float(), threshold=20)
    return g.exp(), b.float().sigmoid()


def causal_conv(x, weight, state, starts, indices, initial):
    """state: [slots, 3, channels]; weight: [channels, 4]."""
    result = torch.zeros_like(x)
    for request, slot in enumerate(indices.tolist()):
        if slot < 0:
            continue
        history = state[slot].clone() if initial[request] else torch.zeros_like(state[slot])
        for token in range(int(starts[request]), int(starts[request + 1])):
            window = torch.cat((history, x[token : token + 1]), dim=0)
            result[token] = F.silu((window.float() * weight.float().T).sum(0)).to(x.dtype)
            history = window[1:]
        state[slot].copy_(history)
    return result


def recurrent(q, k, v, alpha, beta, state, starts, indices, initial):
    """state: [slots, value_heads, value_dim, key_dim], never transposed."""
    result = torch.zeros_like(v)
    repeat = v.shape[1] // q.shape[1]
    for request, slot in enumerate(indices.tolist()):
        if slot < 0:
            continue
        s = state[slot].clone() if initial[request] else torch.zeros_like(state[slot])
        for token in range(int(starts[request]), int(starts[request + 1])):
            qt = q[token].float()
            kt = k[token].float()
            qt = qt * torch.rsqrt(qt.square().sum(-1, keepdim=True) + 1e-6)
            kt = kt * torch.rsqrt(kt.square().sum(-1, keepdim=True) + 1e-6)
            qt = qt.repeat_interleave(repeat, dim=0) * q.shape[-1] ** -0.5
            kt = kt.repeat_interleave(repeat, dim=0)
            s *= alpha[token, :, None, None]
            error = (v[token].float() - (s * kt[:, None, :]).sum(-1)) * beta[token, :, None]
            s += error[:, :, None] * kt[:, None, :]
            result[token] = (s * qt[:, None, :]).sum(-1).to(v.dtype)
        state[slot].copy_(s)
    return result


def kv_append(k, v, cache, slots):
    """cache: [2, blocks, block_size, kv_heads, head_dim]."""
    block_size = cache.shape[2]
    for token, slot in enumerate(slots.tolist()):
        if slot >= 0:
            block, offset = divmod(slot, block_size)
            cache[0, block, offset].copy_(k[token])
            cache[1, block, offset].copy_(v[token])


def paged_attention(q, cache, block_table, seq_lens, starts):
    """Varlen causal prefill and decode reference using physical page IDs."""
    result = torch.zeros_like(q)
    block_size = cache.shape[2]
    repeats = q.shape[1] // cache.shape[3]
    for request, length in enumerate(seq_lens.tolist()):
        begin, end = int(starts[request]), int(starts[request + 1])
        for token in range(begin, end):
            visible = length - (end - token - 1)
            logical = torch.arange(visible)
            physical = block_table[request, logical // block_size].long()
            offset = logical % block_size
            keys = cache[0, physical, offset].repeat_interleave(repeats, dim=1).float()
            values = cache[1, physical, offset].repeat_interleave(repeats, dim=1).float()
            scores = torch.einsum("hd,thd->ht", q[token].float(), keys) * q.shape[-1] ** -0.5
            result[token] = torch.einsum("ht,thd->hd", scores.softmax(-1), values).to(q.dtype)
    return result
