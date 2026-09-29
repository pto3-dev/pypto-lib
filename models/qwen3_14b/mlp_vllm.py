# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under
# the terms and conditions of CANN Open Software License Agreement Version 2.0.
# Please refer to the License for details. You may not use this file except in
# compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the
# License.
# --------------------------------------------------------------------------------
"""Qwen3-14B MLP with vLLM's packed weight layout and BF16 boundaries.

The public input is the output of native post-attention RMSNorm. The returned
tensor replaces only Qwen3MLP.forward; vLLM keeps residual and KV ownership.
The two weight tensors are used in their native ND layout, without repacking.
"""

import pypto.language as pl


BATCH_PAD = 16
HIDDEN = 5120
INTERMEDIATE = 17408
GATE_UP = 2 * INTERMEDIATE
K_TILE = 256
N_TILE = 256


@pl.jit.inline(auto_scope=False)
def qwen3_mlp_inline(
    x: pl.Tensor[[BATCH_PAD, HIDDEN], pl.BF16],
    gate_up_weight: pl.Tensor[[GATE_UP, HIDDEN], pl.BF16],
    down_weight: pl.Tensor[[HIDDEN, INTERMEDIATE], pl.BF16],
    out: pl.Tensor[[BATCH_PAD, HIDDEN], pl.BF16],
):
    """Compute down_proj(silu(gate_proj(x)) * up_proj(x))."""
    gate_up = pl.create_tensor([BATCH_PAD, GATE_UP], dtype=pl.BF16)
    for n0 in pl.parallel(0, GATE_UP, N_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="gate_up"):
            acc = pl.create_tensor([BATCH_PAD, N_TILE], dtype=pl.FP32)
            for k0 in pl.pipeline(0, HIDDEN, K_TILE, stage=2):
                x_tile = x[:, k0 : k0 + K_TILE]
                w_tile = gate_up_weight[n0 : n0 + N_TILE, k0 : k0 + K_TILE]
                acc = pl.matmul_acc(acc, x_tile, w_tile, b_trans=True, init_cond=(k0 == 0))
            gate_up = pl.assemble(gate_up, pl.cast(acc, pl.BF16, mode="rint"), [0, n0])

    activated = pl.create_tensor([BATCH_PAD, INTERMEDIATE], dtype=pl.BF16)
    for n0 in pl.parallel(0, INTERMEDIATE, N_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="silu_and_mul"):
            gate = pl.cast(gate_up[:, n0 : n0 + N_TILE], pl.FP32)
            up = pl.cast(
                gate_up[:, INTERMEDIATE + n0 : INTERMEDIATE + n0 + N_TILE],
                pl.FP32,
            )
            sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate)), 1.0))
            value = pl.mul(pl.mul(gate, sigmoid), up)
            activated = pl.assemble(activated, pl.cast(value, pl.BF16, mode="rint"), [0, n0])

    for n0 in pl.parallel(0, HIDDEN, N_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="down"):
            acc = pl.create_tensor([BATCH_PAD, N_TILE], dtype=pl.FP32)
            for k0 in pl.pipeline(0, INTERMEDIATE, K_TILE, stage=2):
                x_tile = activated[:, k0 : k0 + K_TILE]
                w_tile = down_weight[n0 : n0 + N_TILE, k0 : k0 + K_TILE]
                acc = pl.matmul_acc(acc, x_tile, w_tile, b_trans=True, init_cond=(k0 == 0))
            out = pl.assemble(out, pl.cast(acc, pl.BF16, mode="rint"), [0, n0])
    return out


@pl.jit(auto_scope=False)
def qwen3_mlp_decode(
    x: pl.Tensor[[BATCH_PAD, HIDDEN], pl.BF16],
    gate_up_weight: pl.Tensor[[GATE_UP, HIDDEN], pl.BF16],
    down_weight: pl.Tensor[[HIDDEN, INTERMEDIATE], pl.BF16],
    out: pl.Out[pl.Tensor[[BATCH_PAD, HIDDEN], pl.BF16]],
):
    out = qwen3_mlp_inline(x, gate_up_weight, down_weight, out)
    return out
