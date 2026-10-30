# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Static per-rank layouts for a single distributed Qwen program."""

from dataclasses import dataclass, field
from math import prod
from .geometry import ProductionConfig

CFG = ProductionConfig()
D = CFG.local
I, V, Q, H, KG, VG = (
    D[k] for k in ("intermediate", "vocab", "q_heads", "kv_heads", "gdn_k_heads", "gdn_v_heads")
)
C = (2 * KG + VG) * 128
Z = VG * 128
F = Q * 256


def layer_shapes(index, replicate_small=False):
    shapes = {
        "input_layernorm.weight": (5120,),
        "post_attention_layernorm.weight": (5120,),
        "mlp.gate_proj.weight": (I, 5120),
        "mlp.up_proj.weight": (I, 5120),
        "mlp.down_proj.weight": (5120, I),
    }
    if index % 4 == 3:
        shapes.update(
            {
                "self_attn.q_proj.weight": (2 * F, 5120),
                "self_attn.k_proj.weight": (H * 256, 5120),
                "self_attn.v_proj.weight": (H * 256, 5120),
                "self_attn.o_proj.weight": (5120, F),
                "self_attn.q_norm.weight": (256,),
                "self_attn.k_norm.weight": (256,),
            }
        )
    else:
        small = 48 if replicate_small else VG
        shapes.update(
            {
                "linear_attn.in_proj_qkv.weight": (C, 5120),
                "linear_attn.in_proj_z.weight": (Z, 5120),
                "linear_attn.in_proj_a.weight": (small, 5120),
                "linear_attn.in_proj_b.weight": (small, 5120),
                "linear_attn.out_proj.weight": (5120, F),
                "linear_attn.conv1d.weight": (4, C),
                "linear_attn.norm.weight": (128,),
            }
        )
    return shapes


@dataclass
class WeightBankSpec:
    """Only initialization bank geometry; no compiler switches or scratch layout.

    Four layer positions repeat sixteen times. Old full-matrix offsets are kept
    solely as provenance fields in the vector layout; consumers use small-bank
    offsets computed in bank_layout.layout(). No model is generated from this.
    """

    weights: dict = field(init=False, default_factory=dict)
    float_weights_count: int = 48 * 2 * VG

    def __post_init__(self):
        offset = 0
        for index in range(4):
            items = list(layer_shapes(index).items())
            if index % 4 != 3:
                a = next(i for i, item in enumerate(items) if item[0] == "linear_attn.in_proj_a.weight")
                b = next(i for i, item in enumerate(items) if item[0] == "linear_attn.in_proj_b.weight")
                items[a], items[b] = items[b], items[a]
            for key, shape in items:
                offset = (offset + 31) // 32 * 32
                self.weights[index, key] = (0, offset, shape)
                offset += prod(shape)
