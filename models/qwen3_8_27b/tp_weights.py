# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""TP2 checkpoint slices with semantic Q/K/V and recurrent-head axes."""

import torch
from .weights import TextWeights


class TPWeights(TextWeights):
    def __init__(self, root, rank, replicate_small=False):
        super().__init__(root)
        if rank not in (0, 1):
            raise ValueError("TP2 rank must be 0 or 1.")
        self.rank = rank
        self.replicate_small = replicate_small

    def shard(self, value, axis=0):
        return value.chunk(2, dim=axis)[self.rank].contiguous()

    def vocabulary(self, name, device):
        return self.shard(self.get(name, "cpu")).to(device)

    def layer(self, index, device):
        result = super().layer(index, "cpu")
        column = {
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "linear_attn.in_proj_z.weight",
            "linear_attn.in_proj_a.weight",
            "linear_attn.in_proj_b.weight",
            "linear_attn.A_log",
            "linear_attn.dt_bias",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
        }
        row = {"mlp.down_proj.weight", "linear_attn.out_proj.weight", "self_attn.o_proj.weight"}
        for key in list(result):
            value = result[key]
            if key == "linear_attn.in_proj_qkv.weight":
                value = torch.cat([self.shard(v) for v in value.split((2048, 2048, 6144), dim=0)])
            elif key == "linear_attn.conv1d.weight":
                value = torch.cat([self.shard(v, 1) for v in value.split((2048, 2048, 6144), dim=1)], dim=1)
            elif key in column:
                if not (
                    self.replicate_small
                    and key in {"linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight"}
                ):
                    value = self.shard(value)
            elif key in row:
                value = self.shard(value, 1)
            result[key] = value.to(device)
        return result
