# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Stream text-only checkpoint tensors into balanced pipeline stages."""

import json
import math
from pathlib import Path

import torch
from safetensors import safe_open


class TextWeights:
    def __init__(self, root):
        self.root = Path(root)
        self.mapping = json.loads((self.root / "model.safetensors.index.json").read_text())["weight_map"]

    def get(self, name, device):
        with safe_open(str(self.root / self.mapping[name]), framework="pt", device="cpu") as source:
            tensor = source.get_tensor(name)
            if tensor.dtype not in (torch.bfloat16, torch.float32):
                raise ValueError(f"Unsupported weight dtype for {name}: {tensor.dtype}")
            return tensor.to(device=device)

    def stage_bytes(self, begin, end):
        names = [
            name
            for name in self.mapping
            if any(name.startswith(f"model.language_model.layers.{i}.") for i in range(begin, end))
        ]
        if begin == 0:
            names.append("model.language_model.embed_tokens.weight")
        if end == 64:
            names.extend(("model.language_model.norm.weight", "lm_head.weight"))
        total = 0
        for filename in {self.mapping[name] for name in names}:
            with safe_open(str(self.root / filename), framework="pt", device="cpu") as source:
                for name in names:
                    if self.mapping[name] == filename:
                        tensor = source.get_slice(name)
                        total += math.prod(tensor.get_shape()) * {"BF16": 2, "F32": 4}[tensor.get_dtype()]
        return total

    def layer(self, index, device, packed=False):
        prefix = f"model.language_model.layers.{index}."
        result = {}
        for name in self.mapping:
            if name.startswith(prefix):
                result[name[len(prefix) :]] = self.get(name, "cpu" if packed else device)
        if "linear_attn.conv1d.weight" in result:
            conv = result["linear_attn.conv1d.weight"]
            result["linear_attn.conv1d.weight"] = conv.squeeze(1).T.contiguous()
        if packed:
            groups = {"mlp_up": ["mlp.gate_proj.weight", "mlp.up_proj.weight"]}
            if index % 4 == 3:
                groups["qkv"] = [f"self_attn.{key}_proj.weight" for key in ("q", "k", "v")]
            else:
                groups["qkvz"] = ["linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight"]
                groups["ba"] = ["linear_attn.in_proj_b.weight", "linear_attn.in_proj_a.weight"]
            packed_keys = {key for keys in groups.values() for key in keys}
            for group, keys in groups.items():
                lengths = [result[key].shape[0] for key in keys]
                tensor = torch.cat([result[key] for key in keys]).to(device)
                result["_packed." + group] = tensor
                offset = 0
                for key, length in zip(keys, lengths):
                    result[key] = tensor[offset : offset + length]
                    offset += length
            for key in list(result):
                if key not in packed_keys and not key.startswith("_packed."):
                    result[key] = result[key].to(device)
        return result
