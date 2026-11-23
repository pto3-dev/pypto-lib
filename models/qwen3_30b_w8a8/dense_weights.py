# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -------------------------------------------------------------------------------------------------------
"""CPU checkpoint slices matching the native TP=2 attention weight loader."""

from contextlib import ExitStack
import json
from pathlib import Path

import torch
from safetensors import safe_open

from .config import HIDDEN, LAYERS, TP_SIZE
from .linear import HEAD_DIM, KV_WIDTH, Q_WIDTH


def read_layer_tensors(checkpoint: Path, layer: int, names: tuple[str, ...]) -> dict[str, torch.Tensor]:
    if layer not in range(LAYERS):
        raise ValueError(f"Expected layer in [0,{LAYERS}), got {layer}")
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    with ExitStack() as stack:
        files = {}
        result = {}
        for name in names:
            key = f"model.layers.{layer}.{name}"
            filename = index[key]
            if filename not in files:
                files[filename] = stack.enter_context(
                    safe_open(checkpoint / filename, framework="pt", device="cpu")
                )
            result[name] = files[filename].get_tensor(key).clone()
    return result


def load_attention_weights(checkpoint: Path, layer: int, tp_rank: int) -> dict[str, torch.Tensor]:
    if tp_rank not in range(TP_SIZE):
        raise ValueError(f"Expected TP rank in [0,{TP_SIZE}), got {tp_rank}")
    names = tuple(
        f"self_attn.{kind}_proj.{field}"
        for kind in ("q", "k", "v", "o")
        for field in ("weight", "weight_scale")
    )
    names += (
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "mlp.gate.weight",
    )
    tensors = read_layer_tensors(checkpoint, layer, names)
    weights, scales = [], []
    for kind, width in (("q", Q_WIDTH), ("k", KV_WIDTH), ("v", KV_WIDTH)):
        rows = slice(tp_rank * width, (tp_rank + 1) * width)
        bank = tensors[f"self_attn.{kind}_proj.weight"]
        scale = tensors[f"self_attn.{kind}_proj.weight_scale"]
        if bank.shape != (width * TP_SIZE, HIDDEN) or bank.dtype != torch.int8:
            raise ValueError(f"Unexpected {kind} weight: {bank.shape}/{bank.dtype}")
        if scale.shape != (width * TP_SIZE, 1) or scale.dtype != torch.bfloat16:
            raise ValueError(f"Unexpected {kind} scale: {scale.shape}/{scale.dtype}")
        weights.append(bank[rows])
        scales.append(scale[rows])
    tensors["qkv_weight"] = torch.cat(weights).contiguous()
    tensors["qkv_scale"] = torch.cat(scales).flatten().contiguous()
    bank = tensors["self_attn.o_proj.weight"]
    if bank.shape != (HIDDEN, Q_WIDTH * TP_SIZE) or bank.dtype != torch.int8:
        raise ValueError(f"Unexpected O weight: {bank.shape}/{bank.dtype}")
    tensors["o_weight"] = bank[:, tp_rank * Q_WIDTH : (tp_rank + 1) * Q_WIDTH].contiguous()
    tensors["o_scale"] = tensors["self_attn.o_proj.weight_scale"].flatten().contiguous()
    for key in ("qkv_scale", "o_scale"):
        if not torch.isfinite(tensors[key]).all() or not (tensors[key] > 0).all():
            raise ValueError(f"Invalid {key}: expected finite positive scales")
    for key in ("self_attn.q_norm.weight", "self_attn.k_norm.weight"):
        if tensors[key].shape != (HEAD_DIM,):
            raise ValueError(f"Unexpected head norm weight shape: {tensors[key].shape}")
    return tensors
