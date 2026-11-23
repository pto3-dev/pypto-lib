# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""Unsharded TP1 ND checkpoint loading for the A1 graph path.

No TP2 slice, NZ conversion, device allocation or dtype downgrade is performed.
The existing TP2 loaders remain unchanged.
"""

import json
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

from .config import EXPERTS, HIDDEN, INTERMEDIATE, LAYERS, TOP_K
from .dense_weights import read_layer_tensors


def _validate_config(checkpoint: Path, layer: int) -> None:
    if isinstance(layer, bool) or not isinstance(layer, int) or not 0 <= layer < LAYERS:
        raise ValueError(f"Expected integer layer in [0,{LAYERS}), got {layer}")
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    for key, expected in {
        "hidden_size": HIDDEN,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "num_experts": EXPERTS,
        "num_experts_per_tok": TOP_K,
        "moe_intermediate_size": INTERMEDIATE,
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
        "norm_topk_prob": True,
        "decoder_sparse_step": 1,
        "mlp_only_layers": [],
        "attention_bias": False,
    }.items():
        if config.get(key) != expected:
            raise ValueError(f"Incompatible A1 checkpoint {key}={config.get(key)}; expected {expected}")


def _checked(tensor: torch.Tensor, shape: tuple[int, ...], dtype: torch.dtype, name: str) -> torch.Tensor:
    if tuple(tensor.shape) != shape or tensor.dtype != dtype or tensor.device.type != "cpu":
        raise ValueError(f"{name}: expected CPU {shape}/{dtype}, got {tensor.shape}/{tensor.dtype}")
    if dtype.is_floating_point and not torch.isfinite(tensor).all():
        raise ValueError(f"{name}: non-finite values")
    return tensor


def _scale(tensor: torch.Tensor, width: int, name: str) -> torch.Tensor:
    tensor = _checked(tensor, (width, 1), torch.bfloat16, name).flatten().float()
    if not torch.all(tensor > 0):
        raise ValueError(f"{name}: nonpositive scale")
    return tensor


def load_a1_attention_weights(checkpoint: Path, layer: int) -> dict[str, torch.Tensor]:
    """Load full Q4096/K512/V512/O2048x4096 plus BF16 Norm/Router weights."""
    _validate_config(checkpoint, layer)
    names = tuple(
        f"self_attn.{kind}_proj.{field}"
        for kind in ("q", "k", "v", "o")
        for field in ("weight", "weight_scale")
    ) + (
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "mlp.gate.weight",
    )
    tensors = read_layer_tensors(checkpoint, layer, names)
    weights, scales = [], []
    for kind, width in (("q", 4096), ("k", 512), ("v", 512)):
        name = f"self_attn.{kind}_proj"
        weights.append(_checked(tensors[f"{name}.weight"], (width, HIDDEN), torch.int8, name))
        scales.append(_scale(tensors[f"{name}.weight_scale"], width, name))
    result = {
        "qkv_weight": torch.cat(weights).contiguous(),
        "qkv_scale": torch.cat(scales).reshape(1, 5120).contiguous(),
        "o_weight": _checked(tensors["self_attn.o_proj.weight"], (HIDDEN, 4096), torch.int8, "O"),
        "o_scale": _scale(tensors["self_attn.o_proj.weight_scale"], HIDDEN, "O").reshape(1, HIDDEN),
    }
    for name, shape in (
        ("self_attn.q_norm.weight", (128,)),
        ("self_attn.k_norm.weight", (128,)),
        ("input_layernorm.weight", (HIDDEN,)),
        ("post_attention_layernorm.weight", (HIDDEN,)),
        ("mlp.gate.weight", (EXPERTS, HIDDEN)),
    ):
        result[name] = _checked(tensors[name], shape, torch.bfloat16, name)
    return result


@dataclass(frozen=True)
class A1ExpertLayerWeights:
    """Full ND expert banks; flatten first two weight axes for native launch ABI."""

    w13: torch.Tensor
    w13_scale: torch.Tensor
    w2: torch.Tensor
    w2_scale: torch.Tensor


def load_a1_expert_layer(checkpoint: Path, layer: int) -> A1ExpertLayerWeights:
    """Load 128 full W13[1536,2048] and W2[2048,768] banks on CPU only."""
    _validate_config(checkpoint, layer)
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text(encoding="utf-8"))[
        "weight_map"
    ]
    w13 = torch.empty((EXPERTS, 2 * INTERMEDIATE, HIDDEN), dtype=torch.int8, device="cpu")
    w2 = torch.empty((EXPERTS, HIDDEN, INTERMEDIATE), dtype=torch.int8, device="cpu")
    w13_scale = torch.empty((EXPERTS, 2 * INTERMEDIATE), dtype=torch.float32, device="cpu")
    w2_scale = torch.empty((EXPERTS, HIDDEN), dtype=torch.float32, device="cpu")
    with ExitStack() as stack:
        shards = {}

        def read(name: str) -> torch.Tensor:
            filename = index[name]
            if filename not in shards:
                shards[filename] = stack.enter_context(
                    safe_open(checkpoint / filename, framework="pt", device="cpu")
                )
            return shards[filename].get_tensor(name)

        for expert in range(EXPERTS):
            prefix = f"model.layers.{layer}.mlp.experts.{expert}"
            for kind, offset in (("gate_proj", 0), ("up_proj", INTERMEDIATE)):
                name = f"{prefix}.{kind}"
                w13[expert, offset : offset + INTERMEDIATE].copy_(
                    _checked(read(f"{name}.weight"), (INTERMEDIATE, HIDDEN), torch.int8, name)
                )
                w13_scale[expert, offset : offset + INTERMEDIATE].copy_(
                    _scale(read(f"{name}.weight_scale"), INTERMEDIATE, name)
                )
            name = f"{prefix}.down_proj"
            w2[expert].copy_(_checked(read(f"{name}.weight"), (HIDDEN, INTERMEDIATE), torch.int8, name))
            w2_scale[expert].copy_(_scale(read(f"{name}.weight_scale"), HIDDEN, name))
    return A1ExpertLayerWeights(w13, w13_scale, w2, w2_scale)


def load_a1_vocab_weights(checkpoint: Path) -> dict[str, torch.Tensor]:
    """Load the unsharded BF16 embedding, final Norm and untied LM Head on CPU."""
    _validate_config(checkpoint, 0)
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    if config.get("vocab_size") != 151936 or config.get("tie_word_embeddings") is not False:
        raise ValueError("A1 vocabulary path requires vocab_size=151936 and untied embeddings")
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text(encoding="utf-8"))[
        "weight_map"
    ]
    result = {}
    for name, shape in (
        ("model.embed_tokens.weight", (151936, HIDDEN)),
        ("model.norm.weight", (HIDDEN,)),
        ("lm_head.weight", (151936, HIDDEN)),
    ):
        with safe_open(checkpoint / index[name], framework="pt", device="cpu") as shard:
            result[name] = _checked(shard.get_tensor(name), shape, torch.bfloat16, name)
    return result
