# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -------------------------------------------------------------------------------------------------------
"""CPU packing and exact integer reference; no compiler or device dependency."""

from contextlib import ExitStack
from dataclasses import dataclass
import json
from pathlib import Path

import torch
from safetensors import safe_open

from .config import EXPERTS, HIDDEN, INTERMEDIATE, LAYERS, LOCAL_INTERMEDIATE, TOP_K, TP_SIZE, W13_WIDTH


def pack_w13(gate: torch.Tensor, up: torch.Tensor, tp_rank: int) -> torch.Tensor:
    """Slice output rows per TP rank, then fuse Gate/Up in K-contiguous order."""
    if tp_rank not in range(TP_SIZE):
        raise ValueError(f"tp_rank must be in [0, {TP_SIZE}), got {tp_rank}")
    if gate.shape != up.shape or gate.ndim != 2 or gate.shape[0] % TP_SIZE:
        raise ValueError("Gate/Up must have matching 2-D shapes with TP-divisible output rows")
    if (
        gate.dtype != torch.int8
        or up.dtype != torch.int8
        or gate.device.type != "cpu"
        or up.device.type != "cpu"
    ):
        raise ValueError("Gate/Up must be CPU INT8 tensors")
    width = gate.shape[0] // TP_SIZE
    rows = slice(tp_rank * width, (tp_rank + 1) * width)
    return torch.cat((gate[rows], up[rows]), dim=0).contiguous()


def load_w13_bank(checkpoint: Path, layer: int, tp_rank: int) -> torch.Tensor:
    """Read one layer's 128 expert W13 weights; return [E, N_local, K] INT8."""
    if layer not in range(LAYERS) or tp_rank not in range(TP_SIZE):
        raise ValueError(f"Expected layer in [0, {LAYERS}) and tp_rank in [0, {TP_SIZE})")
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    for key, expected in {
        "hidden_size": HIDDEN,
        "num_hidden_layers": LAYERS,
        "num_experts": EXPERTS,
        "num_experts_per_tok": TOP_K,
        "moe_intermediate_size": INTERMEDIATE,
    }.items():
        if config.get(key) != expected:
            raise ValueError(f"Incompatible checkpoint {key}={config.get(key)}; expected {expected}")
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text(encoding="utf-8"))
    bank = torch.empty((EXPERTS, W13_WIDTH, HIDDEN), dtype=torch.int8)
    with ExitStack() as stack:
        shards = {}

        def read(name: str) -> torch.Tensor:
            filename = index["weight_map"][name]
            if filename not in shards:
                shards[filename] = stack.enter_context(
                    safe_open(checkpoint / filename, framework="pt", device="cpu")
                )
            tensor = shards[filename].get_tensor(name)
            if tensor.shape != (INTERMEDIATE, HIDDEN) or tensor.dtype != torch.int8:
                raise ValueError(f"Unexpected W13 tensor {name}: {tensor.shape}, {tensor.dtype}")
            return tensor

        for expert in range(EXPERTS):
            prefix = f"model.layers.{layer}.mlp.experts.{expert}."
            bank[expert] = pack_w13(
                read(prefix + "gate_proj.weight"), read(prefix + "up_proj.weight"), tp_rank
            )
    return bank


def reference_w13(x: torch.Tensor, expert_ids: torch.Tensor, weight_nt: torch.Tensor) -> torch.Tensor:
    """Exact INT64 reduction, range-checked before converting to the INT32 ABI."""
    if x.dtype != torch.int8 or weight_nt.dtype != torch.int8 or expert_ids.dtype != torch.int32:
        raise ValueError("Expected INT8 inputs/weights and INT32 expert IDs")
    if x.ndim != 2 or weight_nt.ndim != 3 or x.shape[1] != weight_nt.shape[2]:
        raise ValueError("Expected x=[routes,K] and weight=[experts,N,K]")
    if expert_ids.shape != (1, x.shape[0]) or x.shape[0] < 1:
        raise ValueError("Expected a nonempty route set and expert_ids=[1,routes]")
    if not all(t.device.type == "cpu" for t in (x, expert_ids, weight_nt)):
        raise ValueError("The exact reference runs on CPU")
    if torch.any(expert_ids < 0) or torch.any(expert_ids >= weight_nt.shape[0]):
        raise ValueError("Expert ID is outside the resident bank")
    out = torch.empty((x.shape[0], weight_nt.shape[1]), dtype=torch.int32)
    bounds = torch.iinfo(torch.int32)
    for route, expert in enumerate(expert_ids.flatten().tolist()):
        result = weight_nt[expert].to(torch.int64) @ x[route].to(torch.int64)
        if torch.any(result < bounds.min) or torch.any(result > bounds.max):
            raise OverflowError("Expert projection accumulator exceeds INT32")
        out[route] = result.to(torch.int32)
    return out


def exact_int32(actual: torch.Tensor, expected: torch.Tensor, **_context: object) -> tuple[bool, str]:
    """Reject every integer mismatch, including values outside FP32's exact range."""
    if actual.shape != expected.shape or actual.dtype != torch.int32 or expected.dtype != torch.int32:
        return False, "Expected matching INT32 outputs"
    mismatches = int(torch.count_nonzero(actual != expected))
    return mismatches == 0, f"Expert projection INT32 mismatches={mismatches}"


@dataclass(frozen=True)
class ExpertLayerWeights:
    """Rank-local, K-contiguous weights and FP32 per-channel dequantization scales."""

    w13: torch.Tensor
    w13_scale: torch.Tensor
    w2: torch.Tensor
    w2_scale: torch.Tensor
    router: torch.Tensor


def pack_w2(weight: torch.Tensor, tp_rank: int) -> torch.Tensor:
    """Slice W2 input columns, without transposing its output-major storage."""
    if tp_rank not in range(TP_SIZE):
        raise ValueError(f"tp_rank must be in [0, {TP_SIZE}), got {tp_rank}")
    if weight.ndim != 2 or weight.shape[1] % TP_SIZE:
        raise ValueError(f"W2 must have TP-divisible input columns, got {weight.shape}")
    if weight.dtype != torch.int8 or weight.device.type != "cpu":
        raise ValueError("W2 must be a CPU INT8 tensor")
    width = weight.shape[1] // TP_SIZE
    return weight[:, tp_rank * width : (tp_rank + 1) * width].contiguous()


def pack_w13_scales(gate: torch.Tensor, up: torch.Tensor, tp_rank: int) -> torch.Tensor:
    """Match the fused Gate/Up output order, preserving the checkpoint scale values."""
    if tp_rank not in range(TP_SIZE) or gate.shape != up.shape or gate.ndim != 2 or gate.shape[1] != 1:
        raise ValueError("Expected matching [output,1] Gate/Up scales and a valid TP rank")
    if gate.shape[0] % TP_SIZE or gate.dtype != torch.bfloat16 or up.dtype != torch.bfloat16:
        raise ValueError("Expected BF16 scales with TP-divisible output rows")
    width = gate.shape[0] // TP_SIZE
    rows = slice(tp_rank * width, (tp_rank + 1) * width)
    scales = torch.cat((gate[rows], up[rows]), dim=0).flatten().float().contiguous()
    if not torch.isfinite(scales).all() or not torch.all(scales > 0):
        raise ValueError("Expert scales must be finite and positive")
    return scales


def load_expert_layer(checkpoint: Path, layer: int, tp_rank: int) -> ExpertLayerWeights:
    """Read one checkpoint layer's W13/W2/scales/router; no NPU allocation."""
    w13 = load_w13_bank(checkpoint, layer, tp_rank)
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text(encoding="utf-8"))
    w13_scale = torch.empty(EXPERTS, W13_WIDTH, dtype=torch.float32)
    w2 = torch.empty(EXPERTS, HIDDEN, LOCAL_INTERMEDIATE, dtype=torch.int8)
    w2_scale = torch.empty(EXPERTS, HIDDEN, dtype=torch.float32)
    with ExitStack() as stack:
        shards = {}

        def read(name: str, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
            filename = index["weight_map"][name]
            if filename not in shards:
                shards[filename] = stack.enter_context(
                    safe_open(checkpoint / filename, framework="pt", device="cpu")
                )
            tensor = shards[filename].get_tensor(name)
            if tuple(tensor.shape) != shape or tensor.dtype != dtype:
                raise ValueError(f"{name}: expected {shape}/{dtype}, got {tensor.shape}/{tensor.dtype}")
            if dtype.is_floating_point and not torch.isfinite(tensor).all():
                raise ValueError(f"{name}: non-finite values")
            return tensor

        prefix = f"model.layers.{layer}.mlp."
        router = read(prefix + "gate.weight", (EXPERTS, HIDDEN), torch.bfloat16).clone()
        for expert in range(EXPERTS):
            name = prefix + f"experts.{expert}."
            w13_scale[expert] = pack_w13_scales(
                read(name + "gate_proj.weight_scale", (INTERMEDIATE, 1), torch.bfloat16),
                read(name + "up_proj.weight_scale", (INTERMEDIATE, 1), torch.bfloat16),
                tp_rank,
            )
            w2[expert] = pack_w2(read(name + "down_proj.weight", (HIDDEN, INTERMEDIATE), torch.int8), tp_rank)
            down_scale = read(name + "down_proj.weight_scale", (HIDDEN, 1), torch.bfloat16).flatten().float()
            if not torch.all(down_scale > 0):
                raise ValueError(f"{name}down_proj.weight_scale: nonpositive scale")
            w2_scale[expert] = down_scale
    return ExpertLayerWeights(w13, w13_scale, w2, w2_scale, router)
