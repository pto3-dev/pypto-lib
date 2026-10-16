# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Runtime allocation and weight loading for the TP2 graph service."""

from contextlib import contextmanager


import torch


from pypto.runtime import StackedDeviceTensor

from .model_spec import layer_shapes


from .tp_weights import TPWeights

from .tiled_weights import pack_tiles

from .nz_tiled import pack_nz_tiles


@contextmanager
def orchestration_optimization(level):
    """Scope an AICPU compile flag override to this runner and its cache key."""
    if level == "O3":
        yield
        return
    from simpler_setup.toolchain import Aarch64GxxToolchain

    original = Aarch64GxxToolchain.get_compile_flags

    def flags(toolchain, **kwargs):
        return ["-" + level if flag == "-O3" else flag for flag in original(toolchain, **kwargs)]

    Aarch64GxxToolchain.get_compile_flags = flags
    try:
        yield
    finally:
        Aarch64GxxToolchain.get_compile_flags = original


def allocate(runtime, shape, dtype):
    shards = [runtime.alloc_tensor(shape, dtype, worker_id=rank) for rank in (0, 1)]
    return StackedDeviceTensor(shards, (2, *shape), (0, 1))


def load_weights(runtime, spec, checkpoint):
    sources = [TPWeights(checkpoint, r, spec.replicate_small) for r in (0, 1)]
    weights = {}
    floating = torch.empty(2, spec.float_weights_count, dtype=torch.float32)
    for group, count in spec.group_counts.items():
        packed = torch.empty(2, count, dtype=torch.bfloat16)
        for index in range(group * 4, group * 4 + 4):
            for rank, source in enumerate(sources):
                layer = source.layer(index, "cpu")
                for key, shape in layer_shapes(index, spec.replicate_small).items():
                    _, off, _ = spec.weights[index, key]
                    if tuple(layer[key].shape) != shape:
                        raise ValueError((index, key, layer[key].shape, shape))
                    value = layer[key]
                    if spec.packed_weight_policy is not None and len(shape) == 2 and shape[0] >= 128:
                        if spec.fuse_mlp and key == "mlp.up_proj.weight":
                            continue
                        if spec.fuse_mlp and key == "mlp.gate_proj.weight":
                            value = torch.cat((value, layer["mlp.up_proj.weight"]), dim=0)
                        if spec.fuse_gdn_qkv_z and key == "linear_attn.in_proj_z.weight":
                            continue
                        if spec.fuse_gdn_qkv_z and key == "linear_attn.in_proj_qkv.weight":
                            value = torch.cat((value, layer["linear_attn.in_proj_z.weight"]), dim=0)
                        value = (
                            pack_nz_tiles(value, k_tile=spec.weight_tile(*value.shape)[1])
                            if spec.weight_nz(*value.shape)
                            else pack_tiles(value, *spec.weight_tile(*value.shape))
                        )
                    packed[rank, off : off + value.numel()].copy_(value.reshape(-1))
                if index % 4 != 3:
                    g = index - index // 4
                    floating[rank, g * 48 : g * 48 + 24].copy_(layer["linear_attn.A_log"].float().reshape(-1))
                    floating[rank, g * 48 + 24 : g * 48 + 48].copy_(
                        layer["linear_attn.dt_bias"].float().reshape(-1)
                    )
                del layer
        weights[f"weights{group}"] = runtime.alloc_stacked_tensor(packed)
        del packed
        print(f"Loaded resident PTO weight group {group}", flush=True)
    weights["float_weights"] = runtime.alloc_stacked_tensor(floating)
    for name, key in [
        ("embedding", "model.language_model.embed_tokens.weight"),
        ("lm_head", "lm_head.weight"),
    ]:
        shards = []
        for rank, source in enumerate(sources):
            host = source.vocabulary(key, "cpu")
            if name == "lm_head" and spec.packed_weight_policy is not None:
                host = (
                    pack_nz_tiles(host, k_tile=spec.weight_tile(*host.shape)[1])
                    if spec.weight_nz(*host.shape)
                    else pack_tiles(host, *spec.weight_tile(*host.shape))
                )
            shards.append(runtime.alloc_tensor(host.shape, host.dtype, init=host, worker_id=rank))
        weights[name] = StackedDeviceTensor(shards, (2, *shards[0].shape), (0, 1))
    norm = sources[0].get("model.language_model.norm.weight", "cpu").reshape(1, 5120)
    weights["final_norm"] = runtime.alloc_stacked_tensor(norm.unsqueeze(0).repeat(2, 1, 1))
    angles = (
        torch.arange(spec.max_seq, dtype=torch.float32)[:, None]
        * (1e7 ** (-torch.arange(0, 64, 2, dtype=torch.float32) / 64))[None]
    )
    for name, host in [("cos", angles.cos().bfloat16()), ("sin", angles.sin().bfloat16())]:
        weights[name] = runtime.alloc_stacked_tensor(host.unsqueeze(0).repeat(2, 1, 1))
    for name, shape, dtype, _ in spec.arguments:
        if name in ("conv_state", "recurrent_state", "kv_cache"):
            weights[name] = allocate(runtime, shape, {"BF16": torch.bfloat16, "FP32": torch.float32}[dtype])
    return weights
