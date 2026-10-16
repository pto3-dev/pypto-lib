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


def layer_shapes(index, replicate_small=False):
    shapes = {
        "input_layernorm.weight": (5120,),
        "post_attention_layernorm.weight": (5120,),
        "mlp.gate_proj.weight": (8704, 5120),
        "mlp.up_proj.weight": (8704, 5120),
        "mlp.down_proj.weight": (5120, 8704),
    }
    if index % 4 == 3:
        shapes.update(
            {
                "self_attn.q_proj.weight": (6144, 5120),
                "self_attn.k_proj.weight": (512, 5120),
                "self_attn.v_proj.weight": (512, 5120),
                "self_attn.o_proj.weight": (5120, 3072),
                "self_attn.q_norm.weight": (256,),
                "self_attn.k_norm.weight": (256,),
            }
        )
    else:
        small = 48 if replicate_small else 24
        shapes.update(
            {
                "linear_attn.in_proj_qkv.weight": (5120, 5120),
                "linear_attn.in_proj_z.weight": (3072, 5120),
                "linear_attn.in_proj_a.weight": (small, 5120),
                "linear_attn.in_proj_b.weight": (small, 5120),
                "linear_attn.out_proj.weight": (5120, 3072),
                "linear_attn.conv1d.weight": (4, 5120),
                "linear_attn.norm.weight": (128,),
            }
        )
    return shapes


@dataclass
class ModelSpec:
    tokens: int
    layers: int = 64
    max_seq: int = 512
    replicate_small: bool = False
    fuse_mlp: bool = False
    debug_groups: bool = False
    debug_every_layer: bool = False
    independent_scratch: bool = False
    packed_weight_policy: dict | None = None
    teacher_input: bool = False
    native_first_norm: bool = False
    fuse_gdn_qkv_z: bool = False
    fuse_attention_qkv: bool = False
    fuse_gdn_ba: bool = False
    pack_host_controls: bool = False
    nz_weight_shapes: tuple = ()
    device_greedy: bool = False
    graph_stop_guard: bool = False
    native_qk_rope: bool = False
    native_reduce_round: bool = False
    native_weight_order: bool = False
    native_cache_pools: tuple = ()
    native_cache_layout: tuple = ()
    cache_rows: int = 12288
    cache_block_size: int = 1536
    state_slots: int = 8
    weights: dict = field(init=False, default_factory=dict)
    group_counts: dict = field(init=False, default_factory=dict)
    scratch: dict = field(init=False, default_factory=dict)
    floating: dict = field(init=False, default_factory=dict)

    def __post_init__(self):
        if (
            self.fuse_gdn_qkv_z
            and self.packed_weight_policy is None
            and not (self.tokens == 1 and self.native_weight_order)
        ):
            raise ValueError("GDN projection fusion requires resident weight tiles.")
        if self.layers % 4 or not 4 <= self.layers <= 64:
            raise ValueError("Layers must be a multiple of 4 in [4,64].")
        if self.tokens < 1 or self.tokens > self.max_seq or self.max_seq % 128:
            raise ValueError("Invalid token/cache dimensions.")
        for index in range(self.layers):
            group = index // 4
            off = self.group_counts.get(group, 0)
            layer_items = list(layer_shapes(index, self.replicate_small).items())
            if self.native_weight_order and index % 4 != 3:
                a = next(i for i, item in enumerate(layer_items) if item[0] == "linear_attn.in_proj_a.weight")
                b = next(i for i, item in enumerate(layer_items) if item[0] == "linear_attn.in_proj_b.weight")
                layer_items[a], layer_items[b] = layer_items[b], layer_items[a]
            for key, shape in layer_items:
                off = (off + 31) // 32 * 32
                self.weights[index, key] = (group, off, shape)
                off += prod(shape)
            self.group_counts[group] = (off + 31) // 32 * 32
        local_heads = 48 if self.replicate_small else 24
        t = self.tokens
        shapes = {
            "hidden": (t, 5120),
            "residual": (t, 5120),
            "norm": (t, 5120),
            "attention": (t, 5120),
            "mixed": (t, 5120),
            "z": (t, 3072),
            "a": (t, local_heads),
            "b": (t, local_heads),
            "convolved": (t, 5120),
            "q": (t, 8, 128),
            "k": (t, 8, 128),
            "v": (t, 24, 128),
            "gdn_core": (t, 24, 128),
            "gdn_normal": (t * 24, 128),
            "packed_q": (t, 6144),
            "full_q": (t, 12, 256),
            "full_gate": (t, 3072),
            "full_k": (t, 512),
            "full_v": (t, 512),
            "full_qn": (t * 12, 256),
            "full_kn": (t * 2, 256),
            "full_qr": (t, 12, 256),
            "full_kr": (t, 2, 256),
            "full_core": (t, 12, 256),
            "gated": (t, 3072),
            "mlp_norm": (t, 5120),
            "mlp_gate": (t, 8704),
            "mlp_up": (t, 8704),
            "mlp_act": (t, 8704),
            "mlp_down": (t, 5120),
            "head_norm": (t, 5120),
            "logits": (1, 124160),
        }
        if self.fuse_mlp:
            shapes["mlp_packed"] = (t, 17408)
        if self.fuse_gdn_ba:
            shapes["gdn_ba"] = (t, 48)
        shapes["attention_probability"] = (t * 12, self.max_seq)
        off = 0
        for key, shape in shapes.items():
            off = (off + 31) // 32 * 32
            self.scratch[key] = (off, shape)
            off += prod(shape)
        self.scratch_count = (off + 31) // 32 * 32
        shapes = {
            "qn": (t, 8, 128),
            "kn": (t, 8, 128),
            "alpha": (t, 24),
            "beta": (t, 24),
            "partial": (t, 5120),
        }
        shapes["attention_scores"] = (t * 12, self.max_seq)
        off = 0
        for key, shape in shapes.items():
            off = (off + 7) // 8 * 8
            self.floating[key] = (off, shape)
            off += prod(shape)
        self.float_scratch_count = (off + 7) // 8 * 8
        self.gdn_layers = self.layers * 3 // 4
        self.full_layers = self.layers // 4
        self.conv_count = self.gdn_layers * 3 * 5120
        self.state_count = self.gdn_layers * 24 * 128 * 128
        self.kv_count = self.full_layers * 2 * self.max_seq * 2 * 256
        self.float_weights_count = self.gdn_layers * 48
        # Reduction zero is embedding, two row projections per layer follow.
        self.reductions = 1 + 2 * self.layers

    @property
    def debug_count(self):
        return self.layers if self.debug_every_layer else self.layers // 4 if self.debug_groups else 0

    def weight_tile(self, n, k):
        nt, kt, transpose = (self.packed_weight_policy or {}).get(f"{k}x{n}", [128, 256, False])
        if not isinstance(transpose, bool) or nt < 16 or nt % 16 or kt < 256 or kt % 256 or n % nt or k % kt:
            raise ValueError("Invalid resident weight tile policy")
        return nt, kt, transpose

    def weight_nz(self, n, k):
        return f"{k}x{n}" in self.nz_weight_shapes

    def projection_weight_shape(self, shape):
        n, k = shape
        if self.packed_weight_policy is None:
            return shape
        nt, kt, transpose = self.weight_tile(n, k)
        if self.weight_nz(n, k):
            return ((n // 256) * (k // kt), 256, kt)
        return (n // nt, k // kt, kt, nt) if transpose else (n // nt, k // kt, nt, kt)

    @property
    def arguments(self):
        result = [
            ("ids", (self.tokens,), "INT64", ""),
            ("positions", (self.tokens,), "INT64", ""),
            ("metadata", (4,), "INT32", ""),
            ("starts", (2,), "INT32", ""),
            ("indices", (1,), "INT32", ""),
            ("initial", (1,), "INT32", ""),
            ("table", (1, (self.max_seq + self.cache_block_size - 1) // self.cache_block_size), "INT32", ""),
            ("lengths", (1,), "INT32", ""),
        ]
        if self.pack_host_controls:
            result = [
                ("control_i64", (3 * self.tokens,), "INT64", ""),
                (
                    "control_i32",
                    (9 + (self.max_seq + self.cache_block_size - 1) // self.cache_block_size,),
                    "INT32",
                    "",
                ),
            ]
        result += [
            (
                "native_metadata",
                (self.layers, 3 + (self.max_seq + self.cache_block_size - 1) // self.cache_block_size),
                "INT32",
                "",
            )
        ]
        result += [
            ("embedding", (124160, 5120), "BF16", ""),
            ("final_norm", (1, 5120), "BF16", ""),
            ("lm_head", self.projection_weight_shape((124160, 5120)), "BF16", ""),
        ]
        result += [(f"weights{g}", (n,), "BF16", "") for g, n in self.group_counts.items()]
        result += [("float_weights", (self.float_weights_count,), "FP32", "")]
        if self.independent_scratch:
            result += [
                (f"scratch_{key}", shape, "BF16", "InOut") for key, (off, shape) in self.scratch.items()
            ]
            result += [
                (f"float_{key}", shape, "FP32", "InOut") for key, (off, shape) in self.floating.items()
            ]
        else:
            result += [
                ("scratch", (self.scratch_count,), "BF16", "InOut"),
                ("float_scratch", (self.float_scratch_count,), "FP32", "InOut"),
            ]
        if self.native_cache_pools:
            result += [(name, (nbytes // 2,), "BF16", "InOut") for name, nbytes in self.native_cache_pools]
        else:
            for index in range(self.layers):
                if index % 4 == 3:
                    result += [
                        (f"l{index}_keys", (self.cache_rows, 2, 256), "BF16", "InOut"),
                        (f"l{index}_values", (self.cache_rows, 2, 256), "BF16", "InOut"),
                    ]
                else:
                    result += [
                        (f"l{index}_conv_state", (self.state_slots, 3, 5120), "BF16", "InOut"),
                        (f"l{index}_recurrent_state", (self.state_slots, 24, 128, 128), "FP32", "InOut"),
                    ]
        result += [
            ("cos", (self.max_seq, 32), "BF16", ""),
            ("sin", (self.max_seq, 32), "BF16", ""),
            ("out", (1, 124160), "BF16", "Out"),
            ("global_logits", (1, 248320), "FP32", "Out"),
            ("hidden_result", (self.max_seq, 5120), "BF16", "Out"),
        ]
        if self.graph_stop_guard:
            result = [
                (
                    name,
                    shape,
                    dtype,
                    "InOut" if name in ("control_i32", "global_logits", "hidden_result") else direction,
                )
                for name, shape, dtype, direction in result
            ]
        if self.device_greedy:
            result = [
                (name, shape, dtype, "InOut" if name == "out" else direction)
                for name, shape, dtype, direction in result
            ]
            result.append(("greedy", (1, 8), "FP32", "Out"))
        if self.debug_count:
            result.append(("debug_hidden", (self.debug_count, 5120), "BF16", "Out"))
        if self.teacher_input:
            result.extend(
                [
                    ("teacher_hidden", (self.tokens, 5120), "BF16", ""),
                    ("teacher_out", (self.tokens, 5120), "BF16", "Out"),
                ]
            )
        return result
