# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Eager PTO text model using native weights, metadata and cache storage."""

import torch
from vllm.distributed import get_pp_group
from vllm.forward_context import get_forward_context
from vllm.sequence import IntermediateTensors

from vllm_ascend.pypto.qwen3_8_cache import full_attention_views
from vllm_ascend.pypto.qwen3_8_gdn import Qwen38GDNExecutor


class Qwen38ModelExecutor(Qwen38GDNExecutor):
    """Reuse PTO compute helpers without allocating a separate weight/cache pool."""

    def __init__(self, device_id, build_root, max_sequence_length=128, fused_mlp=False):
        super().__init__(device_id, build_root)
        from models.qwen3_8_27b.pure_model import Stage

        self.compute = Stage.__new__(Stage)
        self.compute.device = torch.device(f"npu:{device_id}")
        self.compute.runner = self.runner
        self.compute.kernels = {}
        self.rotary_parameters = {}
        self.cache_bindings = {}
        self.max_sequence_length = max_sequence_length
        self.fused_mlp = fused_mlp

    def embed(self, model, input_ids):
        from models.qwen3_8_27b.layout import make_embedding

        weight = model.embed_tokens.weight
        output = self.compute.allocate((input_ids.numel(), 5120))
        self.compute.call(
            "native_embedding",
            lambda: make_embedding(weight.shape[0]),
            input_ids.to(dtype=torch.int64).contiguous(),
            weight,
            output,
        )
        return output

    def linear(self, layer, x, context):
        t = x.shape[0]
        weight = layer.in_proj_qkvz.weight
        mixed = self.compute.gemm(x, weight[:10240])
        z = self.compute.gemm(x, weight[10240:])
        ba = layer.in_proj_ba.weight
        b = self.compute.gemm(x, ba[:48])
        a = self.compute.gemm(x, ba[48:])
        out = self.compute.allocate((t, 48, 128))
        cache = layer.kv_cache[context.virtual_engine]
        self.execute(layer, context.attn_metadata[layer.prefix], cache, mixed, a, b, out)
        normalized = self.compute.norm(out.view(-1, 128), layer.norm.weight, z.view(-1, 128), offset=0.0)
        self.cache_bindings[layer.prefix] = [value.data_ptr() for value in cache]
        return self.compute.gemm(normalized.view(t, 6144), layer.out_proj.weight)

    def full(self, layer, x, positions, context):
        from models.qwen3_8_27b.attention import make_attention, make_rope
        from models.qwen3_8_27b.kv_cache import make_append
        from models.qwen3_8_27b.layout import make_query_gate_unpack

        t = x.shape[0]
        weight = layer.qkv_proj.weight
        packed = self.compute.gemm(x, weight[:12288])
        q, gate = self.compute.allocate((t, 24, 256)), self.compute.allocate((t, 6144))
        self.compute.call("query_gate_unpack", lambda: make_query_gate_unpack(24), packed, q, gate)
        k = self.compute.gemm(x, weight[12288:13312]).view(t, 4, 256)
        v = self.compute.gemm(x, weight[13312:])
        q = self.compute.norm(q.view(-1, 256), layer.q_norm.weight).view(t, 24, 256)
        k = self.compute.norm(k.view(-1, 256), layer.k_norm.weight).view(t, 4, 256)
        key = layer.attn.layer_name
        cache = layer.rotary_emb.cos_sin_cache
        rotary_key = (cache.data_ptr(), cache.dtype, self.max_sequence_length)
        if rotary_key not in self.rotary_parameters:
            if cache.shape[-1] != 64:
                raise ValueError("Qwen3.8 text requires a 64-element partial RoPE cache.")
            cache = cache[: self.max_sequence_length]
            self.rotary_parameters[rotary_key] = (
                cache[:, :32].to(device=x.device, dtype=torch.bfloat16).contiguous(),
                cache[:, 32:].to(device=x.device, dtype=torch.bfloat16).contiguous(),
            )
        cos, sin = self.rotary_parameters[rotary_key]
        qr, kr = self.compute.allocate(q.shape), self.compute.allocate(k.shape)
        self.compute.call("rope_q", lambda: make_rope(24), q, cos, sin, positions, qr)
        self.compute.call("rope_k", lambda: make_rope(4), k, cos, sin, positions, kr)
        native_cache = layer.attn.kv_cache[context.virtual_engine]
        keys, values = full_attention_views(native_cache, 4)
        block_size = native_cache[0].shape[1]
        meta = context.attn_metadata[key]
        slots = meta.slot_mapping[:t].to(device=x.device, dtype=torch.int64).contiguous()
        table = meta.block_tables.to(device=x.device, dtype=torch.int32).contiguous()
        lengths = getattr(meta, "pypto_seq_lens_device", None)
        if lengths is None:
            lengths = meta.seq_lens
        lengths = lengths.to(device=x.device, dtype=torch.int32).contiguous()
        starts = meta.query_start_loc.to(device=x.device, dtype=torch.int32).contiguous()
        self.compute.call(
            "kv_append",
            lambda: make_append(4),
            kr.view(t, 1024),
            v,
            keys.view(-1, 1024),
            values.view(-1, 1024),
            slots,
        )
        out = self.compute.allocate(q.shape)
        self.compute.call(
            f"paged_attention_{block_size}",
            lambda: make_attention(24, 4, block_size),
            qr,
            keys,
            values,
            table,
            lengths,
            starts,
            out,
        )
        self.cache_bindings[key] = [keys.data_ptr(), values.data_ptr()]
        gated = self.compute.pointwise(gate, out.view(t, 6144), "sigmoid_mul")
        return self.compute.gemm(gated, layer.o_proj.weight)

    def forward(self, model, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        context = get_forward_context()
        if model.aux_hidden_state_layers:
            raise ValueError("The PTO text path does not support auxiliary/speculative hidden states.")
        if positions.ndim == 2:
            if (
                positions.shape[0] != 3
                or not torch.equal(positions[0], positions[1])
                or not torch.equal(positions[0], positions[2])
            ):
                raise ValueError("PTO ordinary text requires identical three-axis MRoPE positions.")
            positions = positions[0]
        positions = positions.to(dtype=torch.int64).contiguous()
        if get_pp_group().is_first_rank:
            hidden = inputs_embeds if inputs_embeds is not None else self.embed(model, input_ids)
            residual = None
        else:
            hidden = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        for index in range(model.start_layer, model.end_layer):
            layer = model.layers[index]
            if layer.layer_scale:
                raise ValueError("Layer scaling is not present in the supported checkpoint.")
            has_residual = residual is not None
            residual = hidden if residual is None else self.compute.pointwise(hidden, residual, "add")
            normalized = self.compute.norm(residual, layer.input_layernorm.weight, native_round=has_residual)
            attention = (
                self.linear(layer.linear_attn, normalized, context)
                if layer.layer_type == "linear_attention"
                else self.full(layer.self_attn, normalized, positions, context)
            )
            residual = self.compute.pointwise(residual, attention, "add")
            weight = layer.mlp.gate_up_proj.weight
            if self.fused_mlp:
                from models.qwen3_8_27b.fused import make_mlp

                tokens = residual.shape[0]
                hidden = self.compute.allocate(residual.shape)
                # Preserve the native hidden/residual PP contract. The fused
                # residual sum is scratch; the next native layer owns that sum.
                self.compute.call(
                    f"fused_mlp_{tokens}_1024",
                    lambda: make_mlp(rows=tokens, vector_tile=1024, native_round=True, logical_cores=20),
                    residual,
                    layer.post_attention_layernorm.weight.view(1, 5120),
                    weight[:17408],
                    weight[17408:],
                    layer.mlp.down_proj.weight,
                    self.compute.allocate(residual.shape),
                    self.compute.allocate((tokens, 17408)),
                    self.compute.allocate((tokens, 17408)),
                    self.compute.allocate((tokens, 17408)),
                    hidden,
                    self.compute.allocate(residual.shape),
                )
            else:
                normalized = self.compute.norm(
                    residual, layer.post_attention_layernorm.weight, native_round=True
                )
                gate = self.compute.gemm(normalized, weight[:17408])
                up = self.compute.gemm(normalized, weight[17408:])
                activated = self.compute.pointwise(gate, up, "silu_mul")
                hidden = self.compute.gemm(activated, layer.mlp.down_proj.weight)
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden, "residual": residual})
        combined = self.compute.pointwise(hidden, residual, "add")
        return self.compute.norm(combined, model.norm.weight, native_round=True)

    def logits(self, model, hidden_states):
        return self.compute.gemm(hidden_states, model.lm_head.weight).float()[:, : model.config.vocab_size]
