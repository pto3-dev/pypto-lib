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
"""Opt-in eager GDN core integration; native vLLM owns all cache allocations."""

from pathlib import Path
from types import MethodType

import torch
from vllm.config import CompilationMode, CUDAGraphMode
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5GatedDeltaNet

from vllm_ascend import envs
from vllm_ascend.pypto.qwen3_8_gdn import Qwen38GDNExecutor


def install():
    original_init = Qwen3_5ForCausalLM.__init__
    original_core = Qwen3_5GatedDeltaNet._forward_core

    def initialize(self, *, vllm_config, prefix=""):
        config = vllm_config.model_config.hf_text_config
        expected = (5120, 17408, 64, 16, 48, 128, 128, 4, 24, 4, 256, 248320)
        actual = (
            config.hidden_size,
            config.intermediate_size,
            config.num_hidden_layers,
            config.linear_num_key_heads,
            config.linear_num_value_heads,
            config.linear_key_head_dim,
            config.linear_value_head_dim,
            config.linear_conv_kernel_dim,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            config.vocab_size,
        )
        if actual != expected:
            raise ValueError(f"PTO Qwen3.8-27B requires text dimensions {expected}, got {actual}.")
        expected_layers = ["full_attention" if index % 4 == 3 else "linear_attention" for index in range(64)]
        if list(config.layer_types) != expected_layers:
            raise ValueError(
                "PTO Qwen3.8 requires the checkpoint's three-GDN/one-full-attention layer order."
            )
        rope = config.rope_parameters
        if rope.get("rope_theta") != 10000000 or rope.get("partial_rotary_factor") != 0.25:
            raise ValueError("PTO Qwen3.8 requires theta=1e7 and the first 64 RoPE dimensions.")
        if get_tensor_model_parallel_world_size() != 1:
            raise ValueError("The initial PTO GDN bridge requires TP=1; use PP=2 for this checkpoint.")
        graph_enabled = not vllm_config.model_config.enforce_eager
        if graph_enabled:
            compilation = vllm_config.compilation_config
            if not envs.VLLM_ASCEND_PYPTO_QWEN38_MODEL:
                raise ValueError("PTO graph mode requires the full model executor.")
            if compilation.mode != CompilationMode.NONE:
                raise ValueError("PTO graph mode requires vLLM compilation mode 0; PyPTO compiles natively.")
            if compilation.cudagraph_mode != CUDAGraphMode.FULL_DECODE_ONLY:
                raise ValueError("PTO graph mode requires FULL_DECODE_ONLY.")
            if vllm_config.scheduler_config.max_num_seqs != 1:
                raise ValueError("PTO whole-decode graph currently requires max_num_seqs=1.")
        if vllm_config.speculative_config is not None or vllm_config.quant_config is not None:
            raise ValueError(
                "PTO Qwen3.8 currently requires BF16 ordinary inference without MTP/quantization."
            )
        if vllm_config.model_config.dtype != torch.bfloat16:
            raise ValueError("PTO Qwen3.8 currently requires BF16 model weights.")
        if envs.VLLM_ASCEND_PYPTO_QWEN38_MODEL and envs.VLLM_ASCEND_ENABLE_NZ == 2:
            raise ValueError(
                "The full PTO text path requires ND BF16 weights; set VLLM_ASCEND_ENABLE_NZ=0 or 1."
            )
        original_init(self, vllm_config=vllm_config, prefix=prefix)
        root = Path(envs.VLLM_ASCEND_PYPTO_QWEN38_BUILD_ROOT)
        executor_type = Qwen38GDNExecutor
        if envs.VLLM_ASCEND_PYPTO_QWEN38_MODEL:
            from vllm_ascend.pypto.qwen3_8_model import Qwen38ModelExecutor

            executor_type = Qwen38ModelExecutor
            if graph_enabled:
                from vllm_ascend.pypto.qwen3_8_graph import Qwen38GraphExecutor, install_graph_hooks

                executor_type = Qwen38GraphExecutor
                install_graph_hooks()
                self.model.do_not_compile = True
        arguments = (torch.npu.current_device(), root / f"rank{torch.npu.current_device()}")
        executor = (
            executor_type(
                *arguments,
                max_sequence_length=vllm_config.model_config.max_model_len,
                fused_mlp=envs.VLLM_ASCEND_PYPTO_QWEN38_MLP_FUSION,
            )
            if envs.VLLM_ASCEND_PYPTO_QWEN38_MODEL
            else executor_type(*arguments)
        )
        self._qwen38_pto_executor = executor
        if graph_enabled:
            executor.causal_model = self
        for layer in self.model.layers:
            if hasattr(layer, "linear_attn"):
                layer.linear_attn._qwen38_pto_executor = executor
        if envs.VLLM_ASCEND_PYPTO_QWEN38_MODEL:
            original_forward = self.model.forward
            original_embed = self.model.embed_input_ids
            original_logits = self.compute_logits

            def model_forward(model, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
                if get_forward_context().attn_metadata is None:
                    return original_forward(input_ids, positions, intermediate_tensors, inputs_embeds)
                return executor.forward(model, input_ids, positions, intermediate_tensors, inputs_embeds)

            def embed(model, input_ids):
                if is_forward_context_available() and get_forward_context().attn_metadata is None:
                    return original_embed(input_ids)
                return executor.embed(model, input_ids)

            def logits(model, hidden_states):
                if is_forward_context_available() and get_forward_context().attn_metadata is None:
                    return original_logits(hidden_states)
                return executor.logits(model, hidden_states)

            self.model.forward = MethodType(model_forward, self.model)
            self.model.embed_input_ids = MethodType(embed, self.model)
            self.compute_logits = MethodType(logits, self)

    def forward_core(self, mixed_qkv, b, a, core_attn_out):
        context = get_forward_context()
        if context.attn_metadata is None:
            return original_core(self, mixed_qkv, b, a, core_attn_out)
        executor = getattr(self, "_qwen38_pto_executor", None)
        if executor is None:
            return original_core(self, mixed_qkv, b, a, core_attn_out)
        metadata = context.attn_metadata[self.prefix]
        cache = self.kv_cache[context.virtual_engine]
        executor.execute(self, metadata, cache, mixed_qkv, a, b, core_attn_out)

    Qwen3_5ForCausalLM.__init__ = initialize
    Qwen3_5GatedDeltaNet._forward_core = forward_core


install()
