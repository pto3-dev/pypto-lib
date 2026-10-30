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
"""PyPTO-native whole decode with fixed addresses and vLLM-owned caches."""

import atexit
from pathlib import Path

import torch
from vllm.config import CUDAGraphMode
from vllm.distributed import get_pp_group
from vllm.sequence import IntermediateTensors

from vllm_ascend.pypto.qwen3_8_cache import full_attention_views, gdn_state_views
from vllm_ascend.pypto.qwen3_8_model import Qwen38ModelExecutor


class Qwen38GraphExecutor(Qwen38ModelExecutor):
    """Keep eager PTO prefill; capture one simpler launch per PP decode stage."""

    def __init__(self, device_id, build_root, max_sequence_length=512, fused_mlp=False):
        super().__init__(device_id, build_root, max_sequence_length, fused_mlp)
        self.graph_runner = None
        self.buffers = None
        self.last_forward_was_graph = False
        self.graph_replays = 0
        self.graph_preparations = 0
        self.build_root = Path(build_root)
        self.device_id = device_id
        self.capture_enqueues = 0
        atexit.register(self.close)

    def prepare(self, model, context):
        if self.graph_runner is not None:
            return
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("PyPTO compilation and simpler prepare must precede capture.")
        from models.qwen3_8_27b.decode_buffers import DecodeBuffers
        from models.qwen3_8_27b.decode_stage import DecodeStageSpec, make_decode_stage
        from models.qwen3_8_27b.graph_runner import GraphRunner

        parameters, small, floating = {}, {}, {}
        slots, cache_rows, block_size = None, None, None
        first_full = None
        for index in range(model.start_layer, model.end_layer):
            layer = model.layers[index]
            if layer.layer_scale:
                raise ValueError("Whole-decode PTO does not support layer scale.")
            parameters[f"l{index}_mlp_up"] = layer.mlp.gate_up_proj.weight
            parameters[f"l{index}_mlp_down"] = layer.mlp.down_proj.weight
            small[(index, "input_norm")] = layer.input_layernorm.weight.view(1, 5120)
            small[(index, "post_norm")] = layer.post_attention_layernorm.weight.view(1, 5120)
            if index % 4 != 3:
                attn = layer.linear_attn
                conv, state = gdn_state_views(attn.kv_cache[context.virtual_engine], 16, 48)
                if slots is not None and slots != conv.shape[0]:
                    raise ValueError("GDN layers must have equal native slot capacity.")
                slots = conv.shape[0]
                parameters.update(
                    {
                        f"l{index}_qkvz": attn.in_proj_qkvz.weight,
                        f"l{index}_ba": attn.in_proj_ba.weight,
                        f"l{index}_out": attn.out_proj.weight,
                        f"l{index}_conv_state": conv,
                        f"l{index}_recurrent_state": state,
                    }
                )
                small[(index, "conv")] = attn.conv1d.weight.view(10240, 4).T.contiguous()
                small[(index, "gdn_norm")] = attn.norm.weight.view(1, 128)
                floating[(index, "a_log")] = attn.A_log.float().view(1, 48)
                floating[(index, "dt_bias")] = attn.dt_bias.float().view(1, 48)
                self.cache_bindings[attn.prefix] = [conv.data_ptr(), state.data_ptr()]
            else:
                attn = layer.self_attn
                native = attn.attn.kv_cache[context.virtual_engine]
                keys, values = full_attention_views(native, 4)
                size = native[0].shape[1]
                if cache_rows is not None and (cache_rows != keys.shape[0] or block_size != size):
                    raise ValueError("Full attention layers must have equal native cache capacity.")
                cache_rows, block_size = keys.shape[0], size
                first_full = attn if first_full is None else first_full
                parameters.update(
                    {
                        f"l{index}_qkv": attn.qkv_proj.weight,
                        f"l{index}_out": attn.o_proj.weight,
                        f"l{index}_keys": keys,
                        f"l{index}_values": values,
                    }
                )
                small[(index, "q_norm")] = attn.q_norm.weight.view(1, 256)
                small[(index, "k_norm")] = attn.k_norm.weight.view(1, 256)
                self.cache_bindings[attn.attn.layer_name] = [keys.data_ptr(), values.data_ptr()]
        if first_full is None or slots is None:
            raise ValueError("Whole-decode stages require both GDN and full attention layers.")
        first = get_pp_group().is_first_rank
        last = get_pp_group().is_last_rank
        if first:
            parameters["embedding_weight"] = model.embed_tokens.weight
        if last:
            small[(64, "final_norm")] = model.norm.weight.view(1, 5120)
            parameters["lm_head_weight"] = self.causal_model.lm_head.weight
        table_cols = (self.max_sequence_length + block_size - 1) // block_size
        spec = DecodeStageSpec(
            model.start_layer,
            model.end_layer,
            slots=slots,
            cache_rows=cache_rows,
            block_size=block_size,
            max_seq=self.max_sequence_length,
            table_cols=table_cols,
            include_embedding=first,
            include_logits=last,
        )
        rotary = first_full.rotary_emb.cos_sin_cache[: self.max_sequence_length]
        cos = rotary[:, :32].to(device=self.compute.device, dtype=torch.bfloat16).contiguous()
        sin = rotary[:, 32:].to(device=self.compute.device, dtype=torch.bfloat16).contiguous()
        self.buffers = DecodeBuffers(spec, self.compute.device, parameters, small, floating, cos, sin)
        fn = make_decode_stage(spec, self.build_root / "whole_decode_source")
        self.graph_runner = GraphRunner.compile(
            self.device_id, fn, self.buffers.arguments, self.build_root / "whole_decode_build"
        )
        self.graph_preparations += 1

    def update_inputs(self, model, context, inputs):
        positions = inputs["positions"]
        if positions.ndim == 2:
            positions = positions[0]
        if positions.numel() != 1:
            raise ValueError("Whole-decode graph currently supports batch1 and one token per request.")
        if get_pp_group().is_first_rank:
            ids = inputs.get("input_ids")
            if ids is None or inputs.get("inputs_embeds") is not None:
                raise ValueError("PTO text decode requires input_ids and no input embedding override.")
            self.buffers.values["token_ids"].copy_(ids)
            self.buffers.values["positions"].copy_(positions)
        else:
            inter = inputs["intermediate_tensors"]
            self.buffers.set_inputs(inter["hidden_states"], inter["residual"], positions)
        self.buffers.set_native_metadata(model, context)

    def forward(self, model, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        from vllm.forward_context import get_forward_context

        context = get_forward_context()
        if context.cudagraph_runtime_mode != CUDAGraphMode.FULL:
            self.last_forward_was_graph = False
            return super().forward(model, input_ids, positions, intermediate_tensors, inputs_embeds)
        if self.graph_runner is None:
            raise RuntimeError("ACLGraph wrapper must prepare and update buffers before capture.")
        self.graph_runner.enqueue()
        self.capture_enqueues += 1
        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {
                    "hidden_states": self.buffers.values["hidden_out"],
                    "residual": self.buffers.values["residual_out"],
                }
            )
        return self.buffers.values["hidden_out"]

    def logits(self, model, hidden_states):
        if self.last_forward_was_graph:
            return self.buffers.values["logits_out"]
        return super().logits(model, hidden_states)

    def close(self):
        if self.graph_runner is not None and not self.graph_runner.closed:
            self.graph_runner.close()
        self.runner.close()


def install_graph_hooks():
    """Update dynamic metadata outside captured compute on every replay."""
    from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
    from vllm_ascend.worker import model_runner_v1

    original_call = ACLGraphWrapper.__call__
    original_update = model_runner_v1.update_full_graph_params

    def graph_call(wrapper, *args, **kwargs):
        from vllm.forward_context import get_forward_context

        causal = getattr(wrapper.runnable, "language_model", wrapper.runnable)
        executor = getattr(causal, "_qwen38_pto_executor", None)
        if not isinstance(executor, Qwen38GraphExecutor):
            return original_call(wrapper, *args, **kwargs)
        context = get_forward_context()
        active = context.cudagraph_runtime_mode == CUDAGraphMode.FULL
        executor.last_forward_was_graph = active
        context.qwen38_pto_whole_graph = active
        if active:
            executor.prepare(causal.model, context)
            executor.update_inputs(causal.model, context, kwargs)
            entry = wrapper.concrete_aclgraph_entries.get(context.batch_descriptor)
            replay = entry is not None and entry.aclgraph is not None
            if replay:
                executor.graph_replays += 1
        result = original_call(wrapper, *args, **kwargs)
        if active:
            entry = wrapper.concrete_aclgraph_entries[context.batch_descriptor]
            executor.graph_runner.track_graph(entry.aclgraph)
        return result

    def update_params(*args, **kwargs):
        context = args[2] if len(args) > 2 else kwargs["forward_context"]
        if getattr(context, "qwen38_pto_whole_graph", False):
            # All PTO attention reads dynamic device metadata directly; there
            # are no captured native attention tasks to update.
            return
        return original_update(*args, **kwargs)

    ACLGraphWrapper.__call__ = graph_call
    model_runner_v1.update_full_graph_params = update_params
