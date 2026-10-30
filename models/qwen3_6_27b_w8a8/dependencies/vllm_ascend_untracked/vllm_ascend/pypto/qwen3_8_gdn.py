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
"""Ordinary-text GDN execution over scheduler-owned Ascend state buffers."""

from pathlib import Path

import torch

from vllm_ascend.pypto.qwen3_8_cache import gdn_state_views

KEY_HEADS = 16
VALUE_HEADS = 48
HEAD_DIM = 128
MIXED_CHANNELS = (2 * KEY_HEADS + VALUE_HEADS) * HEAD_DIM


class Qwen38GDNExecutor:
    """One runtime per pipeline worker, shared by its local GDN layers."""

    def __init__(self, device_id: int, build_root: Path, fused_core=False):
        # Load the optional compiler only when this experimental path is selected.
        from models.qwen3_8_27b.device_runner import DeviceRunner
        from models.qwen3_8_27b.layout import make_gdn_unpack
        from models.qwen3_8_27b.linear_attention import make_conv, make_gates, make_qk_norm, make_recurrent

        self.runner = DeviceRunner(device_id, build_root)
        self.kernels = {
            "conv": make_conv(MIXED_CHANNELS),
            "split": make_gdn_unpack(KEY_HEADS, VALUE_HEADS),
            "qk_norm": make_qk_norm(KEY_HEADS),
            "gates": make_gates(VALUE_HEADS),
            "recurrent": make_recurrent(KEY_HEADS, VALUE_HEADS, "KV"),
        }
        self.parameters = {}
        self.cache_bindings = {}
        self.fused_core = fused_core

    def execute(self, layer, metadata, cache, mixed_qkv, a, b, output):
        if metadata.spec_sequence_masks is not None:
            raise ValueError("PTO Qwen3.8 GDN supports ordinary inference without speculative decoding.")
        conv, recurrent = gdn_state_views(cache, KEY_HEADS, VALUE_HEADS)
        self.cache_bindings[layer.prefix] = [conv.data_ptr(), recurrent.data_ptr()]
        tokens = metadata.num_actual_tokens
        if tokens == 0:
            return
        mixed = mixed_qkv[:tokens].contiguous()
        a, b = a[:tokens].contiguous(), b[:tokens].contiguous()
        if mixed.shape != (tokens, MIXED_CHANNELS) or a.shape != (tokens, VALUE_HEADS):
            raise ValueError("Unsupported Qwen3.8 projection shape.")
        starts = metadata.non_spec_query_start_loc.to(dtype=torch.int32).contiguous()
        indices = metadata.non_spec_state_indices_tensor.to(dtype=torch.int32).contiguous()
        # The native builder omits reset flags for decode-only batches: every
        # active decode request continues its existing scheduler-owned state.
        initial = (
            torch.ones_like(indices)
            if metadata.has_initial_state is None
            else metadata.has_initial_state.to(dtype=torch.int32).contiguous()
        )
        if initial.numel() != indices.numel() or starts.numel() != indices.numel() + 1:
            raise ValueError("GDN request boundaries, state indices, and reset flags do not agree.")
        if layer.prefix not in self.parameters:
            self.parameters[layer.prefix] = (
                layer.conv1d.weight.view(MIXED_CHANNELS, -1).T.contiguous(),
                layer.A_log.float().view(1, VALUE_HEADS),
                layer.dt_bias.float().view(1, VALUE_HEADS),
            )
        conv_weight, a_log, dt_bias = self.parameters[layer.prefix]
        convolved = torch.empty_like(mixed)
        q = torch.empty(tokens, KEY_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=mixed.device)
        k = torch.empty_like(q)
        v = torch.empty(tokens, VALUE_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=mixed.device)
        qn, kn = torch.empty_like(q, dtype=torch.float32), torch.empty_like(k, dtype=torch.float32)
        alpha = torch.empty(tokens, VALUE_HEADS, dtype=torch.float32, device=mixed.device)
        beta = torch.empty_like(alpha)
        out = output[:tokens].view(tokens, VALUE_HEADS, HEAD_DIM)
        if self.fused_core:
            from models.qwen3_8_27b.gdn_fused_draft import make_gdn_core

            name = f"fused_gdn_{tokens}"
            if name not in self.kernels:
                self.kernels[name] = make_gdn_core(rows=tokens)
            self.runner.run(
                name,
                self.kernels[name],
                mixed,
                conv_weight,
                conv,
                starts,
                indices,
                initial,
                a,
                b,
                a_log,
                dt_bias,
                recurrent,
                convolved,
                q,
                k,
                v,
                qn,
                kn,
                alpha,
                beta,
                out,
            )
            return
        self.runner.run(
            "conv", self.kernels["conv"], mixed, conv_weight, conv, starts, indices, initial, convolved
        )
        self.runner.run("split", self.kernels["split"], convolved, q, k, v)
        self.runner.run("qk_norm", self.kernels["qk_norm"], q, k, qn, kn)
        self.runner.run("gates", self.kernels["gates"], a, b, a_log, dt_bias, alpha, beta)
        self.runner.run(
            "recurrent",
            self.kernels["recurrent"],
            qn,
            kn,
            v,
            alpha,
            beta,
            recurrent,
            starts,
            indices,
            initial,
            out,
        )
