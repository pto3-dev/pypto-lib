# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
"""Reuse native weight addresses across prefill ND and decode tile-local NZ."""

from math import prod
import time
import torch

WEIGHT_PHASE_MAX_ROWS = 8192

NZ_SHAPES = ("5120x17408", "5120x5120", "5120x3072", "5120x6144", "5120x512", "3072x5120", "5120x124160")


def pack_projection(value, nz=True, nt=128, kt=256):
    n, k = value.shape
    if nz:
        return (
            value.reshape(n // 256, 16, 16, k // kt, kt // 16, 16)
            .permute(0, 3, 4, 1, 2, 5)
            .contiguous()
            .reshape(-1)
        )
    return value.reshape(n // nt, nt, k // kt, kt).permute(0, 2, 1, 3).contiguous().reshape(-1)


def unpack_projection(value, n, k, nz=True, nt=128, kt=256):
    if nz:
        return (
            value.reshape(n // 256, k // kt, kt // 16, 16, 16, 16)
            .permute(0, 3, 4, 1, 2, 5)
            .contiguous()
            .reshape(n, k)
        )
    return value.reshape(n // nt, k // kt, nt, kt).permute(0, 2, 1, 3).contiguous().reshape(n, k)


class WeightPhaseLayout:
    """One projection-sized temporary; all resident weight addresses stay fixed."""

    def __init__(self, banks, spec):
        self.entries = []
        self.layout = "ND"
        self.records = []
        for (layer, key), (group, offset, shape) in spec.weights.items():
            if key == "mlp.up_proj.weight":
                continue
            if key == "mlp.gate_proj.weight":
                shape = (17408, 5120)
            if len(shape) != 2 or shape[0] < 128:
                continue
            n, k = shape
            raw = banks[f"weights{group}"][offset : offset + prod(shape)]
            self.entries.append((raw, n, k, f"{k}x{n}" in NZ_SHAPES))
        n, k = banks["lm_head"].shape
        head = banks["lm_head"].reshape(-1)
        self.entries.append((head, n, k, True))
        self.device = head.device
        self.stream = torch.npu.SyncLaunchStream(device=self.device)
        self.lm_head_decode_view = head.view((n // 256) * (k // 256), 256, 256)
        self.maximum_temporary_bytes = max(
            min(n, WEIGHT_PHASE_MAX_ROWS) * k * 2 for _, n, k, _ in self.entries
        )

    @torch.inference_mode()
    def set_layout(self, target):
        if target == self.layout:
            return
        if target not in ("ND", "NZ"):
            raise ValueError("Unknown weight phase.")
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("Weight transitions belong outside decode capture.")
        torch.npu.synchronize(self.device)
        begin = time.perf_counter()
        # SyncLaunchStream submits operations directly. One stream orders
        # copy-before-reuse of each temporary without337 host synchronizations
        # or retaining all temporary tensors in Torch's host task queue.
        with torch.npu.stream(self.stream):
            for raw, n, k, nz in self.entries:
                # Outer N storage tiles are independent in both resident
                # layouts. Convert aligned row groups without a full LM-head
                # temporary, preserving exactly the existing physical bytes.
                for start in range(0, n, WEIGHT_PHASE_MAX_ROWS):
                    rows = min(WEIGHT_PHASE_MAX_ROWS, n - start)
                    region = raw[start * k : (start + rows) * k]
                    temporary = (
                        pack_projection(region.view(rows, k), nz)
                        if target == "NZ"
                        else unpack_projection(region, rows, k, nz)
                    )
                    region.copy_(temporary.reshape(-1))
                    del temporary
        self.stream.synchronize()
        self.layout = target
        self.records.append(
            dict(
                to=target,
                ms=(time.perf_counter() - begin) * 1000,
                projection_count=len(self.entries),
                memory_allocated_bytes=torch.npu.memory_allocated(self.device),
                memory_reserved_bytes=torch.npu.memory_reserved(self.device),
                peak_allocated_bytes=torch.npu.max_memory_allocated(self.device),
            )
        )
        torch.npu.empty_cache()

    def stats(self):
        return dict(
            layout=self.layout,
            resident_weight_mirror=False,
            maximum_temporary_bytes=self.maximum_temporary_bytes,
            weight_transitions=self.records,
        )


# Kept as an ordinary-call diagnostic so SDK scheduler/dependency profiling
# can inspect the exact same NZ neural kernels used by the graph candidate.
from vllm.forward_context import get_forward_context
from vllm_ascend.pypto.qwen3_8_mpmd import Qwen38MPMDExecutor


class Qwen38PhaseEagerExecutor(Qwen38MPMDExecutor):
    def __init__(self, causal, config, settings):
        super().__init__(causal, config, settings)
        self.phase_weights = None

    def forward(self, model, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        self._prepare_shared_weights()
        context = get_forward_context()
        native = not context.attn_metadata or positions.shape[-1] != 1
        if not native:
            meta = context.attn_metadata[self.causal.model.layers[0].linear_attn.prefix]
            native = bool(getattr(meta, "num_prefills", 0))
        if native:
            if self.phase_weights is not None:
                self.phase_weights.set_layout("ND")
        else:
            if self.phase_weights is None:
                from qwen3_8_27b_shared_cache.model_spec import ModelSpec

                self.phase_weights = WeightPhaseLayout(
                    self.shared_weights, ModelSpec(tokens=1, fuse_mlp=True, native_weight_order=True)
                )
                self.shared_weights["lm_head"] = self.phase_weights.lm_head_decode_view
            self.phase_weights.set_layout("NZ")
        return super().forward(model, input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs)

    def compute_logits(self, causal, hidden_states):
        result = super().compute_logits(causal, hidden_states)
        if not self.last_real and self.phase_weights is not None:
            self.phase_weights.set_layout("NZ")
        return result

    def stats(self):
        result = super().stats()
        result.update(profiling_ordinary_eager=True, external_rpc_per_decode=True)
        if self.phase_weights is not None:
            result.update(self.phase_weights.stats())
        return result
