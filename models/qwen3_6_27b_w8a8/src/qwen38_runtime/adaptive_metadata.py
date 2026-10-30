# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Reuse native bindings directly while proven stable; retain staged fallback."""

from types import SimpleNamespace
from .captured_metadata import CapturedMetadataPlan
from .metadata import MetadataPlan


class AdaptiveMetadataPlan(CapturedMetadataPlan):
    def build(self, executor, context, inputs):
        super().build(executor, context, inputs)
        self._runtime_ready = False
        self._runtime_canonical_bound = False
        self.runtime_rebinds = 0
        self._capture_direct(executor, context, inputs)
        self.direct_replays = 0
        self.staged_replays = 0
        self.binding_miss_counts = {}
        return self

    def mark_runtime_ready(self):
        self._runtime_ready = True

    def _capture_direct(self, executor, context, inputs):
        import torch

        current = torch.npu.current_stream()
        # A one-time binding boundary. Drain the caller before releasing old
        # graph views; never do this in an ordinary replay.
        self._capture_stream.wait_stream(current)
        self._capture_stream.synchronize()
        # Own the exact views used during capture, even if native replaces them.
        metadata = {}
        self._bound_values = []
        for source in self.sources:
            values = self.values(context, source)
            self._bound_values.extend(values)
            if source.full:
                metadata[source.key] = SimpleNamespace(block_tables=values[0], slot_mapping=values[1])
            else:
                metadata[source.key] = SimpleNamespace(
                    spec_state_indices_tensor=values[0], num_accepted_tokens=values[1]
                )
        self._bound_context = SimpleNamespace(attn_metadata=metadata)
        self._bound_inputs = dict(inputs)
        self._direct_signature = self.current_signature(context, inputs)
        self._capture_stream.wait_stream(current)
        with torch.npu.stream(self._capture_stream):
            MetadataPlan.update(self, executor, self._bound_context, self._bound_inputs)
        self._capture_stream.synchronize()
        self._direct_graph = torch.npu.NPUGraph()
        with torch.npu.graph(self._direct_graph, stream=self._capture_stream):
            MetadataPlan.update(self, executor, self._bound_context, self._bound_inputs)
        current.wait_stream(self._capture_stream)

    def update(self, executor, context, inputs):
        signature = self.current_signature(context, inputs)
        if (
            self._runtime_ready
            and not self._runtime_canonical_bound
            and any(getattr(value, "pto_canonical_only", False) for value in context.attn_metadata.values())
        ):
            if signature != self._direct_signature:
                self._capture_direct(executor, context, inputs)
                self.runtime_rebinds += 1
            self._runtime_canonical_bound = True
        if signature == self._direct_signature:
            self._direct_graph.replay()
            self.replays += 1
            self.direct_replays += 1
            self._validate_if_requested(executor, context, inputs)
        else:
            for index, (actual, captured) in enumerate(zip(signature, self._direct_signature)):
                if actual != captured:
                    key = str(index)
                    self.binding_miss_counts[key] = self.binding_miss_counts.get(key, 0) + 1
            super().update(executor, context, inputs)
            self.staged_replays += 1

    def stats(self):
        result = super().stats()
        result.update(
            method="adaptive_metadata_graph",
            direct_replays=self.direct_replays,
            staged_replays=self.staged_replays,
            direct_staging_copies=0,
            binding_miss_counts=self.binding_miss_counts,
            runtime_rebinds=self.runtime_rebinds,
        )
        return result
