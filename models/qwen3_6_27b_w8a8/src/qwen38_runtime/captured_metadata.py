# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Capture metadata arithmetic only; target stays on prepared DIRECT runtime.

Native source addresses may change. Copy their current values into stable small
staging buffers before replay, rather than retaining stale captured pointers.
"""

from types import SimpleNamespace
from .metadata import MetadataPlan


class CapturedMetadataPlan(MetadataPlan):
    def build(self, executor, context, inputs):
        import torch

        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("Prepare metadata preprocessing before capture")
        super().build(executor, context, inputs)
        self._ids = torch.empty_like(inputs["input_ids"])
        positions = inputs["positions"]
        positions = positions[0] if positions.ndim == 2 else positions
        self._positions = torch.empty_like(positions)
        self._sources = []
        metadata = {}
        for source in self.sources:
            values = self.values(context, source)
            staged = tuple(torch.empty_like(value) for value in values)
            self._sources.append(staged)
            if source.full:
                metadata[source.key] = SimpleNamespace(block_tables=staged[0], slot_mapping=staged[1])
            else:
                metadata[source.key] = SimpleNamespace(
                    spec_state_indices_tensor=staged[0], num_accepted_tokens=staged[1]
                )
        self._context = SimpleNamespace(attn_metadata=metadata)
        self._inputs = dict(input_ids=self._ids, positions=self._positions)
        self._stage(context, inputs)
        current = torch.npu.current_stream()
        self._capture_stream = torch.npu.Stream()
        self._capture_stream.wait_stream(current)
        with torch.npu.stream(self._capture_stream):
            MetadataPlan.update(self, executor, self._context, self._inputs)
        self._capture_stream.synchronize()
        self._graph = torch.npu.NPUGraph()
        with torch.npu.graph(self._graph, stream=self._capture_stream):
            MetadataPlan.update(self, executor, self._context, self._inputs)
        current.wait_stream(self._capture_stream)
        self.replays = 0
        self.validation_limit = int(getattr(executor, "settings", {}).get("metadata_validate_updates", 0))
        self.validated_updates = 0
        self.stage_copies_per_replay = 2 + 2 * len(self.sources)
        return self

    def _stage(self, context, inputs):
        if inputs["input_ids"].numel() != self.rows or inputs.get("inputs_embeds") is not None:
            raise ValueError("Unsupported target input")
        self._ids.copy_(inputs["input_ids"])
        positions = inputs["positions"]
        positions = positions[0] if positions.ndim == 2 else positions
        self._positions.copy_(positions)
        for source, staged in zip(self.sources, self._sources):
            values = self.values(context, source)
            for destination, value in zip(staged, values):
                destination.copy_(value)

    def update(self, executor, context, inputs):
        self._stage(context, inputs)
        self._graph.replay()
        self.replays += 1
        self._validate_if_requested(executor, context, inputs)

    def _validate_if_requested(self, executor, context, inputs):
        if self.validated_updates < self.validation_limit:
            import torch

            fields = ("control_i64", "control_i32", "native_metadata")
            snapshots = [getattr(executor, name).clone() for name in fields]
            MetadataPlan.update(self, executor, context, inputs)
            if not all(torch.equal(value, getattr(executor, name)) for value, name in zip(snapshots, fields)):
                raise RuntimeError("Captured metadata differs from eager metadata")
            self.validated_updates += 1

    def stats(self):
        return dict(
            method="staged_metadata_graph",
            target_capture=False,
            sources=len(self.sources),
            staging_copies_per_replay=self.stage_copies_per_replay,
            replays=self.replays,
            validated_updates=self.validated_updates,
        )
