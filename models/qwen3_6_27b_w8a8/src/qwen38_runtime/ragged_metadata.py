# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DIRECT-only padded verifier metadata. KV/GDN payloads remain native aliases.

Only host query endpoints determine grouping. Prior accepted-state selection uses
all six historical states, independently of this round's valid query length.
"""

from .metadata import Source, tensor_signature


class RaggedMetadataPlan:
    def __init__(self, keys):
        self.keys = tuple(keys)
        self.sources = []
        self.mapping = []
        self.layouts = {}

    @staticmethod
    def raw(context, source):
        meta = context.attn_metadata[source.key]
        return (
            (meta.block_tables, meta.slot_mapping)
            if source.full
            else (meta.spec_state_indices_tensor, meta.num_accepted_tokens)
        )

    def build(self, executor, context, inputs):
        import torch

        seen = {}
        for i, key in enumerate(self.keys):
            source = Source(i, i % 4 == 3, key)
            raw = self.raw(context, source)
            identity = (source.full,) + tuple(tensor_signature(t) for t in raw)
            if identity not in seen:
                seen[identity] = len(self.sources)
                self.sources.append(source)
            self.mapping.append(seen[identity])
        dev = inputs["input_ids"].device
        cols = executor.table_cols
        self.mapping_tensor = torch.tensor(self.mapping, dtype=torch.int64, device=dev)
        self.rows = torch.zeros((len(self.sources), 24 * (3 + cols)), dtype=torch.int32, device=dev)
        self.ids = torch.empty(24, dtype=inputs["input_ids"].dtype, device=dev)
        self.positions = torch.empty(24, dtype=inputs["positions"].dtype, device=dev)
        full = next(source for source in self.sources if source.full)
        self.slots = torch.empty(24, dtype=self.raw(context, full)[1].dtype, device=dev)
        self.tables = torch.zeros((4, cols), dtype=torch.int32, device=dev)
        self.states = torch.zeros((4, 6), dtype=torch.int32, device=dev)
        self.accepted = torch.ones(4, dtype=torch.int64, device=dev)
        self.device = dev
        return self

    def validate(self, context, inputs, *, stable_addresses=False):
        if stable_addresses:
            raise ValueError("Ragged verifier is DIRECT-only")
        # Aliases are checked every call; no representative-only assumption.
        groups = {}
        for i, key in enumerate(self.keys):
            source = Source(i, i % 4 == 3, key)
            raw = self.raw(context, source)
            group = self.mapping[i]
            signature = (source.full,) + tuple(tensor_signature(t) for t in raw)
            if group in groups and groups[group] != signature:
                raise ValueError("Ragged metadata alias groups changed")
            groups[group] = signature

    def update(self, executor, context, inputs, spec):
        import torch

        nreq = len(spec.query_lengths)
        real = sum(spec.query_lengths)
        cols = executor.table_cols
        if spec.input_rows != inputs["input_ids"].numel():
            raise ValueError("Input layout changed")
        key = (spec.query_lengths, spec.input_rows)
        cached = self.layouts.get(key)
        if cached is None:
            cached = (
                torch.tensor(spec.gather_input, dtype=torch.int64, device=self.device),
                torch.tensor(spec.valid_rows, dtype=torch.bool, device=self.device),
                torch.tensor([i < nreq for i in range(4)], dtype=torch.bool, device=self.device),
                torch.tensor(spec.restore_output, dtype=torch.int64, device=self.device),
                torch.tensor(spec.ends, dtype=torch.int32, device=self.device),
            )
            self.layouts[key] = cached
        gather, valid, active, restore, ends = cached
        self.restore = restore
        ids = inputs["input_ids"].reshape(-1)
        positions = inputs["positions"]
        positions = positions[0] if positions.ndim == 2 else positions
        if positions.numel() < spec.input_rows:
            raise ValueError("Position storage is smaller than input")
        torch.index_select(ids, 0, gather, out=self.ids)
        executor.control_i64[:24].copy_(self.ids)
        torch.index_select(positions.reshape(-1), 0, gather, out=self.positions)
        executor.control_i64[24:48].copy_(self.positions)
        full = next(s for s in self.sources if s.full)
        native_slots = self.raw(context, full)[1].reshape(-1)
        if native_slots.numel() < real:
            raise ValueError("Missing slot mapping")
        torch.index_select(native_slots, 0, gather, out=self.slots)
        executor.control_i64[48:72].copy_(torch.where(valid, self.slots, -1))
        self.rows.zero_()
        for i, source in enumerate(self.sources):
            row = self.rows[i]
            values = self.raw(context, source)
            if source.full:
                qends = getattr(context.attn_metadata[source.key], "actual_seq_lengths_q", None)
                expected = []
                total = 0
                for length in spec.query_lengths:
                    total += length
                    expected.append(total)
                if qends != expected:
                    raise ValueError("Full-attention query partitions disagree")
                table = values[0]
                if table.ndim != 2 or table.shape[0] < nreq or table.shape[1] < cols:
                    raise ValueError("Block table shape is insufficient")
                self.tables.zero_()
                self.tables[:nreq].copy_(table[:nreq, :cols])
                row[24:48].fill_(1)
                row[48:72].copy_((self.positions + 1).clamp_min(1))
                row[72:].view(4, 6, cols).copy_(self.tables.unsqueeze(1))
            else:
                state, accepted = values
                if state.ndim != 2 or state.shape[0] < nreq or state.shape[1] != 6 or accepted.numel() < nreq:
                    raise ValueError("Missing historical speculative states")
                self.states.zero_()
                self.states[:nreq].copy_(state[:nreq])
                self.accepted.fill_(1)
                self.accepted[:nreq].copy_(accepted.reshape(-1)[:nreq].clamp(1, 6))
                origin = self.states.gather(1, (self.accepted - 1).reshape(4, 1)).reshape(4)
                row[:4].copy_(torch.where(active, origin, -1))
                row[4:24].copy_(
                    torch.where(active[:, None], self.states[:, 1:], -1).transpose(0, 1).reshape(-1)
                )
                row[24:28].copy_(torch.where(active, self.states[:, 0], -1))
                row[28:32].copy_(row[24:28])
                row[48:52].copy_(self.accepted)
        torch.index_select(self.rows, 0, self.mapping_tensor, out=executor.native_metadata)
        executor.control_i32[9:13].copy_(ends)
        executor.control_i32[29:77].copy_(executor.native_metadata[0, :48])
        n = 24 * cols
        executor.control_i32[77 : 77 + n].copy_(executor.native_metadata[3, 72:])
        executor.control_i32[77 + n : 101 + n].copy_(executor.native_metadata[3, 48:72])
