# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compile an alias plan once and update preallocated ABI-v1 control buffers.

ABI-v1 still requires 64 expanded rows. We preserve it to keep K1/image bytes
unchanged, but remove per-round clones/stack allocations and repeated alias scans.
A compact ABI-v2 needs a matched orchestration image and is not silently enabled.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    layer: int
    full: bool
    key: str


def tensor_signature(value):
    return (
        value.data_ptr(),
        tuple(value.shape),
        tuple(value.stride()),
        value.storage_offset(),
        str(value.dtype),
        str(value.device),
    )


def padded_signature(value, shape):
    """Validate and describe an alias without allocating a torch view."""
    if value is None or len(shape) != value.ndim:
        raise ValueError("Invalid metadata tensor rank")
    stride = value.stride()
    if any(s < 0 for s in stride):
        raise ValueError("Negative strides are unsupported")
    extent = value.storage_offset() + sum((n - 1) * s for n, s in zip(shape, stride)) + 1
    if extent * value.element_size() > value.untyped_storage().nbytes():
        raise ValueError("Metadata backing storage is too small")
    return (
        value.data_ptr(),
        tuple(shape),
        tuple(stride),
        value.storage_offset(),
        str(value.dtype),
        str(value.device),
    )


def padded_view(value, shape):
    padded_signature(value, shape)
    return value.as_strided(shape, value.stride())


class MetadataPlan:
    def __init__(self, keys, batch=4, drafts=5):
        self.keys = tuple(keys)
        self.batch = batch
        self.width = drafts + 1
        self.rows = batch * self.width
        self.sources = []
        self.mapping = []
        self.signature = None
        self.device_rows = None

    def raw_values(self, context, source):
        meta = context.attn_metadata[source.key]
        return (
            (meta.block_tables, meta.slot_mapping)
            if source.full
            else (meta.spec_state_indices_tensor, meta.num_accepted_tokens)
        )

    def shapes(self, source, values):
        return (
            ((self.batch, values[0].shape[1]), (self.rows,))
            if source.full
            else ((self.batch, self.width), (self.batch,))
        )

    def signatures(self, source, values):
        return tuple(
            padded_signature(value, shape) for value, shape in zip(values, self.shapes(source, values))
        )

    def values(self, context, source):
        values = self.raw_values(context, source)
        return tuple(padded_view(value, shape) for value, shape in zip(values, self.shapes(source, values)))

    def build(self, executor, context, inputs):
        import torch

        seen = {}
        self.sources = []
        self.mapping = []
        for i, key in enumerate(self.keys):
            source = Source(i, i % 4 == 3, key)
            values = self.values(context, source)
            identity = (source.full,) + tuple(tensor_signature(v) for v in values)
            if identity not in seen:
                seen[identity] = len(self.sources)
                self.sources.append(source)
            self.mapping.append(seen[identity])
        device = inputs["input_ids"].device
        self.mapping_tensor = torch.tensor(self.mapping, dtype=torch.int64, device=device)
        self.device_rows = torch.zeros(
            (len(self.sources), self.rows * (3 + executor.table_cols)), dtype=torch.int32, device=device
        )
        self.signature = self.current_signature(context, inputs)
        return self

    def current_signature(self, context, inputs):
        values = [tensor_signature(inputs["input_ids"]), tensor_signature(inputs["positions"])]
        for source in self.sources:
            values.extend(self.signatures(source, self.raw_values(context, source)))
        return tuple(values)

    def validate(self, context, inputs, *, stable_addresses):
        if stable_addresses and self.current_signature(context, inputs) != self.signature:
            raise ValueError("Captured metadata bindings changed")
        # Never trust the representatives alone: a nonrepresentative layer can
        # change its binding without changing any representative pointer.
        # Native metadata has no binding-generation token, so validate aliases.
        groups = {}
        for i, key in enumerate(self.keys):
            source = Source(i, i % 4 == 3, key)
            raw = self.raw_values(context, source)
            group = self.mapping[i]
            if group not in groups:
                groups[group] = (raw, (source.full,) + self.signatures(source, raw))
                continue
            representative, expected = groups[group]
            if all(a is b for a, b in zip(raw, representative)):
                continue
            if (source.full,) + self.signatures(source, raw) != expected:
                raise ValueError("Metadata alias groups changed; rebind between completed executions")

    def update(self, executor, context, inputs):
        import torch

        ids = inputs["input_ids"]
        positions = inputs["positions"]
        positions = positions[0] if positions.ndim == 2 else positions
        if ids.numel() != self.rows or inputs.get("inputs_embeds") is not None:
            raise ValueError("Unsupported target input")
        full = next(s for s in self.sources if s.full)
        slots = self.values(context, full)[1]
        active = slots.reshape(self.batch, self.width)[:, 0] >= 0
        executor.control_i64[: self.rows].copy_(ids.reshape(-1))
        executor.control_i64[self.rows : 2 * self.rows].copy_(positions.reshape(-1))
        executor.control_i64[2 * self.rows : 3 * self.rows].copy_(slots)
        for i, source in enumerate(self.sources):
            row = self.device_rows[i]
            values = self.values(context, source)
            if source.full:
                row[self.rows : 2 * self.rows].fill_(1)
                row[2 * self.rows : 3 * self.rows].copy_((positions + 1).clamp_min(1))
                row[3 * self.rows :].view(self.batch, self.width, executor.table_cols).copy_(
                    values[0][:, : executor.table_cols].unsqueeze(1)
                )
            else:
                state = values[0]
                accepted = values[1].clamp(1, self.width).to(torch.int64)
                origin = state.gather(1, (accepted - 1).reshape(self.batch, 1)).reshape(self.batch)
                row[: self.batch].copy_(torch.where(active, origin, -1))
                row[self.batch : self.rows].copy_(
                    torch.where(active[:, None], state[:, 1:], -1).transpose(0, 1).reshape(-1)
                )
                row[self.rows : self.rows + self.batch].copy_(torch.where(active, state[:, 0], -1))
                row[self.rows + self.batch : self.rows + 2 * self.batch].copy_(
                    torch.where(active, state[:, 0], -1)
                )
                row[2 * self.rows : 2 * self.rows + self.batch].copy_(accepted)
        torch.index_select(self.device_rows, 0, self.mapping_tensor, out=executor.native_metadata)
        executor.control_i32[29:77].copy_(executor.native_metadata[0, :48])
        n = self.rows * executor.table_cols
        executor.control_i32[77 : 77 + n].copy_(executor.native_metadata[3, 72:])
        executor.control_i32[77 + n : 101 + n].copy_(executor.native_metadata[3, 48:72])
