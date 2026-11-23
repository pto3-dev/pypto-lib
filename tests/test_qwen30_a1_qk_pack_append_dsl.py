# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""Exact Q/K packing and physical PA-cache updates with changing graph inputs."""

import os
from pathlib import Path

import pytest
import torch


def _pack_reference(cpu, initial, pages):
    expected = {
        "packed_q": torch.zeros((80, 16, 128), dtype=torch.float16),
        "packed_k": cpu["k"]
        .view(torch.int16)
        .reshape(20, 8, 128)[:, :4]
        .reshape(20, 512)
        .clone()
        .view(torch.float16),
        "status": torch.full((80, 16), -7, dtype=torch.int32),
        "append_status": torch.full((20, 16), -7, dtype=torch.int32),
        **{name: value.clone() for name, value in initial.items()},
    }
    expected["packed_q"].view(torch.int16)[:, :8] = (
        cpu["q"].view(torch.int16).reshape(80, 8, 128)
    )
    expected["status"][:, 0] = 0
    slots = cpu["slots"][:, 0].tolist()
    for row, slot in enumerate(slots):
        code = (
            0
            if slot < 0
            else 1
            if slot >= pages * 128
            else 3
            if slots.count(slot) > 1
            else 0
        )
        expected["append_status"][row, 0] = code
        if slot < 0 or code:
            continue
        page, time = divmod(slot, 128)
        for head in range(4):
            expected["key_cache"].view(torch.int16)[
                page, head * 8 : (head + 1) * 8, time
            ] = cpu["k"].view(torch.int16)[row * 8 + head].reshape(8, 16)
            expected["value_cache"].view(torch.int16)[
                page, head * 8 : (head + 1) * 8, time
            ] = (
                cpu["values"]
                .view(torch.int16)[row, head * 128 : (head + 1) * 128]
                .reshape(8, 16)
            )
    return expected


def test_qk_pack_append_dynamic_pages_graph(tmp_path):
    torch_npu = pytest.importorskip("torch_npu")
    isa = os.environ.get("PTO_ISA_ROOT")
    if not isa:
        pytest.skip("PTO_ISA_ROOT is required for the allocated A1 device")
    from simpler_setup import A1BorrowedLaunch

    from models.qwen3_30b_w8a8.compile_modelslim_a1 import compile_modelslim_program
    from models.qwen3_30b_w8a8.qk_pack_append_batched_a1 import pack_qk_append_b20

    image, order = compile_modelslim_program(
        pack_qk_append_b20, tmp_path / "program", Path(isa)
    )
    assert tuple(order) == (
        "q",
        "k",
        "packed_q",
        "packed_k",
        "status",
        "values",
        "key_cache",
        "value_cache",
        "slots",
        "append_status",
    )
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    generator = torch.Generator().manual_seed(310026)

    def raw_half(shape):
        return torch.randint(
            -32768, 32768, shape, dtype=torch.int16, generator=generator
        ).view(torch.float16)

    for pages in (2, 5):
        cpu = {
            "q": raw_half((640, 128)),
            "k": raw_half((160, 128)),
            "values": raw_half((20, 512)),
            "slots": torch.full((20, 8), -9, dtype=torch.int32),
        }
        inputs = {name: value.npu() for name, value in cpu.items()}
        outputs, guards = {}, {}
        for name, shape in [
            ("packed_q", (80, 16, 128)),
            ("packed_k", (20, 512)),
            ("status", (80, 16)),
            ("append_status", (20, 16)),
            ("key_cache", (pages, 32, 128, 16)),
            ("value_cache", (pages, 32, 128, 16)),
        ]:
            dtype = (
                torch.int32 if name in {"status", "append_status"} else torch.float16
            )
            sentinel = -7 if dtype == torch.int32 else -19
            storage = torch.full(
                (torch.Size(shape).numel() + 32,), sentinel, dtype=dtype, device="npu:0"
            )
            guards[name] = (storage, sentinel)
            outputs[name] = storage[16:-16].reshape(shape)
        tensors = inputs | outputs
        with A1BorrowedLaunch(image, device_id=0) as owner:
            call = owner.prepare(*(tensors[name] for name in order), blocks=8)
            graph = torch.npu.NPUGraph()
            try:
                # Warm up the compiled entry before capture; all slots are negative.
                call()
                torch.npu.synchronize()
                with torch.npu.graph(graph):
                    call()
                for state in range(4):
                    for name in ["q", "k", "values"]:
                        cpu[name] = raw_half(cpu[name].shape)
                    cpu["slots"] = torch.full((20, 8), -9, dtype=torch.int32)
                    cpu["slots"][:, 0] = torch.arange(20, dtype=torch.int32)
                    cpu["slots"][-1, 0] = pages * 128 - 1
                    if state == 1:
                        cpu["slots"][:6, 0] = torch.tensor(
                            [-1, -(2**31), pages * 128, 2**31 - 1, 23, 23],
                            dtype=torch.int32,
                        )
                    elif state == 2:
                        cpu["slots"][:, 0].zero_()
                    elif state == 3:
                        cpu["slots"][:, 0] = torch.arange(19, -1, -1, dtype=torch.int32)
                        cpu["slots"][0, 0] = pages * 128 - 1
                    initial = {
                        name: raw_half((pages, 32, 128, 16))
                        for name in ["key_cache", "value_cache"]
                    }
                    expected = _pack_reference(cpu, initial, pages)
                    for name, value in cpu.items():
                        inputs[name].copy_(value)
                    for storage, sentinel in guards.values():
                        storage.fill_(sentinel)
                    for name, value in initial.items():
                        outputs[name].copy_(value)
                    graph.replay()
                    for name, value in outputs.items():
                        assert torch.equal(
                            value.cpu().view(torch.uint8),
                            expected[name].view(torch.uint8),
                        ), (pages, state, name)
                    for name, value in inputs.items():
                        assert torch.equal(
                            value.cpu().view(torch.uint8), cpu[name].view(torch.uint8)
                        ), name
                    for storage, sentinel in guards.values():
                        actual = storage.cpu()
                        assert torch.equal(
                            actual[:16], torch.full_like(actual[:16], sentinel)
                        )
                        assert torch.equal(
                            actual[-16:], torch.full_like(actual[-16:], sentinel)
                        )
            finally:
                graph.reset()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
