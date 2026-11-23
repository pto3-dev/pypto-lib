# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""Bit-preserving QKV preparation with dynamic table metadata and graph replay."""

import os
from pathlib import Path

import pytest
import torch


def test_qkv_prepare_dynamic_table_graph(tmp_path):
    torch_npu = pytest.importorskip("torch_npu")
    isa = os.environ.get("PTO_ISA_ROOT")
    if not isa:
        pytest.skip("PTO_ISA_ROOT is required for the allocated 310P device")
    from simpler_setup import A1BorrowedLaunch

    from models.qwen3_30b_w8a8.compile_modelslim_a1 import compile_modelslim_program
    from models.qwen3_30b_w8a8.qkv_prepare_batched_a1 import prepare_qkv_rows

    image, order = compile_modelslim_program(
        prepare_qkv_rows, tmp_path / "program", Path(isa)
    )
    assert tuple(order) == (
        "qkv",
        "positions",
        "table",
        "queries",
        "keys",
        "values",
        "rope_rows",
        "status",
    )
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    generator = torch.Generator().manual_seed(310025)

    def raw_half(shape):
        return torch.randint(
            -32768, 32768, shape, dtype=torch.int16, generator=generator
        ).view(torch.float16)

    for height in (4096, 8192):
        inputs = {
            "qkv": raw_half((20, 5120)).npu(),
            "table": raw_half((height, 128)).npu(),
            "positions": torch.full((20, 8), -9, dtype=torch.int32, device="npu:0"),
        }
        outputs, guards = {}, {}
        for name, shape in [
            ("queries", (640, 128)),
            ("keys", (160, 128)),
            ("values", (20, 512)),
            ("rope_rows", (20, 128)),
            ("status", (20, 16)),
        ]:
            dtype = torch.int32 if name == "status" else torch.float16
            initial = -7 if name == "status" else -19
            storage = torch.full(
                (shape[0] * shape[1] + 32,), initial, dtype=dtype, device="npu:0"
            )
            guards[name] = (storage, initial)
            outputs[name] = storage[16:-16].reshape(shape)
        tensors = inputs | outputs
        with A1BorrowedLaunch(image, device_id=0) as owner:
            call = owner.prepare(*(tensors[name] for name in order), blocks=8)
            call()
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            try:
                with torch.npu.graph(graph):
                    call()
                for state in range(4):
                    cpu = {
                        "qkv": raw_half((20, 5120)),
                        "table": raw_half((height, 128)),
                        "positions": torch.full((20, 8), -9, dtype=torch.int32),
                    }
                    cpu["positions"][:, 0] = torch.arange(20, dtype=torch.int32) * 127
                    cpu["positions"][-1, 0] = height - 1
                    if state == 1:
                        cpu["positions"][0, 0], cpu["positions"][1, 0] = -1, height
                    elif state == 2:
                        cpu["positions"][:, 0].zero_()
                    elif state == 3:
                        cpu["positions"][:, 0] = height - 1
                    for name, value in cpu.items():
                        inputs[name].copy_(value)
                    for storage, initial in guards.values():
                        storage.fill_(initial)
                    pos = cpu["positions"][:, 0].long()
                    valid = (pos >= 0) & (pos < height)
                    qkv = cpu["qkv"].view(torch.int16).reshape(20, 40, 128)
                    expected = {
                        name: torch.full_like(
                            value.cpu(), -7 if name == "status" else -19
                        )
                        for name, value in outputs.items()
                    }
                    expected["queries"].view(torch.int16).reshape(20, 32, 128)[
                        valid
                    ] = qkv[valid, :32]
                    expected["keys"].view(torch.int16).reshape(20, 8, 128)[valid] = (
                        torch.cat(
                            [qkv[valid, 32:36], torch.zeros_like(qkv[valid, 32:36])],
                            dim=1,
                        )
                    )
                    expected["values"].view(torch.int16).reshape(20, 4, 128)[valid] = (
                        qkv[valid, 36:40]
                    )
                    expected["rope_rows"].view(torch.int16)[valid] = cpu["table"].view(
                        torch.int16
                    )[pos[valid]]
                    expected["status"][:, 0] = (~valid).to(torch.int32)
                    graph.replay()
                    for name, value in outputs.items():
                        assert torch.equal(
                            value.cpu().view(torch.uint8),
                            expected[name].view(torch.uint8),
                        ), (height, state, name)
                    for name, value in inputs.items():
                        assert torch.equal(
                            value.cpu().view(torch.uint8), cpu[name].view(torch.uint8)
                        ), name
                    for storage, initial in guards.values():
                        actual = storage.cpu()
                        assert torch.equal(
                            actual[:16], torch.full_like(actual[:16], initial)
                        )
                        assert torch.equal(
                            actual[-16:], torch.full_like(actual[-16:], initial)
                        )
            finally:
                graph.reset()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
