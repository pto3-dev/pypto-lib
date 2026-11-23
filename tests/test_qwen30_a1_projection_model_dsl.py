# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""Full-width B20 projection agreement with the retained native kernels."""

import os
from contextlib import ExitStack
from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize("kind", ["qkv", "o"])
def test_full_width_modelslim_projection_dsl(kind, tmp_path):
    torch_npu = pytest.importorskip("torch_npu")
    isa = os.environ.get("PTO_ISA_ROOT")
    if not isa:
        pytest.skip("PTO_ISA_ROOT is required")
    from models.qwen3_30b_w8a8.compile_modelslim_a1 import compile_modelslim_program
    from models.qwen3_30b_w8a8.linear_batched_modelslim_a1 import (
        compile_a1_batched_modelslim_projection,
    )
    from models.qwen3_30b_w8a8.linear_batched_modelslim_a1_dsl import (
        make_a1_modelslim_projection,
    )
    from simpler_setup import A1BorrowedLaunch

    inner, width = (2048, 5120) if kind == "qkv" else (4096, 2048)
    program = make_a1_modelslim_projection(kind)
    image, order = compile_modelslim_program(program, tmp_path / "dsl", Path(isa))
    assert order == ("x", "weight", "bias", "scales", "output", "status")
    reference = compile_a1_batched_modelslim_projection(
        kind, Path(isa), wide=True, fused_dequant=True, packed_scale_words=True
    )
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    x = torch.zeros((32, inner), dtype=torch.int8, device="npu:0")
    weight = torch.zeros((width, inner), dtype=torch.int8, device="npu:0")
    weight = torch_npu.npu_format_cast(weight, 29)
    bias = torch.zeros((1, width), dtype=torch.int32, device="npu:0")
    words = torch.zeros((1, 2 * width), dtype=torch.int32, device="npu:0")
    scales = words.view(torch.int64)
    storage = torch.full((20 * width + 32,), -99.0, dtype=torch.float16, device="npu:0")
    output = storage[16:-16].view(20, width)
    native = torch.empty_like(output)
    status = torch.full((8, 16), -7, dtype=torch.int32, device="npu:0")
    native_status = torch.full_like(status, -7)
    generator = torch.Generator().manual_seed(310012 + inner)
    with ExitStack() as owners:
        native_owner = owners.enter_context(A1BorrowedLaunch(reference, device_id=0))
        native_call = native_owner.prepare(x, weight, bias, words, native, native_status,
                                           blocks=8, native_nz_weights=(1,))
        dsl_owner = owners.enter_context(A1BorrowedLaunch(image, device_id=0))
        dsl_call = dsl_owner.prepare(x, weight, bias, scales, output, status,
                                     blocks=8, native_nz_weights=(1,))
        graphs = []
        for call in [native_call, dsl_call]:
            call()
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            owners.callback(graph.reset)
            with torch.npu.graph(graph):
                call()
            graphs.append(graph)
        for replay in range(4):
            source = torch.randint(-128, 128, (32, inner), dtype=torch.int8, generator=generator)
            source[20:].zero_()
            bank = torch.randint(-128, 128, (width, inner), dtype=torch.int8, generator=generator)
            integer_bias = torch.randint(-(2**31), 2**31, (1, width), dtype=torch.int64, generator=generator).int()
            if replay == 0:
                source.zero_()
                integer_bias.zero_()
            elif replay == 1:
                source[0].fill_(127)
                bank[0].fill_(127)
                integer_bias[0, :4] = torch.tensor([1, 2**24 + 1, 2**31 - 1, -(2**31)], dtype=torch.int32)
            channel = torch.exp2(torch.randint(-30, -20, (1, width), generator=generator).float())
            channel *= torch.rand((1, width), generator=generator).add(0.5)
            packed = torch.zeros((1, width, 2), dtype=torch.int32)
            packed[:, :, 0].copy_(channel.contiguous().view(torch.int32))
            x.copy_(source)
            weight.copy_(bank)
            bias.copy_(integer_bias)
            words.copy_(packed.reshape(1, 2 * width))
            storage.fill_(-99.0)
            status.fill_(-7)
            native_status.fill_(-7)
            for graph in graphs:
                graph.replay()
            actual = output.cpu().view(torch.int16)
            expected = native.cpu().view(torch.int16)
            if not torch.equal(actual, expected):
                torch.save({"actual_bits": actual, "native_bits": expected,
                            "input": source, "weight": bank, "bias": integer_bias,
                            "scale_words": packed}, tmp_path / f"mismatch-{replay}.pt")
            assert torch.equal(actual, expected), (kind, replay)
            assert not torch.count_nonzero(status.cpu()[:, 0])
            assert not torch.count_nonzero(native_status.cpu()[:, 0])
            guard = torch.full((16,), -99.0, dtype=torch.float16)
            assert torch.equal(storage[:16].cpu(), guard)
            assert torch.equal(storage[-16:].cpu(), guard)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])



@pytest.mark.parametrize("fused,packed", [(False, False), (True, False), (False, True)])
def test_dsl_projection_rejects_incompatible_scale_abi(fused, packed):
    from models.qwen3_30b_w8a8.decode_batched_modelslim_a1 import A1BatchedModelSlimDecoderLayer

    x = torch.zeros((20, 2048), dtype=torch.float16)
    weights = {"qkv_scale": torch.zeros((1, 10240), dtype=torch.int32),
               "o_scale": torch.zeros((1, 4096), dtype=torch.int32)}
    images = {"execution_options": {"dsl_projection": True,
                                    "fused_projection_dequant": fused,
                                    "packed_projection_scales": packed}}
    with pytest.raises(ValueError, match="require.*fused|require.*packed|require.*dequantization"):
        A1BatchedModelSlimDecoderLayer(images, x, x, None, (), weights, x, x, first_layer=True)
