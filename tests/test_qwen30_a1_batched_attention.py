# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""B20 page-128 FP16 attention against the native 310P paged-attention kernel."""

import os
from contextlib import ExitStack
from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize(
    "direct_pages,cache_format,qk_tile,pv_tile,optimization",
    (
        (False, 2, 16, 16, "base"),
        (True, 2, 16, 16, "base"),
        (True, 29, 16, 16, "base"),
        (True, 2, 128, 16, "base"),
        (True, 29, 128, 16, "base"),
        (True, 2, 128, 128, "base"),
        (True, 29, 128, 128, "base"),
        (True, 2, 128, 128, "padding"),
        (True, 29, 128, 128, "padding"),
        (True, 2, 128, 128, "vector"),
        (True, 29, 128, 128, "vector"),
    ),
)
def test_b20_modelslim_attention_graph(
    tmp_path,
    direct_pages,
    cache_format,
    qk_tile,
    pv_tile,
    optimization,
    *,
    source_override=None,
    control_source=None,
):
    torch_npu = pytest.importorskip("torch_npu")
    isa = os.environ.get("PTO_ISA_ROOT")
    if not isa:
        pytest.skip("PTO_ISA_ROOT is required for the allocated 310P device")
    from models.qwen3_30b_w8a8 import attention_modelslim_a1
    from simpler_setup import A1BorrowedLaunch, KernelCompiler

    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    source = Path(attention_modelslim_a1.__file__).with_name(
        "a1_attention_batched_direct.cpp" if direct_pages else "a1_attention_batched_modelslim.cpp"
    )
    if qk_tile == 128:
        name = "a1_attention_batched_wide.cpp" if pv_tile == 128 else "a1_attention_batched_wide_qk.cpp"
        source = Path(attention_modelslim_a1.__file__).with_name(name)
    if optimization != "base":
        source = Path(attention_modelslim_a1.__file__).with_name(f"a1_attention_batched_{optimization}.cpp")
    if source_override is not None:
        source = source_override
    image = KernelCompiler("a1").compile_single_launch(
        str(source), isa, extra_include_dirs=[str(Path(attention_modelslim_a1.__file__).parent)]
    )
    control_image = None
    if control_source is not None:
        control_image = KernelCompiler("a1").compile_single_launch(
            str(control_source), isa, extra_include_dirs=[str(Path(attention_modelslim_a1.__file__).parent)]
        )
    generator = torch.Generator().manual_seed(310320)
    query = torch.zeros((80, 16, 128), dtype=torch.float16, device="npu:0")
    keys = torch.zeros((80, 4096, 128), dtype=torch.float16, device="npu:0")
    values = torch.zeros_like(keys)
    lengths = torch.zeros((20, 8), dtype=torch.int32, device="npu:0")
    output = torch.zeros((20, 4096), dtype=torch.float16, device="npu:0")
    status = torch.full((80, 16), -7, dtype=torch.int32, device="npu:0")
    native_keys = torch_npu.npu_format_cast(
        torch.zeros((640, 32, 128, 16), dtype=torch.float16, device="npu:0"), cache_format
    )
    native_values = torch_npu.npu_format_cast(torch.zeros_like(native_keys), cache_format)
    table = torch.arange(640, dtype=torch.int32).reshape(20, 32).npu()
    with ExitStack() as owners:

        def prepare(owner):
            if direct_pages:
                return owner.prepare(
                    query,
                    native_keys,
                    native_values,
                    lengths,
                    output,
                    table,
                    status,
                    blocks=8,
                    native_pa_caches=(1, 2) if cache_format == 29 else (),
                )
            return owner.prepare(query, keys, values, lengths, output, status, blocks=8)

        owner = owners.enter_context(A1BorrowedLaunch(image))
        call = prepare(owner)
        call()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        owners.callback(graph.reset)
        control_graph = None
        if control_image is not None:
            control_owner = owners.enter_context(A1BorrowedLaunch(control_image))
            control_call = prepare(control_owner)
            control_call()
            torch.npu.synchronize()
            control_graph = torch.npu.NPUGraph()
            owners.callback(control_graph.reset)
            with torch.npu.graph(control_graph):
                control_call()
        with torch.npu.graph(graph):
            call()
        for case in range(4):
            active_lengths = (
                torch.tensor(
                    [
                        0,
                        1,
                        31,
                        32,
                        63,
                        64,
                        65,
                        127,
                        128,
                        129,
                        191,
                        192,
                        193,
                        255,
                        256,
                        257,
                        511,
                        512,
                        513,
                        4096,
                    ]
                ).int()
                if case % 2 == 0
                else torch.linspace(2048, 2448, 20).int()
            )
            lengths[:, 0].copy_(active_lengths)
            q = torch.randn((20, 32, 128), generator=generator).half()
            k = torch.randn((20, 4096, 4, 128), generator=generator).half()
            v = torch.randn((20, 4096, 4, 128), generator=generator).half()
            if case >= 2 and direct_pages:
                padding = torch.arange(4096)[None, :] >= active_lengths[:, None]
                k[padding] = torch.nan
                v[padding] = torch.nan
            query[:, :8].copy_(q.reshape(80, 8, 128))
            keys.copy_(k.permute(0, 2, 1, 3).reshape_as(keys))
            values.copy_(v.permute(0, 2, 1, 3).reshape_as(values))
            mapping = torch.randperm(640, generator=generator).int().reshape(20, 32)
            table.copy_(mapping)
            physical_k = torch.empty((640, 32, 128, 16), dtype=torch.float16)
            physical_v = torch.empty_like(physical_k)
            physical_k[mapping.flatten().long()] = k.reshape(640, 128, 32, 16).permute(0, 2, 1, 3)
            physical_v[mapping.flatten().long()] = v.reshape(640, 128, 32, 16).permute(0, 2, 1, 3)
            native_keys.copy_(physical_k)
            native_values.copy_(physical_v)
            expected = torch.zeros_like(output)
            active = torch.nonzero(active_lengths).flatten()
            reference = torch.zeros((active.numel(), 32, 128), dtype=torch.float16, device="npu:0")
            torch_npu._npu_paged_attention(
                query=q[active].npu(),
                key_cache=native_keys,
                value_cache=native_values,
                num_kv_heads=4,
                num_heads=32,
                scale_value=128**-0.5,
                block_table=table[active.to("npu:0")],
                context_lens=active_lengths[active].npu(),
                out=reference,
            )
            expected[active.to("npu:0")] = reference.reshape(-1, 4096)
            if control_graph is not None:
                output.fill_(torch.nan)
                status.fill_(-7)
                control_graph.replay()
                assert torch.equal(output.cpu(), expected.cpu()), ("control", case, cache_format)
                assert not torch.count_nonzero(status.cpu()[:, 0])
            output.fill_(torch.nan)
            status.fill_(-7)
            graph.replay()
            actual, golden = output.cpu(), expected.cpu()
            if not torch.equal(actual, golden):
                fixture = tmp_path / f"attention-b20-{case}.pt"
                torch.save(
                    {
                        "actual": actual,
                        "expected": golden,
                        "q": q,
                        "k": k,
                        "v": v,
                        "lengths": active_lengths,
                    },
                    fixture,
                )
                per_row = torch.count_nonzero(actual != golden, dim=1).tolist()
                pytest.fail(
                    f"Exact native attention mismatch by row={per_row}, "
                    f"max_abs={float((actual.float() - golden.float()).abs().max())}, fixture={fixture}"
                )
            assert not torch.count_nonzero(status.cpu()[:, 0])
        import hashlib
        import json
        import statistics

        def measure(replay_graphs, replays_per_sample):
            for _ in range(5):
                for replay_graph in replay_graphs:
                    replay_graph.replay()
            torch.npu.synchronize()
            samples = [[] for _ in replay_graphs]
            for sample in range(30):
                order = range(len(replay_graphs))
                if sample % 2:
                    order = reversed(order)
                for index in order:
                    begin = torch.npu.Event(enable_timing=True)
                    end = torch.npu.Event(enable_timing=True)
                    begin.record()
                    for _ in range(replays_per_sample):
                        replay_graphs[index].replay()
                    end.record()
                    end.synchronize()
                    samples[index].append(begin.elapsed_time(end) / replays_per_sample)
            return samples

        replays_per_sample = 20 if control_graph is not None else 1
        replay_graphs = [graph] if control_graph is None else [graph, control_graph]
        measured = measure(replay_graphs, replays_per_sample)
        times = measured[0]
        control_times = measured[1] if control_graph is not None else None
        (tmp_path / "attention-times.json").write_text(
            json.dumps(
                {
                    "source": str(source),
                    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "image_sha256": hashlib.sha256(image).hexdigest(),
                    "cache_format": cache_format,
                    "lengths": active_lengths.tolist(),
                    "native_exact": True,
                    "replays_per_sample": replays_per_sample,
                    "alternating_control_candidate": control_graph is not None,
                    "samples_ms": times,
                    "median_ms": statistics.median(times),
                    "control_source": str(control_source) if control_source else None,
                    "control_image_sha256": hashlib.sha256(control_image).hexdigest()
                    if control_image
                    else None,
                    "control_samples_ms": control_times,
                    "control_median_ms": statistics.median(control_times) if control_times else None,
                    "scope": "one attention graph; standalone, not service performance",
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
