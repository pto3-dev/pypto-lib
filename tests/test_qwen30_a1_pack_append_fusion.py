# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""Every query/cache/status bit must match the unchanged separate operations."""

import os
from contextlib import ExitStack
from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize("storage_format", (2, 29))
def test_pack_append_composition_graph(tmp_path, storage_format):
    """Composition preserves every cache bit, including invalid and duplicate slots."""
    import json
    import statistics

    torch_npu = pytest.importorskip("torch_npu")
    isa = os.environ.get("PTO_ISA_ROOT")
    if not isa:
        pytest.skip("PTO_ISA_ROOT is required")
    from simpler_setup import A1BorrowedLaunch, KernelCompiler
    from models.qwen3_30b_w8a8 import kv_cache_batched_a1 as cache

    torch.set_num_threads(2)
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    root = Path(cache.__file__).parent
    compiler = KernelCompiler("a1")
    images = {name: compiler.compile_single_launch(str(root / filename), isa) for name, filename in (
        ("pack", "a1_qk_pack_batched.cpp"), ("append", "a1_kv_append_batched.cpp"),
        ("fused", "a1_qk_pack_append_batched.cpp"))}
    generator = torch.Generator().manual_seed(3103273)
    q = torch.randn((640, 128), generator=generator).half().npu()
    k = torch.randn((160, 128), generator=generator).half().npu()
    values = torch.randn((20, 512), generator=generator).half().npu()
    slots = torch.zeros((20, 8), dtype=torch.int32, device="npu:0")
    queries = [torch.empty((80, 16, 128), dtype=torch.float16, device="npu:0") for _ in range(2)]
    keys = [torch.empty((20, 512), dtype=torch.float16, device="npu:0") for _ in range(2)]
    caches = [[torch_npu.npu_format_cast(torch.zeros((24, 32, 128, 16), dtype=torch.float16, device="npu:0"),
                                        storage_format) for _ in range(2)] for _ in range(2)]
    pack_status = [torch.full((80, 16), -7, dtype=torch.int32, device="npu:0") for _ in range(2)]
    append_status = [torch.full((20, 16), -7, dtype=torch.int32, device="npu:0") for _ in range(2)]
    with ExitStack() as owners:
        graphs = []
        for arm in range(2):
            if arm == 0:
                pack = owners.enter_context(A1BorrowedLaunch(images["pack"]))
                append = owners.enter_context(A1BorrowedLaunch(images["append"]))
                calls = [pack.prepare(q, k, queries[arm], keys[arm], pack_status[arm], blocks=8),
                         append.prepare(keys[arm], values, *caches[arm], slots, append_status[arm],
                                        blocks=8, native_pa_caches=(2, 3) if storage_format == 29 else ())]
            else:
                pack = owners.enter_context(A1BorrowedLaunch(images["fused"]))
                calls = [pack.prepare(q, k, queries[arm], keys[arm], pack_status[arm], values,
                                      *caches[arm], slots, append_status[arm], blocks=8,
                                      native_pa_caches=(6, 7) if storage_format == 29 else ())]
            for call in calls:
                call()
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            owners.callback(graph.reset)
            with torch.npu.graph(graph):
                for call in calls:
                    call()
            graphs.append(graph)
        report = {}
        for case in range(4):
            selected = torch.arange(20, dtype=torch.int32) * 128 + 17 + case
            if case == 1:
                selected[7:] = -1
            elif case == 2:
                selected[0] = 24 * 128
            elif case == 3:
                selected[1] = selected[0]
            slots[:, 0].copy_(selected)
            q.copy_(torch.randn((640, 128), generator=generator).half())
            k.copy_(torch.randn((160, 128), generator=generator).half())
            values.copy_(torch.randn((20, 512), generator=generator).half())
            initial = [torch.randn((24, 32, 128, 16), generator=generator).half() for _ in range(2)]
            for arm in range(2):
                for dest, source in zip(caches[arm], initial, strict=True):
                    dest.copy_(source)
                graphs[arm].replay()
            assert torch.equal(queries[0].cpu(), queries[1].cpu())
            assert torch.equal(keys[0].cpu(), keys[1].cpu())
            for index in range(2):
                assert torch.equal(caches[0][index].cpu(), caches[1][index].cpu())
                expected = initial[index].clone()
                source = k.reshape(20, 8, 128)[:, :4].reshape(20, 512).cpu() if index == 0 else values.cpu()
                for row, slot in enumerate(selected.tolist()):
                    if 0 <= slot < 24 * 128 and torch.count_nonzero(selected == slot) == 1:
                        expected[slot // 128, :, slot % 128] = source[row].reshape(32, 16)
                assert torch.equal(caches[1][index].cpu(), expected)
            assert torch.equal(pack_status[0].cpu(), pack_status[1].cpu())
            assert torch.equal(append_status[0].cpu(), append_status[1].cpu())
            times = []
            for graph in graphs:
                for _ in range(5):
                    graph.replay()
                samples = []
                for _ in range(20):
                    begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                    begin.record()
                    graph.replay()
                    end.record()
                    end.synchronize()
                    samples.append(begin.elapsed_time(end))
                times.append(samples)
            report[str(case)] = {"control_ms": times[0], "fused_ms": times[1],
                                 "control_median_ms": statistics.median(times[0]),
                                 "candidate_median_ms": statistics.median(times[1]), "exact": True}
        (tmp_path / "pack-append-times.json").write_text(json.dumps(report, indent=2))
