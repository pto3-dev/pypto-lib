"""Metadata batching preserves addresses, padding, conversion and graph replay."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


def _probe_metadata():
    name = '_all910_metadata_foreach_probe'
    if name not in sys.modules:
        source = Path(__file__).parents[1] / 'models/qwen3_30b_w8a8/kv_cache_batched_a1.py'
        spec = importlib.util.spec_from_file_location(name, source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name].A1BatchedMetadata


@pytest.mark.parametrize("foreach_copy", (False, True))
def test_batched_metadata_preserves_addresses_and_clears_padding(foreach_copy):
    A1BatchedMetadata = _probe_metadata()
    metadata = A1BatchedMetadata.allocate("cpu")
    pointers = [t.data_ptr() for t in vars(metadata).values()]
    for rows in (20, 3, 1, 20):
        positions = torch.arange(rows, dtype=torch.int64) + 2048
        slots = torch.arange(rows, dtype=torch.int32) * 128
        lengths = positions + 1
        table = torch.arange(rows * 20, dtype=torch.int32).reshape(rows, 20)
        metadata.update(positions, slots, lengths, table, foreach_copy=foreach_copy)
        assert [t.data_ptr() for t in vars(metadata).values()] == pointers
        for actual, expected in ((metadata.positions, positions), (metadata.slots, slots),
                                 (metadata.lengths, lengths)):
            assert torch.equal(actual[:rows, 0], expected.int())
        assert torch.equal(metadata.block_table[:rows, :20], table)
        assert not torch.count_nonzero(metadata.block_table[:, 20:])
        assert not torch.count_nonzero(metadata.lengths[rows:])
        assert torch.all(metadata.slots[rows:] == -1)


@pytest.mark.parametrize("invalid", ("batch", "slots", "lengths", "table", "dtype"))
@pytest.mark.parametrize("foreach_copy", (False, True))
def test_batched_metadata_rejects_bad_shapes_before_mutation(invalid, foreach_copy):
    A1BatchedMetadata = _probe_metadata()
    metadata = A1BatchedMetadata.allocate("cpu")
    inputs = [torch.arange(20), torch.arange(20), torch.ones(20, dtype=torch.int32),
              torch.ones((20, 32), dtype=torch.int32)]
    if invalid == "batch":
        inputs[0] = torch.arange(21)
    elif invalid == "slots":
        inputs[1] = torch.arange(19)
    elif invalid == "lengths":
        inputs[2] = torch.ones((20, 1), dtype=torch.int32)
    elif invalid == "table":
        inputs[3] = torch.ones((20, 33), dtype=torch.int32)
    else:
        inputs[0] = inputs[0].float()
    before = [t.clone() for t in vars(metadata).values()]
    with pytest.raises(ValueError):
        metadata.update(*inputs, foreach_copy=foreach_copy)
    assert all(torch.equal(a, b) for a, b in zip(before, vars(metadata).values(), strict=True))




def test_batched_metadata_foreach_graph(tmp_path):
    """Both metadata updates preserve exact padded carriers on changed-input replay."""
    import json
    import statistics

    torch_npu = pytest.importorskip("torch_npu")
    torch.set_num_threads(2)
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    A1BatchedMetadata = _probe_metadata()
    carriers = [A1BatchedMetadata.allocate("npu:0") for _ in range(2)]
    inputs = [torch.zeros(40, dtype=dtype, device="npu:0")[::2]
              for dtype in (torch.int64, torch.int32, torch.int64)]
    inputs.append(torch.zeros((20, 64), dtype=torch.int64, device="npu:0")[:, ::2])
    graphs = []
    try:
        for index, metadata in enumerate(carriers):
            metadata.update(*inputs, foreach_copy=bool(index))
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            graphs.append(graph)
            with torch.npu.graph(graph):
                metadata.update(*inputs, foreach_copy=bool(index))
        for replay in range(4):
            values = [torch.arange(20, dtype=torch.int64) + 2048 + replay,
                      torch.arange(20, dtype=torch.int32) * 128 + replay,
                      torch.arange(20, dtype=torch.int64) + 2049 + replay,
                      torch.arange(20 * 32, dtype=torch.int64).reshape(20, 32) + replay]
            values[1][-3:] = -1
            for tensor, value in zip(inputs, values, strict=True):
                tensor.copy_(value)
            for graph in graphs:
                graph.replay()
            for left, right in zip(vars(carriers[0]).values(), vars(carriers[1]).values(), strict=True):
                assert torch.equal(left.cpu(), right.cpu())
            for metadata in carriers:
                for tensor, expected in zip((metadata.positions, metadata.slots, metadata.lengths), values[:3], strict=True):
                    assert torch.equal(tensor.cpu()[:, 0], expected.int())
                assert torch.equal(metadata.block_table.cpu(), values[3].int())
        samples = []
        for graph in graphs:
            for _ in range(5):
                graph.replay()
            times = []
            for _ in range(40):
                start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                times.append(start.elapsed_time(end))
            samples.append(times)
        (tmp_path / "foreach-metadata-times.json").write_text(json.dumps({
            "control_ms": samples[0], "foreach_ms": samples[1], "exact": True,
            "control_median_ms": statistics.median(samples[0]),
            "candidate_median_ms": statistics.median(samples[1])}, indent=2))
    finally:
        for graph in reversed(graphs):
            graph.reset()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
