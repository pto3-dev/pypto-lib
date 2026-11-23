# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""B20 PTO3 metadata and bit-exact PA-cache graph tests; not a model benchmark."""

import os
from contextlib import ExitStack
from pathlib import Path

import pytest
import torch

from models.qwen3_30b_w8a8.kv_cache_batched_a1 import (
    A1BatchedMetadata,
    compile_a1_batched_cache_images,
)


def test_batched_metadata_preserves_addresses_and_clears_padding():
    metadata = A1BatchedMetadata.allocate("cpu")
    pointers = [t.data_ptr() for t in vars(metadata).values()]
    for rows in (20, 3, 1, 20):
        positions = torch.arange(rows, dtype=torch.int64) + 2048
        slots = torch.arange(rows, dtype=torch.int32) * 128
        lengths = positions + 1
        table = torch.arange(rows * 20, dtype=torch.int32).reshape(rows, 20)
        metadata.update(positions, slots, lengths, table)
        assert [t.data_ptr() for t in vars(metadata).values()] == pointers
        for actual, expected in ((metadata.positions, positions), (metadata.slots, slots),
                                 (metadata.lengths, lengths)):
            assert torch.equal(actual[:rows, 0], expected.int())
        assert torch.equal(metadata.block_table[:rows, :20], table)
        assert not torch.count_nonzero(metadata.block_table[:, 20:])
        assert not torch.count_nonzero(metadata.lengths[rows:])
        assert torch.all(metadata.slots[rows:] == -1)


@pytest.mark.parametrize("invalid", ("batch", "slots", "lengths", "table", "dtype"))
def test_batched_metadata_rejects_bad_shapes_before_mutation(invalid):
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
        metadata.update(*inputs)
    assert all(torch.equal(a, b) for a, b in zip(before, vars(metadata).values(), strict=True))


@pytest.fixture(scope="module")
def cache_images():
    pytest.importorskip("torch_npu")
    root = os.environ.get("PTO_ISA_ROOT")
    if not root:
        pytest.skip("PTO_ISA_ROOT is required for the allocated 310P device")
    torch.npu.set_device(0)
    return compile_a1_batched_cache_images(Path(root))


def _pack_cache(dense):
    return dense.reshape(-1, 128, 32, 16).permute(0, 2, 1, 3).contiguous()


@pytest.mark.parametrize("storage_format", (2, 29))
def test_b20_native_layout_graph_replay(cache_images, storage_format):
    import torch_npu
    from simpler_setup import A1BorrowedLaunch

    generator = torch.Generator().manual_seed(310320)
    device = "npu:0"
    capacity, pages = 4096, 20 * 32
    metadata = A1BatchedMetadata.allocate(device)
    key = torch.zeros((20, 512), dtype=torch.float16, device=device)
    value = torch.zeros_like(key)
    key_cache = torch_npu.npu_format_cast(torch.zeros((pages, 32, 128, 16), dtype=torch.float16, device=device), storage_format)
    value_cache = torch_npu.npu_format_cast(torch.zeros_like(key_cache), storage_format)
    keys = torch.zeros((80, capacity, 128), dtype=torch.float16, device=device)
    values = torch.zeros_like(keys)
    append_status = torch.full((20, 16), -7, dtype=torch.int32, device=device)
    gather_status = torch.full((80, 16), -7, dtype=torch.int32, device=device)
    with ExitStack() as stack:
        append = stack.enter_context(A1BorrowedLaunch(cache_images["append"], device_id=0))
        gather = stack.enter_context(A1BorrowedLaunch(cache_images["gather"], device_id=0))
        append_call = append.prepare(key, value, key_cache, value_cache, metadata.slots,
                                     append_status, blocks=8, native_pa_caches=(2, 3) if storage_format == 29 else ())
        gather_call = gather.prepare(key_cache, value_cache, metadata.block_table, metadata.lengths,
                                     keys, values, gather_status, blocks=8, native_pa_caches=(0, 1) if storage_format == 29 else ())
        if storage_format == 29:
            with pytest.raises(ValueError, match="ND tensor storage"):
                append.prepare(key, value, key_cache, value_cache, metadata.slots, append_status, blocks=8)
            with pytest.raises(ValueError, match="unique argument indices"):
                append.prepare(key, value, key_cache, value_cache, metadata.slots, append_status,
                               blocks=8, native_pa_caches=(2, 2))
            with pytest.raises(ValueError, match="must not overlap"):
                append.prepare(key, value, key_cache, value_cache, metadata.slots, append_status,
                               blocks=8, native_nz_weights=(2,), native_pa_caches=(2, 3))
            assert key_cache.storage_offset() == value_cache.storage_offset() == 0
        else:
            with pytest.raises(ValueError, match="PA borrowing requires"):
                append.prepare(key, value, key_cache, value_cache, metadata.slots, append_status,
                               blocks=8, native_pa_caches=(2, 3))
        append_call()
        gather_call()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        try:
            with torch.npu.graph(graph):
                append_call()
                gather_call()
            for replay, rows in enumerate((20, 7, 1, 20)):
                lengths = (torch.arange(rows, dtype=torch.int32) * 137 + 1 + replay * 31) % 2448 + 1
                lengths[0] = (2048, 2177, 2448, 129)[replay]
                table = torch.randperm(pages, generator=generator).int().reshape(20, 32)[:rows]
                slots = table[torch.arange(rows), (lengths - 1) // 128] * 128 + (lengths - 1) % 128
                metadata.update((lengths - 1).to(device), slots.to(device), lengths.to(device), table.to(device))
                new_key = torch.randn((20, 512), generator=generator).half()
                new_value = torch.randn((20, 512), generator=generator).half()
                dense_key = torch.randn((pages * 128, 512), generator=generator).half()
                dense_value = torch.randn((pages * 128, 512), generator=generator).half()
                key.copy_(new_key)
                value.copy_(new_value)
                key_cache.copy_(_pack_cache(dense_key))
                value_cache.copy_(_pack_cache(dense_value))
                dense_key[slots.long()] = new_key[:rows]
                dense_value[slots.long()] = new_value[:rows]
                graph.replay()
                torch.npu.synchronize()
                assert torch.equal(key_cache.cpu(), _pack_cache(dense_key))
                assert torch.equal(value_cache.cpu(), _pack_cache(dense_value))
                expected_key = torch.zeros((20, 4, capacity, 128), dtype=torch.float16)
                expected_value = torch.zeros_like(expected_key)
                for row, length in enumerate(lengths.tolist()):
                    logical = torch.arange(length)
                    physical = table[row, logical // 128].long() * 128 + logical % 128
                    expected_key[row, :, :length] = dense_key[physical].reshape(length, 4, 128).permute(1, 0, 2)
                    expected_value[row, :, :length] = dense_value[physical].reshape(length, 4, 128).permute(1, 0, 2)
                assert torch.equal(keys.cpu(), expected_key.reshape_as(keys))
                assert torch.equal(values.cpu(), expected_value.reshape_as(values))
                assert not torch.count_nonzero(append_status.cpu()[:, 0])
                assert not torch.count_nonzero(gather_status.cpu()[:, 0])
                query = torch.randn((rows, 32, 128), generator=generator).half().to(device)
                actual_attention = torch.zeros_like(query)
                reference_attention = torch.zeros_like(query)
                reference_keys = torch_npu.npu_format_cast(_pack_cache(dense_key).to(device), 2)
                reference_values = torch_npu.npu_format_cast(_pack_cache(dense_value).to(device), 2)
                for cache_k, cache_v, out in ((key_cache, value_cache, actual_attention),
                                             (reference_keys, reference_values, reference_attention)):
                    torch_npu._npu_paged_attention(
                        query=query, key_cache=cache_k, value_cache=cache_v,
                        num_kv_heads=4, num_heads=32, scale_value=128**-0.5,
                        block_table=table.to(device), context_lens=lengths.to(device), out=out,
                    )
                assert torch.equal(actual_attention.cpu(), reference_attention.cpu())
        finally:
            graph.reset()


def test_b20_cache_rejects_invalid_slots_and_pages_without_writes(cache_images):
    import torch_npu
    from simpler_setup import A1BorrowedLaunch

    device = "npu:0"
    metadata = A1BatchedMetadata.allocate(device)
    key = torch.ones((20, 512), dtype=torch.float16, device=device)
    value = torch.ones_like(key)
    key_cache = torch_npu.npu_format_cast(torch.full((2, 32, 128, 16), 7, device=device).half(), 2)
    value_cache = torch_npu.npu_format_cast(torch.zeros_like(key_cache), 2)
    keys = torch.full((80, 128, 128), 5, dtype=torch.float16, device=device)
    values = torch.full_like(keys, 5)
    append_status = torch.full((20, 16), -7, dtype=torch.int32, device=device)
    gather_status = torch.full((80, 16), -7, dtype=torch.int32, device=device)
    metadata.slots[0, 0] = 256
    metadata.slots[1:3, 0] = 0
    metadata.lengths[0, 0] = 129
    metadata.lengths[1, 0] = 1
    metadata.block_table[1, 0] = -1
    before_key, before_value = key_cache.cpu(), value_cache.cpu()
    with ExitStack() as stack:
        append = stack.enter_context(A1BorrowedLaunch(cache_images["append"], device_id=0))
        gather = stack.enter_context(A1BorrowedLaunch(cache_images["gather"], device_id=0))
        append_call = append.prepare(key, value, key_cache, value_cache, metadata.slots,
                                     append_status, blocks=8)
        gather_call = gather.prepare(key_cache, value_cache, metadata.block_table, metadata.lengths,
                                     keys, values, gather_status, blocks=8)
        append_call()
        gather_call()
        torch.npu.synchronize()
        assert torch.equal(key_cache.cpu(), before_key)
        assert torch.equal(value_cache.cpu(), before_value)
        expected_append = torch.zeros(20, dtype=torch.int32)
        expected_append[:3] = torch.tensor([1, 3, 3])
        expected_gather = torch.zeros(80, dtype=torch.int32)
        expected_gather[:4] = 1
        expected_gather[4:8] = 2
        assert torch.equal(append_status.cpu()[:, 0], expected_append)
        assert torch.equal(gather_status.cpu()[:, 0], expected_gather)
        assert torch.all(keys.cpu()[:8] == 5)
        assert torch.all(values.cpu()[:8] == 5)
        assert not torch.count_nonzero(keys.cpu()[8:])
        assert not torch.count_nonzero(values.cpu()[8:])
