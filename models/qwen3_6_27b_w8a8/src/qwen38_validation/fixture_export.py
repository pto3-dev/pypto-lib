# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Save layer63 after a completed target; never insert work into timed decode.

The B4 case may be a clone of one actual retired prefix. Its provenance is
retained and is not upgraded into four independent production requests.
"""

import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def _hash(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def _pool_view(host, pool_device, view):
    assert pool_device.dtype == view.dtype == host.dtype
    delta = view.data_ptr() - pool_device.data_ptr()
    assert delta >= 0 and delta % view.element_size() == 0
    offset = delta // view.element_size()
    last = offset + sum((s - 1) * stride for s, stride in zip(view.shape, view.stride()))
    assert last < host.numel()
    return torch.as_strided(
        host, tuple(view.shape), tuple(view.stride()), storage_offset=host.storage_offset() + offset
    )


def export(executor, post_outputs, folder, case, bundle):
    """All inputs are retired and synchronized. No allocator/runtime owned here."""
    from qwen36.bank_layout import layout
    from qwen36.model_layout import WeightBankSpec

    layer = 63
    assert executor.max_seq == 43008 and executor.table_cols == 336
    views = {name: executor.cache_views[f"l{layer}_{name}"] for name in ["keys", "values"]}
    keys = views["keys"]
    pool_name = next(
        n
        for n, t in executor.exported.items()
        if n.startswith("cache_pool") and t.untyped_storage().data_ptr() == keys.untyped_storage().data_ptr()
    )
    pool_device = executor.exported[pool_name]
    pool = post_outputs[pool_name].contiguous()
    assert pool.numel() * pool.element_size() == pool_device.numel() * pool_device.element_size()
    kv = {n: _pool_view(pool, pool_device, v).reshape(294912, 256) for n, v in views.items()}
    offsets = {n: (v.data_ptr() - pool_device.data_ptr()) // v.element_size() for n, v in views.items()}
    small, count, _ = layout(WeightBankSpec())
    bank = executor.exported["weights15"].cpu()
    assert count == bank.numel()
    norm = {}
    for name in ["q", "k"]:
        offset, shape, _ = small[3, f"self_attn.{name}_norm.weight"]
        assert tuple(shape) == (256,)
        norm[name] = bank[offset : offset + 256].reshape(1, 256).clone()
    metadata = executor.native_metadata.cpu()[layer].clone()
    controls = executor.control_i64.cpu()
    table = metadata[72:].reshape(24, executor.table_cols).clone()
    lengths = metadata[48:72].clone()
    positions = controls[24:48].clone()
    slots = controls[48:72].clone()
    assert bool((lengths > 0).all()) and bool((lengths <= 43008).all())
    assert torch.equal(positions + 1, lengths.long())
    assert bool((table >= 0).all()) and int(table.max()) < 294912 // 128
    tensors = dict(
        pool=pool,
        table=table,
        lengths=lengths,
        positions=positions,
        slots=slots,
        q_norm=norm["q"],
        k_norm=norm["k"],
        native_metadata=metadata,
        control_i64=controls,
        control_i32=executor.control_i32.cpu(),
    )
    mapping = {
        "q_projected": "scratch_packed_q",
        "k_projected": "scratch_full_k",
        "v_projected": "scratch_full_v",
        "expected_q": "scratch_full_qr",
        "expected_gate": "scratch_full_gate",
        "expected_p": "scratch_attention_probability",
        "expected_scores": "float_attention_scores",
        "expected_partials": "float_attention_partials",
        "expected_gated": "scratch_gated",
        "cos": "cos",
        "sin": "sin",
        "columns": "softmax_columns",
    }
    for name, source in mapping.items():
        tensors[name] = executor.exported[source].cpu().contiguous()
    tensors["k_projected"] = tensors["k_projected"].reshape(24, 1, 256)
    tensors["expected_gated"] = tensors["expected_gated"].reshape(144, 256)
    assert torch.equal(
        tensors["expected_gate"].reshape(24, 6, 256).view(torch.uint8),
        tensors["q_projected"].reshape(24, 6, 512)[:, :, 256:].contiguous().view(torch.uint8),
    )
    assert torch.equal(kv["values"][slots].view(torch.uint8), tensors["v_projected"].view(torch.uint8)), (
        "Last-layer V inputs/cache do not match"
    )
    score_checks = []
    q = tensors["expected_q"].reshape(144, 256).float()
    p = tensors["expected_p"]
    scores = tensors["expected_scores"]
    for row in [0, 35, 36, 72, 143]:
        length = int(lengths[row // 6])
        for column in [0, 128, length - 1]:
            physical = int(table[(row // 36) * 6, column // 128]) * 128 + column % 128
            expected = float((q[row] * kv["keys"][physical].float()).sum())
            actual = float(scores[row, column])
            assert abs(actual - expected) < 1e-4 + abs(expected) * 1e-4, (
                "Stale last-layer scores",
                row,
                column,
                expected,
                actual,
            )
            score_checks.append(dict(row=row, column=column, expected=expected, observed=actual))
        wanted = torch.softmax(scores[row, :length] * 0.0625, dim=-1).bfloat16().float()
        assert torch.allclose(p[row, :length].float(), wanted, rtol=0.01, atol=1e-6), ("P oracle", row)
        assert not bool(p[row, length:].any()), ("P mask", row)
    pv_checks = []
    for request, chunk, channel in [(0, 0, 0), (1, 13, 127), (2, 21, 255), (3, 27, 32)]:
        limit = max(0, min(1536, int(lengths[request * 6 + 5]) - chunk * 1536))
        columns = torch.arange(chunk * 1536, chunk * 1536 + limit)
        physical = table[request * 6, columns // 128].long() * 128 + columns % 128
        wanted = float((p[request * 36, columns].float() * kv["values"][physical, channel].float()).sum())
        actual = float(tensors["expected_partials"][chunk * 144 + request * 36, channel])
        assert abs(actual - wanted) < 1e-6 + abs(wanted) * 1e-4, ("PV oracle", request, chunk, wanted, actual)
        pv_checks.append(
            dict(request=request, chunk=chunk, channel=channel, expected=wanted, observed=actual)
        )
    acc = torch.zeros((144, 256), dtype=torch.float32)
    for chunk in tensors["expected_partials"].reshape(28, 144, 256):
        acc += chunk
    sigmoid = (
        torch.reciprocal(torch.exp(-tensors["expected_gate"].reshape(144, 256).float()) + 1)
        .bfloat16()
        .float()
    )
    wanted = (sigmoid * acc.bfloat16().float()).bfloat16().float()
    assert torch.allclose(tensors["expected_gated"].float(), wanted, rtol=0.02, atol=1e-5), "Gated oracle"
    folder = Path(folder)
    folder.mkdir(exist_ok=False)
    tensor_file = folder / "layer63.safetensors"
    save_file(
        tensors,
        str(tensor_file),
        metadata={
            "scope": "Real checkpoint activations from a synthetic four-active retired-prefix clone; not production ITL."
        },
    )
    allocation = []
    for name, t in executor.exported.items():
        allocation.append(
            dict(
                name=name,
                address=t.data_ptr(),
                storage_base=t.untyped_storage().data_ptr(),
                storage_bytes=t.untyped_storage().nbytes(),
                shape=list(t.shape),
                stride=list(t.stride()),
                dtype=str(t.dtype),
            )
        )
    manifest = dict(
        schema=1,
        layer=layer,
        rank=executor.rank,
        source_build=str(bundle.root),
        source_hashes=bundle.hashes,
        checkpoint=executor.settings.get("checkpoint_provenance"),
        case=case,
        scope="Exact numerical layer63 snapshot of real-weight B4 retired-prefix clone; native full-pool offsets preserved. Neither real independent B4 requests nor a 10K terminal state.",
        pool_name=pool_name,
        pool_bytes=pool.numel() * pool.element_size(),
        cache_offsets_elements=offsets,
        cache_shapes={n: list(v.shape) for n, v in views.items()},
        cache_strides={n: list(v.stride()) for n, v in views.items()},
        source_allocations=allocation,
        source_tensor_map=mapping,
        small_bank_norm_offsets={n: small[3, f"self_attn.{n}_norm.weight"][0] for n in ["q", "k"]},
        tensors={
            n: dict(shape=list(t.shape), dtype=str(t.dtype), sha256=_hash(t)) for n, t in tensors.items()
        },
        file=str(tensor_file),
        file_sha256=hashlib.sha256(tensor_file.read_bytes()).hexdigest(),
        score_checks=score_checks,
        pv_checks=pv_checks,
        math_spot_oracles_passed=True,
        score_comparison="Only columns before each query length are semantically defined; masked QK tails may be undefined.",
        replay_validation="pending",
        no_weights_or_hidden_fallback=True,
    )
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return dict(
        folder=str(folder),
        file_sha256=manifest["file_sha256"],
        bytes=tensor_file.stat().st_size,
        pool_name=pool_name,
        math_spot_oracles_passed=True,
        replay_validation="pending",
    )
