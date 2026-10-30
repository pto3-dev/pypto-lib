# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Audit Qwen3.6 Inner mixed W8A8 without importing torch or changing tensors.

The 263-weight contract covers every MLP, 16 full-attention layers, and MTP.
A converted public checkpoint is a development surrogate, not an exact Inner copy.
"""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import struct

ATTENTION = {"q_proj": (12288, 5120), "k_proj": (1024, 5120), "v_proj": (1024, 5120), "o_proj": (5120, 6144)}
MLP = {"gate_proj": (17408, 5120), "up_proj": (17408, 5120), "down_proj": (5120, 17408)}


def canonical(name):
    # Both HF prefixes are accepted, but collision detection stays strict.
    for prefix in ("model.language_model.", "language_model.model.", "model."):
        if name.startswith(prefix) and name[len(prefix) :].startswith("layers."):
            return name[len(prefix) :]
    return name


def inner_quant_shape(name):
    name = canonical(name)
    match = re.fullmatch(r"(layers\.(\d+)|mtp\.layers\.(\d+))\.(mlp|self_attn)\.([a-z_]+)\.weight", name)
    if not match:
        return None
    _, layer, mtp, module, projection = match.groups()
    if mtp is not None and mtp != "0":
        return None
    if layer is not None and not 0 <= int(layer) < 64:
        return None
    if module == "mlp":
        return MLP.get(projection)
    if mtp is not None or int(layer) % 4 == 3:
        return ATTENTION.get(projection)
    return None


def expected_inner():
    names = {}
    for layer in range(64):
        for projection, shape in MLP.items():
            names[f"layers.{layer}.mlp.{projection}.weight"] = shape
        if layer % 4 == 3:
            for projection, shape in ATTENTION.items():
                names[f"layers.{layer}.self_attn.{projection}.weight"] = shape
    for module, projections in [("mlp", MLP), ("self_attn", ATTENTION)]:
        for projection, shape in projections.items():
            names[f"mtp.layers.0.{module}.{projection}.weight"] = shape
    return names


def read_header(path):
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"Truncated safetensors header: {path}")
        size = struct.unpack("<Q", prefix)[0]
        if size > 64 * 1024 * 1024:
            raise ValueError(f"Implausible safetensors header: {path}")
        header = json.loads(stream.read(size))
    return header, 8 + size


def audit(root, profile="inner", check_values=True):
    root = Path(root)
    index = json.loads((root / "quant_model_weights.safetensors.index.json").read_text())
    description = json.loads((root / "quant_model_description.json").read_text())
    config = json.loads((root / "config.json").read_text())
    errors, records, missing = [], {}, []
    text = config.get("text_config", config)
    for key, expected in [
        ("hidden_size", 5120),
        ("num_hidden_layers", 64),
        ("intermediate_size", 17408),
        ("num_attention_heads", 24),
        ("num_key_value_heads", 4),
        ("head_dim", 256),
        ("vocab_size", 248320),
        ("linear_conv_kernel_dim", 4),
        ("mtp_num_hidden_layers", 1),
    ]:
        if text.get(key) != expected:
            errors.append(f"{key}: expected {expected}, got {text.get(key)}")
    if text.get("layer_types") != ["full_attention" if i % 4 == 3 else "linear_attention" for i in range(64)]:
        errors.append("layer_types differs from the supported 3 GDN : 1 attention schedule")
    if description.get("model_quant_type") != "W8A8_DYNAMIC":
        errors.append("Expected model_quant_type=W8A8_DYNAMIC")
    total = 0
    for shard in sorted(set(index["weight_map"].values())):
        path = root / shard
        if not path.exists():
            missing.append(shard)
            continue
        header, origin = read_header(path)
        end = 0
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            if index["weight_map"].get(name) != shard:
                errors.append(f"Unindexed or incorrectly indexed tensor: {name}")
            key = canonical(name)
            if key in records:
                errors.append(f"Duplicate tensor after prefix normalization: {key}")
            start, stop = entry["data_offsets"]
            if start < 0 or stop < start:
                errors.append(f"Invalid byte range: {name}")
            nbytes = stop - start
            dtype_bytes = {"I8": 1, "BF16": 2, "F32": 4, "F16": 2, "I32": 4, "I64": 8}.get(entry["dtype"])
            if dtype_bytes is None or math.prod(entry["shape"]) * dtype_bytes != nbytes:
                errors.append(f"Invalid dtype or byte size: {name}")
            total += nbytes
            end = max(end, stop)
            records[key] = dict(entry, original_name=name, shard=shard)
            if check_values and (name.endswith("weight_scale") or name.endswith("weight_offset")):
                if entry["dtype"] != "F32":
                    errors.append(f"Checkpoint scale/offset must be F32: {name}")
                    continue
                with path.open("rb") as stream:
                    stream.seek(origin + start)
                    values = stream.read(nbytes)
                for (v,) in struct.iter_unpack("<f", values):
                    if not math.isfinite(v) or (v <= 0 if name.endswith("weight_scale") else v != 0):
                        errors.append(f"Invalid scale or unsupported nonzero offset: {name}")
                        break
        if origin + end != path.stat().st_size:
            errors.append(f"Shard is truncated or contains unindexed trailing bytes: {shard}")
    if not missing:
        if len(records) != len(index["weight_map"]):
            errors.append("Index and actual tensor counts differ")
        if total != int(index["metadata"]["total_size"]):
            errors.append("metadata.total_size does not match tensor bytes")
        actual = {k: v for k, v in records.items() if v["dtype"] == "I8"}
        expected = expected_inner()
        for key, shape in expected.items():
            entry = actual.get(key)
            if entry is None or tuple(entry["shape"]) != shape:
                errors.append(f"Missing or wrong-shaped Inner INT8 tensor: {key}")
        extra = sorted(set(actual) - set(expected))
        if profile == "inner" and extra:
            errors.append(f"Unexpected quantized weights outside Inner scope: {extra}")
        if len(actual) != (263 if profile == "inner" else 440):
            errors.append(f"Unexpected {profile} INT8 weight count: {len(actual)}")
        for key, value in actual.items():
            stem = key.removesuffix(".weight")
            for suffix in ["weight_scale", "weight_offset"]:
                parameter = records.get(stem + "." + suffix)
                if (
                    parameter is None
                    or parameter["dtype"] != "F32"
                    or parameter["shape"] != [value["shape"][0], 1]
                ):
                    errors.append(f"Bad per-output-channel parameter: {stem}.{suffix}")
            if description.get(value["original_name"]) != "W8A8_DYNAMIC":
                errors.append(f"Description disagrees with INT8 payload: {key}")
    return {
        "checkpoint": str(root.resolve()),
        "profile": profile,
        "status": "incomplete" if missing else "failed" if errors else "passed",
        "missing_shards": missing,
        "errors": errors,
        "tensor_count": len(records),
        "dtype_counts": dict(Counter(x["dtype"] for x in records.values())),
        "tensor_payload_bytes": total,
        "value_checks": check_values and not missing,
        "expected_inner_int8_weights": len(expected_inner()),
        "provenance": "public checkpoint; not the unavailable private Inner checkpoint"
        if profile == "public"
        else "must be established by the supplied source",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--profile", choices=["inner", "public"], default="inner")
    parser.add_argument("--headers-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.checkpoint, args.profile, not args.headers_only)
    result = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(result)
    print(result, end="")
    raise SystemExit(0 if report["status"] == "passed" else 2)
