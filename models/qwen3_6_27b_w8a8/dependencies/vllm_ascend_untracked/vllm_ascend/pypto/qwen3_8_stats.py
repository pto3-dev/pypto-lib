# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Worker-local PTO dispatch and native cache alias diagnostics."""


def collect_pto_stats(model):
    """Read worker-local dispatch records and verify native cache storage aliases."""
    from collections import Counter

    causal = getattr(model, "language_model", model)
    mpmd = getattr(causal, "_qwen38_pto_mpmd_executor", None)
    if mpmd is not None:
        return mpmd.stats()
    manager = getattr(causal, "_qwen38_pto_mlp_manager", None)
    if manager is not None:
        return manager.stats()
    executor = getattr(causal, "_qwen38_pto_executor", None)
    if executor is None:
        return {"pto_executor": False}
    records = executor.runner.records
    aliases = {}
    for layer in causal.model.layers:
        if hasattr(layer, "linear_attn"):
            key = layer.linear_attn.prefix
            cache = layer.linear_attn.kv_cache[0]
            expected = [value.data_ptr() for value in cache]
        elif hasattr(layer, "self_attn") and hasattr(executor, "cache_bindings"):
            key = layer.self_attn.attn.layer_name
            cache = layer.self_attn.attn.kv_cache[0]
            expected = [cache[0].data_ptr(), cache[1].data_ptr()]
        else:
            continue
        bindings = getattr(executor, "cache_bindings", {})
        if key not in bindings and not hasattr(executor, "compute"):
            continue
        actual = bindings.get(key)
        aliases[key] = {
            "native_pointers": expected,
            "pto_pointers": actual,
            "same_storage": actual == expected,
        }
    return {
        "pto_executor": True,
        "whole_decode_graph": getattr(executor, "graph_runner", None) is not None,
        "graph_replays": getattr(executor, "graph_replays", 0),
        "graph_preparations": getattr(executor, "graph_preparations", 0),
        "python_decode_enqueues": getattr(executor, "capture_enqueues", 0),
        "model_torch_dynamo_disabled": getattr(causal.model, "do_not_compile", False),
        "fused_mlp": getattr(executor, "fused_mlp", False),
        "source_digest": executor.runner.source_digest,
        "build_generation": str(executor.runner.build_root),
        "operator_counts": dict(Counter(record["operator"] for record in records)),
        "jit_misses": sum(record["jit_miss"] for record in records),
        "cache_aliases": aliases,
    }
