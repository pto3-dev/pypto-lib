# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Opt-in graph/eager comparison from identical native drafter state.

This snapshots device buffers and synchronizes for comparisons. It is a
correctness diagnostic, explicitly disabled for performance measurements.
"""

import copy
import json
from pathlib import Path


def validate_same_state(owner, call, inputs, common, kwargs, batch):
    import torch
    from vllm.config import get_layers_from_vllm_config
    from vllm.distributed import get_tensor_model_parallel_rank
    from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
    from vllm_ascend.compilation.acl_graph import get_draft_graph_params

    tensors, seen = [], set()

    def add(value, label):
        if isinstance(value, torch.Tensor):
            key = (
                str(value.device),
                str(value.dtype),
                value.data_ptr(),
                tuple(value.shape),
                tuple(value.stride()),
            )
            if key not in seen:
                seen.add(key)
                tensors.append((value, value.clone(), label))
        elif isinstance(value, (tuple, list)):
            for i, item in enumerate(value):
                add(item, f"{label}/{i}")

    for key, value in vars(owner).items():
        add(value, key)
    for key, value in owner.model.named_buffers():
        add(value, "model_buffer/" + key)
    layers = get_layers_from_vllm_config(owner.vllm_config, AttentionLayerBase)
    for key in owner.attn_layer_names:
        add(layers[key].kv_cache, "cache/" + key)
    cache_tensors = [(v, label) for v, _, label in tensors if label.startswith("cache/")]
    if not cache_tensors:
        raise RuntimeError("No native MTP KV cache captured for validation")
    frozen_inputs = tuple(v.clone() if isinstance(v, torch.Tensor) else v for v in inputs)

    def cloned_common():
        meta = copy.copy(common)
        for key, value in vars(common).items():
            if isinstance(value, torch.Tensor):
                setattr(meta, key, value.clone())
            elif isinstance(value, (tuple, list, dict)):
                setattr(meta, key, copy.deepcopy(value))
        return meta

    def run(graph):
        supplied = tuple(v.clone() if isinstance(v, torch.Tensor) else v for v in frozen_inputs)
        options = {key: v.clone() if isinstance(v, torch.Tensor) else v for key, v in kwargs.items()}
        return call(values=supplied, common=cloned_common(), options=options, graph=graph)

    # All replay input/state storage is restored before the graph run. The
    # second result/state is left in place, so the real serving call advances once.
    eager = run(False).clone()
    expected_cache = [(value, value.clone(), label) for value, label in cache_tensors]
    for value, original, _ in tensors:
        value.copy_(original)
    graphed = run(True)
    equal = {
        label: bool(
            torch.equal(value.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))
        )
        for value, expected, label in expected_cache
    }
    params = get_draft_graph_params()
    result = dict(
        batch=batch,
        rows=batch * 6,
        comparison="same-state native eager versus native five-step graph",
        tokens_equal=bool(torch.equal(eager, graphed)),
        cache_equal=equal,
        fia_handles={str(k): len(v) for k, v in params.handles.items()},
        eager_token_ids=eager.cpu().tolist(),
        graph_token_ids=graphed.cpu().tolist(),
        performance_measurement=False,
    )
    owner._pto_graph_validation = result
    settings = owner.vllm_config.additional_config["qwen38_pto_mpmd"]
    out = (
        Path(settings["session_dir"])
        / f"rank{get_tensor_model_parallel_rank()}_subbatch_b{batch}_validation.json"
    )
    out.write_text(json.dumps(result, indent=2) + "\n")
    if not result["tokens_equal"] or not all(equal.values()):
        raise RuntimeError(f"Same-state MTP eager/graph differs for B{batch}: {result}")
    owner._pto_graph_validations += 1
    return graphed
