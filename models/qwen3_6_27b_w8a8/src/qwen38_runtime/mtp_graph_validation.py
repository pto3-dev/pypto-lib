# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Diagnostic-only same-state graph replay. Never used for performance timing."""

import copy
import json
from pathlib import Path


def validate_same_state(owner, call, inputs, common, kwargs):
    import torch
    from vllm.config import get_layers_from_vllm_config
    from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
    from vllm.distributed import get_tensor_model_parallel_rank
    from vllm_ascend.compilation.acl_graph import get_draft_graph_params

    tensors = []
    seen = set()

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
            for i, v in enumerate(value):
                add(v, f"{label}/{i}")

    for key, value in vars(owner).items():
        add(value, key)
    for key, value in owner.model.named_buffers():
        add(value, "model_buffer/" + key)
    layers = get_layers_from_vllm_config(owner.vllm_config, AttentionLayerBase)
    for key in owner.attn_layer_names:
        add(layers[key].kv_cache, "cache/" + key)
    if not any(label.startswith("cache/") for _, _, label in tensors):
        raise RuntimeError("No MTP cache in replay validation")
    frozen_inputs = tuple(v.clone() if isinstance(v, torch.Tensor) else v for v in inputs)
    frozen_common = copy.copy(common)
    for key, value in vars(common).items():
        if isinstance(value, torch.Tensor):
            setattr(frozen_common, key, value.clone())
        elif isinstance(value, (tuple, list, dict)):
            setattr(frozen_common, key, copy.deepcopy(value))

    def run():
        supplied = tuple(v.clone() if isinstance(v, torch.Tensor) else v for v in frozen_inputs)
        meta = copy.copy(frozen_common)
        for key, value in vars(frozen_common).items():
            if isinstance(value, torch.Tensor):
                setattr(meta, key, value.clone())
            elif isinstance(value, (tuple, list, dict)):
                setattr(meta, key, copy.deepcopy(value))
        options = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
        return call(values=supplied, common=meta, options=options)

    first = run().clone()
    cache = [(v, v.clone(), label) for v, _, label in tensors if label.startswith("cache/")]
    for value, saved, _ in tensors:
        value.copy_(saved)
    second = run()
    equal = {
        label: bool(torch.equal(value.contiguous().view(torch.uint8), saved.contiguous().view(torch.uint8)))
        for value, saved, label in cache
    }
    params = get_draft_graph_params()
    result = dict(
        tokens_equal=bool(torch.equal(first, second)),
        buffers_and_cache_equal=equal,
        deterministic_algorithms=bool(torch.are_deterministic_algorithms_enabled()),
        comparison="same-state graph versus graph; no eager reference",
        fia_handles={str(k): len(v) for k, v in params.handles.items()},
        first_token_ids=first.cpu().tolist(),
        second_token_ids=second.cpu().tolist(),
    )
    owner._pto_graph_validation = result
    settings = owner.vllm_config.additional_config["qwen38_pto_mpmd"]
    path = (
        Path(settings["session_dir"])
        / f"rank{get_tensor_model_parallel_rank()}_graph_repeat_{owner._pto_graph_validations}.json"
    )
    path.write_text(json.dumps(result, indent=2) + "\n")
    if not result["tokens_equal"] or not all(equal.values()):
        raise RuntimeError("Same-state MTP graph replay differs")
    owner._pto_graph_validations += 1
    return second
