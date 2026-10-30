# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Reuse immutable uniform-MTP index tensors; always gather fresh draft IDs.

No graph/model source transformation and no CPU readback in production. All
five cached tensors depend only on CPU draft/scheduled row counts. They are
read-only to callers; a new SpecDecodeMetadata and CPU length list is returned.
"""

from functools import wraps

FIELDS = (
    "cu_num_draft_tokens",
    "cu_num_sampled_tokens",
    "target_logits_indices",
    "bonus_logits_indices",
    "logits_indices",
)


def eligible(runner, drafts, scheduled, pads):
    settings = runner.vllm_config.additional_config.get("qwen38_pto_mpmd", {})
    if settings.get("spec_metadata_cache") != "uniform_v1":
        return False
    spec = runner.vllm_config.speculative_config
    if spec is None or spec.method != "mtp" or spec.num_speculative_tokens != 5:
        return False
    if runner.pcp_size != 1 or getattr(runner, "dcp_size", 1) != 1 or pads is not None:
        return False
    n = len(drafts)
    if not 1 <= n <= 4 or len(scheduled) != n:
        return False
    return all(int(v) == 5 for v in drafts) and all(int(v) == 6 * (i + 1) for i, v in enumerate(scheduled))


def cached(self, num_draft_tokens, cu_num_scheduled_tokens, num_pcp_pads, *, original):
    if not eligible(self, num_draft_tokens, cu_num_scheduled_tokens, num_pcp_pads):
        self._pto_spec_index_native = getattr(self, "_pto_spec_index_native", 0) + 1
        return original(self, num_draft_tokens, cu_num_scheduled_tokens, num_pcp_pads)
    import torch
    from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

    if torch.npu.is_current_stream_capturing():
        # No lazy cache allocation or new validation barrier inside capture.
        return original(self, num_draft_tokens, cu_num_scheduled_tokens, num_pcp_pads)
    bank = getattr(self, "_pto_spec_index_bank", None)
    if bank is None:
        bank = {}
        self._pto_spec_index_bank = bank
    n = len(num_draft_tokens)
    # At most four entries; each runner owns one device and one input stream.
    if n not in bank:
        result = original(self, num_draft_tokens, cu_num_scheduled_tokens, num_pcp_pads)
        bank[n] = {name: getattr(result, name) for name in FIELDS}
        bank[n]["draft_positions"] = result.logits_indices[result.target_logits_indices + 1]
        self._pto_spec_index_misses = getattr(self, "_pto_spec_index_misses", 0) + 1
        return result
    tensors = bank[n]
    draft_ids = self.input_ids.gpu[tensors["draft_positions"]]
    result = SpecDecodeMetadata(
        draft_token_ids=draft_ids, num_draft_tokens=[5] * n, **{name: tensors[name] for name in FIELDS}
    )
    self._pto_spec_index_hits = getattr(self, "_pto_spec_index_hits", 0) + 1
    settings = self.vllm_config.additional_config["qwen38_pto_mpmd"]
    limit = int(settings.get("spec_metadata_validate_calls", 0))
    checked_batches = getattr(self, "_pto_spec_index_validated_by_batch", {})
    if checked_batches.get(n, 0) < limit:
        expected = original(self, num_draft_tokens, cu_num_scheduled_tokens, num_pcp_pads)
        for name in FIELDS + ("draft_token_ids",):
            a, b = getattr(result, name), getattr(expected, name)
            if a.dtype != b.dtype or tuple(a.shape) != tuple(b.shape) or not torch.equal(a, b):
                raise RuntimeError("Cached spec index field differs: " + name)
        if result.num_draft_tokens != expected.num_draft_tokens:
            raise RuntimeError("Cached CPU draft lengths differ")
        self._pto_spec_index_validated = getattr(self, "_pto_spec_index_validated", 0) + 1
        checked_batches[n] = checked_batches.get(n, 0) + 1
        self._pto_spec_index_validated_by_batch = checked_batches
    return result


def stats(runner):
    bank = getattr(runner, "_pto_spec_index_bank", {})
    return dict(
        enabled=runner.vllm_config.additional_config.get("qwen38_pto_mpmd", {}).get("spec_metadata_cache")
        == "uniform_v1",
        hits=getattr(runner, "_pto_spec_index_hits", 0),
        misses=getattr(runner, "_pto_spec_index_misses", 0),
        native_calls=getattr(runner, "_pto_spec_index_native", 0),
        validated_calls=getattr(runner, "_pto_spec_index_validated", 0),
        validated_by_batch=getattr(runner, "_pto_spec_index_validated_by_batch", {}),
        cached_batches=sorted(bank),
        h2d_copies_avoided=5 * getattr(runner, "_pto_spec_index_hits", 0),
        resident_device_bytes=sum(t.numel() * t.element_size() for e in bank.values() for t in e.values()),
    )


def install():
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    old = NPUModelRunner._calc_spec_decode_metadata
    if getattr(old, "_pto_static_spec_indices", False):
        return

    @wraps(old)
    def calculate(self, num_draft_tokens, cu_num_scheduled_tokens, num_pcp_pads):
        return cached(self, num_draft_tokens, cu_num_scheduled_tokens, num_pcp_pads, original=old)

    calculate._pto_static_spec_indices = True
    NPUModelRunner._calc_spec_decode_metadata = calculate
