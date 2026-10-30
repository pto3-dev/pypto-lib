# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""PTO-only canonical metadata, never consumed by a native GDN kernel.

Only TP4/PP1/B4/MTP5, four real six-row requests and mamba_cache_mode=none
can use the raw views. Other shapes/phases/cache policies use the native builder.
"""

from dataclasses import dataclass
from functools import wraps


@dataclass
class PtoCanonicalGdnMetadata:
    spec_state_indices_tensor: object
    num_accepted_tokens: object
    spec_query_start_loc: object
    num_prefills: int = 0
    num_prefill_tokens: int = 0
    num_decodes: int = 0
    num_decode_tokens: int = 0
    num_spec_decodes: int = 4
    num_spec_decode_tokens: int = 24
    num_actual_tokens: int = 24
    pto_canonical_only: bool = True


def eligibility_reason(builder, metadata, accepted, drafts, common_prefix_len=0):
    config = builder.vllm_config
    settings = config.additional_config.get("qwen38_pto_mpmd", {})
    if (
        settings.get("backend") != "pypto"
        or settings.get("execution") != "direct"
        or settings.get("gdn_metadata") != "raw_canonical"
    ):
        return "disabled"
    parallel = config.parallel_config
    if (
        parallel.tensor_parallel_size,
        parallel.pipeline_parallel_size,
        parallel.decode_context_parallel_size,
        parallel.prefill_context_parallel_size,
    ) != (4, 1, 1, 1):
        return "parallelism"
    if (
        common_prefix_len != 0
        or not builder.use_spec_decode
        or builder.num_spec != 5
        or config.cache_config.mamba_cache_mode != "none"
        or metadata.num_reqs != 4
        or metadata.num_actual_tokens != 24
        or getattr(metadata, "max_query_len", None) != 6
        or accepted is None
        or drafts is None
    ):
        return "phase_or_cache"
    starts = metadata.query_start_loc_cpu
    if starts.device.type != "cpu" or drafts.device.type != "cpu":
        return "non_cpu_routing_metadata"
    if starts.tolist() != [0, 6, 12, 18, 24] or drafts.tolist() != [5, 5, 5, 5]:
        return "not_four_six_row_requests"
    table = metadata.block_table_tensor
    import torch

    if not (
        table.ndim == 2
        and table.shape[0] == 4
        and table.shape[1] >= 6
        and table.dtype == torch.int32
        and accepted.ndim == 1
        and accepted.numel() >= 4
        and accepted.dtype in (torch.int32, torch.int64)
        and table.device.type == "npu"
        and accepted.device == table.device
    ):
        return "tensor_contract"
    import sys

    if accepted.dtype == torch.int64 and (accepted.stride(0) != 1 or sys.byteorder != "little"):
        return "unsupported_int64_view"
    return "eligible"


def eligible(builder, metadata, accepted, drafts, common_prefix_len=0):
    return eligibility_reason(builder, metadata, accepted, drafts, common_prefix_len) == "eligible"


def accepted_int32_view(accepted):
    """Match native INT64->INT32 copy semantics without a device cast task."""
    import torch

    if accepted.dtype == torch.int32:
        return accepted[:4]
    words = accepted[:4].view(torch.int32)
    return words.as_strided((4,), (2,), words.storage_offset())


def build_raw(builder, metadata, accepted):
    builder._pto_raw_builds = getattr(builder, "_pto_raw_builds", 0) + 1
    if not hasattr(builder, "_pto_raw_source_dtypes"):
        builder._pto_raw_source_dtypes = {}
    dtype = str(accepted.dtype)
    builder._pto_raw_source_dtypes[dtype] = builder._pto_raw_source_dtypes.get(dtype, 0) + 1
    return PtoCanonicalGdnMetadata(
        metadata.block_table_tensor[:, :6], accepted_int32_view(accepted), metadata.query_start_loc[:5]
    )


def install():
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder

    original = GDNAttentionMetadataBuilder.build
    if getattr(original, "_pto_raw_canonical", False):
        return

    @wraps(original)
    def build(
        self,
        common_prefix_len,
        common_attn_metadata,
        num_accepted_tokens=None,
        num_decode_draft_tokens_cpu=None,
        fast_build=False,
    ):
        reason = eligibility_reason(
            self, common_attn_metadata, num_accepted_tokens, num_decode_draft_tokens_cpu, common_prefix_len
        )
        if reason == "eligible":
            result = build_raw(self, common_attn_metadata, num_accepted_tokens)
            limit = self.vllm_config.additional_config["qwen38_pto_mpmd"].get(
                "gdn_metadata_validate_builds", 0
            )
            if getattr(self, "_pto_validated_builds", 0) < limit:
                import torch

                native = original(
                    self,
                    common_prefix_len,
                    common_attn_metadata,
                    num_accepted_tokens,
                    num_decode_draft_tokens_cpu,
                    fast_build,
                )
                fields = (
                    "num_prefills",
                    "num_prefill_tokens",
                    "num_decodes",
                    "num_decode_tokens",
                    "num_spec_decodes",
                    "num_spec_decode_tokens",
                    "num_actual_tokens",
                )
                if (
                    any(getattr(native, name) != getattr(result, name) for name in fields)
                    or not torch.equal(
                        native.spec_state_indices_tensor[:4, :6], result.spec_state_indices_tensor
                    )
                    or not torch.equal(native.num_accepted_tokens[:4], result.num_accepted_tokens)
                ):
                    raise RuntimeError("PTO raw GDN metadata differs from native builder")
                self._pto_validated_builds = getattr(self, "_pto_validated_builds", 0) + 1
            return result
        self._pto_native_builds = getattr(self, "_pto_native_builds", 0) + 1
        if not hasattr(self, "_pto_rejection_reasons"):
            self._pto_rejection_reasons = {}
        self._pto_rejection_reasons[reason] = self._pto_rejection_reasons.get(reason, 0) + 1
        return original(
            self,
            common_prefix_len,
            common_attn_metadata,
            num_accepted_tokens,
            num_decode_draft_tokens_cpu,
            fast_build,
        )

    build._pto_raw_canonical = True
    GDNAttentionMetadataBuilder.build = build
