# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Phase classification uses host metadata, never tensor values or token row count."""

from dataclasses import dataclass
from enum import Enum
from numbers import Integral
from .config import Backend


class Phase(str, Enum):
    PROFILE = "profile"
    PREFILL = "prefill"
    MIXED = "mixed_prefill_decode"
    DECODE = "decode"
    VERIFY = "verifier"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Observation:
    phase: Phase
    rows: int
    has_embeddings: bool = False


def _count(meta, name):
    value = getattr(meta, name, 0)
    if value is None:
        return 0
    if not isinstance(value, Integral):
        raise TypeError(f"{name} must be a CPU scalar; do not synchronize a device tensor to route")
    return int(value)


def observe(context, rows, has_embeddings=False):
    metadata = getattr(context, "attn_metadata", None)
    if not metadata:
        return Observation(Phase.PROFILE, rows, has_embeddings)
    if not isinstance(metadata, dict):
        raise TypeError("Pinned vLLM metadata must be a layer-name mapping")
    prefills = specs = decodes = 0
    states = set()
    for meta in metadata.values():
        prefills = max(prefills, _count(meta, "num_prefills"))
        specs = max(specs, _count(meta, "num_spec_decodes"))
        decodes = max(decodes, _count(meta, "num_decodes"))
        state = getattr(meta, "attn_state", None)
        if state is not None:
            states.add(getattr(state, "name", str(state).rsplit(".", 1)[-1]))
    if prefills:
        phase = Phase.MIXED if specs or decodes else Phase.PREFILL
    elif specs or "SpecDecoding" in states:
        phase = Phase.VERIFY
    elif "DecodeOnly" in states or decodes:
        phase = Phase.DECODE
    elif states & {"PrefillNoCache", "PrefillCacheHit", "ChunkedPrefill"}:
        phase = Phase.PREFILL
    else:
        phase = Phase.UNKNOWN
    return Observation(phase, rows, has_embeddings)


@dataclass(frozen=True)
class Decision:
    backend: Backend
    phase: Phase
    reason: str
    fallback: bool = False


class UnsupportedTarget(RuntimeError):
    pass


def decide(settings, observation, target_rows=24):
    phase = observation.phase
    if settings.backend is Backend.NATIVE:
        return Decision(Backend.NATIVE, phase, "backend_disabled")
    if phase in (Phase.PROFILE, Phase.PREFILL, Phase.MIXED):
        return Decision(Backend.NATIVE, phase, phase.value)
    if phase is Phase.VERIFY and observation.rows == target_rows and not observation.has_embeddings:
        return Decision(Backend.PYPTO, phase, "supported_verifier")
    reason = f"unsupported_{phase.value}_rows{observation.rows}"
    if settings.unsupported_decode == "error":
        raise UnsupportedTarget(reason)
    return Decision(Backend.NATIVE, phase, reason, fallback=True)


def guard_metadata_consumer(context, decision):
    if decision.backend is Backend.NATIVE and any(
        getattr(meta, "pto_canonical_only", False) for meta in (context.attn_metadata or {}).values()
    ):
        raise UnsupportedTarget("PTO-only canonical metadata cannot be consumed by native fallback")
