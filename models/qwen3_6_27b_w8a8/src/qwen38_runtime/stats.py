# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Counters are phase/backend facts; no per-step D2H or clock sampling."""

from collections import Counter
from .routing import Phase


class ExecutionStats:
    def __init__(self):
        self.routes = Counter()
        self.fallbacks = Counter()
        self.direct_submissions = 0
        self.full_captures = 0
        self.full_replays = 0

    def record(self, decision):
        self.routes[(decision.backend.value, decision.phase.value)] += 1
        if decision.fallback:
            self.fallbacks[decision.reason] += 1

    def snapshot(self):
        native_prefills = sum(self.routes[("native", p.value)] for p in (Phase.PREFILL, Phase.MIXED))
        native_decodes = sum(self.routes[("native", p.value)] for p in (Phase.DECODE, Phase.VERIFY))
        return dict(
            native_prefill_calls=native_prefills,
            native_decode_calls=native_decodes,
            graph_captures=self.full_captures,
            graph_replays=self.full_replays,
            routes={f"{b}:{p}": n for (b, p), n in self.routes.items()},
            fallback_reasons=dict(self.fallbacks),
            neural_fallback=bool(self.fallbacks),
            direct_submissions=self.direct_submissions,
            full_captures=self.full_captures,
            full_replays=self.full_replays,
        )
