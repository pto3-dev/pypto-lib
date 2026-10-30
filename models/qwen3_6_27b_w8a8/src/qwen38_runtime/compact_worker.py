# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Explicit input-preparation extension; inherited target/cache lifecycle intact."""

from .spec_cache_worker import CachedSpecWorker
from .compact_inputs import install, stats

install()


class CompactInputWorker(CachedSpecWorker):
    def benchmark_mpmd_stats(self):
        result = super().benchmark_mpmd_stats()
        result["compact_inputs"] = stats(self.model_runner)
        return result
