# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Opt-in diagnostic extension of the unchanged native drafter worker."""

from .native_nz_worker import NativeNzWorker
from .host_stage_trace import install, flush

install()


class HostTraceWorker(NativeNzWorker):
    def benchmark_mpmd_stats(self):
        result = NativeNzWorker.benchmark_mpmd_stats(self)
        result["host_stage_trace"] = flush()
        return result
