# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""No tensor audits: combine deterministic native HCCL with input preparation."""

from .compact_worker import CompactInputWorker
from .stable_hccl_worker import install, read_config

install()


class StableCompactWorker(CompactInputWorker):
    def benchmark_mpmd_stats(self):
        result = super().benchmark_mpmd_stats()
        runner = self.model_runner
        audit = dict(runner._stable_hccl_audit)
        audit["after_serving"] = read_config(runner._stable_hccl_handle)
        if audit["after_serving"] != 1:
            raise RuntimeError("HCCL deterministic policy changed")
        result["deterministic_native_tp"] = audit
        return result
