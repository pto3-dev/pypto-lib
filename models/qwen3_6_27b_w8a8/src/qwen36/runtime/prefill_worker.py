# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Collect activation evidence outside timed requests."""

from qwen36.runtime.worker_lifecycle import MPMDBenchmarkWorker


class PrefillWorker(MPMDBenchmarkWorker):
    def benchmark_mpmd_stats(self):
        import sys

        result = super().benchmark_mpmd_stats()
        result["pto_mtp_memory_reservation"] = getattr(self, "_pto_mtp_memory_reservation", {})
        result["prefill_metadata_status"] = getattr(self.model_runner, "_prefill_metadata_status", {})
        result["prefill_metadata_sources"] = {
            name: getattr(sys.modules.get(name), "__file__", None)
            for name in [
                "vllm_ascend.patch.worker.patch_gdn_attn",
                "vllm_ascend.patch.worker.patch_qwen3_5",
                "vllm_ascend.ops.triton.fla.chunk_o",
                "vllm_ascend.ops.triton.fla.chunk",
            ]
        }
        return result
