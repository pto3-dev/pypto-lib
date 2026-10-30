# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from qwen36.runtime.host_timeline import install_worker as install_host_timeline

install_host_timeline()
from qwen36.runtime.prefill_worker import PrefillWorker


class NativeNzWorker(PrefillWorker):
    def benchmark_native_nz_profile(self):
        from qwen36.runtime.native_nz_executor import profile_terminal_step

        model = getattr(self.model_runner.model, "language_model", self.model_runner.model)
        executor = model._qwen38_pto_mpmd_executor
        return profile_terminal_step(executor)

    def benchmark_host_gc_probe(self):
        from qwen36.runtime.host_timeline import retired_gc_probe

        return retired_gc_probe()
