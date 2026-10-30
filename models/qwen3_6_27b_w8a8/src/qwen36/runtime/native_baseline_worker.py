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
from qwen36.runtime.cpu_output_trace import install

install()
"""Native-only diagnostics run after timed generation, without a PTO executor."""


class NativeBenchmarkWorker:
    def benchmark_mpmd_stats(self):
        from types import SimpleNamespace
        from vllm.distributed import get_tensor_model_parallel_rank
        from qwen36.runtime.native_nz_executor import native_cache_bindings

        model = self.get_model()
        causal = getattr(model, "language_model", model)
        if getattr(causal, "_qwen38_pto_mpmd_executor", None) is not None:
            raise RuntimeError("Native baseline unexpectedly has a PTO executor")
        config = getattr(self.model_runner, "model_config", None)
        holder = SimpleNamespace(
            causal=causal, cache_views={}, exported={}, max_seq=getattr(config, "max_model_len", 43008)
        )
        from qwen36.runtime.cpu_output_trace import flush

        return dict(
            cpu_verifier_trace=flush(),
            backend="native",
            rank=get_tensor_model_parallel_rank(),
            pto_executor=False,
            cache_dimensions=native_cache_bindings(holder, SimpleNamespace(virtual_engine=0)),
        )

    def benchmark_mpmd_close(self):
        from qwen36.runtime.cpu_output_trace import flush

        return dict(cpu_verifier_trace=flush(), backend="native", pto_runtime=False, closed=True)

    def benchmark_host_gc_probe(self):
        from qwen36.runtime.host_timeline import retired_gc_probe

        return retired_gc_probe()
