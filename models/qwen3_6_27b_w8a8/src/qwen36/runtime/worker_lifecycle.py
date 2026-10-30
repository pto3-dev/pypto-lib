# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Benchmark lifecycle hooks outside neural execution and request timing."""


def install_weight_initialization():
    import torch
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    original = NPUModelRunner.load_model

    def load_model(runner, *args, **kwargs):
        result = original(runner, *args, **kwargs)
        causal = getattr(runner.model, "language_model", runner.model)
        executor = getattr(causal, "_qwen38_pto_mpmd_executor", None)
        if executor is not None:
            # Weight rebanking has a one-time temporary. Complete it before
            # activation profiling resets its peak-memory counters. Account
            # any permanent allocation change as weights, preserving the
            # normal vLLM available-cache calculation and its checks.
            before = torch.npu.memory_allocated()
            initialization_stream = torch.npu.SyncLaunchStream(device=next(causal.parameters()).device)
            with torch.npu.stream(initialization_stream):
                executor._prepare_shared_weights()
            initialization_stream.synchronize()
            torch.npu.synchronize()
            after = torch.npu.memory_allocated()
            delta = after - before
            runner.model_memory_usage += delta
            executor.shared_weight_info.update(
                initialized_before_activation_profile=True,
                resident_allocation_delta_bytes=delta,
                accounted_weights_bytes=int(runner.model_memory_usage),
            )
        return result

    NPUModelRunner.load_model = load_model


install_weight_initialization()


class MPMDBenchmarkWorker:
    def benchmark_mpmd_stats(self):
        from vllm_ascend.pypto.qwen3_8_stats import collect_pto_stats
        from qwen36.runtime.cpu_output_trace import flush

        result = collect_pto_stats(self.get_model())
        result["cpu_verifier_trace"] = flush()
        return result

    def benchmark_mpmd_close(self):
        from vllm_ascend.pypto.qwen3_8_mpmd import close_mpmd

        return close_mpmd(self.get_model())


# Optional evidence hook is disabled unless QWEN36_CPU_MTP_TRACE is set.
from qwen36.runtime.cpu_output_trace import install as install_cpu_output_trace

install_cpu_output_trace()
