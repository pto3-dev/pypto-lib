# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Worker initialization/statistics hooks, not a neural RPC service."""

_INSTALLED = False


def install_weight_initialization():
    global _INSTALLED
    if _INSTALLED:
        return
    import torch, time
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
    from .adapter import target_of
    from .weight_loading import restore_deferral, deferral_stats

    original = NPUModelRunner.load_model

    def load_model(runner, *args, **kwargs):
        try:
            load_start = time.perf_counter()
            result = original(runner, *args, **kwargs)
            native_load_ms = (time.perf_counter() - load_start) * 1000
            executor = target_of(runner.model)
            if executor is not None:
                before = torch.npu.memory_allocated()
                stream = torch.npu.SyncLaunchStream(device=next(executor.causal.parameters()).device)
                with torch.npu.stream(stream):
                    executor._prepare_shared_weights()
                stream.synchronize()
                torch.npu.synchronize()
                delta = torch.npu.memory_allocated() - before
                runner.model_memory_usage += delta
                executor.shared_weight_info.update(
                    weight_deferral=deferral_stats(),
                    native_model_and_drafter_load_ms=native_load_ms,
                    initialized_before_activation_profile=True,
                    resident_allocation_delta_bytes=delta,
                    accounted_weights_bytes=int(runner.model_memory_usage),
                )
            return result
        finally:
            restore_deferral()
            from .preloaded_weights import restore_preload_hooks

            restore_preload_hooks()

    capture_original = NPUModelRunner.capture_model

    def capture_model(runner, *args, **kwargs):
        result = capture_original(runner, *args, **kwargs)
        from .mtp_async_merged_graph import activate_target_padding

        activate_target_padding(runner)
        executor = target_of(runner.model)
        if (
            executor is not None
            and executor.policy.execution.value == "direct"
            and executor.borrowed_runtime is not None
        ):
            if executor.borrowed_runtime.release_warmup_binding():
                executor.direct_warmup_reprepares = getattr(executor, "direct_warmup_reprepares", 0) + 1
            if executor.metadata_plan is not None and hasattr(executor.metadata_plan, "mark_runtime_ready"):
                executor.metadata_plan.mark_runtime_ready()
        return result

    NPUModelRunner.capture_model = capture_model
    NPUModelRunner.load_model = load_model
    _INSTALLED = True


install_weight_initialization()


class MPMDBenchmarkWorker:
    def benchmark_mpmd_stats(self):
        from .adapter import target_of

        executor = target_of(self.get_model())
        from .cpu_output_trace import flush

        result = executor.stats() if executor is not None else {"pto_executor": False}
        result["cpu_verifier_trace"] = flush()
        return result

    def benchmark_mpmd_close(self):
        from .adapter import close_mpmd

        return close_mpmd(self.get_model())

    def benchmark_retired_pair(self):
        from .adapter import target_of
        from qwen38_validation.retired_pair import validate

        return validate(target_of(self.get_model()))


from .cpu_output_trace import install as install_cpu_output_trace

install_cpu_output_trace()
