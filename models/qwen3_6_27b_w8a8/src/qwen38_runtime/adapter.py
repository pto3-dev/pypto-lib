# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Two explicit framework compatibility points, scoped to attached target instances.

No AST source patch, no replacement of the generic module-level parameter updater,
no inherited service executor, and no changes to native drafter execution.
"""

from types import MethodType
from .config import Backend, Execution

_INSTALLED = False


def target_of(model):
    runnable = getattr(model, "runnable", model)
    causal = getattr(runnable, "language_model", runnable)
    return getattr(causal, "_qwen38_pto_mpmd_executor", None)


def attach_mpmd(causal, config, settings):
    from .config import RuntimeSettings

    if RuntimeSettings.from_mapping(settings).backend is Backend.NATIVE:
        return None
    from .executor import TargetExecutor

    executor = TargetExecutor(causal, config, settings)
    if settings.get("gdn_metadata") == "raw_canonical":
        if executor.ragged_enabled:
            raise ValueError("Raw GDN metadata candidate requires canonical verifier")
        from .gdn_metadata_fast import install

        install()
    install_framework_adapter()
    causal._qwen38_pto_mpmd_executor = executor
    causal.model.forward = MethodType(executor.forward, causal.model)
    causal.compute_logits = MethodType(executor.compute_logits, causal)
    return executor


def install_framework_adapter():
    global _INSTALLED
    if _INSTALLED:
        return
    from .framework_contract import validate_framework

    validate_framework()
    from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    original_call = ACLGraphWrapper.__call__
    original_update = NPUModelRunner._update_full_graph_params_if_needed
    if getattr(ACLGraphWrapper, "_qwen38_mpmd_graph_hook", False):
        raise RuntimeError("Legacy graph overlay is active; use only the explicit integration package")

    def call(wrapper, *args, **kwargs):
        executor = target_of(wrapper)
        if executor is None or executor.policy.backend is Backend.NATIVE:
            return original_call(wrapper, *args, **kwargs)
        from vllm.forward_context import get_forward_context

        context = get_forward_context()
        decision = executor.route(context, kwargs)
        if decision.backend is Backend.NATIVE:
            return original_call(wrapper, *args, **kwargs)
        if executor.policy.execution is Execution.DIRECT:
            # Leave native drafter graph configuration untouched, but never capture this target.
            return wrapper.runnable(*args, **kwargs)
        return full_call(wrapper, executor, context, decision, args, kwargs)

    def update(runner, context, num_tokens_padded, positions):
        executor = target_of(runner.model)
        if executor is not None and executor.policy.backend is Backend.PYPTO:
            from .routing import observe, decide

            observation = observe(context, num_tokens_padded)
            from .routing import Phase

            if executor.ragged_enabled and observation.phase is Phase.VERIFY and 1 <= num_tokens_padded <= 24:
                executor.query_layout(context, num_tokens_padded)
                return
            decision = decide(executor.policy, observation)
            from .routing import guard_metadata_consumer

            guard_metadata_consumer(context, decision)
            if decision.backend is Backend.PYPTO:
                return
        return original_update(runner, context, num_tokens_padded, positions)

    ACLGraphWrapper.__call__ = call
    NPUModelRunner._update_full_graph_params_if_needed = update
    _INSTALLED = True


def full_call(wrapper, executor, context, decision, args, kwargs):
    import torch, weakref

    if getattr(context.cudagraph_runtime_mode, "name", str(context.cudagraph_runtime_mode)) != "FULL":
        raise RuntimeError("FULL target selected but the framework supplied no FULL capture/replay context")
    key = context.batch_descriptor
    executor.prepare_resources(context, kwargs["positions"].device)
    executor.prepare_inputs(context, kwargs)
    entry = executor.graph_entries.get(key)
    if entry is None:
        graph = torch.npu.NPUGraph()
        executor.metadata_plan.validate(context, kwargs, stable_addresses=True)
        was_capturing = getattr(context, "capturing", False)
        context.capturing = True
        try:
            with torch.npu.graph(graph, pool=wrapper.graph_pool, stream=executor.capture_stream):
                output = wrapper.runnable(*args, **kwargs)
        finally:
            context.capturing = was_capturing
        executor.graph_entries[key] = (graph, output)
        executor.caller_graphs.append(weakref.ref(graph))
        executor.last_replay_stream = torch.npu.current_stream().npu_stream
        if executor.settings.get("initial_replay_check", True):
            executor.guard_epoch_budget()
            executor.control_i32[3].fill_(1)
            graph.replay()
            from .stream_completion import synchronize_stream

            synchronize_stream(executor.last_replay_stream, executor.settings.get("stream_timeout_ms", 30000))
            executor.control_i32[1].zero_()
            executor.control_i32[3].zero_()
        return output
    executor.guard_epoch_budget()
    executor.metadata_plan.validate(context, kwargs, stable_addresses=True)
    # All target metadata updates and kernel joins are on the captured dependency chain.
    # Keep the prior barrier as an explicit baseline policy until device validation.
    executor.last_replay_stream = torch.npu.current_stream().npu_stream
    if executor.settings.get("full_sync", "required") == "required":
        from .stream_completion import synchronize_stream

        synchronize_stream(executor.last_replay_stream, executor.settings.get("stream_timeout_ms", 30000))
    elif executor.settings.get("full_sync") != "stream":
        raise ValueError("full_sync must be required or stream")
    entry[0].replay()
    executor.mark_replay(decision)
    return entry[1]


def close_mpmd(model):
    executor = target_of(model)
    if executor is not None:
        executor.close()
    return {"mpmd_closed": executor is not None}
