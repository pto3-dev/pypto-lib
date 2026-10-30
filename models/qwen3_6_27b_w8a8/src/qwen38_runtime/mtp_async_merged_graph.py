# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Native five-step MTP graphs for B1..B4, with no extra generated tokens.

Each row count owns its five FIA task groups via vLLM's GraphParams dictionaries.
One completion chain spans ALL row counts because the proposer input buffers are
shared. Target execution remains Simpler DIRECT; drafter math remains native.
"""

from collections import Counter
from contextlib import nullcontext
from functools import wraps
from contextvars import ContextVar

GRAPH_MODE = "merged_subbatches"
_IN_DRAFTER = ContextVar("qwen36_subbatch_drafter_dispatch", default=False)


class TargetPaddingDispatcher:
    """Keep the fixed 24-row PTO target independent of drafter graph shapes.

    Initialization uses the real dispatch keys so native capture can prepare
    all drafter shapes. Serving target decode pads to 24 exactly as before;
    only a scoped MTP proposal sees the exact 6/12/18/24-row dispatcher.
    """

    def __init__(self, dispatcher):
        object.__setattr__(self, "_dispatcher", dispatcher)
        object.__setattr__(self, "serving", False)
        object.__setattr__(self, "target_padding_calls", 0)

    def __getattr__(self, key):
        return getattr(self._dispatcher, key)

    def __setattr__(self, key, value):
        if key in ("_dispatcher", "serving", "target_padding_calls"):
            object.__setattr__(self, key, value)
        else:
            setattr(self._dispatcher, key, value)

    def dispatch(self, num_tokens, uniform_decode=False, **kwargs):
        result = self._dispatcher.dispatch(num_tokens=num_tokens, uniform_decode=uniform_decode, **kwargs)
        mode, _ = result
        if (
            self.serving
            and not _IN_DRAFTER.get()
            and uniform_decode
            and 0 < num_tokens < 24
            and getattr(mode, "name", "") == "FULL"
        ):
            if num_tokens != 24:
                self.target_padding_calls += 1
            return self._dispatcher.dispatch(num_tokens=24, uniform_decode=uniform_decode, **kwargs)
        return result


def activate_target_padding(runner):
    dispatcher = runner.cudagraph_dispatcher
    if isinstance(dispatcher, TargetPaddingDispatcher):
        dispatcher.serving = True


def canonical_batch(common, ids, hidden, next_ids, descriptor, kwargs, *, subbatches):
    """Check CPU metadata only; never synchronize/read NPU values here."""
    batch = int(common.num_reqs)
    if batch not in ((1, 2, 3, 4) if subbatches else (4,)):
        return None
    rows = batch * 6
    if not (
        common.num_actual_tokens == rows
        and ids.shape[0] == rows
        and hidden.shape[0] == rows
        and next_ids.shape[0] == batch
        and descriptor is not None
        and bool(descriptor.uniform)
        and kwargs.get("num_prefill_reqs", 0) == 0
        and kwargs.get("mm_embed_inputs") is None
        and common.query_start_loc_cpu.device.type == "cpu"
        and common.query_start_loc_cpu.tolist() == list(range(0, rows + 1, 6))
    ):
        return None
    return batch


def stats(owner):
    graph = getattr(owner, "_pto_merged_mtp_graph", None)
    return dict(
        enabled=bool(getattr(owner, "_pto_mtp_graph_enabled", False)),
        native_math=True,
        use_cuda_graph=bool(getattr(owner, "use_cuda_graph", False)),
        async_scheduling=bool(getattr(owner, "use_async_scheduling", False)),
        subbatch_graphs=bool(getattr(owner, "_pto_mtp_subbatch_graphs", False)),
        captures=graph.captures if graph else 0,
        replays=graph.replays if graph else 0,
        eager_calls=graph.fallback_calls if graph else 0,
        graph_calls_by_batch=dict(graph.calls_by_batch) if graph else {},
        fallback_by_shape=dict(graph.fallback_by_shape) if graph else {},
        fia_handles_by_rows=graph.fia_handles_by_rows if graph else {},
        fixed_target_padding_calls=getattr(
            getattr(getattr(owner, "runner", None), "cudagraph_dispatcher", None), "target_padding_calls", 0
        ),
        validations_by_batch=getattr(owner, "_pto_graph_validated_by_batch", {}),
        validated_calls=getattr(owner, "_pto_graph_validations", 0),
        validation_result=getattr(owner, "_pto_graph_validation", None),
    )


def install():
    import torch
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import get_forward_context
    from vllm_ascend.compilation.acl_graph import ACLGraphWrapper, get_draft_graph_params
    from vllm_ascend.spec_decode.eagle_proposer import SpecDecodeBaseProposer
    from vllm_ascend.utils import enable_sp

    old_init = SpecDecodeBaseProposer.__init__
    if getattr(old_init, "_pto_graph_runtime", False):
        return
    old_share = SpecDecodeBaseProposer._maybe_share_lm_head
    old_propose = SpecDecodeBaseProposer._propose
    old_update = SpecDecodeBaseProposer._update_full_graph_params

    class Graph:
        def __init__(self, owner):
            self.owner, self.raw = owner, owner._run_merged_draft
            self.wrapper = ACLGraphWrapper(
                self.raw,
                owner.vllm_config,
                runtime_mode=CUDAGraphMode.FULL,
                use_eagle=True,
                enable_enpu=False,
            )
            self.captures = self.replays = self.fallback_calls = 0
            self.calls_by_batch, self.fallback_by_shape = Counter(), Counter()
            self.fia_handles_by_rows = {}
            self.completions = [torch.npu.Event(), torch.npu.Event()]
            self.previous, self.next_completion = None, 0

        def __call__(self, **kwargs):
            context = get_forward_context()
            batch, rows = kwargs["batch_size"], kwargs["num_tokens"]
            batches = (1, 2, 3, 4) if self.owner._pto_mtp_subbatch_graphs else (4,)
            selected = (
                context.cudagraph_runtime_mode == CUDAGraphMode.FULL
                and batch in batches
                and rows == 6 * batch
                and kwargs["num_input_tokens"] == rows
                and not kwargs.get("is_prefill")
            )
            entries = self.wrapper.concrete_aclgraph_entries
            # Legacy mode retains its previously measured coverage exactly.
            if (
                selected
                and not self.owner._pto_mtp_subbatch_graphs
                and entries
                and context.batch_descriptor not in entries
            ):
                selected = False
            self.owner._pto_graph_call_used = selected
            if not selected:
                self.fallback_calls += 1
                self.fallback_by_shape[f"B{batch}/rows{rows}/prefill{bool(kwargs.get('is_prefill'))}"] += 1
                mode = context.cudagraph_runtime_mode
                context.cudagraph_runtime_mode = CUDAGraphMode.NONE
                try:
                    return self.raw(**kwargs)
                finally:
                    context.cudagraph_runtime_mode = mode
            entry = entries.get(context.batch_descriptor)
            capturing = entry is None or entry.aclgraph is None
            # ACLGraphWrapper enforces initialization-only capture. A missing
            # graph at serving time fails visibly instead of silently recapturing.
            output = self.wrapper(**kwargs)
            if capturing:
                self.captures += 1
                params = get_draft_graph_params()
                handles = params.handles[rows]
                expected = self.owner.num_speculative_tokens * len(self.owner.attn_layer_names)
                if len(handles) != expected or len({id(h) for h in handles}) != expected:
                    raise RuntimeError(
                        f"MTP {rows} rows needs {expected} distinct FIA handles; got {len(handles)}"
                    )
                self.fia_handles_by_rows[str(rows)] = len(handles)
            else:
                self.replays += 1
            self.calls_by_batch[str(batch)] += 1
            return output

        def update(self, *args, **kwargs):
            if self.previous is not None:
                self.owner.update_stream.wait_event(self.previous)
            result = old_update(self.owner, *args, **kwargs)
            event = self.completions[self.next_completion]
            event.record(torch.npu.current_stream())
            self.previous, self.next_completion = event, 1 - self.next_completion
            return result

    @wraps(old_init)
    def init(self, *args, **kwargs):
        old_init(self, *args, **kwargs)
        cfg = self.vllm_config
        settings = cfg.additional_config.get("qwen38_pto_mpmd", {})
        self._pto_mtp_graph_enabled = False
        mode = settings.get("mtp_async_graph")
        if mode not in ("merged_canonical", GRAPH_MODE):
            return
        p = cfg.parallel_config
        if (
            settings.get("backend") not in ("native", "pypto")
            or self.method != "mtp"
            or self.num_speculative_tokens != 5
            or not self.use_async_scheduling
            or self.speculative_config.enforce_eager
            or self.speculative_config.disable_padded_drafter_batch
            or not self.runner._use_aclgraph()
            or self.enable_enpu
            or enable_sp(cfg)
            or cfg.scheduler_config.max_num_seqs != 4
            or (
                p.tensor_parallel_size,
                p.pipeline_parallel_size,
                p.prefill_context_parallel_size,
                p.decode_context_parallel_size,
            )
            != (4, 1, 1, 1)
        ):
            raise ValueError("MTP graph requires TP4/PP1/maxB4/MTP5, async, padded, no CP/SP/ENPU")
        self._pto_mtp_subbatch_graphs = mode == GRAPH_MODE
        if self._pto_mtp_subbatch_graphs:
            sizes = set(cfg.compilation_config.cudagraph_capture_sizes)
            if not {6, 12, 18, 24} <= sizes:
                raise ValueError("Subbatch MTP graphs require capture sizes [6,12,18,24]")
            if settings.get("backend") == "pypto" and not isinstance(
                self.runner.cudagraph_dispatcher, TargetPaddingDispatcher
            ):
                self.runner.cudagraph_dispatcher = TargetPaddingDispatcher(self.runner.cudagraph_dispatcher)
        if settings.get("mtp_graph_validate_calls", 0):
            import torch_npu

            torch_npu.npu.set_deterministic_level(1)
        self._pto_mtp_graph_enabled, self._pto_graph_call_used = True, False
        self._pto_graph_validations, self._pto_graph_validated_by_batch = 0, {}
        self.use_cuda_graph, self.maybe_eager_context = True, nullcontext()

    @wraps(old_share)
    def share(self, *args, **kwargs):
        if not getattr(self, "_pto_mtp_graph_enabled", False):
            return old_share(self, *args, **kwargs)
        self.use_cuda_graph = False
        try:
            result = old_share(self, *args, **kwargs)
        finally:
            self.use_cuda_graph = True
        self.update_stream = torch.npu.Stream()
        self._pto_merged_mtp_graph = Graph(self)
        self._runnable = self._pto_merged_mtp_graph
        return result

    @wraps(old_update)
    def update(self, *args, **kwargs):
        if not getattr(self, "_pto_mtp_graph_enabled", False):
            return old_update(self, *args, **kwargs)
        if self._pto_graph_call_used:
            return self._pto_merged_mtp_graph.update(*args, **kwargs)

    @wraps(old_propose)
    def propose(
        self,
        target_token_ids,
        target_positions,
        target_hidden_states,
        next_token_ids,
        token_indices_to_sample,
        common_attn_metadata,
        target_model_batch_desc,
        sampling_metadata,
        *args,
        **kwargs,
    ):
        inputs = (
            target_token_ids,
            target_positions,
            target_hidden_states,
            next_token_ids,
            token_indices_to_sample,
        )
        if not getattr(self, "_pto_mtp_graph_enabled", False):
            return old_propose(
                self,
                *inputs,
                common_attn_metadata,
                target_model_batch_desc,
                sampling_metadata,
                *args,
                **kwargs,
            )
        batch = canonical_batch(
            common_attn_metadata,
            target_token_ids,
            target_hidden_states,
            next_token_ids,
            target_model_batch_desc,
            kwargs,
            subbatches=self._pto_mtp_subbatch_graphs,
        )
        selected = batch is not None and not args

        def call(values=inputs, common=common_attn_metadata, options=kwargs, graph=None):
            self.use_cuda_graph = selected if graph is None else graph
            token = _IN_DRAFTER.set(True)
            try:
                return old_propose(
                    self, *values, common, target_model_batch_desc, sampling_metadata, *args, **options
                )
            finally:
                _IN_DRAFTER.reset(token)
                self.use_cuda_graph = True

        settings = self.vllm_config.additional_config["qwen38_pto_mpmd"]
        limit = settings.get("mtp_graph_validate_calls", 0)
        if selected and self._pto_graph_validated_by_batch.get(str(batch), 0) < limit:
            from .mtp_subbatch_validation import validate_same_state

            result = validate_same_state(self, call, inputs, common_attn_metadata, kwargs, batch)
            self._pto_graph_validated_by_batch[str(batch)] = (
                self._pto_graph_validated_by_batch.get(str(batch), 0) + 1
            )
            return result
        return call()

    init._pto_graph_runtime = True
    SpecDecodeBaseProposer.__init__ = init
    SpecDecodeBaseProposer._maybe_share_lm_head = share
    SpecDecodeBaseProposer._update_full_graph_params = update
    SpecDecodeBaseProposer._propose = propose
