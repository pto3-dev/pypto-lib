# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Per-TP-rank target integration. No service process, socket, or model rewriting."""

from __future__ import annotations
import atexit, contextvars, os
from dataclasses import dataclass
from .config import RuntimeSettings, Backend, Execution
from .routing import observe, decide
from .stats import ExecutionStats
from .route_trace import traced_forward


@dataclass(frozen=True)
class OutputTicket:
    backend: Backend
    sequence: int
    restore: object = None


class TargetExecutor:
    def __init__(self, causal, config, settings):
        from vllm.distributed import get_tensor_model_parallel_rank

        self.causal = causal
        self.settings = dict(settings)
        self.policy = RuntimeSettings.from_mapping(settings)
        parallel = config.parallel_config
        text = config.model_config.hf_text_config
        if (parallel.tensor_parallel_size, parallel.pipeline_parallel_size) != (4, 1):
            raise ValueError("Target artifact requires TP4/PP1")
        if (text.hidden_size, text.num_hidden_layers, text.vocab_size) != (5120, 64, 248320):
            raise ValueError("Unsupported Qwen3.8 target geometry")
        if config.lora_config:
            raise ValueError("LoRA is outside this artifact contract")
        if config.model_config.quantization not in ("ascend", "modelslim"):
            raise ValueError("Expected native mixed W8A8 checkpoint loading")
        if not config.speculative_config or config.speculative_config.num_speculative_tokens != 5:
            raise ValueError("Target artifact requires MTP5")
        self.rank = get_tensor_model_parallel_rank()
        self.max_seq = config.model_config.max_model_len
        if self.max_seq <= 0 or self.max_seq % 3072:
            raise ValueError("Unsupported sequence-capacity alignment")
        from .native_compiled_view import prepare_native_compile_view

        self.native_prefill_view = prepare_native_compile_view(causal.model)
        self.native_forward = (
            self.native_prefill_view if self.native_prefill_view is not None else causal.model.forward
        )
        self.native_logits = causal.compute_logits
        self.shared_weights = None
        self.shared_weight_info = None
        self.hidden = None
        self.logits = None
        self.exported = {}
        self.cache_views = {}
        self.topology = None
        self.generation = None
        self.borrowed_runtime = None
        self.bridge = None
        self.metadata_plan = None
        self.caller_graphs = []
        self.graph_entries = {}
        self.counters = ExecutionStats()
        self.sequence = 0
        self._output_ticket = contextvars.ContextVar("qwen38_target_output", default=None)
        self.closed = False
        self.ragged_enabled = os.environ.get("QWEN36_RAGGED_VERIFIER") == "1"
        self.ragged_metadata = None
        self._ends_dirty = False
        if self.ragged_enabled and self.policy.execution is not Execution.DIRECT:
            raise ValueError("Ragged verifier requires DIRECT")
        self.communication_max_steps = 67108864
        self.communication_submissions = 0
        if self.policy.weight_strategy == "device_preloaded_nd_to_nz":
            from .preloaded_weights import install_device_destinations
            from .model_layout import WeightBankSpec

            self.preloaded_banks = install_device_destinations(causal, WeightBankSpec())
        if self.policy.weight_strategy in ("direct_cpu_nz", "preloaded_nd_to_nz"):
            from .preloaded_weights import install_cpu_destinations
            from .model_layout import WeightBankSpec

            self.preloaded_banks = install_cpu_destinations(causal, WeightBankSpec())
        if self.policy.weight_strategy in (
            "batched_nd_to_nz",
            "direct_cpu_nz",
            "preloaded_nd_to_nz",
            "device_preloaded_nd_to_nz",
        ):
            from .weight_loading import install_target_deferral

            install_target_deferral(causal)
        causal.model.do_not_compile = True
        atexit.register(self.close)

    def _prepare_shared_weights(self):
        from .resources import prepare_weights

        prepare_weights(self)

    def _ensure_outputs(self, device):
        import torch

        if self.hidden is None:
            self.hidden = torch.zeros((24, 5120), dtype=torch.bfloat16, device=device)
            self.logits = torch.zeros((24, 248320), dtype=torch.float32, device=device)

    def prepare_resources(self, context, device):
        from .resources import prepare_resources

        prepare_resources(self, context, device)

    def prepare_inputs(self, context, inputs):
        from .metadata import MetadataPlan

        spec = self.query_layout(context, inputs["input_ids"].numel()) if self.ragged_enabled else None
        if spec is not None and not spec.canonical:
            if self.ragged_metadata is None:
                from .ragged_metadata import RaggedMetadataPlan

                keys = [
                    layer.self_attn.attn.layer_name if i % 4 == 3 else layer.linear_attn.prefix
                    for i, layer in enumerate(self.causal.model.layers)
                ]
                self.ragged_metadata = RaggedMetadataPlan(keys).build(self, context, inputs)
            return
        if self.metadata_plan is None:
            keys = [
                layer.self_attn.attn.layer_name if i % 4 == 3 else layer.linear_attn.prefix
                for i, layer in enumerate(self.causal.model.layers)
            ]
            mode = self.settings.get("metadata_preprocess", "eager")
            plan_type = MetadataPlan
            if mode in ("staged_graph", "adaptive_graph"):
                if self.policy.execution is not Execution.DIRECT:
                    raise ValueError("Metadata graph candidate currently requires DIRECT target")
                from .captured_metadata import CapturedMetadataPlan

                plan_type = CapturedMetadataPlan
                if mode == "adaptive_graph":
                    from .adaptive_metadata import AdaptiveMetadataPlan

                    plan_type = AdaptiveMetadataPlan
            elif mode != "eager":
                raise ValueError("Unknown metadata preprocessing mode")
            self.metadata_plan = plan_type(keys).build(self, context, inputs)

    def _ticket(self, backend, restore=None):
        self.sequence += 1
        self._output_ticket.set(OutputTicket(backend, self.sequence, restore))

    def query_layout(self, context, rows):
        from .ragged_layout import layout

        key = self.causal.model.layers[3].self_attn.attn.layer_name
        meta = context.attn_metadata[key]
        return layout(getattr(meta, "actual_seq_lengths_q", None), rows)

    def route(self, context, inputs):
        ids = inputs.get("input_ids")
        rows = (
            ids.numel()
            if ids is not None
            else (inputs["inputs_embeds"].shape[0] if inputs.get("inputs_embeds") is not None else 0)
        )
        observation = observe(context, rows, inputs.get("inputs_embeds") is not None)
        from .routing import Phase, Decision

        if (
            self.ragged_enabled
            and self.policy.backend is Backend.PYPTO
            and observation.phase is Phase.VERIFY
            and 1 <= rows <= 24
            and not observation.has_embeddings
        ):
            self.query_layout(context, rows)  # Reject invalid CPU metadata before changing cache.
            return Decision(Backend.PYPTO, Phase.VERIFY, "supported_ragged_verifier")
        decision = decide(self.policy, observation)
        from .routing import guard_metadata_consumer

        guard_metadata_consumer(context, decision)
        return decision

    @traced_forward
    def forward(self, model, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        import torch
        from vllm.forward_context import get_forward_context

        context = get_forward_context()
        inputs = dict(input_ids=input_ids, positions=positions, inputs_embeds=inputs_embeds)
        decision = self.route(context, inputs)
        self.counters.record(decision)
        if decision.backend is Backend.NATIVE:
            self._ticket(Backend.NATIVE)
            return self.native_forward(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )
        self.prepare_resources(context, positions.device)
        spec = self.query_layout(context, input_ids.numel()) if self.ragged_enabled else None
        if self.metadata_plan is None or (
            spec is not None and not spec.canonical and self.ragged_metadata is None
        ):
            if torch.npu.is_current_stream_capturing():
                raise RuntimeError("Prepare metadata binding before capture")
            self.prepare_inputs(context, inputs)
        if self.policy.execution is Execution.DIRECT and torch.npu.is_current_stream_capturing():
            raise RuntimeError("Direct target must bypass the ACL wrapper capture path")
        self.guard_epoch_budget()
        restore = None
        if spec is not None and not spec.canonical:
            self.ragged_metadata.validate(context, inputs)
            self.ragged_metadata.update(self, context, inputs, spec)
            restore = self.ragged_metadata.restore
            self._ends_dirty = True
        else:
            if self._ends_dirty:
                self.control_i32[9:13].copy_(self.canonical_ends)
                self._ends_dirty = False
            self.metadata_plan.validate(
                context, inputs, stable_addresses=self.policy.execution is Execution.FULL
            )
            self.metadata_plan.update(self, context, inputs)
        self.bridge.enqueue()
        self._ticket(Backend.PYPTO, restore)
        if self.policy.execution is Execution.DIRECT:
            self.counters.direct_submissions += 1
        else:
            self.counters.full_captures += 1
        return torch.index_select(self.hidden, 0, restore) if restore is not None else self.hidden[:24]

    def compute_logits(self, causal, hidden_states):
        ticket = self._output_ticket.get()
        if ticket is None:
            raise RuntimeError("compute_logits has no forward/replay output ticket in this context")
        self._output_ticket.set(None)
        if ticket.backend is Backend.PYPTO:
            if hidden_states.shape[0] > 24:
                raise ValueError("Target output row count exceeds artifact")
            if ticket.restore is not None:
                import torch

                return torch.index_select(self.logits, 0, ticket.restore)
            return self.logits[: hidden_states.shape[0]]
        return self.native_logits(hidden_states)

    def guard_epoch_budget(self):
        if self.communication_submissions >= self.communication_max_steps:
            raise RuntimeError(
                "Communication epoch budget exhausted; synchronized signal/epoch reinitialization required"
            )
        self.communication_submissions += 1

    def mark_replay(self, decision):
        self.counters.record(decision)
        self.counters.full_replays += 1
        self._ticket(Backend.PYPTO)

    def stats(self):
        return dict(
            pto_executor=self.policy.backend is Backend.PYPTO,
            full_mpmd_model=True,
            metadata_preprocess=self.metadata_plan.stats()
            if self.metadata_plan is not None and hasattr(self.metadata_plan, "stats")
            else {"method": "eager"},
            native_prefill_compiler_enabled=self.native_prefill_view is not None,
            native_prefill_compiled=bool(
                self.native_prefill_view is not None and getattr(self.native_prefill_view, "compiled", False)
            ),
            ragged_verifier=self.ragged_enabled,
            route_trace=str(getattr(self, "_route_trace_path", "")),
            tp=4,
            pp=1,
            model_target=self.policy.model_target,
            checkpoint_provenance=self.policy.checkpoint_provenance,
            execution=self.policy.execution.value,
            topology=self.topology,
            shared_weight_info=self.shared_weight_info,
            cache_contents_copied=False,
            borrowed_cache_tensors=len(self.cache_views),
            build_generation=self.generation,
            external_rpc_per_decode=False,
            direct_warmup_reprepares=getattr(self, "direct_warmup_reprepares", 0),
            artifact_hashes=getattr(self, "artifact_hashes", {}),
            communication_max_steps=self.communication_max_steps,
            communication_submissions=self.communication_submissions,
            **self.counters.snapshot(),
        )

    def close(self):
        if self.closed:
            return
        from .resources import close

        close(self)
        self.closed = True
