# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Uniform MTP5 input preparation; no changes to target/drafter mathematics.

Only CPU metadata controls eligibility. Fresh sampled and draft IDs are consumed
on every call; no token values are cached. Nonuniform, multimodal and reordered
batches retain the upstream path. Per-runner graph sources are address-checked.
"""

from collections import Counter
from functools import wraps


def settings(runner):
    return runner.vllm_config.additional_config.get("qwen38_pto_mpmd", {})


def uniform_batch(runner, scheduler, rows):
    batch = runner.input_batch
    count = batch.num_reqs
    if not 1 <= count <= 4 or rows != 6 * count:
        return None
    if runner.pcp_size != 1 or getattr(runner, "dcp_size", 1) != 1:
        return None
    if runner.enable_prompt_embeds or runner.num_spec_tokens != 5:
        return None
    for index, req_id in enumerate(batch.req_ids):
        req = runner.requests[req_id]
        if req.prompt_token_ids is None or req.prompt_embeds is not None:
            return None
        if int(batch.num_computed_tokens_cpu[index]) < len(req.prompt_token_ids):
            return None
        if scheduler.num_scheduled_tokens[req_id] != 6:
            return None
        if len(scheduler.scheduled_spec_decode_tokens.get(req_id, ())) != 5:
            return None
    return count


def pack_eligible(runner, scheduler, rows):
    count = uniform_batch(runner, scheduler, rows)
    if count is None:
        return None
    batch = runner.input_batch
    previous = batch.prev_req_id_to_index
    if previous is None or batch.prev_sampled_token_ids is None:
        return None
    if any(previous.get(req_id) != index for index, req_id in enumerate(batch.req_ids)):
        return None
    draft = runner._draft_token_ids
    if draft is None or draft.ndim != 2 or draft.shape[0] < count or draft.shape[1] != 5:
        return None
    if batch.prev_sampled_token_ids.ndim != 2 or batch.prev_sampled_token_ids.shape[0] < count:
        return None
    return count


def pack_tokens(runner, count):
    import torch

    output = runner.input_ids.gpu[: 6 * count].view(count, 6)
    sampled = runner.input_batch.prev_sampled_token_ids[:count, :1]
    drafts = runner._draft_token_ids[:count, :5]
    # vLLM sampler output is normally int32; handle int64 without host readback.
    torch.cat((sampled.to(dtype=output.dtype), drafts.to(dtype=output.dtype)), dim=1, out=output)


class PositionGraph:
    """Compute text M-RoPE from the already-uploaded sequence lengths."""

    def __init__(self, runner, count):
        import torch

        self.source = runner.seq_lens.gpu
        self.output = runner.mrope_positions.gpu
        self.source_ptr = self.source.data_ptr()
        self.output_ptr = self.output.data_ptr()
        self.count = count
        self.offsets = torch.arange(6, dtype=torch.int64, device=self.source.device)
        current = torch.npu.current_stream()
        self.stream = torch.npu.Stream()
        self.stream.wait_stream(current)
        with torch.npu.stream(self.stream):
            self.calculate()
        self.stream.synchronize()
        self.graph = torch.npu.NPUGraph()
        with torch.npu.graph(self.graph, stream=self.stream):
            self.calculate()
        current.wait_stream(self.stream)

    def calculate(self):
        positions = (
            self.source[: self.count].to(dtype=self.offsets.dtype)[:, None] - 6 + self.offsets[None, :]
        ).reshape(-1)
        self.output[:, : 6 * self.count].copy_(positions[None, :].expand(3, -1))

    def replay(self, runner):
        if (
            runner.seq_lens.gpu.data_ptr() != self.source_ptr
            or runner.mrope_positions.gpu.data_ptr() != self.output_ptr
        ):
            raise RuntimeError("Position graph buffers changed; rebuild required")
        self.graph.replay()


def counter(runner):
    if not hasattr(runner, "_compact_input_counts"):
        runner._compact_input_counts = Counter()
    return runner._compact_input_counts


def stats(runner):
    return dict(
        counts=dict(counter(runner)),
        position_graph_batches=sorted(getattr(runner, "_compact_position_graphs", {})),
        target_or_drafter_math_changed=False,
        validation_limit=settings(runner).get("compact_validate_calls", 0),
        diagnostic_calls=getattr(runner, "_compact_call_records", []),
    )


def install():
    import torch
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    original_ids = NPUModelRunner._prepare_input_ids
    original_sample = NPUModelRunner.sample_tokens
    original_positions = NPUModelRunner._prepare_mrope_inputs
    if getattr(original_ids, "_compact_decode_inputs", False):
        return

    @wraps(original_ids)
    def prepare_ids(self, scheduler_output, total_num_scheduled_tokens, cu_num_tokens):
        config = settings(self)
        mode = config.get("compact_inputs", "off")
        if mode not in ("off", "tokens", "positions", "both", "alternating"):
            raise ValueError("Unknown compact_inputs mode: " + mode)
        counts = counter(self)
        calls = counts["calls"]
        counts["calls"] += 1
        self._compact_current = mode
        if mode == "alternating":
            # Counter, not clock; all TP ranks execute the same ordered forwards.
            block = int(config.get("compact_block_calls", 32))
            initial = int(config.get("compact_initial_native_calls", 0))
            if block < 16 or initial < 0:
                raise ValueError("Invalid diagnostic block sizes")
            self._compact_current = "both" if calls >= initial and ((calls - initial) // block) % 2 else "off"
        self._compact_call = calls
        count = pack_eligible(self, scheduler_output, total_num_scheduled_tokens)
        if config.get("compact_record_calls", False):
            import time

            if not hasattr(self, "_compact_call_records"):
                self._compact_call_records = []
            self._compact_call_records.append(
                dict(
                    call=calls,
                    mode=self._compact_current,
                    monotonic_ns=time.monotonic_ns(),
                    batch=self.input_batch.num_reqs,
                    rows=total_num_scheduled_tokens,
                    token_fast_eligible=count is not None,
                )
            )
        use = (
            self._compact_current in ("tokens", "both")
            and count is not None
            and not torch.npu.is_current_stream_capturing()
        )
        if not use:
            counts["tokens_native"] += 1
            return original_ids(self, scheduler_output, total_num_scheduled_tokens, cu_num_tokens)
        limit = int(config.get("compact_validate_calls", 0))
        key = f"tokens_validated_B{count}"
        expected = None
        if counts[key] < limit:
            original_ids(self, scheduler_output, total_num_scheduled_tokens, cu_num_tokens)
            expected = self.input_ids.gpu.clone()
        pack_tokens(self, count)
        if expected is not None:
            if not torch.equal(expected, self.input_ids.gpu):
                raise RuntimeError("Compact token pack differs from upstream")
            counts[key] += 1
        counts[f"tokens_fast_B{count}"] += 1
        counts["token_index_h2d_avoided"] += 4

    @wraps(original_positions)
    def prepare_positions(self, scheduler_output, total_num_scheduled_tokens):
        count = uniform_batch(self, scheduler_output, total_num_scheduled_tokens)
        mode = getattr(self, "_compact_current", "off")
        eligible = count is not None and all(
            self.requests[req_id].mrope_position_delta == 0 for req_id in self.input_batch.req_ids
        )
        if mode not in ("positions", "both") or not eligible or torch.npu.is_current_stream_capturing():
            counter(self)["positions_native"] += 1
            return original_positions(self, scheduler_output, total_num_scheduled_tokens)
        # Keep the native CPU mirror, including inactive columns, for fallback.
        self._calc_mrope_positions(scheduler_output)
        bank = getattr(self, "_compact_position_graphs", None)
        if bank is None:
            bank = self._compact_position_graphs = {}
        if count not in bank:
            bank[count] = PositionGraph(self, count)
        bank[count].replay(self)
        counts = counter(self)
        key = f"positions_validated_B{count}"
        if counts[key] < int(settings(self).get("compact_validate_calls", 0)):
            if not torch.equal(self.mrope_positions.gpu.cpu(), self.mrope_positions.cpu):
                raise RuntimeError("Device M-RoPE differs from complete upstream CPU bank")
            counts[key] += 1
        counts[f"positions_fast_B{count}"] += 1
        counts["mrope_h2d_avoided"] += 1
        counts["mrope_h2d_bytes_avoided"] += (
            self.mrope_positions.cpu.numel() * self.mrope_positions.cpu.element_size()
        )

    @wraps(original_sample)
    def sample_tokens(self, *args, **kwargs):
        result = original_sample(self, *args, **kwargs)
        if settings(self).get("compact_record_calls", False) and hasattr(result, "get_output"):
            result._compact_submission_tag = dict(
                call=getattr(self, "_compact_call", None), mode=getattr(self, "_compact_current", "off")
            )
        return result

    NPUModelRunner.sample_tokens = sample_tokens
    prepare_ids._compact_decode_inputs = True
    NPUModelRunner._prepare_input_ids = prepare_ids
    NPUModelRunner._prepare_mrope_inputs = prepare_positions
