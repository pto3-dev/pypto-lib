# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Opt-in host submission trace. Never reads NPU tensor values or synchronizes."""

import functools, json, os, time
from pathlib import Path
from numbers import Integral


def traced_forward(fn):
    @functools.wraps(fn)
    def wrapped(self, model, input_ids, positions, *args, **kwargs):
        destination = os.environ.get("QWEN36_ROUTE_TRACE")
        if not destination:
            return fn(self, model, input_ids, positions, *args, **kwargs)
        from vllm.forward_context import get_forward_context

        ctx = get_forward_context()
        meta = getattr(ctx, "attn_metadata", None) or {}
        observation = self.route(
            ctx, dict(input_ids=input_ids, positions=positions, inputs_embeds=kwargs.get("inputs_embeds"))
        )
        qloc = None
        key = None
        for name, value in meta.items():
            candidate = getattr(value, "actual_seq_lengths_q", None)
            if isinstance(candidate, (list, tuple)) and all(isinstance(v, Integral) for v in candidate):
                qloc = list(map(int, candidate))
                key = name
                break
        path = getattr(self, "_route_trace_path", None)
        if path is None:
            path = Path(destination) / f"rank{self.rank}_{os.getpid()}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            self._route_trace_path = path
            self._route_trace_count = 0
        self._route_trace_count += 1
        sequence = self._route_trace_count
        start = time.monotonic_ns()
        record = dict(
            event="begin",
            sequence=sequence,
            pid=os.getpid(),
            rank=self.rank,
            monotonic_ns=start,
            input_rows=int(input_ids.numel()),
            backend=observation.backend.value,
            reason=observation.reason,
            phase=observation.phase.value,
            query_ends_cpu=qloc,
            query_source=key,
            graph_mode=getattr(ctx.cudagraph_runtime_mode, "name", str(ctx.cudagraph_runtime_mode)),
        )
        with path.open("a") as out:
            out.write(json.dumps(record) + "\n")
        try:
            result = fn(self, model, input_ids, positions, *args, **kwargs)
        except BaseException as error:
            with path.open("a") as out:
                out.write(
                    json.dumps(
                        dict(
                            event="error",
                            sequence=sequence,
                            monotonic_ns=time.monotonic_ns(),
                            error=type(error).__name__,
                        )
                    )
                    + "\n"
                )
            raise
        ticket = self._output_ticket.get()
        end = time.monotonic_ns()
        with path.open("a") as out:
            out.write(
                json.dumps(
                    dict(
                        event="end",
                        sequence=sequence,
                        monotonic_ns=end,
                        cpu_submit_us=(end - start) / 1000,
                        actual_backend=ticket.backend.value if ticket else None,
                    )
                )
                + "\n"
            )
        return result

    return wrapped
