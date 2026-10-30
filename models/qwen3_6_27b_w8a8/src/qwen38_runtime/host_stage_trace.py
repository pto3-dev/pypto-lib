# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Diagnostic-only CPU spans; no device reads, barriers or per-call IO."""

import atexit
from functools import wraps
import gc
import json
import os
from pathlib import Path
import threading
import time

_rows = []
_round = 0
_installed = False
_gc_start = {}


def enabled():
    return 1500 <= _round <= 3500


def flush():
    folder = Path(os.environ["QWEN36_HOST_STAGE_TRACE"])
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"host_stages_{os.getpid()}.json"
    path.write_text(
        json.dumps(
            dict(
                pid=os.getpid(),
                diagnostic_only=True,
                scope="CPU wall/thread CPU spans; no device time, no tensor values",
                observed_execute_calls=_round,
                records=_rows,
            ),
            separators=(",", ":"),
        )
        + "\n"
    )
    return str(path)


def graph_stats(owner):
    graph = getattr(owner, "_pto_merged_mtp_graph", None)
    return dict(
        graph_replays=getattr(graph, "replays", None),
        raw_fallbacks=getattr(graph, "fallback_calls", None),
        graph_call_used=getattr(owner, "_pto_graph_call_used", None),
    )


def wrap(cls, name, label, *, counter=False, graph=False):
    old = getattr(cls, name)

    @wraps(old)
    def call(self, *args, **kwargs):
        global _round
        if counter:
            _round += 1
        if not enabled():
            return old(self, *args, **kwargs)
        import torch

        start = time.monotonic_ns()
        cpu = time.thread_time_ns()
        epoch = _round
        before = graph_stats(self) if graph else None
        try:
            with torch.autograd.profiler.record_function(label):
                return old(self, *args, **kwargs)
        finally:
            if len(_rows) >= 100000:
                raise RuntimeError("Host stage diagnostic limit reached")
            _rows.append(
                dict(
                    name=label,
                    round=epoch,
                    thread=threading.get_native_id(),
                    start_ns=start,
                    end_ns=time.monotonic_ns(),
                    thread_cpu_ns=time.thread_time_ns() - cpu,
                    graph_before=before,
                    graph_after=graph_stats(self) if graph else None,
                )
            )

    setattr(cls, name, call)


def gc_callback(phase, info):
    if not enabled():
        return
    key = (threading.get_native_id(), info["generation"])
    if phase == "start":
        _gc_start[key] = (time.monotonic_ns(), time.thread_time_ns(), _round)
    elif key in _gc_start:
        start, cpu, epoch = _gc_start.pop(key)
        _rows.append(
            dict(
                name="Host.GC",
                round=epoch,
                thread=key[0],
                generation=key[1],
                start_ns=start,
                end_ns=time.monotonic_ns(),
                thread_cpu_ns=time.thread_time_ns() - cpu,
            )
        )


def install():
    global _installed
    if _installed or not os.environ.get("QWEN36_HOST_STAGE_TRACE"):
        return
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
    from vllm_ascend.spec_decode.eagle_proposer import SpecDecodeBaseProposer
    from .executor import TargetExecutor
    from .borrowed_runtime import BorrowedRuntime
    from .adaptive_metadata import AdaptiveMetadataPlan

    wrap(NPUModelRunner, "execute_model", "Runner.execute", counter=True)
    wrap(NPUModelRunner, "sample_tokens", "Runner.sample")
    wrap(NPUModelRunner, "_prepare_inputs", "Runner.inputs")
    wrap(NPUModelRunner, "propose_draft_token_ids", "Runner.propose")
    wrap(TargetExecutor, "forward", "PTO.forward")
    wrap(AdaptiveMetadataPlan, "update", "PTO.metadata")
    wrap(BorrowedRuntime, "enqueue", "PTO.enqueue")
    wrap(SpecDecodeBaseProposer, "_propose", "MTP.propose", graph=True)
    wrap(SpecDecodeBaseProposer, "_update_full_graph_params", "MTP.params", graph=True)
    wrap(SpecDecodeBaseProposer, "set_inputs_first_pass", "MTP.inputs")
    gc.callbacks.append(gc_callback)
    atexit.register(flush)
    _installed = True
