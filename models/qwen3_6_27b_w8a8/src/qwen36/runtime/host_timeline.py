# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Opt-in host clocks/GC evidence; no new NPU readback, event or stream wait.

Records wall+thread-CPU spans around existing worker operations. GC callbacks
never enumerate objects. The explicit retired-heap probe is separate from all
request/performance intervals, and restores GC freeze status before returning.
"""

import atexit, functools, gc, json, os, sys, time
from pathlib import Path

ENABLED = bool(os.environ.get("QWEN36_HOST_TRACE"))
_EVENTS = []
_STARTS = {}
_COUNTS = {}
_LIMIT = 200000
_DROPPED = 0
_FLUSHING = False
_INSTALLED = False
_PHASE = "startup"
_STEP = 0


def begin():
    return time.monotonic_ns(), time.thread_time_ns()


def end(kind, start, **fields):
    global _DROPPED
    if not ENABLED or _FLUSHING:
        return
    stop = time.monotonic_ns()
    cpu = time.thread_time_ns()
    if len(_EVENTS) >= _LIMIT:
        _DROPPED += 1
        return
    _EVENTS.append(
        dict(
            kind=kind,
            start_ns=start[0],
            end_ns=stop,
            thread_cpu_ns=cpu - start[1],
            phase=_PHASE,
            step=_STEP,
            **fields,
        )
    )


def phase(value):
    global _PHASE
    _PHASE = value


def _gc(when, info):
    generation = info.get("generation", -1)
    if when == "start":
        _STARTS[generation] = begin()
    elif generation in _STARTS:
        start = _STARTS.pop(generation)
        _COUNTS[generation] = _COUNTS.get(generation, 0) + 1
        end(
            "gc",
            start,
            generation=generation,
            collected=info.get("collected"),
            uncollectable=info.get("uncollectable"),
        )


def flush():
    global _FLUSHING
    if not ENABLED:
        return None
    _FLUSHING = True
    try:
        folder = Path(os.environ["QWEN36_HOST_TRACE"])
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"host_{os.getpid()}.json"
        payload = dict(
            pid=os.getpid(),
            argv=sys.argv,
            clock="CLOCK_MONOTONIC + thread CPU; all host processes same machine",
            wall_minus_monotonic_ns=time.time_ns() - time.monotonic_ns(),
            phase=_PHASE,
            gc_enabled=gc.isenabled(),
            gc_thresholds=gc.get_threshold(),
            frozen_objects=gc.get_freeze_count(),
            gc_counts=_COUNTS,
            dropped=_DROPPED,
            events=tuple(_EVENTS),
            scope="Host instrumentation only; spans may include device waits, scheduling or host descheduling. GC is attributed only on time overlap, not assumed from wall duration.",
        )
        path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
        return str(path)
    finally:
        _FLUSHING = False


def install_global():
    global _INSTALLED
    if not ENABLED or _INSTALLED:
        return
    gc.callbacks.append(_gc)
    atexit.register(flush)
    _INSTALLED = True


def install_worker():
    if not ENABLED:
        return
    install_global()
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
    from vllm_ascend.worker.worker import NPUWorker
    from vllm.v1.worker.gpu_model_runner import AsyncGPUModelRunnerOutput

    def wrap(cls, name, *, execute=False, ready=False):
        original = getattr(cls, name)
        if getattr(original, "_qwen36_host_timeline", False):
            return

        @functools.wraps(original)
        def traced(self, *args, **kwargs):
            global _STEP
            if execute:
                _STEP += 1
            if ready:
                phase("warmup")
            start = begin()
            try:
                return original(self, *args, **kwargs)
            finally:
                end(name, start)
                if ready:
                    phase("serving")

        traced._qwen36_host_timeline = True
        setattr(cls, name, traced)

    for name in ["execute_model", "sample_tokens", "_model_forward"]:
        wrap(NPUModelRunner, name, execute=name == "execute_model")
    wrap(NPUWorker, "compile_or_warm_up_model", ready=True)
    wrap(AsyncGPUModelRunnerOutput, "get_output")


def retired_gc_probe():
    """One explicit GC2 before/after freeze; NOT evidence of a natural pause."""
    old_phase = _PHASE
    phase("retired_gc_probe")
    initial = gc.get_freeze_count()
    result = dict(
        pid=os.getpid(),
        initial_frozen_objects=initial,
        scope="Explicit retired worker heap GC2 experiment, outside request timing. Not proof that a previous natural ITL spike was GC.",
    )
    try:
        if initial:
            result["skipped"] = "Existing frozen heap must remain unchanged"
            return result
        start = begin()
        collected = gc.collect(2)
        stop = begin()
        result.update(
            before_gc2_ms=(stop[0] - start[0]) / 1e6,
            before_gc2_cpu_ms=(stop[1] - start[1]) / 1e6,
            collected_before=collected,
        )
        gc.freeze()
        result["frozen_objects"] = gc.get_freeze_count()
        start = begin()
        collected = gc.collect(2)
        stop = begin()
        result.update(
            after_gc2_ms=(stop[0] - start[0]) / 1e6,
            after_gc2_cpu_ms=(stop[1] - start[1]) / 1e6,
            collected_after=collected,
        )
        return result
    finally:
        if not initial:
            gc.unfreeze()
        phase(old_phase)
        result["trace"] = flush()
