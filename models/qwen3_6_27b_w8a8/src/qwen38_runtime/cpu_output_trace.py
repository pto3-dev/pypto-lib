# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Opt-in verifier emission trace, after vLLM's existing asynchronous host copy.

No extra tensor .cpu(), .tolist(), stream wait or device event is introduced.
The CPU list reports verifier output lengths, not SSE batches; final scheduler
truncation and cancellation can make these differ from delivered token counts.
"""

import atexit, json, os, time
from pathlib import Path

_RECORDS = []
_LIMIT = 100000


def flush():
    output = os.environ.get("QWEN36_CPU_MTP_TRACE")
    if not output:
        return None
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    file = folder / f"cpu_verifier_output_{os.getpid()}.json"
    file.write_text(
        json.dumps(
            dict(
                pid=os.getpid(),
                scope="CPU verifier outputs before scheduler truncation; NOT per-token device times",
                records=_RECORDS,
            ),
            separators=(",", ":"),
        )
        + "\n"
    )
    return str(file)


def install():
    output = os.environ.get("QWEN36_CPU_MTP_TRACE")
    if not output:
        return False
    from vllm.v1.worker.gpu_model_runner import AsyncGPUModelRunnerOutput

    cls = AsyncGPUModelRunnerOutput
    if getattr(cls, "_qwen36_cpu_trace", False):
        return True
    original = cls.get_output

    def traced(self):
        width = self.sampled_token_ids_cpu.shape[-1]
        value = original(self)
        if width > 1:
            rows = value.sampled_token_ids
            if not isinstance(rows, list) or any(not isinstance(r, list) for r in rows):
                raise TypeError("Expected already materialized CPU lists")
            if len(_RECORDS) >= _LIMIT:
                raise RuntimeError("CPU MTP trace limit reached; refuse incomplete evidence")
            _RECORDS.append(
                dict(
                    monotonic_ns=time.monotonic_ns(),
                    req_ids=list(value.req_ids),
                    sampled_token_ids=[list(row) for row in rows],
                    valid_output_lengths=[len(row) for row in rows],
                    padded_output_width=int(width),
                )
            )
            tag = getattr(self, "_compact_submission_tag", None)
            if tag is not None:
                _RECORDS[-1]["compact_submission"] = tag
                if len(_RECORDS) >= 1000 and not getattr(traced, "_compact_profile_ready", False):
                    import torch.distributed as dist

                    if dist.get_rank() == 0 and len(value.req_ids) == 4 and all(len(r) > 0 for r in rows):
                        traced._compact_profile_ready = True
                        target = Path(output).parent / "compact_profile_ready.json"
                        target.write_text(json.dumps(dict(rows=len(_RECORDS), submission=tag)) + "\n")
        return value

    cls.get_output = traced
    cls._qwen36_cpu_trace = True
    atexit.register(flush)
    return True
