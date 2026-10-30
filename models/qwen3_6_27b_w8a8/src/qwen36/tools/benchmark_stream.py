# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Replay four independent HTTP requests with frozen token IDs.

This is a trace-preserving replay client, not the official bench serve CLI.
No token timestamps are interpolated inside an MTP/SSE output batch.
"""

import argparse, concurrent.futures, hashlib, json, statistics, threading, time, urllib.request
from pathlib import Path
from qwen36.tools.itl_metrics import collect


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.request.read_bytes()
    template = json.loads(raw)
    prompts = template.pop("prompt")
    count = template["max_tokens"]
    if len(prompts) != 4 or not all(isinstance(x, list) and len(x) == 32768 for x in prompts):
        raise ValueError("Four frozen 32768-token prompts required")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "request_template.json").write_bytes(raw)
    barrier = threading.Barrier(4)
    lock = threading.Lock()
    records = [[] for _ in prompts]
    sequences = [[] for _ in prompts]
    usages = [None] * 4
    starts = [None] * 4
    finishes = [None] * 4
    with (args.output / "events.jsonl").open("w", buffering=1) as log:

        def run(index):
            body = dict(template, prompt=prompts[index])
            body["stream_options"] = dict(body.get("stream_options") or {}, include_usage=True)
            request_bytes = json.dumps(body, separators=(",", ":")).encode()
            (args.output / f"request_{index}.json").write_bytes(request_bytes)
            request = urllib.request.Request(
                args.url,
                data=request_bytes,
                headers={"Content-Type": "application/json", "X-Request-Id": f"qwen36-frozen-{index}"},
            )
            barrier.wait(timeout=30)
            starts[index] = time.monotonic_ns()
            with urllib.request.urlopen(request, timeout=1200) as response:
                for line in response:
                    stamp = time.monotonic_ns()
                    if not line.startswith(b"data: "):
                        continue
                    payload = line[6:].strip()
                    if payload == b"[DONE]":
                        break
                    event = json.loads(payload)
                    if "error" in event:
                        raise RuntimeError(event)
                    with lock:
                        log.write(
                            json.dumps(
                                dict(request_index=index, monotonic_ns=stamp, response=event),
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                    if event.get("usage"):
                        usages[index] = event["usage"]
                    for choice in event.get("choices", []):
                        if choice["index"] != 0:
                            raise ValueError("Expected one independent completion per request")
                        ids = choice.get("token_ids") or []
                        if ids:
                            sequences[index].extend(ids)
                            records[index].append((stamp, len(ids), len(sequences[index]) - 1))
            finishes[index] = time.monotonic_ns()

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(run, i) for i in range(4)]
            for future in futures:
                future.result()
    if list(map(len, sequences)) != [count] * 4:
        raise ValueError(dict(output_counts=list(map(len, sequences))))
    (args.output / "token_sequences.json").write_text(json.dumps(sequences, separators=(",", ":")) + "\n")
    for usage in usages:
        if usage is not None and (usage["prompt_tokens"] != 32768 or usage["completion_tokens"] != count):
            raise ValueError(usage)
    rows = []
    for index, events in enumerate(records):
        rows.append(
            dict(
                request_index=index,
                input_tokens=32768,
                output_tokens=count,
                start_monotonic_ns=starts[index],
                first_output_monotonic_ns=events[0][0],
                last_output_monotonic_ns=events[-1][0],
                http_finish_monotonic_ns=finishes[index],
                ttft_ms=(events[0][0] - starts[index]) / 1e6,
                whole_decode_tpot_ms=(events[-1][0] - events[0][0]) / 1e6 / (count - 1),
                sse_output_batches=len(events),
                coalesced_events=sum(n > 1 for _, n, _ in events),
                input_token_ids_sha256=hashlib.sha256(
                    json.dumps(prompts[index], separators=(",", ":")).encode()
                ).hexdigest(),
                output_token_ids_sha256=hashlib.sha256(
                    json.dumps(sequences[index], separators=(",", ":")).encode()
                ).hexdigest(),
            )
        )
    result = dict(
        client="four independent HTTP requests; frozen official-generator input token IDs",
        requests=rows,
        request_sha256=hashlib.sha256(raw).hexdigest(),
        usage=usages,
        usage_reported=[u is not None for u in usages],
        input_tokens_each=32768,
        output_tokens_each=count,
        average_tpot_ms=statistics.mean(x["whole_decode_tpot_ms"] for x in rows),
        ttft_ms=[x["ttft_ms"] for x in rows],
        exact_generation_itl_available=False,
        timestamp_note="Client arrival timestamps of actual SSE batches; never per-token device generation timestamps",
    )
    result["itl_distributions"] = collect(records, count, args.output)
    (args.output / "token_sequences.json").write_text(json.dumps(sequences, separators=(",", ":")) + "\n")
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print("COMPLETE", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
