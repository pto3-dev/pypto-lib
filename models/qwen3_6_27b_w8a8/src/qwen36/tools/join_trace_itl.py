# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Join exact verifier output IDs to SSE boundaries, without new device sampling."""

import argparse, collections, json, re, statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    a = parser.parse_args()
    result = {}
    for label in ["native", "pto"]:
        folder = a.run / label
        delivered = json.loads((folder / "measurement/token_sequences.json").read_text())
        data = max(
            (json.loads(f.read_text()) for f in (folder / "cpu_verifier_trace").glob("*.json")),
            key=lambda d: len(d["records"]),
        )
        lists = collections.defaultdict(list)
        events = collections.defaultdict(list)
        for row in data["records"]:
            for rid, ids in zip(row["req_ids"], row["sampled_token_ids"]):
                m = re.search(r"qwen36-frozen-(\d)", rid)
                if m and ids:
                    i = int(m.group(1))
                    lists[i].extend(ids)
                    events[i].append((row["monotonic_ns"], len(ids)))
        stamps = {}
        for i, ids in lists.items():
            actual = delivered[i]
            offsets = [
                o
                for o in range(7)
                if ids[: min(len(ids), len(actual) - o)] == actual[o : o + min(len(ids), len(actual) - o)]
            ]
            if not offsets:
                raise ValueError("CPU/SSE output ID sequences cannot be aligned")
            offset = min(offsets)
            count = offset
            for stamp, n in events[i]:
                count += n
                stamps[i, count - 1] = stamp
        sse = {}
        cumulative = [0] * 4
        for line in (folder / "measurement/events.jsonl").read_text().splitlines():
            row = json.loads(line)
            i = row["request_index"]
            ids = [x for c in row["response"].get("choices", []) for x in c.get("token_ids") or []]
            if ids:
                cumulative[i] += len(ids)
                sse[i, cumulative[i] - 1] = row["monotonic_ns"]
        joined = []
        misses = 0
        for line in (folder / "measurement/itl_intervals.jsonl").read_text().splitlines():
            row = json.loads(line)
            if not row["stable"]:
                continue
            i = row["request_index"]
            k0 = i, row["from_token_index"]
            k1 = i, row["to_token_index"]
            if k0 not in stamps or k1 not in stamps:
                misses += 1
                continue
            cpu_gap = (stamps[k1] - stamps[k0]) / 1e6
            sse_gap = row["sse_batch_gap_ms"]
            joined.append(
                dict(
                    **row,
                    cpu_verifier_gap_ms=cpu_gap,
                    delivery_gap_difference_ms=sse_gap - cpu_gap,
                    delivery_latency_before_ms=(sse[k0] - stamps[k0]) / 1e6,
                    delivery_latency_after_ms=(sse[k1] - stamps[k1]) / 1e6,
                )
            )
        ordered = sorted(joined, key=lambda d: d["sse_batch_gap_ms"], reverse=True)
        result[label] = dict(
            joined_stable_intervals=len(joined),
            unjoined_stable_intervals=misses,
            largest_sse_gaps=ordered[:8],
            mean_cpu_verifier_gap_ms=statistics.mean(r["cpu_verifier_gap_ms"] for r in joined),
            mean_delivery_gap_difference_ms=statistics.mean(r["delivery_gap_difference_ms"] for r in joined),
            scope="Host verifier materialization to client SSE delivery; CPU gap can include device wait, sampling, drafter, scheduling or host descheduling. No sub-kernel attribution from this trace alone.",
        )
    (a.run / "cpu_sse_itl_attribution.json").write_text(json.dumps(result, indent=2) + "\n")
    for label, r in result.items():
        print(
            label,
            "joined",
            r["joined_stable_intervals"],
            "missed",
            r["unjoined_stable_intervals"],
            "largest",
            json.dumps(r["largest_sse_gaps"][:2]),
        )


if __name__ == "__main__":
    main()
