# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""SSE interval distributions; never synthesize individual MTP token timestamps."""

import json, math, statistics


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    pos = (len(values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def distribution(values):
    return dict(
        count=len(values),
        mean_ms=statistics.mean(values) if values else None,
        median_ms=percentile(values, 0.5),
        p90_ms=percentile(values, 0.9),
        p99_ms=percentile(values, 0.99),
        max_ms=max(values) if values else None,
        percentile_method="linear interpolation",
    )


def weighted_distribution(pairs):
    pairs = sorted(pairs)
    total = sum(w for _, w in pairs)

    def quantile(q):
        if not total:
            return None
        threshold = q * total
        seen = 0
        for value, weight in pairs:
            seen += weight
            if seen >= threshold:
                return value
        return pairs[-1][0]

    return dict(
        token_intervals=total,
        mean_ms=sum(value * weight for value, weight in pairs) / total if total else None,
        median_ms=quantile(0.5),
        p90_ms=quantile(0.9),
        p99_ms=quantile(0.99),
        max_ms=max((v for v, w in pairs), default=None),
        percentile_method="token-weighted nearest rank",
    )


def collect(records, count, folder):
    overlap_start = max(x[0][0] for x in records)
    overlap_end = min(x[-1][0] for x in records)
    rows = []
    all_batches = []
    all_amortized = []
    weighted = []
    stable_batches = []
    stable_amortized = []
    stable_weighted = []
    with (folder / "itl_intervals.jsonl").open("w") as output:
        for rid, events in enumerate(records):
            gaps = []
            amortized = []
            pairs = []
            stable_gaps = []
            stable_values = []
            stable_pairs = []
            buckets = {}
            for previous, current in zip(events, events[1:]):
                elapsed = (current[0] - previous[0]) / 1e6
                advance = current[2] - previous[2]
                if advance <= 0:
                    raise ValueError("Non-increasing output token index")
                value = elapsed / advance
                stable = (
                    128 <= previous[2]
                    and current[2] <= count - 129
                    and overlap_start <= previous[0]
                    and current[0] <= overlap_end
                )
                interval = dict(
                    request_index=rid,
                    from_token_index=previous[2],
                    to_token_index=current[2],
                    token_advance=advance,
                    sse_batch_gap_ms=elapsed,
                    amortized_interval_ms=value,
                    stable=stable,
                    context_tokens=32768 + current[2] + 1,
                )
                output.write(json.dumps(interval) + "\n")
                gaps.append(elapsed)
                amortized.append(value)
                pairs.append((value, advance))
                bucket = current[2] // 1024
                buckets.setdefault(bucket, []).append((value, advance))
                if stable:
                    stable_gaps.append(elapsed)
                    stable_values.append(value)
                    stable_pairs.append((value, advance))
            exact = all(n == 1 for _, n, _ in events)
            rows.append(
                dict(
                    request_index=rid,
                    exact_token_itl_available=exact,
                    sse_batch_gap=distribution(gaps),
                    amortized_sse_interval=distribution(amortized),
                    token_weighted_amortized_interval=weighted_distribution(pairs),
                    stable_sse_batch_gap=distribution(stable_gaps),
                    stable_amortized_sse_interval=distribution(stable_values),
                    stable_token_weighted_amortized_interval=weighted_distribution(stable_pairs),
                    output_1k_buckets={str(k): weighted_distribution(v) for k, v in buckets.items()},
                )
            )
            all_batches += gaps
            all_amortized += amortized
            weighted += pairs
            stable_batches += stable_gaps
            stable_amortized += stable_values
            stable_weighted += stable_pairs
    result = dict(
        metric_scope="client-observed SSE arrivals",
        device_token_generation_times_available=False,
        requests=rows,
        sse_batch_gap=distribution(all_batches),
        amortized_sse_interval=distribution(all_amortized),
        token_weighted_amortized_interval=weighted_distribution(weighted),
        stable_sse_batch_gap=distribution(stable_batches),
        stable_amortized_sse_interval=distribution(stable_amortized),
        stable_token_weighted_amortized_interval=weighted_distribution(stable_weighted),
        caveat="MTP can deliver multiple tokens per SSE event. Amortized intervals divide an observed gap by token advance; they are not individually measured token generation times.",
    )
    (folder / "itl_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
