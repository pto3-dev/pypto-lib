# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Read-only paired result reconciliation, exact counter survival histogram."""

import argparse, collections, json, re
from pathlib import Path


def prom(path):
    result = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        key, value = line.rsplit(None, 1)
        result[key] = float(value)
    return result


def mtp(folder):
    before = prom(folder / "metrics_before.prom")
    after = prom(folder / "metrics_after.prom")
    diff = {k: v - before.get(k, 0) for k, v in after.items()}

    def total(name):
        return sum(v for k, v in diff.items() if k.split("{")[0] == name)

    drafts = total("vllm:spec_decode_num_drafts_total")
    tokens = total("vllm:spec_decode_num_draft_tokens_total")
    accepted = total("vllm:spec_decode_num_accepted_tokens_total")
    positions = [
        sum(
            v
            for k, v in diff.items()
            if k.startswith("vllm:spec_decode_num_accepted_tokens_per_pos_total{") and f'position="{i}"' in k
        )
        for i in range(5)
    ]
    if any(a < b for a, b in zip([drafts, *positions], positions)) or abs(sum(positions) - accepted) > 0.01:
        raise ValueError("Invalid acceptance survival counters")
    histogram = {str(i): int(a - b) for i, (a, b) in enumerate(zip([drafts, *positions], [*positions, 0]))}
    return dict(
        draft_rounds=drafts,
        drafted_tokens=tokens,
        accepted_draft_tokens=accepted,
        acceptance_rate=accepted / tokens if tokens else None,
        mean_accepted_drafts=accepted / drafts if drafts else None,
        mean_acceptance_length_with_bonus=1 + accepted / drafts if drafts else None,
        accepted_draft_count_histogram=histogram,
        acceptance_length_with_bonus_histogram={str(int(k) + 1): v for k, v in histogram.items()},
        per_position_accepted_tokens=positions,
        all_rounds_have_five_drafts=abs(tokens - 5 * drafts) < 0.01,
        preemptions=total("vllm:num_preemptions_total"),
        histogram_method="prefix-survival counter differences D-c0,c0-c1,...,c4; accepted drafts exclude bonus; verifier scheduler truncation may reduce delivered tokens",
    )


def repetition(sequence):
    four = [tuple(sequence[i : i + 4]) for i in range(len(sequence) - 3)]
    counts = collections.Counter(sequence)
    return dict(
        tokens=len(sequence),
        unique_tokens=len(counts),
        distinct_token_fraction=len(counts) / len(sequence),
        unique_4gram_fraction=len(set(four)) / len(four),
        most_frequent_token=counts.most_common(1)[0][0],
        most_frequent_fraction=counts.most_common(1)[0][1] / len(sequence),
    )


def trace(folder, sequences):
    files = []
    for path in sorted((folder / "cpu_verifier_trace").glob("*.json")):
        data = json.loads(path.read_text())
        rows = data["records"]
        emitted = collections.defaultdict(list)
        hist = collections.Counter()
        stable = collections.Counter()
        for r in rows:
            for rid, ids in zip(r["req_ids"], r["sampled_token_ids"]):
                if not ids:
                    continue
                m = re.search(r"qwen36-frozen-(\d)", rid)
                if m is None:
                    continue
                i = int(m.group(1))
                start = len(emitted[i])
                emitted[i].extend(ids)
                hist[len(ids)] += 1
                if start >= 128 and len(emitted[i]) <= 10240 - 128:
                    stable[len(ids)] += 1
        evidence = {}
        for i, ids in emitted.items():
            actual = sequences[i]
            offsets = [
                offset
                for offset in range(7)
                if ids[: min(len(ids), len(actual) - offset)]
                == actual[offset : offset + min(len(ids), len(actual) - offset)]
            ]
            offset = min(offsets) if offsets else None
            evidence[str(i)] = dict(
                emitted_cpu_tokens=len(ids),
                delivered_sse_tokens=len(actual),
                covered_token_ids_match=offset is not None,
                initial_untraced_tokens=offset,
                final_untraced_tokens=max(0, len(actual) - offset - len(ids)) if offset is not None else None,
                extra_after_scheduler_truncation=max(0, len(ids) - (len(actual) - offset))
                if offset is not None
                else None,
                note="Initial pure-prefill or final singleton output can have width1 and is intentionally outside the MTP trace; all output IDs remain in SSE evidence",
            )
        files.append(
            dict(
                file=str(path),
                records=len(rows),
                request_prefix_validation=evidence,
                verifier_output_length_histogram=dict(hist),
                stable_verifier_output_length_histogram=dict(stable),
            )
        )
    return dict(
        files=files,
        scope="CPU verifier output lengths before scheduler truncation, includes bonus; not SSE batch distribution",
        nonempty_files=sum(bool(f["records"]) for f in files),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    folder = a.run
    status = json.loads((folder / "status.json").read_text())
    if status["status"] != "complete":
        raise ValueError("Pair is not complete")
    sides = {}
    outputs = {}
    bodies = {}
    for label in ["native", "pto"]:
        f = folder / label
        s = json.loads((f / "measurement/summary.json").read_text())
        sequences = json.loads((f / "measurement/token_sequences.json").read_text())
        outputs[label] = sequences
        audit = {}
        for path in (f / "request_audit").glob("request_*.json"):
            r = json.loads(path.read_text())
            audit[r["request_id"]] = r["body"]
        if set(audit) != {f"qwen36-frozen-{i}" for i in range(4)}:
            raise ValueError("Request audit incomplete")
        bodies[label] = audit
        sides[label] = dict(
            average_tpot_ms=s["average_tpot_ms"],
            ttft_ms=s["ttft_ms"],
            requests=s["requests"],
            itl=s["itl_distributions"],
            mtp=mtp(f),
            output_repetition=[repetition(seq) for seq in sequences],
            cpu_verifier_trace=trace(f, sequences),
            worker_stats=str(f / "worker_stats.json"),
            source_snapshot=status["source_snapshot"],
        )
    if bodies["native"] != bodies["pto"]:
        raise ValueError("Actual server request bodies differ")
    divergent = []
    for i, (x, y) in enumerate(zip(outputs["native"], outputs["pto"])):
        first = next((j for j, (v, z) in enumerate(zip(x, y)) if v != z), None)
        divergent.append(
            dict(
                request_index=i,
                first_different_output_index=first,
                bitwise_output_ids_equal=first is None and len(x) == len(y),
            )
        )
    result = dict(
        status="complete",
        devices=status["devices"],
        requests_identical=True,
        checkpoint="public-inner-surrogate; not private equivalent",
        scenario="TP4/PP1/B4 MTP5,32768 input +10240 output each",
        inputs="four distinct contiguous real-text excerpts via official CustomDataset; exact IDs frozen; not production traffic",
        sides=sides,
        output_divergence=divergent,
        scope="single native/PTO pair, client SSE arrivals; no device token times; stochastic free-generation is not numerical acceptance",
        formal_itl_improvement_percent=(
            1
            - sides["pto"]["itl"]["stable_token_weighted_amortized_interval"]["mean_ms"]
            / sides["native"]["itl"]["stable_token_weighted_amortized_interval"]["mean_ms"]
        )
        * 100,
    )
    (folder / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print("PAIR_SUMMARY", json.dumps({k: v for k, v in result.items() if k != "sides"}), flush=True)
    for label, s in sides.items():
        print(
            label,
            json.dumps(
                dict(
                    tpot_ms=s["average_tpot_ms"],
                    ttft_ms=s["ttft_ms"],
                    stable=s["itl"]["stable_token_weighted_amortized_interval"],
                    sse=s["itl"]["sse_batch_gap"],
                    mtp=s["mtp"],
                    repetition=s["output_repetition"],
                    trace_nonempty=s["cpu_verifier_trace"]["nonempty_files"],
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
