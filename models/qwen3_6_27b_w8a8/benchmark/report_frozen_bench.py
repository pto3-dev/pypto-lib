# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Join official client ITLs with pass-through server SSE IDs; never interpolate token timestamps."""

import argparse, hashlib, json
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer
from qwen36.tools.itl_metrics import collect


def counter(path, name, pos=None):
    vals = [
        float(l.split()[-1])
        for l in path.read_text().splitlines()
        if l.startswith(name + "{") and (pos is None or f'position="{pos}"' in l)
    ]
    if len(vals) != 1:
        raise ValueError((path, name, pos, vals))
    return vals[0]


def mtp(folder):
    def d(name, pos=None):
        return counter(folder / "metrics_after.prom", name, pos) - counter(
            folder / "metrics_before.prom", name, pos
        )

    n = d("vllm:spec_decode_num_drafts_total")
    tokens = d("vllm:spec_decode_num_draft_tokens_total")
    accepted = d("vllm:spec_decode_num_accepted_tokens_total")
    pos = [d("vllm:spec_decode_num_accepted_tokens_per_pos_total", i) for i in range(5)]
    assert accepted == sum(pos) and n > 0
    hist = [n - pos[0]] + [pos[i] - pos[i + 1] for i in range(4)] + [pos[-1]]
    return dict(
        drafts=n,
        draft_tokens=tokens,
        accepted_draft_tokens=accepted,
        acceptance_percent=100 * accepted / tokens,
        mean_acceptance_length_including_bonus=1 + accepted / n,
        accepted_prefix_histogram={str(i): int(v) for i, v in enumerate(hist)},
    )


def sse_events(path):
    buf = ""
    for line in path.read_text().splitlines():
        entry = json.loads(line)
        buf += entry["body"]
        while "\n\n" in buf:
            block, buf = buf.split("\n\n", 1)
            for part in block.splitlines():
                if part.startswith("data: ") and part[6:] != "[DONE]":
                    yield entry["monotonic_ns"], json.loads(part[6:])
    if buf.strip():
        raise ValueError("Truncated SSE capture")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--native-baseline", type=Path)
    a = ap.parse_args()
    root = a.root
    fixture = json.loads((root / "dataset_manifest.json").read_text())
    model = Path("/home/g00893677/pto-dev/qwen3_6_27b-inner-w8a8/checkpoints/public-inner-surrogate")
    tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
    mapping = {r["prompt_sha256"]: r["request_index"] for r in fixture["requests"]}
    result = dict(
        scope="Public inner-format surrogate, frozen dataset "
        + fixture["generator"]
        + ", TP4/B4/32768in/10240out/MTP5",
        official_client_unmodified=True,
        runs={},
    )
    if a.native_baseline:
        retained = json.loads((a.native_baseline / "comparison.json").read_text())
        result["runs"]["native"] = retained["runs"]["native"]
        result["comparison_design"] = "current PTO versus retained native, not a freshly interleaved pair"
        result["native_reference_root"] = str(a.native_baseline.resolve())
    for label in ["pto"] if a.native_baseline else ["native", "pto"]:
        folder = root / label
        status = json.loads((folder / "status.json").read_text())
        assert status["status"] == "complete"
        official = json.loads((folder / "official_result.json").read_text())
        out = folder / "measurement"
        out.mkdir(exist_ok=True)
        records = [[] for _ in range(4)]
        sequences = [[] for _ in range(4)]
        inputs = [None] * 4
        starts = [None] * 4
        usages = [None] * 4
        requests = [None] * 4
        choice_sizes = [[] for _ in range(4)]
        server_text = [""] * 4
        with (out / "events.jsonl").open("w") as log:
            for request_file in sorted((folder / "request_audit").glob("request_*.json")):
                request = json.loads(request_file.read_text())
                body = request["body"]
                prompt = body["prompt"]
                i = mapping[hashlib.sha256(prompt.encode()).hexdigest()]
                ids = tok(prompt).input_ids
                assert ids == fixture["requests"][i]["input_token_ids"]
                inputs[i] = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
                starts[i] = request["start_monotonic_ns"]
                requests[i] = body
                assert (
                    body["max_tokens"] == 10240
                    and body["ignore_eos"]
                    and body["seed"] == 1024
                    and body["temperature"] == 1.0
                    and body["top_k"] == 20
                    and body["top_p"] == 0.95
                )
                (out / f"request_{i}.json").write_text(
                    json.dumps(dict(body, input_token_ids=ids), ensure_ascii=False) + "\n"
                )
                for stamp, event in sse_events(
                    request_file.with_name("sse_" + request_file.stem.split("_")[-1] + ".jsonl")
                ):
                    log.write(
                        json.dumps(
                            dict(
                                request_index=i,
                                monotonic_ns=stamp,
                                response=event,
                                timestamp_origin="server ASGI send",
                            ),
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    if event.get("usage"):
                        usages[i] = event["usage"]
                    for choice in event.get("choices", []):
                        assert choice["index"] == 0
                        new = choice.get("token_ids") or []
                        choice_sizes[i].append(len(new))
                        server_text[i] += choice.get("text") or ""
                        if new:
                            sequences[i].extend(new)
                            records[i].append((stamp, len(new), len(sequences[i]) - 1))
        assert list(map(len, sequences)) == [10240] * 4
        assert all(u["prompt_tokens"] == 32768 and u["completion_tokens"] == 10240 for u in usages)
        (out / "token_sequences.json").write_text(json.dumps(sequences, separators=(",", ":")) + "\n")
        itl = collect(records, 10240, out)
        client_records = [[] for _ in range(4)]
        with (out / "client_receive_events.jsonl").open("w") as joined:
            for i in range(4):
                assert len(choice_sizes[i]) == len(official["itls"][i]) + 1, (i, "SSE message count mismatch")
                assert server_text[i] == official["generated_texts"][i], (i, "SSE text mismatch")
                elapsed = official["ttfts"][i]
                cursor = 0
                for n, size in enumerate(choice_sizes[i]):
                    if n:
                        elapsed += official["itls"][i][n - 1]
                    stamp = round((official["start_times"][i] + elapsed) * 1e9)
                    batch = sequences[i][cursor : cursor + size]
                    cursor += size
                    joined.write(
                        json.dumps(
                            dict(
                                request_index=i,
                                choice_index=n,
                                monotonic_ns=stamp,
                                token_ids=batch,
                                timestamp_origin="official client measured start+TTFT+cumulative SSE ITLs; one timestamp per actual received choice, no token interpolation",
                            )
                        )
                        + "\n"
                    )
                    if size:
                        client_records[i].append((stamp, size, cursor - 1))
                assert cursor == 10240
        client_dir = out / "client_receive"
        client_dir.mkdir(exist_ok=True)
        client_itl = collect(client_records, 10240, client_dir)
        # Official client arrays follow its fixed custom dataset order, matching fixture order.
        repeated = []
        for seq in sequences:
            grams = Counter(tuple(seq[j : j + 4]) for j in range(len(seq) - 3))
            tail = seq[-1024:]
            repeated.append(
                dict(
                    unique_4gram_ratio=len(grams) / (len(seq) - 3),
                    most_common_4gram_count=max(grams.values()),
                    tail1024_unique_tokens=len(set(tail)),
                )
            )
        summary = dict(
            average_tpot_ms=official["mean_tpot_ms"],
            ttft_ms=[v * 1000 for v in official["ttfts"]],
            official_client=dict(
                mean_itl_ms=official["mean_itl_ms"],
                median_itl_ms=official["median_itl_ms"],
                p90_itl_ms=official.get("p90_itl_ms"),
                p99_itl_ms=official.get("p99_itl_ms"),
                metric_note="Official client ITL is received SSE choice-chunk gap, not individual-token device time",
            ),
            client_receive_itl={k: v for k, v in client_itl.items() if k not in ["requests", "caveat"]},
            server_send_itl={k: v for k, v in itl.items() if k not in ["requests", "caveat"]},
            server_send_ttft_ms=[(records[i][0][0] - starts[i]) / 1e6 for i in range(4)],
            mtp=mtp(folder),
            input_token_sha256=inputs,
            output_token_sha256=[
                hashlib.sha256(json.dumps(x, separators=(",", ":")).encode()).hexdigest() for x in sequences
            ],
            output_repetition=repeated,
            worker_stats=json.loads((folder / "worker_stats.json").read_text()),
            raw=str(folder),
            timestamp_note="Client batch timestamps reconstructed from official measured start/TTFT/SSE ITLs and joined by exact message count/order/text to server token IDs; no token interpolation. Server send timestamps separately retained",
        )
        if label == "native":
            assert all(w.get("pto_executor") is False for w in summary["worker_stats"])
        else:
            assert all(
                w["topology"]["native_pid"] == w["topology"]["runtime_pid"]
                and not w["external_rpc_per_decode"]
                and w["graph_captures"] == w["graph_replays"] == 0
                for w in summary["worker_stats"]
            )
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        result["runs"][label] = summary
    left, right = result["runs"]["native"], result["runs"]["pto"]
    assert left["input_token_sha256"] == right["input_token_sha256"]
    aseq = json.loads(((a.native_baseline or root) / "native/measurement/token_sequences.json").read_text())
    bseq = json.loads((root / "pto/measurement/token_sequences.json").read_text())
    result["outputs"] = [
        dict(
            request_index=i,
            equal=aseq[i] == bseq[i],
            first_different_index=next((j for j, (a, b) in enumerate(zip(aseq[i], bseq[i])) if a != b), None),
        )
        for i in range(4)
    ]
    result["observed_tpot_improvement_percent"] = 100 * (
        1 - right["average_tpot_ms"] / left["average_tpot_ms"]
    )
    result["causal_limit"] = (
        "Same prompts/sampling do not close long-output numerical divergence. Acceptance changes remain a confounder; serving speedup is not isolated kernel speedup."
    )
    result["dataset_representativeness"] = fixture.get("representativeness")
    result["dataset_generator"] = fixture["generator"]
    status = json.loads((root / "pto/status.json").read_text())
    cmd = status["server_command"]
    cfg = json.loads(cmd[cmd.index("--additional-config") + 1])["qwen38_pto_mpmd"]
    result["actual_pto_implementation"] = dict(
        devices=status["devices"],
        build_root=cfg["build_root"],
        execution=cfg["execution"],
        compact_inputs=cfg.get("compact_inputs", "off"),
        kernel_pair_build=cfg.get("kernel_pair_build"),
        worker=cmd[cmd.index("--worker-extension-cls") + 1],
    )
    (root / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# 冻结输入：native/PTO MTP5配对",
        "",
        "公开inner格式surrogate，TP4/B4，每请求32768输入、10240输出；输入生成一次并冻结，再由未修改的vllm bench serve custom入口读取；构造方法和来源见dataset_manifest.json。不是私有生产权重/真实业务输入。",
        "",
        "| 指标 | native | PTO |",
        "|---|---:|---:|",
    ]

    def row(title, fn):
        lines.append(f"|{title}|{fn(left):.4f}|{fn(right):.4f}|")

    row("全程平均TPOT ms", lambda r: r["average_tpot_ms"])
    row("官方客户端SSE间隔中位数 ms", lambda r: r["official_client"]["median_itl_ms"])
    for title, key in [
        ("稳定token加权摊销均值", "mean_ms"),
        ("稳定token加权摊销中位数", "median_ms"),
        ("稳定token加权摊销P90", "p90_ms"),
        ("稳定token加权摊销P99", "p99_ms"),
        ("稳定token加权摊销最大值", "max_ms"),
    ]:
        row(
            title + " ms（客户端接收时间）",
            lambda r, key=key: r["client_receive_itl"]["stable_token_weighted_amortized_interval"][key],
        )
    row("MTP接受率 %", lambda r: r["mtp"]["acceptance_percent"])
    row("平均接受长度（含bonus）", lambda r: r["mtp"]["mean_acceptance_length_including_bonus"])
    lines += ["", "| 请求 | native TTFT秒 | PTO TTFT秒 | 首次输出分叉索引 |", "|---|---:|---:|---:|"]
    for i in range(4):
        lines.append(
            f"|{i}|{left['ttft_ms'][i] / 1000:.4f}|{right['ttft_ms'][i] / 1000:.4f}|{result['outputs'][i]['first_different_index']}|"
        )
    lines += [
        "",
        "完整输入ID/输出ID、原始SSE服务器发送时间戳、客户端官方ITL数组、接受长度分布和逐请求重复度见comparison.json及各侧measurement/request_audit。客户端批次时间由官方实测start/TTFT/ITL累加恢复，与服务器SSE消息数量、顺序、文本逐项校验后关联token ID；没有插值逐token。服务器发送间隔另存，不与客户端接收间隔混用。稳定窗口沿用四请求公共输出时间及排除头128/尾129token的规则。",
        "",
        "相同输入尚不代表数值一致；若输出分叉或接受长度不一致，不把端到端差值全部归因于kernel。本次实际卡组、target编译产物、worker和输入准备配置见comparison.json的actual_pto_implementation；不同卡组及接受轨迹变化均不得算作独立kernel收益。",
    ]
    if a.native_baseline:
        lines += [
            "",
            "原生数据复用既有正式参照："
            + str(a.native_baseline.resolve())
            + "。本轮仅刷新PTO；不是新采的原生/PTO交错配对。",
        ]
    (root / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            dict(
                metrics={
                    k: {
                        "tpot_ms": v["average_tpot_ms"],
                        "client_sse_median_ms": v["official_client"]["median_itl_ms"],
                        "stable_amortized_ms": v["client_receive_itl"][
                            "stable_token_weighted_amortized_interval"
                        ]["mean_ms"],
                        "acceptance_length": v["mtp"]["mean_acceptance_length_including_bonus"],
                    }
                    for k, v in result["runs"].items()
                },
                outputs=result["outputs"],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
