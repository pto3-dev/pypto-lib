# Complete serving test entry

Run `bash e2e_pypto_vllm_910b4.sh help` inside the existing CANN9.1/vLLM0.23
execution container. The entry composes the frozen launch command, existing
chat client, official bench, a custom SSE window client and outer CANN capture.
It never edits vLLM, the model or generated kernels, and does not use HBG.

## Explicit configuration

```bash
export SERVER_COMMAND_FILE=/existing/validated/status.json
export BACKEND=pto
export MODEL=/checkpoint
export CORPUS=/frozen-corpus/zh_qa.jsonl
export PTO_TARGET_SOURCE=/frozen-attachment
export ASCEND_RT_VISIBLE_DEVICES=1,3,6,7
export BASE_URL=http://127.0.0.1:8253
export RUN_DIR=/new/run-directory
bash e2e_pypto_vllm_910b4.sh env
bash e2e_pypto_vllm_910b4.sh check
bash e2e_pypto_vllm_910b4.sh run
```

`SERVER_COMMAND_FILE` is a saved validated status JSON with `server_command`,
or its literal JSON argv list. The selected environment launcher and all
referenced source/build/runtime paths must still exist. The entry rejects
protocol drift rather than replacing the launcher or reconstructing its imports.
`PTO_TARGET_SOURCE` is the explicit frozen attachment directory containing
`src/qwen38_runtime`; the already validated environment script consumes it.
For native, use `BACKEND=native` and the saved native command. Both comparisons
must select the same idle four-card group; the shown card IDs are examples,
not a fixed group. No process belonging to another job is stopped.

The frozen protocol is TP4/PP1, B4, input32768/output10240 per request, MTP5,
max-model-len45056, memory0.65, KV cache6GiB, prefill chunk2048, mamba-cache align,
NZ enabled and the263-INT8-matrix mixed checkpoint. PTO target is mode0/prepared
AICPU DIRECT; native target is mode3/FULL. Both use the same native mode3 MTP
drafter, asynchronous scheduling, padding and sampling. The entry validates
these settings and changes only the service port/host, result session directory
and official profiler output configuration. Profiling is inactive during bench.

## 6.3.3 Conversation test

Entry: [e2e_pypto_vllm_910b4.sh](e2e_pypto_vllm_910b4.sh), `qa` or `run`.
Client: [chat_qa.py](chat_qa.py). Existing-service use:

```bash
export RUN_DIR=/new/chat-results
bash e2e_pypto_vllm_910b4.sh qa
```

Four concurrent two-turn conversations extract records into specified JSON,
then use actual conversation history to calculate remaining work and budget.
The executed client previously passed8/8 requests and32/32 fields; its function
bodies are retained. [Recorded responses](CHAT_RESULT.md) belong to that source
validation. They do not certify a fresh device run of the new combined shell entry.
The donor's long-context single-turn QA is a different case; it is not claimed
as the same functional workload.

## 6.3.4 Stress tests

Custom measurement: [e2e_vllm_window.py](e2e_vllm_window.py), called with `window`:

```bash
bash e2e_pypto_vllm_910b4.sh window /corpus/zh_qa.jsonl /new/window-results
```

It checks the frozen corpus manifest, issues four barrier-aligned SSE requests,
saves complete requests and raw message payloads/token IDs with real arrival
stamps, samples Prometheus counters every0.5s, and calculates common-B4-window
client/server token-amortized time and acceptance. An SSE batch can contain
several tokens; individual token arrival times are not fabricated. It saves
`requests.json`, `metrics-samples.json`, raw before/after metrics and `result.json`.

Official benchmark: [e2e_pypto_vllm_profile.py](e2e_pypto_vllm_profile.py), `bench`:

```bash
export RUN_DIR=/new/official-results
bash e2e_pypto_vllm_910b4.sh bench
```

This invokes actual `vllm bench serve`, including save-result/save-detailed,
explicit sampling and TTFT/TPOT/ITL/E2EL percentiles. Use existing zh_qa,
en_technical and code_review custom corpora; do not relabel a random dataset
as one of these scenes. Exact input token IDs, corpus hash, command and complete
raw official result are retained. The older `bench_serve.py` remains a standalone
convenience client; it is not the canonical full workflow or a historical-run original.

## Outer timeline collection and statistics

The same [profile script](e2e_pypto_vllm_profile.py) supplies `capture` and `analyse`.
Existing-service capture requires an explicit raw profiler directory from that
service's configuration; capture does not stop the service:

```bash
export RUN_DIR=/new/diagnostic-results
bash e2e_pypto_vllm_910b4.sh capture --raw /service/profiler
```

It waits for all prefills to finish with four active requests, starts the official
profiler over HTTP, records a bounded50-step diagnostic window, stops it and
cancels only its own diagnostic client. This request is labelled cancelled and
never represented as a completed10K measurement. Raw four-rank profiles are kept;
`torch_npu.profiler.profiler.analyse` runs offline outside daemon workers.
Outputs include complete `trace_view.json`, `kernel_details.csv` and `op_statistic.csv`.
Open the unmodified trace in Perfetto; this is the outer vLLM/CANN timeline.

`run` owns one server: QA ->32-token warmup -> unprofiled official10K -> separate
diagnostic capture -> own-server cleanup -> offline export. `serve` starts in
foreground; `qa`, `bench`, `window`, `capture` and `analyse` work independently.
Builds remain separate normal DSL/AICPU compilations; no build or kernel
replacement is hidden in these tests.

For numerical timeline statistics, use `timeline` and the existing
[analyze_outer_trace.py](analyze_outer_trace.py):

```bash
bash e2e_pypto_vllm_910b4.sh timeline --backend pto --output /new/timeline.json \
  --rank-csv /rank0/kernel_details.csv /rank1/kernel_details.csv \
  /rank2/kernel_details.csv /rank3/kernel_details.csv
```

Verify is `matched_full_verify`, the complete target including embedding,
head and logits gather. PTO uses the union of concurrent AICPU/persistent AICore
spans. Decode is `mtp_start_alternate`, native five-step drafter graph start to
next graph start. Rejection-to-rejection timing remains a separately labelled
field. Original rows, graph/task/stream IDs and raw files remain available.
Unknown kernel naming or incomplete B4 boundaries fail explicitly.

## Publication validation and provenance

These completed entries consolidate the existing measured responsibilities;
they are new integration entries, not byte-identical historical scripts.
The original scripts and execution-chain hashes remain in the local archival
manifest. Publication checks cover syntax, frozen-command preservation,
SSE token/timestamp retention and existing boundary-reader agreement. This
change does not claim a fresh NPU run or refreshed10K performance numbers.
