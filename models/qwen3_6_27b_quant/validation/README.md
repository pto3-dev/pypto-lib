# Qwen3.6 mixed-W8A8 serving validation

Run inside the installed CANN9.1/vLLM0.23 environment. Configure paths once in
`serving.local.sh`, using [serving.local.example.sh](serving.local.example.sh)
on a new machine. This local file is ignored by Git. Select four idle cards;
startup checks their process table before launching.

## Commands

Start the service in one terminal:

```bash
bash serve_vllm_910b4.sh pto
```

In a second terminal, select one task:

```bash
# Four exact32768-token context-extraction requests.
bash e2e_pypto_vllm_910b4.sh pto

# Official vLLM bench serve: four requests,32768 input/10240 output tokens each.
bash bench_vllm_910b4.sh pto

# Separate outer capture, four-rank export and Verify/Decode statistics.
bash profile_vllm_910b4.sh pto
```

Replace `pto` with `native` for the native service and clients. The client scripts
accept `zh_qa` (default), `en_technical` or `code_review` as a second argument.
The long-context QA contract requires the record corpus and uses `zh_qa`.
There are no per-command exports or manually entered profiler/CSV paths.

Clients use the `service.latest` session recorded by startup, validate the
backend, checkpoint, endpoint and protocol, wait for readiness and reject
concurrent clients or an already busy service. They do not start or stop servers.
Startup stays in the foreground; stop it from its owning terminal. Each task
creates a fresh result directory under the configured `RESULTS_ROOT`.

## Configuration and protocol

[serving_config.sh](serving_config.sh) loads the single machine configuration.
Set checkpoint, corpus root, PTO/native environment wrappers, compiled target,
Simpler runtime, physical weight reference, card group and output root there.
`VLLM_LAUNCHER` is the selected executable environment wrapper which accepts
Python arguments; it is not the `vllm` binary. Startup retains visible vLLM flags
in [serve_vllm_910b4.sh](serve_vllm_910b4.sh). Use its `--print-command` option to
inspect effective argv without NPU access.

TP4/PP1, B4, MTP5; max-model-len45056, memory0.65, KV cache6GiB, chunk2048,
mamba-cache align, asynchronous scheduling, no prefix caching and the
263-INT8-matrix mixed checkpoint. PTO target uses mode0/prepared AICPU DIRECT;
native target uses mode3/FULL. Both retain the same native mode3 MTP drafter,
padding and sampling. No model math, generated kernel or framework is modified.
No HBG is used. [pto_target_config.json](pto_target_config.json) contains the
validated PTO runtime options with explicit path placeholders.

## Files used by the entries

| File | Purpose |
|---|---|
| [e2e_pypto_vllm_profile.py](e2e_pypto_vllm_profile.py) | Shared startup checks, official benchmark, GDD capture lifecycle and export |
| [serving_workflow.py](serving_workflow.py) | Service-session checks, QA/benchmark/profile dispatch and automatic rank discovery |
| [long_context_qa.py](long_context_qa.py) | Exact32K input construction and strict JSON answer validation |
| [analyze_outer_trace.py](analyze_outer_trace.py) | GDD Decode, reference-marker Verify and matched full-target boundaries |
| [e2e_vllm_window.py](e2e_vllm_window.py) | Custom SSE batch/timestamp and MTP acceptance-window measurements |

The Python files are dependencies or measurement implementations, not duplicate
shell entry points. The custom window client retains raw SSE batches and token
IDs; it never fabricates individual token timestamps.

## Outputs

- QA: exact input IDs/hashes, complete requests/responses and `qa-result.json`.
- Benchmark: official command, detailed `result.json`, log and input gate.
- Profile: complete `trace_view.json`, `kernel_details.csv`, `op_statistic.csv`,
  rank mapping, per-cycle `timeline.json` and a readable `RESULTS.md`.

Open complete traces in Perfetto. Verify is the full target, including embedding,
head and logits gather. The main Decode follows GDD: consecutive B4 rejection
sampling starts. MTP-start-to-next-MTP-start intervals are an additional metric.
Four-rank P50 aggregation
uses the median of rank P50 values; the mean aggregates rank means.

The profile command follows the GDD capture lifecycle: complete a full four-request
32768/10240 warmup, issue another four-request32768/10240 workload, wait for all
prefills to finish and B4 decode to generate at least256 tokens, call
`/start_profile`, let that workload finish, call `/stop_profile`, then export
offline. `max_iterations=50` bounds the device collection; it does not shorten
or cancel requests. This profiled workload is diagnostic and is never substituted
for the separate formal benchmark with profiling inactive.

## Boundary definitions and reference alignment

Reference: GDD_ESCC/pypto, `dynamic-batch-20261021`, commit
`a62c3c176cc173e2ec84f35054a2e7bb403bc24d`. The collection source is
`e2e_pypto_vllm_profile.py::capture`; the metric source is
`e2e_vllm_timeline.py::read_rank` in that repository's end-to-end test directory.

| Metric | Start | End | Relation to GDD |
|---|---|---|---|
| Decode, `repository_decode` | B4 rejection_greedy_sample_triton start | Next B4 rejection start | Same definition; adjacent output shapes must both be4,6 |
| Native reference Verify, `repository_verify` | First GemmaRmsNorm in the target graph | Last AddRmsNormBias in that graph | Same definition; target graph has16 FIA operations |
| PTO reference-marker adapter, `repository_verify` | Persistent AICore task start | That task's end | Simpler has no PyptoRun; this is explicitly an adapter, not the literal GDD marker |
| Matched full-target Verify, `matched_full_verify` | Full target begins, including embedding | Logits gather completes | Additional equal-work boundary; PTO includes the union span of its AICPU/AICore tasks |
| Additional MTP interval, `mtp_start_alternate` | Five-step drafter graph starts | Next five-step drafter graph starts | Additional field; never relabelled as GDD Decode |

Native reference Verify excludes embedding/head stages covered by full-target
Verify. These fields are retained separately, with original CSV row numbers,
model/task/stream IDs and rejected-interval counts. An unknown PTO marker or
incomplete native graph must not be silently interpreted as a matching boundary.
GDD emits per-rank results; our additional four-rank summary uses the aggregation
documented above and retains all per-rank values.

Service configuration remains our frozen protocol above. GDD service defaults
such as memory0.80, disabled chunked prefill and mamba-cache none are not copied.
The checkpoint, target/runtime implementation and native drafter stay unchanged.

## Verified long-context result

[LONG_CONTEXT_QA.md](LONG_CONTEXT_QA.md) contains the actual four-request32K
result:4/4 passed,60/60 fields correct, all responses naturally completed.
[LONG_CONTEXT_QA_RESULT.json](LONG_CONTEXT_QA_RESULT.json) retains responses
and execution proof; [LONG_CONTEXT_INPUT_AUDIT.json](LONG_CONTEXT_INPUT_AUDIT.json)
records input construction. The checked records cover the beginning, middle
and end without repeated padding.

This is functional QA evidence, not a refreshed10K baseline or long-output
native/PTO equivalence claim. Current wrapper checks are CPU-only, including
native/PTO dispatch and analysis of read-only historical timelines. Earlier
script versions remain in Git history; original run evidence is preserved.
