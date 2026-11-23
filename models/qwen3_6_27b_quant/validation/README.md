# Serving validation clients

These clients use an already running, explicitly selected service. They do not
launch workers, choose cards, modify vLLM, or change target/drafter settings.
Use the same checkpoint, service configuration and idle-card group for paired runs.
The target uses prepared AICPU DIRECT, not HBG; the drafter remains native.
Do not compare profiler windows as unprofiled 10K results.

## Conversation check

```bash
python chat_qa.py --base-url http://127.0.0.1:8000 --output /new/chat-results
```

Four concurrent conversations, two turns each. The first turn extracts records
as JSON; the second carries the actual history and computes remaining work and
budget. Temperature and seed are zero, thinking is disabled, and EOS is allowed.
The client saves exact requests, responses, field scores and finish reasons.
[Executed response examples](CHAT_RESULT.md) are functional evidence only.

## Official serving benchmark

```bash
python bench_serve.py --base-url http://127.0.0.1:8000 --model MODEL_ID \
  --tokenizer /checkpoint --corpus /corpus/zh_qa.jsonl --output /new/bench-results
```

Use existing `zh_qa`, `en_technical`, and `code_review` JSONL files for the three
scenes; each must contain four prompts of exactly 32768 tokens. Output is 10240
tokens per request, concurrency four, ignore-EOS, temperature zero, seed zero.
The wrapper invokes official `vllm bench serve` and preserves `command.json`,
`bench.log`, and official `result.json`, including detailed TTFT/TPOT/ITL and
speculative-decode metrics when exposed by the installed version.
Omit `--corpus` for official random generation; use the same seed on both arms.
Random input is a separate workload and is not interchangeable with the three scenes.
The wrapper is a portable extraction of the existing benchmark command, not a
newly measured baseline. It does not select only favorable acceptance rates.

For MTP, official detailed ITL describes received stream-message intervals.
Neither that interval nor TPOT is a device Verify duration; do not invent
individual token generation timestamps from batched messages.

## Complete outer device timeline

Use the service's official profiling configuration and `/start_profile` and
`/stop_profile` endpoints in a separate diagnostic window. Export the four raw
rank profiles with `torch_npu.profiler.profiler.analyse` outside daemon workers.
Retain the complete `trace_view.json` and `kernel_details.csv`; open the JSON in
Perfetto. This is the CANN/vLLM outer timeline, not a PyPTO-internal leaf trace.

```bash
python analyze_outer_trace.py --backend pto --output /new/timeline.json \
  --rank-csv /rank0/kernel_details.csv /rank1/kernel_details.csv \
  /rank2/kernel_details.csv /rank3/kernel_details.csv
```

Use `--backend native` for the native arm. Each CSV must have the corresponding
`trace_view.json` beside it. The reader preserves the executed full-target
matching rules and per-cycle original row numbers, graph IDs, task IDs and streams.
`matched_full_verify` includes the whole target through logits gathering;
PTO uses the union span of concurrent AICPU and persistent AICore tasks.
`mtp_start_alternate` is Decode from one native five-step drafter graph start
to the next. `repository_decode` retains rejection-to-rejection timing separately.
The reader requires at least eight complete B4 cycles and explicitly rejects
unresolved boundaries. It is specific to the validated v0.23 kernel naming/ABI;
unknown names are errors, not inferred measurements. Ranks execute concurrently,
so the aggregate is the median of rank P50 and mean of rank means.
