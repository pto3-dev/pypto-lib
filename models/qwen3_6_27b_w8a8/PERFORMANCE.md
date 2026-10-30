# Performance evidence and unresolved issues

The fixed engineering-context fixture has four independent saved requests,
32768 input tokens and 10240 output tokens each, TP4/PP1/B4, MTP5,
async scheduling, prefill chunk2048, memory utilization0.96, cards1/2/3/4.
The checkpoint is the public mixed-W8A8 inner-format surrogate, not unavailable
private production weights. Both runs replay the same saved input token IDs
and sampling settings. This is one paired formal run, not a repeated average.

| Metric | Same-card historical-source replay | Current fused + compact |
| --- | ---: | ---: |
| Mean TPOT, ms/token | 17.336742 | 16.463486 |
| Stable SSE batch gap mean, ms | 50.627053 | 47.811743 |
| Stable SSE batch gap median, ms | 49.085000 | 47.231300 |
| Stable SSE batch gap P99, ms | 73.412069 | 60.731313 |
| Stable token-weighted amortized mean, ms/token | 15.909376 | 15.081841 |
| Mean acceptance length including bonus | 3.210898 | 3.188400 |
| Accepted draft token fraction | 44.2353% | 43.7796% |

Current request TTFTs:9.252125,19.153451,28.417903,37.583669 seconds.
All four requests finish all10240 output tokens. SSE gaps are real received
output batches; amortization divides by batch token counts and does not invent
individual token device timestamps. Full distributions and acceptance histograms
are retained in `evidence/performance.json`.

Observed mean TPOT improvement over the same-card old replay is5.037%.
The result restores the historical16.498971 ms range. It does not establish a
statistically significant improvement over that historical record. The native
baseline was retained, not freshly rerun for this checkpoint. Global acceptance
length did not rise, but local trajectories/admission timing can still confound
an end-to-end attribution. The old and current free-generation outputs differ.

## Separately verified findings

- The compact-input same-run comparison removed four token-index uploads and
  a roughly49KB M-RoPE upload. The late upload itself took only5-7us, while its
  admission could wait17-21ms after CPU submission. Start skew fell from a
  diagnostic P99 of21.768ms to0.0288ms. The underlying CANN admission mechanism
  is not fully attributed; this does not prove a driver defect.
- An unprofiled, same-state target comparison measured the current fused target
  at31.2303ms versus32.5058ms for the older target on rank0, approximately3.9%
  faster; the other ranks agree. Logits, hidden and all16 complete physical
  cache pools (including all KV/GDN/convolution states) were byte-identical.
  This is an actual retired B1-state diagnostic, not B4 production ITL.
- That pair tool calls the served new target `reference` and the supplied old
  comparison target `candidate`. These labels must not be interpreted backwards.
  Current formal artifacts match the accepted pair artifacts on all four ranks.
- Shared-memory exhaustion caused a separate SIGBUS in vLLM's broadcast write.
  Authorized orphan reclamation removed that startup/run blocker. It is not
  asserted to explain the entire historical16-to17ms latency drift.
- Diagnostic HCCL determinism is not enabled for the formal performance result.
  Both sides use native default policy. It is distinct from Torch global
  deterministic algorithms and has a separate numerical/communication cost.

## Still open

- Native/PTO long-output equality and the native BF16 reduction-order issue are
  not closed. Short fixed-state equality does not replace long-output acceptance.
- P99 has improved but long tails remain; current maximum stable SSE gap is
  924.372ms. Not every tail sample has a full device-level causal attribution.
- Rare irregular-shape native fallbacks remain. Ordinary B1-B4 drafter graph
  coverage does not imply all prefill/profile/ragged cases are graph replay.
- The low-level static C++ dispatcher and explicit source registration remain.
  Fresh-machine portability and upstream library integration need separate
  acceptance; this commit preserves the latest tested checkpoint first.
