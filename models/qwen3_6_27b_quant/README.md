# Qwen3.6-27B mixed W8A8 target

This is the TP4/PP1, batch-four, MTP5 target/verifier. Its 24 token rows
are four requests times five drafts plus one verifier token. The native
vLLM drafter and prefill retain ownership of their weights and state.
The target shares their NZ weights and KV/GDN/convolution storage in the
same worker process and executes through Simpler AICPU prepared DIRECT.

The mixed checkpoint contains 263 INT8 matrices: all MLPs and full-attention
projections are W8A8; GDN QKV/Z projections remain BF16. Public surrogate
validation does not certify a private checkpoint. Only text inference is
covered; the MTP drafter is not reimplemented in PyPTO.

## Read the source

- [Whole target and explicit rank selection](qwen36/model.py)
- [Packed INT8 gate/up and fused SwiGLU/quantization](qwen36/operators/packed_act_quant.py)
- [INT32 down producer](qwen36/operators/down_i32.py)
- [Communication/dequantization/Norm fusion](qwen36/operators/collective_dequant_row.py)
- [Row-ready communication/Norm/quantization](qwen36/operators/collective_row_ready.py)
- [BF16 projections and shared-A groups](qwen36/operators/projection_bf16.py)
- [GDN convolution, Q/K normalization and recurrence](qwen36/operators/gdn.py)
- [QK](qwen36/operators/attention.py), [FD plan](qwen36/operators/attention_fd_plan.py),
  [softmax/PV/combine](qwen36/operators/attention_fd.py)
- [Hardware queries and tuning](qwen36/tuning.py)

The `qwen36` package name and module-level `pl.function` form are retained
for the validated compiler/interface contract. Newer DSL migrations are
not implicit in this source publication. Unselected variants remain available
for existing imports; the published build entry selects only `row_ready`.

## Build through normal DSL compilation

Use the matching `qwen3.6-27b-quant` branches of the compiler, runtime and
integration repositories; exact revisions are recorded in `stack_versions.json`.
Install/build dependencies in their own worktrees. Set `PYTHONPATH` to the
selected PyPTO `python`, Simpler `python` and source directories, and the
integration `src` directory; select PTOAS 0.65 in `PTOAS_ROOT`.

```bash
python models/qwen3_6_27b_quant/compile_target.py \
  --compiler-source "$PYPTO_SOURCE" --geometry-dir "$GEOMETRY_DIR" \
  --output-root "$BUILD_ROOT" --soc 910b4
python models/qwen3_6_27b_quant/seal_bundle.py \
  --root "$BUILD_ROOT" --geometry-dir "$GEOMETRY_DIR"
```

`compile_target.py` compiles all four ranks, or one `--rank`. It snapshots
the input DSL and checks that every registered generated source belongs to
the current compilation. `seal_bundle.py` checks the measured cache ABI and
emits interface contracts; it never changes generated kernel registrations,
metadata or mathematics. Existing output directories are not overwritten.

The sample `geometry/` files describe the measured 45056-capacity, 128-token
page, fixed-6-GiB-cache specialization. Measure geometry with the frozen
interface before deploying different cache settings. Geometry is a compile
input, not a fallback to another source snapshot. `--soc 910b3` is an explicit
configuration entry; device validation currently covers 910B4 only.

## Ownership and validation

Model mathematics is maintained here. DSL primitives, IR, passes and codegen
belong to `pto3-dev/pypto`; scheduling/physical bindings/prepared-run support
belong to `pto3-dev/simpler`; instruction implementations belong to
`pto3-dev/pto-isa`. The frozen attachment lives in
`pto3-dev/vllm-ascend/integrations/qwen3_6_27b_quant`, not in this model.

The source before publication passed a four-rank 4x32768-input/10240-output
same-state run: 2896 actual target invocations per rank, with byte-exact
logits, hidden and 128 state-write tensors. An uninstrumented asynchronous
10K run and subsequent multi-turn requests also completed. These are source
validation facts, not a new performance measurement of the relocated package.
Formatting/import cleanup is audited against the executed mathematical AST.

The current performance refresh is separate. No historical TPOT or local
timings are promoted to a new baseline. See the current source-only
[optimization explanation](../../docs/models/qwen3_6_27b_quant/index.md).
