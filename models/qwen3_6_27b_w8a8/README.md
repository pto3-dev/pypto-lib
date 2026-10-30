# Qwen3.6-27B mixed W8A8 target: tested implementation checkpoint

This is the complete experimental source checkpoint behind the restored
16.463486 ms/token result, committed for review and continued integration.
It is not a claim of completed production numerical acceptance or a portable
one-command build. The current target is PyPTO; prefill and the MTP drafter
are native vLLM. Target execution is DIRECT in the vLLM worker process.
Native FULL_DECODE_ONLY/mode3 graph replay is used for the drafter, not the
PTO target. NZ weights and KV/GDN/convolution caches are shared.

## Reading order

1. `src/qwen36/model.py` and `src/qwen36/operators/`: directly readable target
   composition and mixed INT8/BF16 operators, with explicit rank selection.
2. `src/qwen38_runtime/serve.py` and `fast_recovery_worker.py`: service options
   and the actual selected worker. The qwen38 namespace is retained ABI naming,
   not a different checkpoint or a second target process.
3. `src/qwen38_runtime/compact_inputs.py`: consume fresh sampled/draft device
   token IDs, generate text M-RoPE positions on device, preserve CPU mirrors
   and fall back on unsupported/reordered/ragged/multimodal inputs. No token
   values are cached across steps. Position graphs are distinct from target
   capture. Regular B1/B2/B3/B4 drafter graphs cover serving tails.
4. `src/qwen38_runtime/{adapter,resources,runtime_factory,borrowed_runtime}.py`:
   dispatch, typed shared caches, prepared runtime, streams and lifetime.
5. `dsl/qwen36_attention_dsl/{paged_attention,softmax_explicit}.py` and
   `dsl/qwen36_fusion/gate_quant.py`: selected attention/softmax and fusion DSL.
6. `static_runtime/`, `link_contract/rank*/sources.json` and
   `registered_sources/`: the exact static orchestration, owned leaf overrides and hashes of generated
   leaves. Compiler outputs are regenerated from the owned DSL, not committed;
   this is not a new claim that all C++ registration/ABI adaptation is gone.
7. `dependencies/lock.json`: actual source revisions, dirty-checkout patches,
   previously untracked extensions/tests, and the frozen tested runtime overlay.
   The overlay takes precedence over the ordinary Simpler checkout patch.
8. `benchmark/run_fast_fused.py`, `recovery_stage.py` and `PERFORMANCE.md`:
   original formal protocol, resource guards, measurement and limitations.

`src/qwen36/build/compile.py` is the direct frontend compile entry. Its legacy
`link_backend.py` is retained baseline code; it does not by itself recreate the
selected fused target. The selected low-level registration is in `link_contract`.
The selected DSL builder and compile scripts are in `build_tools/`. They still
reference the original workspace. This checkpoint makes those dependencies
explicit rather than pretending the baseline entry rebuilds every optimization.

## Source and artifact identity

The original tested workspace and artifacts remain untouched. Python publication
copies receive license headers, formatting and removal of unused imports.
`source_provenance.json` records original hashes, published hashes and normalized
AST identity. Mathematical bodies, constants and execution order are checked
against the original source; C++ changes only remove indentation on empty lines.
Unified dependency patches retain necessary blank-line context verbatim.

Run the CPU-only publication audit (no NPU initialization):

```bash
python models/qwen3_6_27b_w8a8/verify_snapshot.py
```

On the original machine, also check every source and binary against the recorded
originals:

```bash
python models/qwen3_6_27b_w8a8/verify_snapshot.py --originals --artifacts
```

Generated binaries, weights, raw prompts and large profiling datasets are not
committed. `evidence/artifacts.json` identifies the retained binaries by path,
size and SHA256. `evidence/LATEST_VERIFIED_RESULT_20261030.json` points to the
actual measured runtime/source/DSL paths and full raw evidence. Source paths are
recorded explicitly for the requested shared workspace handoff.

## Reproduction in the original workspace

The exact environment and dependency selection are in `benchmark/env_used.sh`.
Do not substitute newer default toolchains or clean upstream Simpler libraries
for this ABI. On an equivalent machine the dependency patches, runtime overlay,
artifact roots and service paths must first be rebuilt/configured and accepted.
A CPU audit alone does not prove relocated device acceptance.

The original checked command is:

```bash
bash /home/g00893677/pto-dev/qwen3_6_27b-inner-w8a8/work/restore_16ms_20261030/env.sh \
  /home/g00893677/pto-dev/qwen3_6_27b-inner-w8a8/work/restore_16ms_20261030/tools/run_fast_fused.py \
  --name NEW_EVIDENCE_NAME --devices 1,2,3,4 --skip-terminal-validation
```

Omit `--devices` to select an idle four-card group. Use a new evidence name.
The skip option is only for binaries whose full-state acceptance already exists;
otherwise keep the default terminal validation. Never reclaim another job's
shared memory or cards: the benchmark only checks available shared memory and
refuses startup below its deployment reserve. The authorized orphan cleanup is
a completed machine operation, not an automatic part of this implementation.

See [performance and open issues](PERFORMANCE.md) before interpreting the result.
