# Qwen3.8-27B TP2 shared-cache decode

The model provides BF16 TP2/PP1 text decode for native vLLM prefill with
shared ND weights and KV/GDN caches. The selected implementation uses fused
B/A gates, convolution/QK normalization, attention and MLP vector operations,
reduce/norm, the recorded projection tiles, and two-AIV logits gather.
`projection_tiles.json` and `l1_reuse_selection.json` preserve the selected
policies; they do not apply one buffer policy to every matmul.

After PyPTO compilation, apply `fixed_mlp_adapter.install_fixed_mlp`,
`attention_program_adapter.install_attention_program`, then
`mlp_input_adapter.install_mlp_input`. These guarded adapters require the
selected generated ABI. They combine attention leaf calls and the MLP input
region while retaining dynamic boundaries. They request unique core placement
without a global drain or sync-start barrier. MLP vector work stays on lane 0.
Generated sources, manifests and plan images belong under the build directory.

The Simpler dependency must provide `simpler_setup.device_program`,
`simpler_setup.static_program` and fixed-placement dispatch including pure-AIC
early release. PyPTO must emit the `static_block_mapping` attribute. Caller-owned
graph execution and mailbox wakeups are also required by the service. Use
matching compiler/runtime source builds rather than mixing extension binaries.
The source checkout of this model is selected explicitly by the vLLM-Ascend
`VLLM_ASCEND_PYPTO_LIB_ROOT` setting; no historical work directories are searched.

## Selected source and measurement boundary

This integration follows `INTEGRATION_BASELINE_20261017.md`, generation
`ea00209693a8e241`. The recorded B1/input300/output128/maxlen512 HTTP run has
three-request median 39.311919 ms/token on cards 6/7. The same code recorded
39.420060 ms/token on cards 1/3. These are separate groups, not an inferred
hardware comparison or a guarantee for the relocated package. The output SHA256
was `3caa37ade2065bae574b201dfe2c3805c8e6814c4f7fb25b8e226d565786ed87`.

The recorded PTOAS executable was version 0.65, SHA256
`49b44fcce9ca7f2a2b16d60030bffeefea66d2a0440d63c5705c51be58523b60`;
the recorded PTO-ISA checkout was `418de18dde8fa33773dc5c384fdfe5506463b628`.
These identify the evidence toolchain, not bundled binaries or new global pins.
Audit generated weight cache hints and instructions when using newer toolchains.

The relocated integration requires independent device output, graph lifecycle
and TPOT acceptance. CPU and compile-only checks do not establish that result.
MLP lane rotation, split-half variants and unselected GDN/PV experiments are
excluded. No 32K-context or B4 whole-model support is claimed.
