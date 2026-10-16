# Qwen3.8-27B PTO decode

This source module implements the dense BF16 text model used by the experimental
vLLM-Ascend TP2/PP1 bridge. Native prefill and PTO decode share native KV/GDN
storage. Decode uses one CPU L3 and two L2 workers, without L1 workers.

The model includes reduction/residual/normalization fusion, convolution/QK
normalization fusion, Q/RoPE fusion, ND weight readers, gate/up/SwiGLU fusion,
with staged projection loads.

The model builder uses ordinary dynamic dispatch. It does not request static
block mapping and requires no local static-scheduler or L1Reuse compiler pass.
Early resolve remains enabled independently of static block mapping.

Run from the repository root with the configured PyPTO environment and the root
on PYTHONPATH. Use the toolchain versions pinned by that PyPTO checkout.

```bash
python -m models.qwen3_8_27b.validate --case gemm_gates --compile-only --output build_output/qwen38-check
python -m models.qwen3_8_27b.validate --device DEVICE --case all --output build_output/qwen38-check
```

Select an allocated device for execution. The default correctness cases use
independent Torch references; they do not establish end-to-end model parity.

`model_spec.ModelSpec` describes weight banks and cache metadata.
`model_program.build_model` generates the complete distributed decode program.
The vLLM bridge supplies native cache layouts, fixed metadata storage and graph
lifecycle management; standalone model timing is not HTTP decode timing.

`PYPTO_QWEN_WEIGHT_CACHE_BYPASS=1` changes only read-only decode weight loads.
Enable it only when the bridge owns weights in ordinary L2 allocations: an IPC
import of a native-owned allocation cannot safely use the bypass offset.

The source was extracted from a recorded 40.540270 ms/token candidate with
static scheduling removed. That historical timing does not measure this
changed dynamic-dispatch combination. Fixed-prompt output hashes from that
candidate do not establish full native same-prefix tensor validation.
