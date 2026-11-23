# Qwen3.6-27B mixed W8A8: retained optimizations

The [model entry](../../../models/qwen3_6_27b_quant/README.md) describes the
validated execution and build boundary. This page explains only mechanisms
present in the selected implementation, without adding historical benefits.

## MLP: keep activation processing together

An unfused boundary writes BF16 SwiGLU activations to GM and reloads them
for row-wise quantization. The current packed gate/up AIC producer emits
INT32. One AIV then processes a complete 4352-column row in UB:

```text
packed gate/up -> INT32 GM -> dequant -> SwiGLU -> absmax -> INT8 GM -> down
                                 one AIV row lifetime
```

[The current code](../../../models/qwen3_6_27b_quant/qwen36/operators/packed_act_quant.py)
retains projection and activation BF16 rounding. SwiGLU uses direct FP32
division before multiplying the up branch. INT32 GM traffic remains; the
removed round trip is the BF16 activation between SwiGLU and quantization.

## Down: fuse the consumer's dequantization

Instead of `INT32 -> down AIV dequant -> FP32 GM -> communication reload`,
the [down producer](../../../models/qwen3_6_27b_quant/qwen36/operators/down_i32.py)
writes INT32 directly. The
[communication consumer](../../../models/qwen3_6_27b_quant/qwen36/operators/collective_dequant_row.py)
applies scales and BF16 rounding once before publishing its row. This removes
an AIV handoff; it does not remove the communication or its GM staging.

## Communication: reuse the normalized row

The row-ready consumer performs notification, wait, four-rank reads,
residual, Norm and quantization in one AIV function. A separate quantizer
would reload the normalized row; the selected implementation consumes its
UB `result` directly. Notification/polling/remote reads remain. Their wait
time is not pure RMSNorm arithmetic, and row notification alone does not
guarantee tile-level release through a following SPMD boundary.

## L1 reuse and buffering

The GDN output projection groups four adjacent N64 blocks in one SPMD
kernel and reuses one `[32,1536]` A tile. It does not dynamically grow fusion
without limit and does not reuse the four distinct B blocks. The 20 logical
groups are work items; physical launch limits are queried from the backend.

Packed gate/up uses N224 (N192 tail), outer K512/stage3 and inner K128/stage2,
with explicit two-slot L0B. Generated L1 B addresses are 163840, 278528 and
393216; the maximal live allocation ends at 507904 bytes. INT8 down uses
N128/K256 with L0B addresses 0/32768. These allocations were checked against
queried capacities. All these kernels have a single L0C address; no claim of
double buffering at every level follows from a `pipeline` annotation.

BF16 GDN/Z uses B4 N32/K512/K64 and explicit native accumulation-order
branches for smaller effective row counts. The shared-A projection and
lm_head have their own tilings. They do not inherit one universal buffer rule.

## GDN: consume convolution outputs and reuse state

[Convolution](../../../models/qwen3_6_27b_quant/qwen36/operators/gdn.py)
runs on AIV, directly normalizing the newly produced Q/K tile. Per-request
history and weights remain in the loop while multiple draft tokens advance.
Recurrent state is FP32, split by 64 V columns, with native BF16 rounding
boundaries. Requests and accepted-state slots remain independent.

## Attention: process visible cache ranges

The device FD plan drives normal DSL QK, 512-column softmax partitions, PV
and combine. PV visits only visible partitions and pipelines 128-token
loads. Softmax uses `row_expand_expdif` with native partition/reduction
semantics and invalid-region masking. These are separate tasks with GM
intermediates, not an entirely on-chip FlashAttention kernel.

## Cache policy and prepared execution

Large weights and flattened K/V views explicitly use BYPASS. This controls
cache behavior; it does not create a second model-weight copy in L2.
Native prefill and PTO target alias the same final NZ banks and caches.
Necessary initial upload/layout conversion remains, while redundant ND
stacking and NZ-to-bank copies are absent in the selected loading strategy.

The separate runtime repository owns prepared AICPU DIRECT submission and
physical binding. The target uses no HBG and no postcompile kernel
replacement. Native prefill and the MTP drafter remain native vLLM.

## Precision and performance scope

Current exactness guards include Native FD partition/reduction order,
metadata flatten-before-slice, BF16 intermediate rounding, scalar Norm DIV,
SwiGLU DIV, typed `0-x` K-RoPE, and native BF16 GEMM accumulation order.
The source manifest records original source hashes; no weight data, raw
requests, generated binaries or machine-local evidence is included.

The source is validated; newly relocated entry points require their own
build checks. Independent per-optimization end-to-end gains are not inferred
from structural changes. Performance comparisons must keep the native MTP
drafter, padding, sampling and cache settings aligned and report target
Verify, MTP-cycle Decode and streaming-message ITL separately.
