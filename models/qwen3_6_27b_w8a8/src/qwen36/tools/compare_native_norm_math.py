# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""One-card mathematical precision diagnosis, no whole-model performance claim."""

import argparse, ctypes, hashlib, json, time
from pathlib import Path
import torch, torch_npu
from safetensors import safe_open
from pypto import ir
from qwen36.runtime.borrowed_graph_runtime import BorrowedGraphRuntime
from qwen36.tools.idle_guard import _query, wait_for_idle

W = Path(__file__).resolve().parents[3]
ROOT = W.parents[1]
BINARY = Path(
    "/home/g00893677/pto-dev/qwen3_8_27b-perf/work/inprocess_native_nz_20261023/simpler/build/lib/a2a3"
)


def compare(a, b):
    a = a.detach().cpu()
    b = b.detach().cpu().reshape(a.shape)
    diff = a.float() - b.float()
    neq = a.ne(b)
    return dict(
        bitwise_equal=torch.equal(a.view(torch.uint8), b.view(torch.uint8)),
        different_elements=int(neq.sum()),
        elements=a.numel(),
        max_abs=float(diff.abs().max()),
        mean_abs=float(diff.abs().mean()),
        different_rows=int(neq.reshape(a.shape[0], -1).any(dim=1).sum()),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    a = p.parse_args()
    if Path(a.name).name != a.name:
        raise ValueError("single-component name required")
    output = W / "evidence" / a.name
    output.mkdir(exist_ok=False)
    status = dict(status="waiting_idle_card", pid=__import__("os").getpid())

    def save():
        (output / "status.json").write_text(json.dumps(status, indent=2) + "\n")

    save()
    card = None
    deadline = time.monotonic() + 300
    while card is None and time.monotonic() < deadline:
        for c in range(8):
            try:
                if "No process in device." in _query(c, "proc-mem"):
                    wait_for_idle([c], timeout=10, stable_seconds=2)
                    card = c
                    break
            except (RuntimeError, OSError):
                pass
        if card is None:
            time.sleep(2)
    if card is None:
        raise TimeoutError("No idle card")
    status.update(status="running", device=card)
    save()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.npu.set_device(card)
    model = ROOT / "checkpoints/public-inner-surrogate"
    index = json.loads((model / "quant_model_weights.safetensors.index.json").read_text())["weight_map"]
    name = "model.language_model.layers.3.post_attention_layernorm.weight"
    with safe_open(model / index[name], framework="pt", device="cpu") as f:
        gamma_cpu = f.get_tensor(name).reshape(1, 5120)
    weight = gamma_cpu.npu()
    gamma = (1 + weight).flatten()
    program = ir.CompiledProgram.from_dir(W / "build/norm_math_diagnosis", platform="a2a3")
    chip = program.chip_callable
    image = ctypes.string_at(chip.buffer_ptr(), chip.buffer_size())
    x = torch.empty((24, 5120), dtype=torch.bfloat16, device=weight.device)
    res = torch.empty_like(x)
    outputs = [
        torch.empty_like(x),
        torch.empty_like(x),
        torch.empty_like(x, dtype=torch.int8),
        torch.empty((24, 1), dtype=torch.float32, device=x.device),
    ]
    stream = torch.npu.Stream()
    runtime = BorrowedGraphRuntime(BINARY, card, image, aicpu_threads=0)
    graph = torch.npu.NPUGraph()
    rng = torch.Generator().manual_seed(20261027)
    cases = []
    try:
        runtime.bind([x, res, weight, *outputs])
        runtime.prepare_graph()
        with torch.npu.graph(graph, stream=stream):
            runtime.enqueue_graph(stream.npu_stream)
        for scale in [0.01, 1.0, 32.0]:
            xv = (torch.randn((24, 5120), generator=rng) * scale).bfloat16()
            rv = (torch.randn((24, 5120), generator=rng) * scale).bfloat16()
            xv[0] = 0
            rv[0] = 0
            xv[1] *= 1e-7
            rv[1] *= 1e-7
            with torch.npu.stream(stream):
                x.copy_(xv)
                res.copy_(rv)
                graph.replay()
                norm, inv, added = torch_npu.npu_add_rms_norm(x, res, gamma, 1e-6)
                quant, qscale = torch_npu.npu_dynamic_quant(norm)
                custom_norm, custom_inv, custom_added = torch.ops._C_ascend.npu_add_rms_norm_bias(
                    x, res, gamma, None, 1e-6
                )
                custom_quant, custom_scale = torch_npu.npu_dynamic_quant(custom_norm)
                fused = torch.ops.npu.npu_add_rms_norm_dynamic_quant(
                    x, res, gamma, epsilon=1e-6, output_mask=[True, True]
                )
            stream.synchronize()
            hypotheses = {}
            gamma_ref = (gamma_cpu + 1).bfloat16().float()
            for add_round in [True, False]:
                for norm_round in [True, False]:
                    total = xv.float() + rv.float()
                    if add_round:
                        total = total.bfloat16().float()
                    value = total * torch.rsqrt(total.square().mean(dim=-1, keepdim=True) + 1e-6) * gamma_ref
                    if norm_round:
                        value = value.bfloat16().float()
                    magnitude = value.abs().amax(dim=-1, keepdim=True)
                    ratio = torch.where(magnitude == 0, torch.zeros_like(magnitude), 127 / magnitude)
                    expected_q = torch.round(value * ratio).clamp(-128, 127).to(torch.int8)
                    expected_s = magnitude / 127
                    hypotheses[f"round_add_{add_round}_round_norm_{norm_round}"] = dict(
                        quantized=compare(expected_q, fused[0]), scales=compare(expected_s, fused[3])
                    )
            cases.append(
                dict(
                    pto_vs_native_custom=dict(
                        added=compare(outputs[0], custom_added),
                        normalized=compare(outputs[1], custom_norm),
                        quantized=compare(outputs[2], custom_quant),
                        scales=compare(outputs[3], custom_scale),
                    ),
                    cpu_fp32_hypotheses_vs_native_fused=hypotheses,
                    input_scale=scale,
                    pto_vs_native_separate=dict(
                        added=compare(outputs[0], added),
                        normalized=compare(outputs[1], norm),
                        quantized=compare(outputs[2], quant),
                        scales=compare(outputs[3], qscale),
                    ),
                    pto_vs_native_fused=dict(
                        added=compare(outputs[0], fused[2]),
                        quantized=compare(outputs[2], fused[0]),
                        scales=compare(outputs[3], fused[3]),
                    ),
                    native_separate_vs_fused=dict(
                        added=compare(added, fused[2]),
                        quantized=compare(quant, fused[0]),
                        scales=compare(qscale.reshape(24, 1), fused[3]),
                    ),
                    fused_shapes=[list(t.shape) for t in fused],
                )
            )
        from types import SimpleNamespace
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch._inductor.pattern_matcher import PatternMatcherPass
        from vllm_ascend.compilation.passes.norm_quant_fusion_pass import (
            AddRMSNormDynamicQuantPattern,
            AddRMSNormDynamicQuantPatternWithBias,
        )

        def native_slice(a, b, g):
            n, _, r = torch.ops._C_ascend.npu_add_rms_norm_bias(a, b, g, None, 1e-6)
            q, s = torch.ops.npu.npu_dynamic_quant(n)
            return q, s, r

        gm = make_fx(native_slice)(x, res, gamma)
        (output / "native_custom_before_fusion.py").write_text(gm.code)
        matcher = PatternMatcherPass(pass_name="qwen36_norm_diagnosis")
        config = SimpleNamespace(model_config=SimpleNamespace(dtype=torch.bfloat16))
        for cls in [AddRMSNormDynamicQuantPattern, AddRMSNormDynamicQuantPatternWithBias]:
            cls(config, eps=1e-6).register(matcher)
        match_count = matcher.apply(gm.graph)
        gm.recompile()
        (output / "native_custom_after_fusion.py").write_text(gm.code)
        result = dict(
            native_custom_slice_dynamic_fusion_matches=match_count,
            status="complete",
            device=card,
            checkpoint=str(model),
            weight_name=name,
            weight_sha256=hashlib.sha256(gamma_cpu.view(torch.uint8).numpy().tobytes()).hexdigest(),
            cases=cases,
            scope="Already BF16-rounded reduction + add + norm + quant, identical inputs; synthetic activations with real public gamma. Does not diagnose allreduce rounding or prove the cause of long-generation divergence.",
        )
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        status.update(status="complete")
        save()
        print(json.dumps(result), flush=True)
    except BaseException as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise
    finally:
        stream.synchronize()
        graph.reset()
        runtime.close()


if __name__ == "__main__":
    main()
