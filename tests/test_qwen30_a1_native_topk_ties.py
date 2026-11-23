"""Probe real repeated-native divergence for exact routing ties."""

import os
from pathlib import Path

import pytest
import torch


def test_native_topk_ties():
    root = os.environ.get("PTO3_B20_NATIVE_REPEAT_ROOT")
    if not root:
        pytest.skip("PTO3_B20_NATIVE_REPEAT_ROOT is required")
    import torch_npu
    from safetensors import safe_open

    root = Path(root)
    a = torch.load(root / "native-b20-layer-trace-v3-first/step-1.pt",map_location="cpu",weights_only=True)
    b = torch.load(root / "native-b20-layer-trace-v3/step-1.pt",map_location="cpu",weights_only=True)
    index = next(i for i,(x,y) in enumerate(zip(a["layers"],b["layers"])) if not torch.equal(x[0],y[0]))
    rows = torch.nonzero((a["layers"][index][0] != b["layers"][index][0]).any(-1)).flatten()
    x = a["stages"][index]["post_norm"][0]
    assert torch.equal(x,b["stages"][index]["post_norm"][0])
    key = f"model.layers.{index}.mlp.gate.weight"
    checkpoint = os.environ.get("A1_MODELSLIM_CHECKPOINT")
    if not checkpoint:
        pytest.skip("A1_MODELSLIM_CHECKPOINT is required")
    for path in Path(checkpoint).glob("*.safetensors"):
        with safe_open(path,framework="pt") as shard:
            if key in shard.keys():
                weight = shard.get_tensor(key).half()
                break
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    device_x = x.npu()
    device_w = torch_npu.npu_format_cast(weight.npu(),29)
    logits = torch.nn.functional.linear(device_x,device_w)
    values, order = torch.sort(logits.cpu(),descending=True,stable=True)
    print("REAL_NATIVE_DIFFERENCE",index,rows.tolist())
    for row in rows.tolist():
        print("CUTOFF",row,values[row,:12].tolist(),order[row,:12].tolist())
    selected, ids, _ = torch_npu.npu_moe_gating_top_k_softmax(logits,k=8)
    samples, probability_samples = [], []
    graph = torch.npu.NPUGraph()
    try:
        with torch.npu.graph(graph):
            selected, ids, _ = torch_npu.npu_moe_gating_top_k_softmax(logits,k=8)
        for _ in range(20):
            graph.replay()
            samples.append(ids.cpu())
            probability_samples.append(selected.cpu())
        print("TOPK_REPEAT_DIFFERENCES",[int(torch.count_nonzero(value != samples[0])) for value in samples])
        print("TOPK_REAL_ROWS",[value[rows].tolist() for value in samples[:3]])
        print("TOPK_PROB_REPEAT",[int(torch.count_nonzero(value != probability_samples[0])) for value in probability_samples])
    finally:
        graph.reset()
    for deterministic,padded in ((False,False),(False,True),(True,False),(True,True)):
        torch.use_deterministic_algorithms(deterministic)
        graph = torch.npu.NPUGraph()
        try:
            with torch.npu.graph(graph):
                captured_logits = torch.nn.functional.linear(device_x,device_w)
                source_logits = torch.nn.functional.pad(captured_logits,(0,0,0,12)) if padded else captured_logits
                selected, ids, _ = torch_npu.npu_moe_gating_top_k_softmax(source_logits,k=8)
            logit_samples, id_samples, probability_samples = [], [], []
            for _ in range(30):
                graph.replay()
                logit_samples.append(captured_logits.cpu())
                id_samples.append(ids.cpu())
                probability_samples.append(selected.cpu()[:20])
            print("GATE_GRAPH",deterministic,padded,
                  [int(torch.count_nonzero(value != logit_samples[0])) for value in logit_samples],
                  [int(torch.count_nonzero(value != id_samples[0])) for value in id_samples],
                  [int(torch.count_nonzero(value != probability_samples[0])) for value in probability_samples])
        finally:
            graph.reset()
    torch.use_deterministic_algorithms(False)
