"""Replay actual native dispatch/combine boundaries without PTO arithmetic."""

import os

import pytest
import torch


def test_actual_native_moe_boundaries():
    path = os.environ.get("PTO3_B20_REAL_SNAPSHOT")
    if not path:
        pytest.skip("PTO3_B20_REAL_SNAPSHOT must name a native MoE trace")
    import torch_npu

    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    snapshot = torch.load(path,map_location="cpu",weights_only=True)
    differences = []
    for index in range(48):
        logits, selected, ids = snapshot["topk"][index]
        sorted_hidden, inverse, groups = snapshot["dispatch"][index]
        hidden = snapshot["stages"][index]["post_norm"][0]
        expected_groups = torch.bincount(ids.flatten().long(),minlength=128).cumsum(0)
        record = {"layer":index,"groups":int(torch.count_nonzero(groups != expected_groups)),
                  "dispatch":int(torch.count_nonzero(sorted_hidden[inverse.long()] != hidden.repeat_interleave(8,0)))}
        first, group13, up_gate = snapshot["gmm"][index][2048]
        activated, group2, expert_y = snapshot["gmm"][index][768]
        record["gmm1_input"] = int(torch.count_nonzero(first != sorted_hidden))
        record["gmm_groups"] = int(torch.count_nonzero(group13 != groups)) + int(torch.count_nonzero(group2 != groups))
        native_activation = torch_npu.npu_swiglu(up_gate.npu())
        record["activation"] = int(torch.count_nonzero(native_activation.cpu() != activated))
        probability = selected.npu()
        probability = probability / probability.sum(-1,keepdim=True)
        combined = torch_npu.npu_moe_token_unpermute(permuted_tokens=expert_y.npu(),
                                                    sorted_indices=inverse.npu(),probs=probability)
        record["combine"] = int(torch.count_nonzero(combined.cpu() != snapshot["layers"][index][0]))
        if any(value for name,value in record.items() if name != "layer"):
            differences.append(record)
    print("NATIVE_MOE_BOUNDARY_DIFFERENCES", differences)
    assert not differences
