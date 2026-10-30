# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import json
from pathlib import Path

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test

W = Path("/home/g00893677/pto-dev/qwen3_8_27b-perf/work/device_program_model_20261016")
B = next((W / "candidate_build").glob("*/tokens1/build/next_levels/rank0"))
R = json.loads((B.parents[1] / "attention_program_manifest.json").read_text())[0]
NAMES = [
    (23, "aic", "compute_scores_kernel_aic", 8),
    (24, "aiv", "compute_scores_kernel_aiv", 8),
    (25, "aiv", "softmax_scores_kernel", 2),
    (26, "aic", "compute_values_kernel_aic", 8),
    (27, "aiv", "compute_values_kernel_aiv", 8),
    (28, "aiv", "gate_attention_kernel", 3),
    (35, "aic", "device_attention_program_aic", 22),
    (36, "aiv", "device_attention_program_aiv", 22),
]


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestDeviceAttentionModel(SceneTestCase):
    CALLABLE = {
        "orchestration": {
            "source": "kernels/program.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN] * 6 + [D.INOUT] * 11,
        },
        "incores": [
            {
                "func_id": fid,
                "source": str(B / "kernels" / lane / (name + ".cpp")),
                "core_type": lane,
                "signature": ([D.INOUT] * (argc - 1) + [D.SCALAR] if fid >= 35 else [D.INOUT] * argc),
                "extra_include_dirs": [
                    str(W.parent / "device_program_20261016/simpler/src/a2a3/platform/onboard/aicore")
                ],
            }
            for fid, lane, name, argc in NAMES
        ],
    }
    CASES = [{"name": "original_attention_kernels_three_epochs", "platforms": ["a2a3"], "params": {}}]

    def generate_args(self, params):
        torch.manual_seed(11839)
        storage = torch.zeros(56320, dtype=torch.uint8)
        image = (B / "attention_program.bin").read_bytes()
        storage[: len(image)].copy_(torch.tensor(list(image), dtype=torch.uint8))
        return TaskArgsBuilder(
            TensorArg("q", torch.randn(1, 12, 256, dtype=torch.bfloat16) * 0.1),
            TensorArg("k", torch.randn(12288, 2, 256, dtype=torch.bfloat16) * 0.1),
            TensorArg("v", torch.randn(12288, 2, 256, dtype=torch.bfloat16) * 0.1),
            TensorArg("table", torch.arange(4, dtype=torch.int32).reshape(1, 4)),
            TensorArg("lengths", torch.tensor([300], dtype=torch.int32)),
            TensorArg("gate", torch.randn(1, 3072, dtype=torch.bfloat16)),
            *[
                TensorArg(name, torch.zeros(*shape, dtype=dtype))
                for name, shape, dtype in [
                    ("rs", (12, 512), torch.float32),
                    ("rp", (12, 512), torch.bfloat16),
                    ("rv", (1, 12, 256), torch.bfloat16),
                    ("rg", (1, 3072), torch.bfloat16),
                    ("s", (12, 512), torch.float32),
                    ("p", (12, 512), torch.bfloat16),
                    ("val", (1, 12, 256), torch.bfloat16),
                    ("g", (1, 3072), torch.bfloat16),
                    ("sp", (8192,), torch.float32),
                    ("vp", (16384,), torch.float32),
                ]
            ],
            TensorArg("storage", storage),
        )

    def compute_golden(self, args, params):
        pass

    def compare_outputs(self, a, golden, names, params):
        for x, y in [("rs", "s"), ("rp", "p"), ("rv", "val"), ("rg", "g")]:
            torch.testing.assert_close(getattr(a, x), getattr(a, y), rtol=0, atol=0)
        assert torch.isfinite(a.g).all() and a.g.abs().max() > 0
        trace = a.storage[15360:17920].view(torch.int64).reshape(20, 16)
        assert (trace[:, 4] == 3).all()
        by_name = dict(zip(R["event_names"], trace.tolist()))
        for b in range(2):
            assert by_name["pv_c" + str(b)][3] == by_name["score_c" + str(b)][3]
        e = W.parents[1] / "evidence/device_program_model_scene_20261016"
        e.mkdir(parents=True, exist_ok=True)
        torch.save({"trace": trace, "result": a.g, "reference": a.rg}, e / "attention.pt")
        (e / "validation.json").write_text(
            json.dumps(
                dict(
                    passed=True,
                    epochs=3,
                    all_intermediates_bitwise_equal=True,
                    original_model_kernels=True,
                    trace=by_name,
                ),
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
