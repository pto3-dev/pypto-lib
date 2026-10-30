# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import ctypes
from pathlib import Path

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestRopeAppend(SceneTestCase):
    RTOL = 0.01
    ATOL = 0.01
    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/program.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN] * 6 + [D.INOUT, D.INOUT, D.IN, D.INOUT, D.SCALAR],
        },
        "incores": [
            {"func_id": i, "source": "kernels/aiv/" + name + ".cpp", "core_type": "aiv", "signature": sig}
            for i, name, sig in [
                (0, "norm", [D.IN] * 7 + [D.INOUT]),
                (1, "append", [D.IN, D.IN, D.INOUT, D.INOUT, D.IN]),
                (2, "fused", [D.IN] * 6 + [D.INOUT, D.INOUT, D.IN, D.IN, D.INOUT]),
                (3, "static", [D.IN] * 6 + [D.INOUT, D.INOUT, D.IN, D.IN, D.INOUT, D.INOUT, D.IN]),
            ]
        ],
    }
    CASES = [
        {"name": name, "platforms": ["a2a3"], "params": {"mode": mode}}
        for mode, name in [(0, "control"), (1, "static_sequence"), (2, "ub_fusion")]
    ]

    def generate_args(self, params):
        torch.manual_seed(42)
        return TaskArgsBuilder(
            TensorArg("x", torch.randn(1, 2, 256, dtype=torch.bfloat16)),
            TensorArg("w", torch.randn(1, 256, dtype=torch.bfloat16) * 0.02),
            TensorArg("cos", torch.randn(512, 32, dtype=torch.bfloat16)),
            TensorArg("sin", torch.randn(512, 32, dtype=torch.bfloat16)),
            TensorArg("position", torch.tensor([300])),
            TensorArg("v", torch.randn(1, 512, dtype=torch.bfloat16)),
            TensorArg("key", torch.full((12288, 512), 0.125, dtype=torch.bfloat16)),
            TensorArg("value", torch.full((12288, 512), 0.125, dtype=torch.bfloat16)),
            TensorArg("slot", torch.tensor([300])),
            TensorArg("temp", torch.zeros(1, 2, 256, dtype=torch.bfloat16)),
            Scalar("mode", ctypes.c_int64(params["mode"])),
        )

    def compute_golden(self, args, params):
        x = args.x.float()
        normalized = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-6) * (1 + args.w).bfloat16().float()
        first = normalized[..., :32]
        second = normalized[..., 32:64]
        c = args.cos[300].float()
        s = args.sin[300].float()
        out = normalized.clone()
        out[..., :32] = first * c - second * s
        out[..., 32:64] = second * c + first * s
        args.key[300].copy_(out.bfloat16().reshape(512))
        args.value[300].copy_(args.v.reshape(512))

    def compare_outputs(self, test_args, golden_args, output_names, params):
        super().compare_outputs(test_args, golden_args, [n for n in output_names if n != "temp"], params)
        dest = Path(__file__).resolve().parents[5] / "outputs" / "static_program_validation"
        dest.mkdir(parents=True, exist_ok=True)
        reference = dest / "rope_control.pt"
        got = {"key": test_args.key.clone(), "value": test_args.value.clone()}
        if params["mode"] == 0:
            torch.save(got, reference)
        else:
            old = torch.load(reference, weights_only=True)
            assert all(torch.equal(got[k], old[k]) for k in old), (
                "Expected bitwise agreement with original PTO kernels"
            )
        torch.save(got, dest / ("rope_mode" + str(params["mode"]) + ".pt"))


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
