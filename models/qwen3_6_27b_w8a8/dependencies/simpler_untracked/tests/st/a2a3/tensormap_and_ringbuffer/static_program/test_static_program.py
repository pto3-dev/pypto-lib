# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import ctypes

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestStaticProgram(SceneTestCase):
    RTOL = 1e-4
    ATOL = 2e-5
    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/program.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.IN, D.INOUT, D.INOUT, D.INOUT, D.SCALAR, D.SCALAR],
        },
        "incores": [
            {
                "func_id": i,
                "source": "kernels/aic/" + name + ".cpp",
                "core_type": "aic",
                "extra_include_dirs": ["../../../../../src/a2a3/platform/onboard/aicore"],
                "signature": sig,
            }
            for i, name, sig in [
                (0, "project", [D.IN, D.IN, D.INOUT]),
                (1, "sequence", [D.IN, D.IN, D.INOUT, D.INOUT]),
                (2, "region", [D.IN, D.IN, D.INOUT, D.INOUT, D.INOUT, D.SCALAR]),
            ]
        ],
    }
    CASES = [
        {"name": name, "platforms": ["a2a3"], "params": {"mode": mode}}
        for mode, name in [(0, "dynamic_boundaries"), (1, "static_sequence"), (2, "static_cross_core_gate")]
    ]

    def generate_args(self, params):
        torch.manual_seed(42)
        return TaskArgsBuilder(
            TensorArg("x", torch.randn(1, 256, dtype=torch.bfloat16)),
            TensorArg("w", torch.randn(256, 256, dtype=torch.bfloat16) * 0.02),
            TensorArg("a", torch.zeros(1, 256)),
            TensorArg("b", torch.zeros(1, 256)),
            TensorArg("state", torch.zeros(2, 128, 16, dtype=torch.int64)),
            Scalar("mode", ctypes.c_int64(params["mode"])),
            Scalar("epoch", ctypes.c_int64(1)),
        )

    def compare_outputs(self, test_args, golden_args, output_names, params):
        super().compare_outputs(test_args, golden_args, [n for n in output_names if n != "state"], params)
        if params["mode"] == 2:
            populated = test_args.state.flatten()[::16]
            populated = populated[populated != 0]
            assert populated.numel() > 0 and populated.numel() % 2 == 0
            assert torch.all(populated == 1)

    def compute_golden(self, args, params):
        expected = args.x.float() @ args.w.float().T
        args.a.copy_(expected)
        args.b.copy_(expected)
        # State is protocol scratch, not a numerical model output.


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
