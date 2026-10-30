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


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestStaticForkJoin(SceneTestCase):
    RTOL = 1e-4
    ATOL = 2e-5
    CALLABLE = {
        "orchestration": {
            "source": "kernels/program.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.IN] + [D.INOUT] * 4,
        },
        "incores": [
            {
                "func_id": i,
                "source": "kernels/" + lane + ".cpp",
                "core_type": "aic" if lane == "aic" else "aiv",
                "extra_include_dirs": ["../../../../../src/a2a3/platform/onboard/aicore"],
                "signature": [D.IN, D.IN] + [D.INOUT] * 4 + [D.SCALAR],
            }
            for i, lane in enumerate(("aic", "aiv0", "aiv1"))
        ],
    }
    CASES = [{"name": "three_epochs", "platforms": ["a2a3"], "params": {}}]

    def generate_args(self, params):
        torch.manual_seed(42)
        return TaskArgsBuilder(
            TensorArg("x", torch.randn(1, 256, dtype=torch.bfloat16)),
            TensorArg("w", torch.randn(256, 256, dtype=torch.bfloat16) * 0.02),
            TensorArg("a", torch.zeros(1, 256)),
            TensorArg("b", torch.zeros(1, 256)),
            TensorArg("state", torch.zeros(4 * 128 * 16, dtype=torch.int64)),
            TensorArg("trace", torch.zeros(4 * 128 * 16, dtype=torch.int64)),
        )

    def compute_golden(self, args, params):
        out = args.x.float() @ args.w.float().T
        args.a.copy_(out)
        args.b.copy_(out)

    def compare_outputs(self, test_args, golden_args, output_names, params):
        target = Path(__file__).resolve().parents[5] / "outputs" / "static_program_validation"
        target.mkdir(parents=True, exist_ok=True)
        torch.save(
            {name: getattr(test_args, name).clone() for name in ("a", "b", "state", "trace")},
            target / "fork_join_actual_v3.pt",
        )
        super().compare_outputs(test_args, golden_args, ["a", "b"], params)
        events = test_args.state[::16]
        populated = events[events != 0]
        assert populated.numel() % 4 == 0 and torch.all(populated == 3)
        grid = populated.numel() // 4
        trace = test_args.trace[: 4 * grid * 16].reshape(4, grid, 16)[:, :, :5]
        assert torch.all(trace[:, :, 4] == 3)
        assert int(trace[1, :, 1].min()) >= int(trace[0, :, 2].max())
        assert int(trace[2, :, 1].min()) >= int(trace[0, :, 2].max())
        assert int(trace[3, :, 1].min()) >= int(trace[1:3, :, 2].max())
        assert torch.equal(trace[0, :, 3], trace[3, :, 3]), (
            "AIC position must be invariant across internal steps"
        )
        e = Path(__file__).resolve().parents[5] / "outputs" / "static_program_validation"
        e.mkdir(parents=True, exist_ok=True)
        torch.save(trace, e / "fork_join_trace.pt")
        (e / "fork_join_validation.json").write_text(
            json.dumps(
                dict(
                    passed=True,
                    epochs=3,
                    grid=grid,
                    nodes=4,
                    numerical_output=True,
                    dependency_order=True,
                    aic_position_invariant=True,
                    physical_core_ids=trace[:, :, 3].tolist(),
                    trace=trace.tolist(),
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
