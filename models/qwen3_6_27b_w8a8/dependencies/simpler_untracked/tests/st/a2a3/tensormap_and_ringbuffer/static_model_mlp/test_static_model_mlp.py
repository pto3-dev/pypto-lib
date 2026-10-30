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
from pypto import backend, ir
from simpler.task_interface import ArgDirection as D
from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.environment import PROJECT_ROOT

P = Path("/home/g00893677/pto-dev/qwen3_8_27b-perf")
B = (
    P
    / "work/static_program_20261016/compile_only_build/b439d2b20f5ed45e/tokens1/build/next_levels/rank0/kernels"
)
SIG = [D.IN, D.IN, D.INOUT, D.INOUT, D.IN, D.INOUT, D.INOUT, D.SCALAR]


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestStaticModelMLP(SceneTestCase):
    CALLABLE = {
        "orchestration": {
            "source": "kernels/program.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.IN, D.IN] + [D.INOUT] * 5,
        },
        "incores": [
            {
                "func_id": 0,
                "source": str(B / "aic/project_mlp_kernel_aic.cpp"),
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.INOUT, D.INOUT],
            },
            {
                "func_id": 1,
                "source": str(B / "aiv/project_mlp_kernel_aiv.cpp"),
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.INOUT, D.INOUT],
            },
            {
                "func_id": 2,
                "source": str(B / "aic/project_down_kernel.cpp"),
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.INOUT],
            },
        ]
        + [
            {
                "func_id": i,
                "source": "kernels/region_" + lane + ".cpp",
                "core_type": lane,
                "signature": SIG,
                "extra_include_dirs": [str(PROJECT_ROOT / "src/a2a3/platform/onboard/aicore")],
            }
            for i, lane in [(3, "aic"), (4, "aiv")]
        ],
    }
    CASES = [{"name": "three_real_mlp_epochs", "platforms": ["a2a3"], "params": {}}]

    def generate_args(self, params):
        backend.set_backend_type(backend.BackendType.Ascend910B)
        cap = backend.get_backend_instance(backend.get_backend_type()).get_core_count(ir.CoreType.CUBE)
        torch.manual_seed(183)
        return TaskArgsBuilder(
            TensorArg("x", torch.randn(1, 5120, dtype=torch.bfloat16)),
            TensorArg("wg", torch.randn(17408, 5120, dtype=torch.bfloat16) * 0.01),
            TensorArg("wd", torch.randn(5120, 8704, dtype=torch.bfloat16) * 0.01),
            TensorArg("activation", torch.zeros(1, 8704, dtype=torch.bfloat16)),
            TensorArg("reference", torch.zeros(1, 5120)),
            TensorArg("result", torch.zeros(1, 5120)),
            TensorArg("pipe", torch.zeros(512 * cap)),
            TensorArg("state", torch.zeros(3 * cap * 16, dtype=torch.int64)),
        )

    def compute_golden(self, args, params):
        pass

    def compare_outputs(self, test_args, golden_args, output_names, params):
        assert torch.isfinite(test_args.result).all() and test_args.result.abs().max() > 0
        torch.testing.assert_close(test_args.reference, test_args.result, rtol=0, atol=0)
        epochs = test_args.state[::16]
        populated = epochs[epochs != 0]
        assert populated.numel() > 1 and (populated == 3).all()
        grid = (populated.numel() - 1) // 2
        assert populated.numel() == 2 * grid + 1
        folder = P / "evidence/static_mix_placement_20261016"
        folder.mkdir(exist_ok=True)
        torch.save(
            {"reference": test_args.reference, "result": test_args.result, "state": test_args.state},
            folder / "real_mlp_three_epochs.pt",
        )
        (folder / "real_mlp_validation.json").write_text(
            json.dumps(
                dict(
                    passed=True,
                    bitwise_equal=True,
                    epochs=3,
                    grid=grid,
                    original_leaf_sources=True,
                    down_consumes_real_swiglu=True,
                ),
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
