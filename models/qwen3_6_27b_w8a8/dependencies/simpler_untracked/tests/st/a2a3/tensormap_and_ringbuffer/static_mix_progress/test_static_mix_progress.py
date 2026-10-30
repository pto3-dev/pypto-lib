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


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestStaticMixProgress(SceneTestCase):
    CALLABLE = {
        "orchestration": {
            "source": "kernels/program.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT, D.INOUT],
        },
        "incores": [
            {
                "func_id": i,
                "source": "kernels/" + lane + ".cpp",
                "core_type": "aic" if i < 2 else "aiv",
                "signature": ([D.IN, D.OUT] if i == 4 else [D.INOUT, D.INOUT]),
                "extra_include_dirs": [
                    str(PROJECT_ROOT / "src/a2a3/platform/onboard/aicore"),
                    str(PROJECT_ROOT / "src/a2a3/platform/include"),
                ],
            }
            for i, lane in enumerate(["blocker", "aic", "aiv0", "aiv1", "marker"])
        ],
    }
    CASES = [{"name": "vector_runs_while_target_cube_busy", "platforms": ["a2a3"], "params": {}}]

    def generate_args(self, params):
        backend.set_backend_type(backend.BackendType.Ascend910B)
        cap = backend.get_backend_instance(backend.get_backend_type()).get_core_count(ir.CoreType.CUBE)
        return TaskArgsBuilder(
            TensorArg("state", torch.zeros((2 + 2 * cap) * 16, dtype=torch.int64)),
            TensorArg("trace", torch.zeros((1 + 3 * cap) * 16, dtype=torch.int64)),
        )

    def compute_golden(self, args, params):
        pass

    def compare_outputs(self, test_args, golden_args, output_names, params):
        data = test_args.trace.reshape(-1, 16)
        grid = int(data[1, 3])
        assert grid > 0
        trace = data[1 : 1 + 3 * grid].reshape(3, grid, 16)
        assert (trace[:, :, 0] > 0).all() and (trace[:, :, 1] >= trace[:, :, 0]).all()
        assert trace[1, 0, 0] < data[0, 1] <= trace[0, 0, 0], "AIV must start before blocked AIC exits"
        assert data[0, 2] == trace[0, 0, 2], "Fixed AIC target must stay behind its own blocker"
        for lane in range(3):
            assert len(set(trace[lane, :, 2].tolist())) == grid
        assert len(set(trace[1:, :, 2].flatten().tolist())) == 2 * grid
        e = Path("/home/g00893677/pto-dev/qwen3_8_27b-perf/evidence/aic_fixed_reuse_20261017")
        e.mkdir(parents=True, exist_ok=True)
        (e / "independent_progress.json").write_text(
            json.dumps(
                dict(
                    passed=True,
                    grid=grid,
                    independent_vector_start=True,
                    unique_cube_and_vector_placement=True,
                    blocker=data[0, :4].tolist(),
                    trace=trace[:, :, :4].tolist(),
                ),
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
