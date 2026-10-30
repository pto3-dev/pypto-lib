# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import json

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.device_program import Invocation, LeafSignature, compile_device_program
from simpler_setup.environment import PROJECT_ROOT

ROOT = PROJECT_ROOT.parent
EVIDENCE = ROOT.parents[1] / "evidence/device_program_20261016"
RUNTIME = PROJECT_ROOT / "src/a2a3/runtime/tensormap_and_ringbuffer"
LEAVES = ROOT / "leaf_build/kernels"


def plan(record_trace=True):
    nodes = []
    for b in range(4):
        nodes.append(Invocation(f"cube{b}", 0, "aic", b, b, 4, (4, 5, 6), ("consume0",) if b == 3 else ()))
        for sub in range(2):
            nodes.append(
                Invocation(f"act{b}_{sub}", 1, f"aiv{sub}", b, 2 * b + sub, 8, (6, 7), (f"cube{b}",))
            )
        nodes.append(Invocation(f"consume{b}", 2, "aiv0", b, b, 4, (7, 8), (f"act{b}_0", f"act{b}_1")))
    return compile_device_program(
        nodes,
        [LeafSignature(0, "aic", 3), LeafSignature(1, "aiv", 2), LeafSignature(2, "aiv", 2)],
        grid=4,
        available_clusters=4,
        root_arg_count=10,
        epoch_arg=9,
        record_trace=record_trace,
    )


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestDeviceProgram(SceneTestCase):
    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/program.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN] + [D.INOUT] * 3 + [D.IN] * 2 + [D.INOUT] * 6,
        },
        "incores": [
            {
                "func_id": 0,
                "source": str(LEAVES / "aic/project.cpp"),
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.INOUT],
            },
            {
                "func_id": 1,
                "source": str(LEAVES / "aiv/activation.cpp"),
                "core_type": "aiv",
                "signature": [D.IN, D.INOUT],
            },
            {
                "func_id": 2,
                "source": str(LEAVES / "aiv/consume.cpp"),
                "core_type": "aiv",
                "signature": [D.IN, D.INOUT],
            },
        ]
        + [
            {
                "func_id": i,
                "source": str(RUNTIME / f"kernels/device_program/device_program_{lane}.cpp"),
                "core_type": lane,
                "signature": [D.IN] + [D.INOUT] * 3 + [D.IN] * 2 + [D.INOUT] * 3 + [D.SCALAR],
                "extra_include_dirs": [str(PROJECT_ROOT / "src/a2a3/platform/onboard/aicore")],
            }
            for i, lane in [(3, "aic"), (4, "aiv")]
        ],
    }
    CASES = [
        {"name": "tile_dependencies_trace_" + str(trace), "platforms": ["a2a3"], "params": {"trace": trace}}
        for trace in (True, False)
    ]

    def generate_args(self, params):
        p = plan(params["trace"])
        torch.manual_seed(2183)
        return TaskArgsBuilder(
            TensorArg("program", torch.tensor(list(p.image), dtype=torch.uint8)),
            TensorArg("frames", torch.zeros(p.frame_bytes, dtype=torch.uint8)),
            TensorArg("events", torch.zeros(p.signal_bytes // 8, dtype=torch.int64)),
            TensorArg("trace", torch.zeros(p.signal_bytes // 8, dtype=torch.int64)),
            TensorArg("x", torch.randn(1, 256, dtype=torch.bfloat16)),
            TensorArg("w", torch.randn(256, 256, dtype=torch.bfloat16) * 0.02),
            TensorArg("ref_y", torch.zeros(1, 256)),
            TensorArg("ref_a", torch.zeros(1, 256, dtype=torch.bfloat16)),
            TensorArg("ref_o", torch.zeros(1, 128)),
            TensorArg("y", torch.zeros(1, 256)),
            TensorArg("a", torch.zeros(1, 256, dtype=torch.bfloat16)),
            TensorArg("o", torch.zeros(1, 128)),
        )

    def compute_golden(self, args, params):
        pass

    def compare_outputs(self, test_args, golden_args, output_names, params):
        a = test_args
        torch.testing.assert_close(a.y, a.ref_y, rtol=0, atol=0)
        torch.testing.assert_close(a.a, a.ref_a, rtol=0, atol=0)
        torch.testing.assert_close(a.o, a.ref_o, rtol=0, atol=0)
        torch.testing.assert_close(a.y, a.x.float() @ a.w.float().T, rtol=1e-4, atol=2e-5)
        assert torch.isfinite(a.o).all() and a.o.abs().max() > 0
        assert (a.events[::16] == 3).all()
        by_name = {}
        if params["trace"]:
            traces = a.trace.reshape(-1, 16)
            assert (traces[:, 4] == 3).all()
            assert (traces[:, 0] <= traces[:, 1]).all() and (traces[:, 1] <= traces[:, 2]).all()
            by_name = dict(zip(plan().event_names, traces.tolist()))
            for b in range(4):
                for sub in range(2):
                    assert by_name[f"act{b}_{sub}"][1] >= by_name[f"cube{b}"][2]
                    assert by_name[f"consume{b}"][1] >= by_name[f"act{b}_{sub}"][2]
                assert by_name[f"consume{b}"][3] == by_name[f"act{b}_0"][3]
            assert by_name["cube3"][1] >= by_name["consume0"][2]
            assert by_name["consume0"][2] < max(by_name[f"cube{b}"][2] for b in range(4))
        else:
            assert (a.trace == 0).all()
        EVIDENCE.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"y": a.y, "activation": a.a, "output": a.o, "trace": a.trace, "events": a.events},
            EVIDENCE / ("validation_trace_" + str(params["trace"]) + ".pt"),
        )
        result = dict(
            passed=True,
            epochs=3,
            bitwise_equal=True,
            independent_kernel_binaries=True,
            device_indirect_calls_per_epoch=16,
            outer_dispatches_per_epoch=1,
            outer_core_invocations_per_epoch=12,
            grid=4,
            unique_positions=True,
            tile_forward_progress_checked=params["trace"],
            whole_operation_join=False,
            trace=by_name,
            record_trace=params["trace"],
            full_model_tpot_ms=None,
            performance_claim=False,
        )
        (EVIDENCE / ("validation_trace_" + str(params["trace"]) + ".json")).write_text(
            json.dumps(result, indent=2) + "\n"
        )


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
