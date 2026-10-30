# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compile two complete down intervals from explicit source, no rewriting."""

from pathlib import Path
import argparse
from pypto import backend, ir
import pypto.language as pl

W = Path(__file__).resolve().parents[3]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["reference", "candidate"], required=True)
    args = p.parse_args()
    backend.set_backend_type(backend.BackendType.Ascend910B)
    from qwen36.operators.quantized import quant_4352

    if args.variant == "reference":
        from qwen36.operators.quantized import project_down_kernel, project_down
    else:
        from qwen36.operators.down_prefetch import project_down_kernel, project_down

    @pl.function(type=pl.FunctionType.Orchestration)
    def forward(
        x: pl.Tensor[[24, 4352], pl.BF16],
        weight: pl.Tensor[[16, 4352, 5120], pl.INT8, pl.NZ],
        scales: pl.Tensor[[16, 5120], pl.FP32],
        out: pl.Out[pl.Tensor[[24, 5120], pl.FP32]],
    ) -> pl.Tensor[[24, 5120], pl.FP32]:
        out = project_down(x, weight, out, 0, scales)
        return out

    program = ir.Program(
        [quant_4352, project_down_kernel, project_down, forward], f"Down_{args.variant}", forward.span
    )
    ir.compile(
        program, platform="a2a3", output_dir=W / "build" / f"down_probe_{args.variant}", dump_passes=True
    )
    print("DOWN_PROBE_COMPILED", args.variant, flush=True)


if __name__ == "__main__":
    main()
