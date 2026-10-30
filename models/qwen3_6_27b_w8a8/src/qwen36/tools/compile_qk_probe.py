# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compile the existing actual24-row/B4 QK kernel without model/communication."""

from pathlib import Path
from pypto import backend, ir
import pypto.language as pl
from qwen36.operators.attention import compute_scores

W = Path(__file__).resolve().parents[3]


@pl.function(type=pl.FunctionType.Orchestration)
def forward(
    q: pl.Tensor[[24, 6, 256], pl.BF16],
    cache: pl.Tensor[[294912, 1, 256], pl.BF16],
    table: pl.Tensor[[24, 336], pl.INT32],
    lengths: pl.Tensor[[24], pl.INT32],
    out: pl.Out[pl.Tensor[[144, 43008], pl.FP32]],
):
    out = compute_scores(q, cache, table, lengths, out)
    return out


def main():
    backend.set_backend_type(backend.BackendType.Ascend910B)
    ir.compile(
        ir.Program([compute_scores, forward], "ActualB4QK", forward.span),
        platform="a2a3",
        output_dir=W / "build/qk_b4_diagnosis",
        dump_passes=True,
    )
    print("QK_B4_DIAGNOSIS_COMPILED", flush=True)


if __name__ == "__main__":
    main()
