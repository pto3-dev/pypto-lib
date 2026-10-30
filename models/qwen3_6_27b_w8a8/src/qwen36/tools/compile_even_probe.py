# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""64 even collectives in actual3 AddNorm:1 Add order; reference vs lane."""

import argparse
from pathlib import Path
from pypto import backend, ir
import pypto.language as pl
import pypto.language.distributed as pld

W = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", action="store_true")
    args = parser.parse_args()
    backend.set_backend_type(backend.BackendType.Ascend910B)
    if args.lane:
        from qwen36.operators.collective_even_lanes import reduce_add_norm_kernel, reduce_add_kernel
    else:
        from qwen36.operators.collectives import reduce_add_norm_kernel, reduce_add_kernel

    @pl.function(type=pl.FunctionType.Orchestration)
    def forward(
        x: pl.Tensor[[24, 5120], pl.FP32],
        residual: pl.Tensor[[24, 5120], pl.BF16],
        weight: pl.Tensor[[1, 5120], pl.BF16],
        metadata: pl.Tensor[[4], pl.INT32],
        data: pl.InOut[pld.DistributedTensor[[3096, 5120], pl.BF16]],
        signal: pl.InOut[pld.DistributedTensor[[516, 1], pl.INT32]],
        out: pl.Out[pl.Tensor[[24, 5120], pl.BF16]],
        normalized: pl.Out[pl.Tensor[[24, 5120], pl.BF16]],
    ):
        width = pl.min(pl.system.available_aiv_count(), 24)
        for group in pl.range(16):
            for inner in pl.range(3):
                operation = group * 8 + inner * 2 + 2
                with pl.spmd(width, allow_early_resolve=True):
                    out, normalized, data, signal = reduce_add_norm_kernel(
                        x, residual, out, weight, normalized, data, signal, metadata, operation
                    )
            operation_end = group * 8 + 8
            with pl.spmd(width, allow_early_resolve=True):
                out, data, signal = reduce_add_kernel(x, residual, out, data, signal, metadata, operation_end)
        return out, normalized

    program = ir.Program(
        [reduce_add_norm_kernel, reduce_add_kernel, forward], "EvenCollectives", forward.span
    )
    ir.compile(
        program,
        platform="a2a3",
        output_dir=W / "build" / ("even_probe_lane" if args.lane else "even_probe_reference"),
        dump_passes=True,
    )
    print("EVEN_PROBE_COMPILED", args.lane, flush=True)


if __name__ == "__main__":
    main()
