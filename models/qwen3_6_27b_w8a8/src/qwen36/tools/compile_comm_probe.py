# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Quota-only communication+Norm+Quant probe, same math and shared-memory ABI."""

import argparse
from pathlib import Path
from pypto import backend, ir
import pypto.language as pl
import pypto.language.distributed as pld

W = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quota", type=int, choices=[12, 24], required=True)
    parser.add_argument("--lane", action="store_true")
    parser.add_argument("--series", action="store_true")
    args = parser.parse_args()
    quota = args.quota
    iterations = 64 if args.series else 1
    if args.lane and not args.series:
        raise ValueError("Lane protocol must exercise all64 odd operations per epoch")
    backend.set_backend_type(backend.BackendType.Ascend910B)
    if args.lane:
        from qwen36.operators.collective_lanes import reduce_add_norm_quant_kernel
    else:
        from qwen36.operators.collectives import reduce_add_norm_quant_kernel

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
        quantized: pl.Out[pl.Tensor[[24, 5120], pl.INT8]],
        scales: pl.Out[pl.Tensor[[24, 1], pl.FP32]],
    ):
        width = pl.min(pl.system.available_aiv_count(), quota)
        for odd_index in pl.range(iterations):
            operation = odd_index * 2 + 1
            with pl.spmd(width, allow_early_resolve=True):
                out, normalized, data, signal, quantized, scales = reduce_add_norm_quant_kernel(
                    x, residual, out, weight, normalized, data, signal, metadata, operation, quantized, scales
                )
        return out, normalized, quantized, scales

    program = ir.Program([reduce_add_norm_quant_kernel, forward], f"CommQuota{quota}", forward.span)
    ir.compile(
        program,
        platform="a2a3",
        output_dir=W
        / "build"
        / f"comm_probe_{'lane' if args.lane else ''}{'series' if args.series else ''}{quota}",
        dump_passes=True,
    )
    print("COMM_PROBE_COMPILED", quota, flush=True)


if __name__ == "__main__":
    main()
