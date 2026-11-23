# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""A1 ModelSlim INT8 projections and integer-bias FP16 dequantization."""

from pathlib import Path
from typing import Any

import torch


def decode_a1_modelslim_scale(encoded: torch.Tensor) -> torch.Tensor:
    """Decode checked CANN INT64 scale words without numeric INT64 conversion."""
    if encoded.dtype != torch.int64 or encoded.device.type != "cpu":
        raise ValueError(
            f"Expected CPU INT64 encoded dequantization scale, got {encoded.device}/{encoded.dtype}"
        )
    if torch.count_nonzero(encoded >> 32):
        raise ValueError("A1 ModelSlim scale requires zero upper 32 bits")
    decoded = (encoded & 0xFFFFFFFF).to(torch.int32).contiguous().view(torch.float32)
    if not torch.isfinite(decoded).all() or not torch.all(decoded > 0):
        raise ValueError("A1 ModelSlim scale must encode finite positive FP32 values")
    return decoded


def make_a1_modelslim_projection(output_width: int, input_width: int) -> Any:
    """Build the exact INT32 accumulator stage for QKV or O."""
    import pypto.language as pl

    if (output_width, input_width) not in ((5120, 2048), (2048, 4096)):
        raise ValueError(f"Expected TP1 QKV/O projection, got N={output_width}, K={input_width}")
    # tiling
    M_TILE = 16
    N_TILE = 16
    K_TILE = 32

    @pl.jit.incore
    def modelslim_projection_a1(
        x: pl.Tensor[[M_TILE, input_width], pl.INT8],
        weight: pl.Tensor[[output_width, input_width], pl.INT8],
        acc_out: pl.Out[pl.Tensor[[M_TILE, output_width], pl.INT32]],
    ):
        for column in pl.range(0, output_width, N_TILE):
            lhs = pl.load(x, [0, 0], [M_TILE, K_TILE], target_memory=pl.MemorySpace.Mat)
            rhs = pl.load(weight, [column, 0], [N_TILE, K_TILE], target_memory=pl.MemorySpace.Mat)
            accumulator = pl.matmul(lhs, pl.tile.transpose_view(rhs))
            for inner in pl.range(K_TILE, input_width, K_TILE):
                lhs_next = pl.load(x, [0, inner], [M_TILE, K_TILE], target_memory=pl.MemorySpace.Mat)
                rhs_next = pl.load(
                    weight, [column, inner], [N_TILE, K_TILE], target_memory=pl.MemorySpace.Mat
                )
                accumulator = pl.matmul_acc(accumulator, lhs_next, pl.tile.transpose_view(rhs_next))
            pl.store(accumulator, [0, column], acc_out)
        return acc_out

    return modelslim_projection_a1


def compile_a1_modelslim_dequant(pto_isa_root: Path) -> bytes:
    """Compile integer bias addition followed by encoded-scale FP16 output."""
    # Scale decoding remains usable without the optional hardware compiler.
    from simpler_setup import KernelCompiler  # noqa: PLC0415

    source = Path(__file__).with_name("a1_modelslim_dequant.cpp")
    return KernelCompiler("a1").compile_single_launch(str(source), str(pto_isa_root))
