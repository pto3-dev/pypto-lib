# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""BF16-preserving A1 vocabulary projection with resident FP32-expanded weights."""

from pathlib import Path

import torch


def prepack_a1_lm_head(weight: torch.Tensor) -> torch.Tensor:
    """Exactly expand BF16[N,2048] to contiguous FP32[2048,N] once.

    This is not a weight quantization or an FP16 model conversion. The prepared
    projection preserves the BF16 output boundary before publishing FP32 logits.
    """
    if weight.ndim != 2 or weight.shape[1] != 2048 or weight.shape[0] % 128 != 0:
        raise ValueError(f"Expected LM Head BF16[N,2048], N divisible by 128; got {weight.shape}")
    if weight.dtype != torch.bfloat16 or not torch.isfinite(weight).all():
        raise ValueError(f"Expected finite BF16 LM Head weights, got {weight.dtype}")
    return weight.float().t().contiguous()


def compile_a1_lm_head(pto_isa_root: str | Path) -> bytes:
    """Compile the eight-block BF16-semantics LM Head vector projection."""
    from simpler_setup import KernelCompiler

    return KernelCompiler("a1").compile_single_launch(
        str(Path(__file__).with_name("a1_lm_head.cpp")), str(pto_isa_root)
    )
