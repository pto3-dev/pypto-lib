# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -------------------------------------------------------------------------------------------------------
"""Borrow native W8A8 expert buffers without converting or copying weight banks."""

from dataclasses import dataclass

import torch

from .config import EXPERTS, HIDDEN, LOCAL_INTERMEDIATE, W13_WIDTH


@dataclass(frozen=True)
class NativeNZWeights:
    """Keep native owners alive for the lifetime of their captured callable."""

    w13: torch.Tensor
    w13_scale: torch.Tensor
    w2: torch.Tensor
    w2_scale: torch.Tensor
    native_w2_scale: torch.Tensor

    @classmethod
    def borrow(
        cls,
        w13: torch.Tensor,
        w13_scale: torch.Tensor,
        w2: torch.Tensor,
        w2_scale: torch.Tensor,
    ) -> "NativeNZWeights":
        """Validate existing native NZ buffers; widen only the small W2 scale once.

        Call during initialization, outside capture. W13 scales must be the
        native FP32 scales, not the fused INT64 representation. The caller must
        retain this object until all graph replays have drained and finalized.
        """
        expected = (
            ("w13", w13, (EXPERTS, HIDDEN, W13_WIDTH), (torch.int8,)),
            ("w13_scale", w13_scale, (EXPERTS, W13_WIDTH), (torch.float32,)),
            ("w2", w2, (EXPERTS, LOCAL_INTERMEDIATE, HIDDEN), (torch.int8,)),
            ("w2_scale", w2_scale, (EXPERTS, HIDDEN), (torch.bfloat16, torch.float32)),
        )
        for name, tensor, shape, dtypes in expected:
            if tuple(tensor.shape) != shape or tensor.dtype not in dtypes:
                raise ValueError(
                    f"{name}: expected shape={shape}, dtype in {dtypes}; got {tensor.shape}, {tensor.dtype}"
                )
            if tensor.device.type != "npu" or tensor.device != w13.device:
                raise ValueError(f"{name}: expected the same NPU device as w13, got {tensor.device}")
            if tensor.storage_offset() != 0 or not tensor.is_contiguous():
                raise ValueError(f"{name}: expected a contiguous, zero-offset native buffer")
        import torch_npu

        for name, tensor in (("w13", w13), ("w2", w2)):
            if torch_npu.get_npu_format(tensor) != 29:
                raise ValueError(f"{name}: expected native FRACTAL_NZ (29)")
        for name, tensor in (("w13_scale", w13_scale), ("w2_scale", w2_scale)):
            if torch_npu.get_npu_format(tensor) != 2:
                raise ValueError(f"{name}: expected native ND (2)")
        return cls(w13, w13_scale, w2, w2_scale.to(torch.float32), w2_scale)
