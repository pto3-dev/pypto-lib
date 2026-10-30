# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Validate one weight family with two device scalar reads, not three per layer."""


def validate_scale_bank(scales, offsets):
    import torch

    if not scales or len(scales) != len(offsets):
        raise ValueError("Incomplete quantization metadata family")
    scale_bank = torch.stack(scales).contiguous()
    offset_bank = torch.cat([v.detach().reshape(-1) for v in offsets])
    if torch.count_nonzero(offset_bank).item() != 0:
        raise ValueError("Nonzero weight_offset is unsupported")
    valid = torch.isfinite(scale_bank).all() & (scale_bank > 0).all()
    if not bool(valid.item()):
        raise ValueError("Weight scales must be finite and positive")
    return scale_bank
