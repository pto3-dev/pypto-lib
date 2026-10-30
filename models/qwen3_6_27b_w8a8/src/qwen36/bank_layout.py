# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared bank geometry and vector parameter offsets; no source generation."""

from math import prod
from .model_layout import I, H, C, Z, F

GROUPS = 16


def _float_layout(spec):
    small = {}
    offset = 0
    for (index, key), (_, old, shape) in spec.weights.items():
        if index >= 4:
            continue
        if (
            len(shape) == 2
            and key != "linear_attn.conv1d.weight"
            and (not key.startswith("linear_attn.in_proj_a"))
            and (not key.startswith("linear_attn.in_proj_b"))
        ):
            continue
        offset = (offset + 31) // 32 * 32
        small[index, key] = (offset, shape, old)
        offset += prod(shape)
    count = (offset + 31) // 32 * 32
    banks = []
    for i in range(4):
        for kind, n, k in (
            [("gateup", 2 * I, 5120), ("down", 5120, I), ("out", 5120, F), ("qkvz", C + Z, 5120)]
            if i != 3
            else [
                ("gateup", 2 * I, 5120),
                ("down", 5120, I),
                ("out", 5120, F),
                ("qkv", 2 * F + 2 * H * 256, 5120),
            ]
        ):
            banks.append((f"bank{i}_{kind}", (GROUPS, n, k), "BF16", ""))
    return (small, count, banks)


def is_quant_bank(name):
    return name.endswith(("_gateup", "_down")) or name in ("bank3_qkv", "bank3_out")


def layout(spec):
    small, count, banks = _float_layout(spec)
    return (
        small,
        count,
        [
            (
                name,
                (shape[0], shape[2], shape[1]) if is_quant_bank(name) else shape,
                "INT8" if is_quant_bank(name) else dtype,
                direction,
            )
            for name, shape, dtype, direction in banks
        ],
    )
