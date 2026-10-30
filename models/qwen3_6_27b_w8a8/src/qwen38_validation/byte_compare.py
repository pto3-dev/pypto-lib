# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exact CPU snapshot comparison without a threaded elementwise scan.

Used only by retired diagnostics, never model execution or production sampling.
"""

import ctypes
import time

_memcmp = ctypes.CDLL(None).memcmp
_memcmp.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
_memcmp.restype = ctypes.c_int
STATS = dict(calls=0, bytes=0, seconds=0.0)


def equal(a, b):
    if a.device.type != "cpu" or b.device.type != "cpu":
        raise ValueError("Only completed CPU snapshots may be compared")
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    a, b = a.contiguous(), b.contiguous()
    size = a.numel() * a.element_size()
    begin = time.monotonic()
    result = a.data_ptr() == b.data_ptr() or not size or _memcmp(a.data_ptr(), b.data_ptr(), size) == 0
    STATS["calls"] += 1
    STATS["bytes"] += size
    STATS["seconds"] += time.monotonic() - begin
    return result
