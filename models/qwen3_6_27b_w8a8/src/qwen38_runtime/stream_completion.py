# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bounded completion for owned/caller streams; never reset a shared device."""

import ctypes as C


def synchronize_stream(stream, timeout_ms=30000):
    if not 0 < timeout_ms <= 2147483647:
        raise ValueError("Completion timeout must be a positive int32")
    lib = C.CDLL("libascendcl.so")
    fn = lib.aclrtSynchronizeStreamWithTimeout
    fn.argtypes = [C.c_void_p, C.c_int32]
    fn.restype = C.c_int
    rc = fn(int(stream), timeout_ms)
    if rc:
        raise RuntimeError(f"Caller stream completion failed or timed out: {rc}; resources retained")
