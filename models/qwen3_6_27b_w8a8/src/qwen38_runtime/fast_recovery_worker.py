# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Performance worker: compact inputs, native default HCCL; no tensor audits."""

import ctypes as C
from .compact_worker import CompactInputWorker


class FastRecoveryWorker(CompactInputWorker):
    def benchmark_mpmd_stats(self):
        import torch

        result = super().benchmark_mpmd_stats()

        class Value(C.Union):
            _fields_ = [("value", C.c_int32)]

        lib = C.CDLL("/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libhccl.so")
        lib.HcclGetConfig.argtypes = [C.c_int, C.POINTER(Value)]
        lib.HcclGetConfig.restype = C.c_int
        v = Value()
        rc = lib.HcclGetConfig(0, C.byref(v))
        if rc:
            raise RuntimeError(f"HcclGetConfig failed: {rc}")
        result["native_hccl_policy"] = dict(
            deterministic=int(v.value),
            torch_global_deterministic=torch.are_deterministic_algorithms_enabled(),
            set_by_this_worker=False,
        )
        return result
