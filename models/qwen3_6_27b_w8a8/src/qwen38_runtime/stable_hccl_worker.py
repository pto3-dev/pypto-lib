# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Serving candidate: native HCCL determinism only, with no tensor audits.

Explicitly opt in using native_tp_reduce=deterministic_bf16. Initialization
queries/sets CANN's public configuration; the token path is unchanged.
"""

import ctypes as C
from functools import wraps
from .native_nz_worker import NativeNzWorker


class ConfigValue(C.Union):
    _fields_ = [("value", C.c_int32)]


def read_config(handle):
    value = ConfigValue()
    rc = handle.HcclGetConfig(0, C.byref(value))
    if rc:
        raise RuntimeError(f"HcclGetConfig failed: {rc}")
    return int(value.value)


def configure():
    import torch

    handle = C.CDLL("/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libhccl.so")
    handle.HcclGetConfig.argtypes = [C.c_int, C.POINTER(ConfigValue)]
    handle.HcclGetConfig.restype = C.c_int
    handle.HcclSetConfig.argtypes = [C.c_int, ConfigValue]
    handle.HcclSetConfig.restype = C.c_int
    before = read_config(handle)
    value = ConfigValue()
    value.value = 1
    rc = handle.HcclSetConfig(0, value)
    if rc:
        raise RuntimeError(f"HcclSetConfig failed: {rc}")
    after = read_config(handle)
    if after != 1:
        raise RuntimeError("HCCL deterministic configuration did not persist")
    return handle, dict(
        before=before,
        after=after,
        torch_global_deterministic=torch.are_deterministic_algorithms_enabled(),
        tensor_audits=False,
    )


def install():
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    original = NPUModelRunner.load_model
    if getattr(original, "_stable_hccl_installed", False):
        return

    @wraps(original)
    def load(runner, *args, **kwargs):
        settings = runner.vllm_config.additional_config["qwen38_pto_mpmd"]
        if settings.get("native_tp_reduce") != "deterministic_bf16":
            raise ValueError("StableHcclWorker requires native_tp_reduce=deterministic_bf16")
        result = original(runner, *args, **kwargs)
        runner._stable_hccl_handle, runner._stable_hccl_audit = configure()
        return result

    load._stable_hccl_installed = True
    NPUModelRunner.load_model = load


install()


class StableHcclWorker(NativeNzWorker):
    def benchmark_mpmd_stats(self):
        result = super().benchmark_mpmd_stats()
        audit = dict(self.model_runner._stable_hccl_audit)
        audit["after_serving"] = read_config(self.model_runner._stable_hccl_handle)
        if audit["after_serving"] != 1:
            raise RuntimeError("HCCL determinism changed during serving")
        result["deterministic_native_tp"] = audit
        return result
