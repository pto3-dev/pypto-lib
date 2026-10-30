# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Choose the matched runtime from an explicit artifact contract."""

from .borrowed_runtime import BorrowedRuntime
from .host_graph_runtime import HostGraphRuntime


def create_target_runtime(bundle, binaries, device, **kwargs):
    runtime = bundle.contract.get("runtime", "tensormap_and_ringbuffer")
    if runtime == "host_build_graph":
        return HostGraphRuntime(binaries, device, bundle.image, **kwargs)
    if runtime == "tensormap_and_ringbuffer":
        return BorrowedRuntime(binaries, device, bundle.image, **kwargs)
    raise ValueError("Unsupported target runtime: " + str(runtime))
