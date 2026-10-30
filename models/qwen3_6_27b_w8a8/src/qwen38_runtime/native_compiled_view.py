# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent native compiler entry sharing every parameter and cache module.

Create before replacing the target instance's forward. The shallow module view
does not copy tensor storage; its compiler callable must be recreated because
copying an existing callable would keep it bound to the target instance.
"""

from copy import copy


def prepare_native_compile_view(model):
    if getattr(model, "do_not_compile", True):
        return None
    if "forward" in model.__dict__:
        raise ValueError("Native compile view requires the original class forward before target attachment")
    if getattr(model, "compiled", False):
        raise ValueError("Create native compile view before any model execution")
    from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

    if not isinstance(model, TorchCompileWithNoGuardsWrapper):
        raise TypeError("Expected the current vLLM native compiler wrapper")
    view = copy(model)
    if view is model or view.__dict__ is model.__dict__:
        raise RuntimeError("Native view must have independent instance attributes")
    for name in ["_modules", "_parameters", "_buffers"]:
        if getattr(view, name) is not getattr(model, name):
            raise RuntimeError("Shallow native view unexpectedly copied tensor/module registries")
    # This rebinds both the compiled callable and the bytecode hook to `view`.
    # The original target can independently bypass Dynamo for PyPTO execution.
    TorchCompileWithNoGuardsWrapper.__init__(view)
    return view
