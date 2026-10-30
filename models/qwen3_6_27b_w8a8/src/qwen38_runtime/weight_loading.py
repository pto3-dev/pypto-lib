# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Optional W1: defer only this target's NZ casts until family batching.

Native drafter loading is unaffected. Hooks exist only during load_model and are
restored before activation profiling; W1 is an explicitly selected candidate.
"""

import contextvars, sys, weakref
from types import MethodType

_DEFER = contextvars.ContextVar("qwen38_defer_target_nz", default=False)
_OWNED = weakref.WeakSet()
_METHODS = []
_GLOBALS = []
_DEFERRED = {}


def install_target_deferral(causal, *, utilities=None):
    if utilities is None:
        import vllm_ascend.utils as utilities
    utils = utilities
    original_cast = utils.maybe_trans_nz

    def deferred_cast(weight):
        if not _DEFER.get():
            return original_cast(weight)
        key = (str(weight.dtype), tuple(weight.shape))
        _DEFERRED[key] = _DEFERRED.get(key, 0) + 1
        return weight

    for module in tuple(sys.modules.values()):
        namespace = getattr(module, "__dict__", {})
        if namespace.get("maybe_trans_nz") is original_cast:
            _GLOBALS.append((namespace, original_cast))
            namespace["maybe_trans_nz"] = deferred_cast
    seen = set()
    for module in causal.modules():
        if module is causal:
            continue
        method = getattr(module, "quant_method", None)
        if method is None or not hasattr(method, "process_weights_after_loading"):
            continue
        _OWNED.add(module)
        if id(method) in seen:
            continue
        seen.add(id(method))
        original = method.process_weights_after_loading

        def process(self, layer, *args, __original=original, **kwargs):
            token = _DEFER.set(layer in _OWNED)
            try:
                callback = getattr(layer, "_pypto_postload_no_copy", None)
                if layer in _OWNED and callback is not None:
                    return callback()
                return __original(layer, *args, **kwargs)
            finally:
                _DEFER.reset(token)

        _METHODS.append((method, original))
        method.process_weights_after_loading = MethodType(process, method)


def restore_deferral():
    while _METHODS:
        method, original = _METHODS.pop()
        method.process_weights_after_loading = original
    while _GLOBALS:
        namespace, original = _GLOBALS.pop()
        namespace["maybe_trans_nz"] = original
    _OWNED.clear()
    _DEFERRED.clear()


def deferral_stats():
    return dict(
        deferred_cast_calls=sum(_DEFERRED.values()),
        by_dtype_shape=[
            dict(dtype=key[0], shape=list(key[1]), calls=count) for key, count in sorted(_DEFERRED.items())
        ],
    )
