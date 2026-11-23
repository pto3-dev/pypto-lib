# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""Cold binding replacement for the independent fused Q/K branch."""

from collections.abc import Callable
from typing import Any

from simpler_setup.a1_kernel_plan import A1KernelBinding, A1KernelRole


def prepare_fused_qk_call(
    heads: Any, images: dict[str, Any], launch_factory: Callable[[bytes, int], Any], *,
    write_norm: bool = True,
) -> None:
    """Share the original tensors and owner lifetime; leave prepare/pack unchanged."""
    calls = heads._calls
    if len(calls) != 4 or not all(isinstance(call, A1KernelBinding) for call in calls):
        raise ValueError("Fused Q/K requires the selected four B20 head bindings")
    query, key = calls[1:3]
    if query.kernel_image != images["q"][0] or key.kernel_image != images["k"][0]:
        raise ValueError("Fused Q/K requires the selected resident head images")
    qargs, kargs = query.tensors, key.tensors
    if len(qargs) != 5 or len(kargs) != 5 or qargs[2] is not kargs[2]:
        raise ValueError("Fused Q/K requires one shared prepared rotary table")
    image, order = images["fused_qk"]
    expected = ("query", "query_weight", "cos_sin", "query_norm", "query_rotated",
                "key", "key_weight", "key_norm", "key_rotated")
    if tuple(order) != expected:
        raise ValueError(f"Fused Q/K tensor order differs: {order}")
    if not write_norm:
        owner = heads._owners.enter_context(
            launch_factory(A1KernelRole("qk.fused_qk_rotated"), qargs[0].device.index)
        )
        calls[1:3] = [owner.prepare(*qargs[:3], qargs[4], *kargs[:2], kargs[4], blocks=8)]
        del heads.buffers["q_norm"], heads.buffers["k_norm"]
        return
    owner = heads._owners.enter_context(launch_factory(image, qargs[0].device.index))
    fused = owner.prepare(*qargs, kargs[0], kargs[1], kargs[3], kargs[4], blocks=8)
    calls[1:3] = [fused]
