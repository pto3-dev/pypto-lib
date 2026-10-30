# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Explicit checkpoint destinations, before any target weight is loaded.

Primary device banks preserve checkpoint order and keep all layout conversion on
NPU. CPU staging banks are retained solely as a measured, rejected experiment.
Existing vLLM TP/fused loaders and Parameter identities are preserved.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import weakref
import torch
from .bank_layout import layout, is_quant_bank

_OWNED = weakref.WeakSet()
_CONTEXT_ORIGINAL = None


def bank_module(causal, name, group):
    position = int(name[4])
    kind = name.split("_", 1)[1]
    layer = causal.model.layers[4 * group + position]
    if kind == "gateup":
        return layer.mlp.gate_up_proj
    if kind == "down":
        return layer.mlp.down_proj
    if kind == "out":
        return layer.self_attn.o_proj if position == 3 else layer.linear_attn.out_proj
    if kind == "qkv":
        return layer.self_attn.qkv_proj
    if kind == "qkvz":
        return layer.linear_attn.in_proj_qkvz
    raise ValueError("Unknown weight family " + name)


@dataclass
class PreloadedBanks:
    banks: dict
    modules: dict
    cpu_bytes: int
    contexts_kept_on_cpu: int = 0
    device_checkpoint_layout: bool = False
    postload_no_copy_count: int = 0

    def validate_family(self, name, parameters):
        bank = self.banks[name]
        for index, parameter in enumerate(parameters):
            expected = bank[index].T if self.device_checkpoint_layout and is_quant_bank(name) else bank[index]
            if (
                parameter.device != bank.device
                or parameter.dtype != bank.dtype
                or parameter.shape != expected.shape
                or parameter.stride() != expected.stride()
                or parameter.data_ptr() != expected.data_ptr()
            ):
                raise ValueError(f"{name}/{index}: native processing replaced direct-loaded storage")
        return bank


def install_cpu_destinations(causal, spec):
    global _CONTEXT_ORIGINAL
    if _CONTEXT_ORIGINAL is not None:
        raise RuntimeError("Only one target loading session is allowed")
    banks = {}
    modules = {}
    total = 0
    for name, shape, _, _ in layout(spec)[2]:
        dtype = torch.int8 if is_quant_bank(name) else torch.bfloat16
        bank = torch.empty(shape, dtype=dtype, device="cpu")
        banks[name] = bank
        members = []
        for group in range(shape[0]):
            module = bank_module(causal, name, group)
            parameter = module.weight
            destination = bank[group].T if is_quant_bank(name) else bank[group]
            if parameter.dtype != dtype or parameter.shape != destination.shape:
                raise ValueError(f"{name}: checkpoint loading destination shape/dtype mismatch")
            # Keep Parameter identity and its TP/packed-loader metadata intact.
            parameter.data = destination
            members.append(module)
            _OWNED.add(module)
        modules[name] = tuple(members)
        total += bank.numel() * bank.element_size()
    state = PreloadedBanks(banks, modules, total)
    causal._pypto_preloaded_banks = state
    from vllm.model_executor.model_loader import utils

    _CONTEXT_ORIGINAL = utils.device_loading_context
    original = _CONTEXT_ORIGINAL

    @contextmanager
    def loading_context(module, target_device):
        if module in _OWNED:
            # Other parameters (scales/offsets) remain on their original NPU.
            # Deferred maybe_trans_nz skips only the owned target's weight cast.
            if module.weight.device.type != "cpu":
                raise ValueError("Direct-load weight left CPU before processing")
            state.contexts_kept_on_cpu += 1
            yield module
        else:
            with original(module, target_device):
                yield module

    utils.device_loading_context = loading_context
    return state


def restore_preload_hooks():
    global _CONTEXT_ORIGINAL
    if _CONTEXT_ORIGINAL is not None:
        from vllm.model_executor.model_loader import utils

        utils.device_loading_context = _CONTEXT_ORIGINAL
        _CONTEXT_ORIGINAL = None
    _OWNED.clear()
    for module in list(_DEVICE_OWNED):
        delattr(module, "_pypto_postload_no_copy")
    _DEVICE_OWNED.clear()


_DEVICE_OWNED = weakref.WeakSet()


def install_device_destinations(causal, spec):
    """Direct H2D loading into NPU banks in the original checkpoint order.

    INT8 postprocessing defers the transpose materialization to the final batch
    format conversion. Scale/offset processing is identical to the pinned method.
    """
    from vllm_ascend.quantization.methods.w8a8_dynamic import AscendW8A8DynamicLinearMethod

    banks = {}
    modules = {}
    state = PreloadedBanks(banks, modules, 0, device_checkpoint_layout=True)
    for name, shape, _, _ in layout(spec)[2]:
        quantized = is_quant_bank(name)
        native_shape = (shape[0], shape[2], shape[1]) if quantized else shape
        first = bank_module(causal, name, 0).weight
        bank = torch.empty(native_shape, dtype=first.dtype, device=first.device)
        banks[name] = bank
        members = []
        for group in range(shape[0]):
            module = bank_module(causal, name, group)
            parameter = module.weight
            if parameter.shape != bank[group].shape or parameter.dtype != bank.dtype:
                raise ValueError(f"{name}: native checkpoint destination mismatch")
            parameter.data = bank[group]
            members.append(module)
            if quantized:
                if type(module.quant_method.quant_method) is not AscendW8A8DynamicLinearMethod:
                    raise TypeError("Review direct loading for this quantization scheme")

                def postprocess(layer=module):
                    layer.weight.data = layer.weight.data.transpose(0, 1)
                    layer.weight_scale.data = layer.weight_scale.data.flatten()
                    layer.weight_scale_fp32 = layer.weight_scale.data.to(torch.float32)
                    layer.weight_offset.data = layer.weight_offset.data.flatten()
                    state.postload_no_copy_count += 1

                module._pypto_postload_no_copy = postprocess
                _DEVICE_OWNED.add(module)
        modules[name] = tuple(members)
    causal._pypto_preloaded_banks = state
    return state


# Historical test/tool spelling; primary lifecycle calls the general name.
restore_cpu_destinations_hook = restore_preload_hooks
