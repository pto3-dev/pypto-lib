# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Native/PTO shared banks: native NZ large matrices, ND vector parameters."""

import gc, time
from math import prod
import torch, torch_npu
from qwen38_runtime.native_nz_storage import (
    NativeNzBank,
    validate_batched_physical_bytes,
    from_cpu_bank,
    from_preloaded_nd_bank,
    from_checkpoint_device_bank,
)
from qwen38_runtime.bank_layout import layout, is_quant_bank
from qwen38_runtime.model_layout import C, VG
from qwen38_runtime.weight_validation import validate_scale_bank


def bank_native_weights(causal, spec, *, strategy="native_nz", validate_weight_bytes=False):
    if strategy not in (
        "native_nz",
        "batched_nd_to_nz",
        "direct_cpu_nz",
        "preloaded_nd_to_nz",
        "device_preloaded_nd_to_nz",
    ):
        raise ValueError("Unknown bank strategy")
    device = causal.lm_head.weight.device
    small, count, banks = layout(spec)
    family_metrics = []
    reference_nz_calls = 0
    validation_ms = 0.0
    result = {}
    owners = []
    copied = 0
    conv_mirror = 0
    scale_mirror = 0
    # Aliases use raw storage: retain owners even if a later initialization step fails.
    causal._pypto_mixed_nz_owners = owners
    with torch.no_grad():
        # Move one family at a time: native parameters release their old allocation
        # before the next family is packed. The byte order remains native format29.
        for name, shape, _, _ in banks:
            position = int(name[4])
            kind = name.split("_", 1)[1]
            parameters = []
            scales = []
            offsets = []
            for group in range(16):
                layer = causal.model.layers[4 * group + position]
                if kind == "gateup":
                    module = layer.mlp.gate_up_proj
                elif kind == "down":
                    module = layer.mlp.down_proj
                elif kind == "out":
                    module = layer.self_attn.o_proj if position == 3 else layer.linear_attn.out_proj
                elif kind == "qkv":
                    module = layer.self_attn.qkv_proj
                else:
                    module = layer.linear_attn.in_proj_qkvz
                parameter = module.weight
                expected_dtype = torch.int8 if is_quant_bank(name) else torch.bfloat16
                if parameter.dtype != expected_dtype:
                    raise ValueError(f"{name}: expected {expected_dtype}, got {parameter.dtype}")
                if is_quant_bank(name):
                    if module.weight_scale.numel() != shape[-1]:
                        raise ValueError(f"{name}: per-channel scale size mismatch")
                    native_scale = module.weight_scale.detach().float().flatten()
                    scales.append(native_scale)
                    offsets.append(module.weight_offset)
                if tuple(parameter.shape) != tuple(shape[1:]):
                    raise ValueError(f"Native matrix shape mismatch {name}: {parameter.shape}")
                if strategy in ("direct_cpu_nz", "preloaded_nd_to_nz"):
                    if parameter.device.type != "cpu":
                        raise ValueError("Direct-loaded family is not on CPU")
                else:
                    expected_formats = (29,) if strategy == "native_nz" else (0, 2)
                    if torch_npu.get_npu_format(parameter) not in expected_formats:
                        raise ValueError(f"{name}: source format incompatible with {strategy}")
                parameters.append(parameter)
            scale_bank = validate_scale_bank(scales, offsets) if scales else None
            factory = NativeNzBank.from_weights if strategy == "native_nz" else NativeNzBank.from_nd_weights
            family_start = time.perf_counter()
            allocated_before = torch.npu.memory_allocated(device)
            if strategy in ("direct_cpu_nz", "preloaded_nd_to_nz", "device_preloaded_nd_to_nz"):
                preloaded = causal._pypto_preloaded_banks
                source_bank = preloaded.validate_family(name, parameters)
                if strategy == "device_preloaded_nd_to_nz":
                    bank = from_checkpoint_device_bank(source_bank, quantized=is_quant_bank(name))
                else:
                    bank = (
                        from_cpu_bank(source_bank, device)
                        if strategy == "direct_cpu_nz"
                        else from_preloaded_nd_bank(source_bank, device)
                    )
            else:
                bank = factory([p.detach() for p in parameters])
            family_ms = (time.perf_counter() - family_start) * 1000
            family = dict(
                name=name,
                layers=len(parameters),
                strategy=strategy,
                prepare_ms=family_ms,
                allocated_before=allocated_before,
                allocated_after=torch.npu.memory_allocated(device),
                peak_allocated_cumulative=torch.npu.max_memory_allocated(device),
                nz_bank_bytes=bank.storage.numel(),
            )
            if validate_weight_bytes and strategy in (
                "batched_nd_to_nz",
                "direct_cpu_nz",
                "preloaded_nd_to_nz",
                "device_preloaded_nd_to_nz",
            ):
                validation_start = time.perf_counter()
                calls = validate_batched_physical_bytes(bank, parameters)
                elapsed = (time.perf_counter() - validation_start) * 1000
                family.update(physical_bytes_verified=True, reference_nz_calls=calls, validation_ms=elapsed)
                reference_nz_calls += calls
                validation_ms += elapsed
            family_metrics.append(family)
            owners.append(bank)  # Retain before the first raw alias is rebound.
            for parameter, alias in zip(parameters, bank.weights):
                parameter.data = alias
            result[name] = bank.storage.view(torch.int8 if is_quant_bank(name) else torch.bfloat16)
            if scales:
                result[name + "_scale"] = scale_bank
                scale_mirror += result[name + "_scale"].numel() * 4
            copied += bank.storage.numel()
            if strategy in ("direct_cpu_nz", "preloaded_nd_to_nz", "device_preloaded_nd_to_nz"):
                del source_bank
                del preloaded.banks[name]
            del parameters
            gc.collect()
            torch.npu.empty_cache()
        for group in range(16):
            storage = torch.empty(count, dtype=torch.bfloat16, device=device)
            for pos in range(4):
                layer = causal.model.layers[4 * group + pos]
                values = {
                    "input_layernorm.weight": layer.input_layernorm.weight,
                    "post_attention_layernorm.weight": layer.post_attention_layernorm.weight,
                }
                if pos == 3:
                    values.update(
                        {
                            "self_attn.q_norm.weight": layer.self_attn.q_norm.weight,
                            "self_attn.k_norm.weight": layer.self_attn.k_norm.weight,
                        }
                    )
                else:
                    attn = layer.linear_attn
                    ba = torch_npu.npu_format_cast(attn.in_proj_ba.weight, 2)
                    b, a = ba.split(VG, dim=0)
                    values.update(
                        {
                            "linear_attn.in_proj_b.weight": b,
                            "linear_attn.in_proj_a.weight": a,
                            "linear_attn.conv1d.weight": attn.conv1d.weight.view(C, 4).T,
                            "linear_attn.norm.weight": attn.norm.weight,
                        }
                    )
                for (position, key), (offset, shape, _) in small.items():
                    if position != pos:
                        continue
                    view = storage[offset : offset + prod(shape)].view(shape)
                    view.copy_(values[key])
                    if key == "linear_attn.conv1d.weight":
                        conv_mirror += view.numel() * 2
                    elif key.startswith(("linear_attn.in_proj_a", "linear_attn.in_proj_b")):
                        pass
                    else:
                        value = values[key]
                        value.data = view
                if pos != 3:
                    offset = small[pos, "linear_attn.in_proj_b.weight"][0]
                    if small[pos, "linear_attn.in_proj_a.weight"][0] != offset + VG * 5120:
                        raise ValueError("BA bank adjacency violated")
                    attn.in_proj_ba.weight.data = storage[offset : offset + 2 * VG * 5120].view(2 * VG, 5120)
            torch.npu.synchronize()
            result[f"weights{group}"] = storage
        floating = torch.empty(spec.float_weights_count, dtype=torch.float32, device=device)
        for index, layer in enumerate(causal.model.layers):
            if index % 4 == 3:
                continue
            ordinal = index - index // 4
            attn = layer.linear_attn
            for j, parameter in enumerate((attn.A_log, attn.dt_bias)):
                view = floating[(ordinal * 2 + j) * VG : (ordinal * 2 + j + 1) * VG]
                view.copy_(parameter)
                parameter.data = view
        # The LM head is not always routed through the unquantized-linear hook.
        # Convert once at initialization if needed, then both consumers share it.
        if torch_npu.get_npu_format(causal.lm_head.weight) != 29:
            causal.lm_head.weight.data = torch_npu.npu_format_cast(causal.lm_head.weight, 29)
        result.update(
            float_weights=floating,
            embedding=causal.model.embed_tokens.weight.detach(),
            final_norm=causal.model.norm.weight.detach().view(1, 5120),
            lm_head=causal.lm_head.weight.detach(),
        )
    info = dict(
        layout="native FRACTAL_NZ shared with prefill",
        large_weight_rebanked_bytes=copied if strategy == "native_nz" else 0,
        weight_strategy=strategy,
        nd_stack_bytes=copied if strategy == "batched_nd_to_nz" else 0,
        batch_nz_calls=len(banks) if strategy == "batched_nd_to_nz" else 0,
        full_weight_mirror=False,
        small_conv_transpose_mirror_bytes=conv_mirror,
        fp32_scale_mirror_bytes=scale_mirror,
        quantization="mixed W8A8_DYNAMIC",
        weight_banks=len(banks),
        owner_pid=__import__("os").getpid(),
    )
    info.update(
        families=family_metrics,
        reference_nz_calls=reference_nz_calls,
        validation_ms=validation_ms,
        large_parameter_matrices=sum(f["layers"] for f in family_metrics),
    )
    if strategy in ("direct_cpu_nz", "preloaded_nd_to_nz", "device_preloaded_nd_to_nz"):
        info.update(
            cpu_nz_pack_bytes=copied if strategy == "direct_cpu_nz" else 0,
            required_large_weight_h2d_bytes=copied,
            device_nd2nz_calls=0 if strategy == "direct_cpu_nz" else len(banks),
            batch_nz_calls=0 if strategy == "direct_cpu_nz" else len(banks),
            nz_to_bank_copy_bytes=0,
            preloaded_cpu_bank_bytes=preloaded.cpu_bytes,
            loading_contexts_kept_cpu=preloaded.contexts_kept_on_cpu,
            all_postload_family_aliases_preserved=True,
            reference_h2d_calls=reference_nz_calls
            if validate_weight_bytes and strategy != "device_preloaded_nd_to_nz"
            else 0,
            direct_device_bank_loading=strategy == "device_preloaded_nd_to_nz",
            int8_postload_transpose_views=preloaded.postload_no_copy_count,
        )
        if strategy == "device_preloaded_nd_to_nz":
            info.update(
                device_layout_conversion_calls=len(banks),
                native_format_cast_calls=sum(not is_quant_bank(name) for name, *_ in banks),
                fused_int8_pack_calls=sum(is_quant_bank(name) for name, *_ in banks),
                reference_nd_materializations=sum(x[1][0] for x in banks) if validate_weight_bytes else 0,
            )
    return result, owners, info
