# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Native/PTO shared banks: native NZ large matrices, ND vector parameters."""

import gc
import torch, torch_npu
from qwen36.runtime.native_nz_storage import NativeNzBank
from qwen36.bank_layout import layout, is_quant_bank
from qwen36.model_layout import C, VG


def bank_native_weights(causal, spec):
    device = causal.lm_head.weight.device
    small, count, banks = layout(spec)
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
                    if torch.count_nonzero(module.weight_offset).item() != 0:
                        raise ValueError(f"{name}: nonzero weight_offset is unsupported")
                    native_scale = module.weight_scale.detach().float().flatten()
                    if not bool(torch.isfinite(native_scale).all()) or not bool((native_scale > 0).all()):
                        raise ValueError(f"{name}: invalid scale")
                    scales.append(native_scale)
                if tuple(parameter.shape) != tuple(shape[1:]):
                    raise ValueError(f"Native matrix shape mismatch {name}: {parameter.shape}")
                if torch_npu.get_npu_format(parameter) != 29:
                    raise ValueError(f"{name} requires native NZ=2 before bank construction")
                parameters.append(parameter)
            bank = NativeNzBank.from_weights([p.detach() for p in parameters])
            for parameter, alias in zip(parameters, bank.weights):
                parameter.data = alias
            result[name] = bank.storage.view(torch.int8 if is_quant_bank(name) else torch.bfloat16)
            if scales:
                result[name + "_scale"] = torch.stack(scales).contiguous()
                scale_mirror += result[name + "_scale"].numel() * 4
            owners.append(bank)
            copied += bank.storage.numel()
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
                    view = storage[offset : offset + torch.tensor(shape).prod().item()].view(shape)
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
        large_weight_rebanked_bytes=copied,
        full_weight_mirror=False,
        small_conv_transpose_mirror_bytes=conv_mirror,
        fp32_scale_mirror_bytes=scale_mirror,
        quantization="mixed W8A8_DYNAMIC",
        weight_banks=len(banks),
        owner_pid=__import__("os").getpid(),
    )
    return result, owners, info
