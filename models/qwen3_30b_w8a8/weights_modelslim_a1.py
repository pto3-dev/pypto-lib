# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""A1 FP16 ModelSlim static-Attention and dynamic-expert checkpoint loading."""

import json
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open

from .weights_a1 import A1ExpertLayerWeights, _checked, _validate_config


class A1ModelSlimWeights:
    """Read unchanged ModelSlim shards into checked TP1 ND weight banks."""

    def __init__(self, checkpoint: Path) -> None:
        self.checkpoint = checkpoint.resolve()
        _validate_config(self.checkpoint, 0)
        config = json.loads((self.checkpoint / "config.json").read_text())
        dtype = config.get("dtype", config.get("torch_dtype"))
        quantize = config.get("quantize")
        if dtype != "float16" or quantize != "w8a8_dynamic":
            raise ValueError(f"Expected FP16 ModelSlim checkpoint, got dtype={dtype}, quantize={quantize}")
        index_path = self.checkpoint / "quant_model_weight_w8a8_dynamic.safetensors.index.json"
        self.index = json.loads(index_path.read_text())["weight_map"]
        self.description = json.loads((self.checkpoint / "quant_model_description.json").read_text())
        for filename in set(self.index.values()):
            source = (self.checkpoint / filename).resolve()
            if not source.is_relative_to(self.checkpoint) or source.suffix != ".safetensors":
                raise ValueError(f"Invalid ModelSlim shard path: {filename}")

    def _read(self, names: tuple[str, ...]) -> dict[str, torch.Tensor]:
        with ExitStack() as stack:
            shards = {}
            result = {}
            for name in names:
                filename = self.index[name]
                if filename not in shards:
                    path = self.checkpoint / filename
                    shards[filename] = stack.enter_context(safe_open(path, framework="pt", device="cpu"))
                result[name] = shards[filename].get_tensor(name)
            return result

    @staticmethod
    def _floating(tensor: torch.Tensor, shape: tuple[int, ...], name: str) -> torch.Tensor:
        if not tensor.dtype.is_floating_point:
            raise ValueError(f"{name}: expected floating checkpoint storage, got {tensor.dtype}")
        _checked(tensor, shape, tensor.dtype, name)
        result = tensor.to(torch.float16).contiguous()
        if not torch.isfinite(result).all():
            raise ValueError(f"{name}: cannot represent checkpoint values as finite FP16")
        return result

    def attention(self, layer: int) -> dict[str, torch.Tensor]:
        """Load packed QKV and O with original integer bias and encoded scale."""
        _validate_config(self.checkpoint, layer)
        prefix = f"model.layers.{layer}."
        fields = ("weight", "input_scale", "input_offset", "quant_bias", "deq_scale")
        names = tuple(f"self_attn.{kind}_proj.{field}" for kind in ("q", "k", "v", "o") for field in fields)
        norm_names = ("input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm")
        names += tuple(f"{name}.{field}" for name in norm_names for field in ("weight", "bias"))
        names += ("mlp.gate.weight",)
        tensors = self._read(tuple(prefix + name for name in names))
        result = {}
        qkv_weights, qkv_biases, qkv_scales = [], [], []
        for kind, output_width, input_width in (
            ("q", 4096, 2048),
            ("k", 512, 2048),
            ("v", 512, 2048),
            ("o", 2048, 4096),
        ):
            name = f"{prefix}self_attn.{kind}_proj"
            if self.description[f"{name}.weight"] != "W8A8":
                raise ValueError(f"{name}: expected static W8A8")
            weight = _checked(tensors[f"{name}.weight"], (output_width, input_width), torch.int8, name)
            bias = _checked(tensors[f"{name}.quant_bias"], (output_width,), torch.int32, name)
            safe_bias = 2**31 - 1 - input_width * 16384
            if torch.any(bias.long().abs() > safe_bias):
                raise ValueError(f"{name}: integer bias may overflow the INT32 projection accumulator")
            scale = _checked(tensors[f"{name}.deq_scale"], (output_width,), torch.int64, name)
            input_scale = self._floating(tensors[f"{name}.input_scale"], (1,), name)
            offset = tensors[f"{name}.input_offset"].float()
            if offset.shape != (1,) or not torch.isfinite(offset).all():
                raise ValueError(f"{name}: invalid scalar input offset")
            if not torch.equal(offset, offset.round()) or not torch.all((offset >= -128) & (offset <= 127)):
                raise ValueError(f"{name}: input offset must be representable as INT8")
            if not torch.all(input_scale > 0):
                raise ValueError(f"{name}: nonpositive static input scale")
            output_name = "o" if kind == "o" else "qkv"
            if kind in ("q", "o"):
                result[f"{output_name}_input_scale"] = input_scale
                result[f"{output_name}_input_offset"] = offset.to(torch.int8)
            else:
                if not torch.equal(input_scale, result["qkv_input_scale"]):
                    raise ValueError(f"{name}: QKV input scales differ")
                if not torch.equal(offset.to(torch.int8), result["qkv_input_offset"]):
                    raise ValueError(f"{name}: QKV input offsets differ")
            if kind == "o":
                result["o_weight"] = weight
                result["o_quant_bias"] = bias.reshape(1, -1)
                result["o_deq_scale"] = scale.reshape(1, -1)
            else:
                qkv_weights.append(weight)
                qkv_biases.append(bias)
                qkv_scales.append(scale)
        result["qkv_weight"] = torch.cat(qkv_weights).contiguous()
        result["qkv_quant_bias"] = torch.cat(qkv_biases).reshape(1, -1).contiguous()
        result["qkv_deq_scale"] = torch.cat(qkv_scales).reshape(1, -1).contiguous()
        for name in norm_names:
            width = 128 if name.startswith("self_attn") else 2048
            for field in ("weight", "bias"):
                key = f"{name}.{field}"
                result[key] = self._floating(tensors[prefix + key], (width,), prefix + key)
                if name.startswith("self_attn") and field == "bias" and torch.count_nonzero(result[key]):
                    raise ValueError(f"{prefix + key}: head Norm/RoPE currently requires zero bias")
        result["mlp.gate.weight"] = self._floating(tensors[prefix + "mlp.gate.weight"], (128, 2048), prefix)
        return result

    def experts(self, layer: int) -> A1ExpertLayerWeights:
        """Load all 128 experts without sharding or changing scale precision."""
        _validate_config(self.checkpoint, layer)
        prefix = f"model.layers.{layer}.mlp.experts."
        fields = ("weight", "weight_scale", "weight_offset")
        names = tuple(
            f"{prefix}{expert}.{kind}.{field}"
            for expert in range(128)
            for kind in ("gate_proj", "up_proj", "down_proj")
            for field in fields
        )
        tensors = self._read(names)
        w13 = torch.empty((128, 1536, 2048), dtype=torch.int8)
        w2 = torch.empty((128, 2048, 768), dtype=torch.int8)
        w13_scale = torch.empty((128, 1536), dtype=torch.float32)
        w2_scale = torch.empty((128, 2048), dtype=torch.float32)
        for expert in range(128):
            for kind, width, input_width, start in (
                ("gate_proj", 768, 2048, 0),
                ("up_proj", 768, 2048, 768),
                ("down_proj", 2048, 768, 0),
            ):
                name = f"{prefix}{expert}.{kind}"
                if self.description[f"{name}.weight"] != "W8A8_DYNAMIC":
                    raise ValueError(f"{name}: expected dynamic W8A8")
                weight = _checked(tensors[f"{name}.weight"], (width, input_width), torch.int8, name)
                raw_scale = tensors[f"{name}.weight_scale"]
                if not raw_scale.dtype.is_floating_point:
                    raise ValueError(f"{name}: expert scale must use floating checkpoint storage")
                _checked(raw_scale, (width, 1), raw_scale.dtype, name)
                scale = raw_scale.float().flatten()
                if not torch.all(scale > 0):
                    raise ValueError(f"{name}: nonpositive expert scale")
                offset = tensors[f"{name}.weight_offset"]
                if offset.shape != (width, 1) or torch.count_nonzero(offset):
                    raise ValueError(f"{name}: asymmetric expert weights are unsupported")
                if kind == "down_proj":
                    w2[expert].copy_(weight)
                    w2_scale[expert].copy_(scale)
                else:
                    w13[expert, start : start + width].copy_(weight)
                    w13_scale[expert, start : start + width].copy_(scale)
        return A1ExpertLayerWeights(w13, w13_scale, w2, w2_scale)

    def vocabulary(self) -> dict[str, torch.Tensor]:
        """Load FP16 embedding, final Norm weight/bias and untied LM Head."""
        shapes = {
            "model.embed_tokens.weight": (151936, 2048),
            "model.norm.weight": (2048,),
            "model.norm.bias": (2048,),
            "lm_head.weight": (151936, 2048),
        }
        tensors = self._read(tuple(shapes))
        return {name: self._floating(tensors[name], shape, name) for name, shape in shapes.items()}
