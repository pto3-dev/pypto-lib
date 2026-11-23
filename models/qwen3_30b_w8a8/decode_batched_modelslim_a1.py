# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""PTO3 A1 B20 decoder over native weights and PA KV storage.

This module binds precompiled images without importing the PyPTO compiler in
the vLLM worker. The compiler and worker may use different Python minor
versions, but every image must be built from this same source checkout.
"""

import hashlib
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable

import torch
from simpler_setup import A1BorrowedLaunch, KernelCompiler

from .kv_cache_batched_a1 import A1BatchedMetadata
from .moe_batched_modelslim_a1 import A1BatchedModelSlimMoe
from .qk_batched_modelslim_a1 import A1BatchedHeadNormRope


def compile_a1_batched_decoder_images(directory: Path, isa: Path, *, direct_pages: bool = False,
                                     wide_projection: bool = False, batched_heads: bool = False,
                                     qk_tile: int = 16, pv_tile: int = 16,
                                     cached_expert_inputs: bool = False, expert_k_step: int = 32,
                                     attention_probability_padding: bool = False,
                                     attention_vector_copy: bool = False,
                                     fused_projection_dequant: bool = False, expert_pipeline: bool = False,
                                     vector_swiglu_amax: bool = False, router_batch_rows: int = 1,
                                     vector_static_quant: bool = False, expert_wide_output: bool = False,
                                     packed_swiglu_scales: bool = False, expert_n_tile: int = 256,
                                     retained_moe_quant: bool = False, fuse_next_input_norm: bool = False,
                                     direct_head_rows: bool = False, fused_head_cache: bool = False,
                                     metadata_foreach: bool = False, grouped_w13_dequant: bool = False,
                                     packed_projection_scales: bool = False, vector_dynamic_quant: bool = False,
                                     dsl_projection: bool = False, resident_norm: bool = False,
                                     dsl_direct_heads: bool = False, dsl_combine_norm: bool = False,
                                     dsl_attention: bool = False,
                                     attention_physical_pages: int | None = None,
                                     dsl_topk: bool = False) -> dict:
    """Build the common images once, outside model loading/capture."""
    if dsl_attention:
        if not (direct_pages and qk_tile == 128 and pv_tile == 128
                and attention_probability_padding and attention_vector_copy):
            raise ValueError("A1 DSL Attention requires direct QK128/PV128, retained padding and vector copy")
        if (type(attention_physical_pages) is not int
                or not 1 <= attention_physical_pages <= ((1 << 32) - 1) // (32 * 128 * 16)):
            raise ValueError("A1 DSL Attention requires the explicit positive physical PA page extent")
    elif attention_physical_pages is not None:
        raise ValueError("attention_physical_pages requires dsl_attention=True")
    from .compile_modelslim_a1 import compile_modelslim_program, compile_modelslim_source
    from .kv_cache_batched_a1 import compile_a1_batched_cache_images
    from .linear_batched_modelslim_a1 import (
        compile_a1_batched_modelslim_dequant,
        compile_a1_batched_modelslim_projection,
    )
    from .moe_batched_modelslim_a1 import compile_a1_batched_moe_images
    from .norm_modelslim_a1 import make_a1_modelslim_norm
    from .qk_batched_modelslim_a1 import compile_a1_batched_qk_images
    from .quant_batched_modelslim_a1 import make_a1_batched_modelslim_static_quant
    from .quant_batched_modelslim_a1_vector import make_a1_batched_modelslim_vector_quant

    if qk_tile not in (16, 128) or pv_tile not in (16, 128) or (pv_tile == 128 and qk_tile != 128):
        raise ValueError(f"Expected attention tiles 16/16,128/16 or128/128, got {qk_tile}/{pv_tile}")
    if qk_tile == 128 and not direct_pages:
        raise ValueError("A1 wide attention requires direct native PA cache access")
    if attention_probability_padding and (not direct_pages or qk_tile != 128 or pv_tile != 128):
        raise ValueError("A1 retained probability padding requires direct QK128/PV128 attention")
    if attention_vector_copy and not attention_probability_padding:
        raise ValueError("A1 vector attention requires retained probability padding")
    if fused_projection_dequant and not wide_projection:
        raise ValueError("A1 fused projection dequantization requires wide projections")
    if packed_projection_scales and not fused_projection_dequant:
        raise ValueError("A1 native projection scales require fused projection dequantization")
    if dsl_projection and not (wide_projection and fused_projection_dequant and packed_projection_scales):
        raise ValueError("A1 DSL projections require wide tiles and fused packed-channel dequantization")
    if dsl_combine_norm and not fuse_next_input_norm:
        raise ValueError("A1 DSL combine-Norm requires the fused next-input Norm path")
    images = {"dsl": {}, "native": compile_a1_batched_cache_images(isa),
              "execution_options": {"direct_pages": direct_pages, "wide_projection": wide_projection,
                                    "batched_heads": batched_heads, "metadata_foreach": metadata_foreach, "fused_head_cache": fused_head_cache, "direct_head_rows": direct_head_rows, "qk_tile": qk_tile, "pv_tile": pv_tile,
                                    "cached_expert_inputs": cached_expert_inputs, "expert_k_step": expert_k_step,
                                    "attention_probability_padding": attention_probability_padding,
                                    "attention_vector_copy": attention_vector_copy,
                                    "vector_dynamic_quant": vector_dynamic_quant, "dsl_projection": dsl_projection,
                                    "resident_norm": resident_norm, "dsl_direct_heads": dsl_direct_heads,
                                    "dsl_combine_norm": dsl_combine_norm,
                                    "packed_projection_scales": packed_projection_scales, "fused_projection_dequant": fused_projection_dequant, "expert_pipeline": expert_pipeline, "router_batch_rows": router_batch_rows, "vector_static_quant": vector_static_quant, "expert_wide_output": expert_wide_output, "fuse_next_input_norm": fuse_next_input_norm, "retained_moe_quant": retained_moe_quant, "expert_n_tile": expert_n_tile, "packed_swiglu_scales": packed_swiglu_scales, "vector_swiglu_amax": vector_swiglu_amax, "grouped_w13_dequant": grouped_w13_dequant}}
    if dsl_topk:
        images["execution_options"]["dsl_topk"] = True
    if dsl_attention:
        images["execution_options"]["dsl_attention"] = True
        images["execution_options"]["attention_physical_pages"] = attention_physical_pages
    for name, program in (("plain_norm", make_a1_modelslim_norm(fused_add=False, rows=20, balanced_reduction=True)),
                          ("norm", make_a1_modelslim_norm(fused_add=True, rows=20, resident_row=resident_norm)),
                          ("qkv_quant", make_a1_batched_modelslim_static_quant(2048)),
                          ("o_quant", make_a1_batched_modelslim_static_quant(4096))):
        if name == "norm" and not resident_norm:
            image = compile_modelslim_source("a1_post_norm_batched.cpp", directory / name, isa)
            images["dsl"][name] = (image, ("x", "residual", "weight", "bias", "y", "residual_out"))
        elif vector_static_quant and name in ("qkv_quant", "o_quant"):
            width = 2048 if name == "qkv_quant" else 4096
            vector_program = make_a1_batched_modelslim_vector_quant(width)
            images["dsl"][name] = compile_modelslim_program(vector_program, directory / name, isa)
        else:
            images["dsl"][name] = compile_modelslim_program(program, directory / name, isa)
    for name in ("qkv", "o"):
        if dsl_projection:
            from .linear_batched_modelslim_a1_dsl import make_a1_modelslim_projection

            program = make_a1_modelslim_projection(name)
            image, order = compile_modelslim_program(program, directory / name, isa)
            if order != ("x", "weight", "bias", "scales", "output", "status"):
                raise ValueError(f"Unexpected A1 DSL projection argument order: {order}")
            images["native"][name] = image
        else:
            images["native"][name] = compile_a1_batched_modelslim_projection(
                name, isa, wide=wide_projection, fused_dequant=fused_projection_dequant,
                packed_scale_words=packed_projection_scales)
    images["native"]["dequant"] = compile_a1_batched_modelslim_dequant(isa)
    if dsl_combine_norm:
        from .combine_modelslim_a1 import resident_expert_combine_norm

        image, order = compile_modelslim_program(
            resident_expert_combine_norm, directory / "combine_norm", isa)
        expected = ("acc", "input_scale", "ids", "weight_scale", "route_weights",
                    "expert_y", "y", "status", "residual", "gamma", "beta", "norm_y", "residual_out")
        if tuple(order) != expected:
            raise ValueError(f"A1 DSL combine-Norm tensor ABI changed: {order}")
        images["native"]["combine_norm"] = image
    else:
        images["native"]["combine_norm"] = compile_modelslim_source(
            "a1_expert_combine_norm_batched.cpp", directory / "combine_norm", isa)
    if dsl_attention:
        from .attention_batched_modelslim_a1 import make_attention_b20

        image, order = compile_modelslim_program(
            make_attention_b20(physical_pages=attention_physical_pages), directory / "attention", isa)
        expected = ("query", "keys", "values", "lengths", "output", "block_table", "status")
        if tuple(order) != expected:
            raise ValueError(f"A1 DSL Attention tensor ABI changed: {order}")
        images["native"]["attention"] = image
    else:
        filename = ("a1_attention_batched_vector.cpp" if attention_vector_copy else
                    "a1_attention_batched_padding.cpp" if attention_probability_padding else
                    "a1_attention_batched_wide.cpp" if pv_tile == 128 else
                    "a1_attention_batched_wide_qk.cpp" if qk_tile == 128 else
                    "a1_attention_batched_direct.cpp" if direct_pages else "a1_attention_batched_modelslim.cpp")
        source = Path(__file__).with_name(filename)
        images["native"]["attention"] = KernelCompiler("a1").compile_single_launch(str(source), str(isa))
    images["qk"] = compile_a1_batched_qk_images(directory / "qk", isa, batched_heads=batched_heads, direct_head_rows=direct_head_rows, pack_append=fused_head_cache, dsl_direct_heads=dsl_direct_heads)
    images["moe"] = compile_a1_batched_moe_images(
        directory / "moe", isa, cached_inputs=cached_expert_inputs, k_step=expert_k_step,
        pipeline=expert_pipeline, wide_output=expert_wide_output, n_tile=expert_n_tile,
        vector_swiglu_amax=vector_swiglu_amax, packed_swiglu=packed_swiglu_scales,
        router_batch_rows=router_batch_rows, retained_quant=retained_moe_quant,
        grouped_w13_dequant=grouped_w13_dequant, vector_dynamic_quant=vector_dynamic_quant,
        dsl_topk=dsl_topk,
    )
    images["source_hashes"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in Path(__file__).parent.iterdir() if path.suffix in (".py", ".cpp", ".h")
    }
    return images


class A1BatchedModelSlimDecoderLayer:
    """All decoder arithmetic is PTO; native ownership of weights/KV is retained."""

    def __init__(self, images: dict, hidden: torch.Tensor, residual: torch.Tensor,
                 metadata: A1BatchedMetadata, cache: tuple[torch.Tensor, torch.Tensor],
                 weights: dict[str, torch.Tensor], output: torch.Tensor, residual_out: torch.Tensor,
                 *, first_layer: bool, staged_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
                 launch_factory: Callable[[bytes, int], Any] = A1BorrowedLaunch):
        for tensor in (hidden, residual, output, residual_out):
            if tensor.shape != (20, 2048) or tensor.dtype != torch.float16 or tensor.device != hidden.device:
                raise ValueError("B20 decoder activations must be FP16[20,2048] on one NPU")
        device = hidden.device
        fused_projection = images.get("execution_options", {}).get("fused_projection_dequant", False)
        if images.get("execution_options", {}).get("packed_projection_scales", False):
            if not fused_projection:
                raise ValueError("A1 native projection scales require fused dequantization")
            for kind, width in (("qkv", 5120), ("o", 2048)):
                scale = weights[f"{kind}_scale"]
                if scale.dtype != torch.int32 or tuple(scale.shape) != (1, 2 * width):
                    raise ValueError(f"A1 {kind} native scale words require INT32[1,{2 * width}]")
        dsl_projection = images.get("execution_options", {}).get("dsl_projection", False)
        if dsl_projection and not (fused_projection and images["execution_options"].get("packed_projection_scales")):
            raise ValueError("A1 DSL projections require fused packed-channel dequantization")
        # Retain zero-copy packed-word views for the whole captured-graph lifetime.
        self.projection_scales = {
            kind: weights[f"{kind}_scale"].view(torch.int64) if dsl_projection else weights[f"{kind}_scale"]
            for kind in ("qkv", "o")
        }
        self._owners = ExitStack()
        self._calls = []
        self.status = {}
        self.output, self.residual_out = output, residual_out
        self.buffers = {
            name: torch.zeros(shape, dtype=dtype, device=device)
            for name, shape, dtype in (
                ("norm", (20, 2048), torch.float16), ("skip", (20, 2048), torch.float16),
                ("qkv_quant", (32, 2048), torch.int8), ("qkv_acc", (32, 5120), torch.int32),
                ("qkv", (20, 5120), torch.float16), ("attention", (20, 4096), torch.float16),
                ("o_quant", (32, 4096), torch.int8), ("o_acc", (32, 2048), torch.int32),
                ("o", (20, 2048), torch.float16), ("post_norm", (20, 2048), torch.float16),
            )
        }
        if fused_projection:
            del self.buffers["qkv_acc"], self.buffers["o_acc"]
        direct_pages = images.get("execution_options", {}).get("direct_pages", False)
        if staged_cache is None and not direct_pages:
            staged_cache = tuple(torch.zeros((80, 4096, 128), dtype=torch.float16, device=device) for _ in range(2))
        self.staged_cache = staged_cache
        b = self.buffers

        def dsl(name, tensors):
            image, order = images["dsl"][name]
            owner = self._owners.enter_context(launch_factory(image, device.index))
            return owner.prepare(*(tensors[label] for label in order), blocks=8)

        def native(name, tensors, status_rows, *, image_name=None, nz=(), pa=()):
            image = images["native"][image_name or name]
            owner = self._owners.enter_context(launch_factory(image, device.index))
            self.status[name] = torch.full((status_rows, 16), -7, dtype=torch.int32, device=device)
            return owner.prepare(*tensors, self.status[name], blocks=8, native_nz_weights=nz, native_pa_caches=pa)

        def norm(name, x, skip, prefix, y, skip_out):
            return dsl(name, {"x": x, "residual": skip, "weight": weights[f"{prefix}_weight"].reshape(1, 2048),
                              "bias": weights[f"{prefix}_bias"].reshape(1, 2048), "y": y, "residual_out": skip_out})

        try:
            self._calls.append(norm("plain_norm" if first_layer else "norm", hidden, residual,
                                    "input_norm", b["norm"], b["skip"]))
            self._calls.append(dsl("qkv_quant", {"x": b["norm"], "reciprocal": weights["qkv_reciprocal"],
                                                  "offset": weights["qkv_offset"], "quant": b["qkv_quant"][:20]}))
            if fused_projection:
                self._calls.append(native("qkv", (b["qkv_quant"], weights["qkv_weight"], weights["qkv_bias"],
                                                  self.projection_scales["qkv"], b["qkv"]), 8, nz=(1,)))
            else:
                self._calls.append(native("qkv", (b["qkv_quant"], weights["qkv_weight"], b["qkv_acc"]), 8, nz=(1,)))
                self._calls.append(native("qkv_dequant", (b["qkv_acc"], weights["qkv_bias"], weights["qkv_scale"], b["qkv"]),
                                           8, image_name="dequant"))
            self.heads = A1BatchedHeadNormRope(images["qk"], b["qkv"], metadata.positions,
                                              weights["rope"], weights["q_norm"], weights["k_norm"], cache=cache, slots=metadata.slots,
                                              launch_factory=launch_factory)
            self._owners.callback(self.heads.close)
            self._calls.append(self.heads.enqueue)
            hb = self.heads.buffers
            import torch_npu
            cache_nz = torch_npu.get_npu_format(cache[0]) == 29
            if images["qk"].get("pack_append", False):
                self.status["append"] = self.heads.status["append"]
            else:
                self._calls.append(native("append", (hb["key"], hb["value"], *cache, metadata.slots), 20,
                                           pa=(2, 3) if cache_nz else ()))
            if "attention_o_quant" in images["native"]:
                # Invalid metadata leaves the previous row untouched, so the
                # initial INT8 row must match quantization of the old zero buffer.
                offset = weights["o_offset"].float()
                initial = offset.clamp(-128, 127).round().to(torch.int8)
                b["o_quant"][:20].copy_(initial.expand(20, -1))
                del b["attention"]
                self._calls.append(native(
                    "attention_o_quant",
                    (hb["query"], *cache, metadata.lengths, b["o_quant"][:20],
                     metadata.block_table, weights["o_reciprocal"], weights["o_offset"]),
                    80, pa=(1, 2) if cache_nz else (),
                ))
            else:
                if direct_pages:
                    self._calls.append(native("attention", (hb["query"], *cache, metadata.lengths,
                                                              b["attention"], metadata.block_table), 80,
                                              pa=(1, 2) if cache_nz else ()))
                else:
                    self._calls.append(native("gather", (*cache, metadata.block_table, metadata.lengths, *staged_cache), 80,
                                               pa=(0, 1) if cache_nz else ()))
                    self._calls.append(native("attention", (hb["query"], *staged_cache, metadata.lengths, b["attention"]), 80))
                self._calls.append(dsl("o_quant", {"x": b["attention"], "reciprocal": weights["o_reciprocal"],
                                                    "offset": weights["o_offset"], "quant": b["o_quant"][:20]}))
            if fused_projection:
                self._calls.append(native("o", (b["o_quant"], weights["o_weight"], weights["o_bias"],
                                                self.projection_scales["o"], b["o"]), 8, nz=(1,)))
            else:
                self._calls.append(native("o", (b["o_quant"], weights["o_weight"], b["o_acc"]), 8, nz=(1,)))
                self._calls.append(native("o_dequant", (b["o_acc"], weights["o_bias"], weights["o_scale"], b["o"]),
                                           8, image_name="dequant"))
            self._calls.append(norm("norm", b["o"], b["skip"], "post_norm", b["post_norm"], residual_out))
            self.moe = A1BatchedModelSlimMoe(images["moe"], b["post_norm"], weights["router"], weights["w13"],
                                            weights["w13_scale"], weights["w2"], weights["w2_scale"], output,
                                            packed_swiglu=images.get("execution_options", {}).get("packed_swiglu_scales", False),
                                            grouped_w13_dequant=images.get("execution_options", {}).get("grouped_w13_dequant", False),
                                            launch_factory=launch_factory)
            self._owners.callback(self.moe.close)
            self._calls.append(self.moe.enqueue)
        except BaseException:
            self._owners.close()
            raise

    def enqueue(self) -> None:
        """Capture the ordered decoder on the caller stream, with no host routing."""
        for call in self._calls:
            call()

    def close(self) -> None:
        """Release immutable resources after enclosing graph destruction."""
        self._owners.close()
