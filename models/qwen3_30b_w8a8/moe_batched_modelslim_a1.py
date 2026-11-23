# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""B20/160-route ModelSlim MoE using the native resident NZ expert banks."""

from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable

import torch
from simpler_setup import A1BorrowedLaunch, KernelCompiler

from .compile_modelslim_a1 import compile_modelslim_program, compile_modelslim_source


def compile_a1_batched_moe_images(
    directory: Path, isa: Path, *, cached_inputs: bool = False, k_step: int = 32, pipeline: bool = False,
    vector_swiglu_amax: bool = False, router_batch_rows: int = 1, wide_output: bool = False,
    packed_swiglu: bool = False, n_tile: int = 256, retained_quant: bool = False,
    grouped_w13_dequant: bool = False, vector_dynamic_quant: bool = False,
    dsl_topk: bool = False,
) -> dict[str, Any]:
    """Build this checkout's FP16 epilogues and explicit physical-NZ projections."""
    if router_batch_rows not in (1, 16):
        raise ValueError("Router row tile must be1 or16")
    if k_step not in (32, 64) or (not cached_inputs and k_step != 32):
        raise ValueError(f"Expert K step {k_step} requires cached inputs and a retained 32/64 tile")
    if pipeline and (not cached_inputs or k_step != 64):
        raise ValueError("Pipelined experts require cached inputs and K64")
    if wide_output and not pipeline:
        raise ValueError("Wide integer output requires the retained expert pipeline")
    if packed_swiglu and not vector_swiglu_amax:
        raise ValueError("Packed hardware dequant requires the retained vector-amax SwiGLU")
    if n_tile not in (256, 512) or (n_tile == 512 and not wide_output):
        raise ValueError("Expert N512 requires the retained wide-output pipeline")
    if grouped_w13_dequant and not (packed_swiglu and pipeline and wide_output and n_tile == 512):
        raise ValueError("Grouped W13 dequantization requires packed scales and the N512 expert pipeline")
    if vector_dynamic_quant and not retained_quant:
        raise ValueError("DSL dynamic quantization requires retained-row input quantization")
    images = {}
    for name, filename in (("router", "a1_modelslim_router_batched.cpp"),
                           ("topk", "a1_modelslim_topk_batched.cpp"),
                           ("quant", "a1_modelslim_dynamic_quant.cpp"),
                           ("swiglu", "a1_swiglu_quant_batched.cpp")):
        if name == "topk" and dsl_topk:
            from .topk_batched_a1 import topk_b20

            image, order = compile_modelslim_program(topk_b20, directory / name, isa)
            expected = ("logits", "ids", "weights", "status")
            if tuple(order) != expected:
                raise ValueError(f"A1 DSL TopK tensor ABI changed: {order}")
            images[name] = image
            continue
        if name == "quant" and vector_dynamic_quant:
            from .quant_dynamic_modelslim_a1_vector import make_a1_modelslim_dynamic_quant

            program = make_a1_modelslim_dynamic_quant(20, 2048)
            image, order = compile_modelslim_program(program, directory / name, isa)
            if order != ("x", "quant", "scale", "status"):
                raise ValueError(f"A1 dynamic quantization tensor ABI changed: {order}")
            images[name] = image
            continue
        if name == "quant" and retained_quant:
            filename = "a1_modelslim_dynamic_quant_retained.cpp"
        if name == "swiglu" and vector_swiglu_amax:
            filename = "a1_swiglu_quant_batched_vector_max.cpp"
        if name == "swiglu" and packed_swiglu:
            filename = "a1_swiglu_quant_batched_vector_packed.cpp"
        if name == "swiglu" and grouped_w13_dequant:
            from .swiglu_quant_batched_a1 import swiglu_quant_b20

            image, order = compile_modelslim_program(swiglu_quant_b20, directory / name, isa)
            expected = ("input", "input_scale", "ids", "weight_scale", "quant", "scale", "status")
            if tuple(order) != expected:
                raise ValueError(f"A1 DSL SwiGLU tensor ABI changed: {order}")
            images[name] = image
            continue
        if name == "router" and router_batch_rows == 16:
            from .router_batched_a1 import make_router_b20

            image, order = compile_modelslim_program(make_router_b20(), directory / name, isa)
            expected = ("input", "weight", "output", "status")
            if tuple(order) != expected:
                raise ValueError(f"A1 DSL Router tensor ABI changed: {order}")
            images[name] = image
            continue
        images[name] = compile_modelslim_source(filename, directory / name, isa)
    from .combine_modelslim_a1 import resident_expert_combine

    image, order = compile_modelslim_program(
        resident_expert_combine, directory / "combine", isa)
    expected = ("acc", "input_scale", "ids", "weight_scale",
                "route_weights", "expert_y", "y", "status")
    if tuple(order) != expected:
        raise ValueError(f"A1 DSL expert combine tensor ABI changed: {order}")
    images["combine"] = image
    for name in ("w13", "w2"):
        if (
            name == "w13"
            and cached_inputs
            and pipeline
            and wide_output
            and n_tile == 512
            and k_step == 64
            and grouped_w13_dequant
        ):
            from .expert_w13_batched_a1 import make_w13_b20

            image, order = compile_modelslim_program(
                make_w13_b20(column_partition=True), directory / name, isa)
            expected = ("input", "weight", "ids", "expert_counts", "route_indices", "output", "channel_scale", "status")
            if tuple(order) != expected:
                raise ValueError(f"A1 DSL W13 tensor ABI changed: {order}")
            images[name] = image
            images["w13_shared_routes"] = True
            continue
        if (
            name == "w2"
            and cached_inputs
            and pipeline
            and wide_output
            and n_tile == 512
            and k_step == 64
        ):
            from .expert_w2_batched_a1 import make_w2_b20

            image, order = compile_modelslim_program(
                make_w2_b20(column_partition=False), directory / name, isa)
            expected = ("input", "weight", "ids", "expert_counts", "route_indices", "output", "status")
            if tuple(order) != expected:
                raise ValueError(f"A1 DSL W2 tensor ABI changed: {order}")
            images[name] = image
            images["w2_shared_routes"] = True
            continue
        suffix = ("_cached" + (f"_k{k_step}" if k_step != 32 else "")) if cached_inputs else ""
        if pipeline:
            suffix = "_cached_pipeline_wide" if wide_output else "_cached_pipeline"
            if n_tile == 512:
                suffix = "_cached_pipeline_wide_n512"
        source = Path(__file__).with_name(f"a1_expert_{name}_nz_batched{suffix}.cpp")
        if name == "w13" and grouped_w13_dequant:
            source = Path(__file__).with_name("a1_expert_w13_nz_batched_grouped_dequant.cpp")
        images[name] = KernelCompiler("a1").compile_single_launch(str(source), str(isa))
    if images.get("w13_shared_routes") or images.get("w2_shared_routes"):
        from .route_metadata_batched_a1 import route_metadata_b20

        image, order = compile_modelslim_program(
            route_metadata_b20, directory / "route_metadata", isa)
        if tuple(order) != ("ids", "expert_counts", "route_indices", "status"):
            raise ValueError(f"A1 routing metadata tensor ABI changed: {order}")
        images["route_metadata"] = image
    return images


class A1BatchedModelSlimMoe:
    """Capturable MoE; quantized expert storage stays owned by the native model."""

    def __init__(self, images: dict[str, Any], hidden: torch.Tensor, router: torch.Tensor,
                 w13: torch.Tensor, w13_scale: torch.Tensor, w2: torch.Tensor,
                 w2_scale: torch.Tensor, output: torch.Tensor, *, packed_swiglu: bool = False,
                 grouped_w13_dequant: bool = False,
                 launch_factory: Callable[[bytes, int], Any] = A1BorrowedLaunch):
        if hidden.shape != (20, 2048) or hidden.dtype != torch.float16 or not hidden.is_contiguous():
            raise ValueError("B20 MoE requires contiguous FP16[20,2048]")
        if output.shape != hidden.shape or output.dtype != hidden.dtype or output.device != hidden.device:
            raise ValueError("B20 MoE output must match its hidden-state contract")
        if grouped_w13_dequant and not packed_swiglu:
            raise ValueError("Grouped W13 output requires resident packed channel scales")
        fused = "expert" in images
        if fused and not (packed_swiglu and grouped_w13_dequant):
            raise ValueError("Fused experts require resident packed channel scales")
        device = hidden.device
        self._owners = ExitStack()
        self._calls = []
        gate_key = "channel13" if grouped_w13_dequant else "acc13"
        gate_dtype = torch.float16 if grouped_w13_dequant else torch.int32
        self.buffers = {
            name: torch.zeros(shape, dtype=dtype, device=device)
            for name, shape, dtype in (
                ("quant", (20, 2048), torch.int8), ("scale", (20, 16), torch.float32),
                ("logits", (20, 128), torch.float16), ("ids", (20, 8), torch.int32),
                ("weights", (20, 8), torch.float32), (gate_key, (160, 1536), gate_dtype),
                ("quant13", (160, 768), torch.int8), ("scale13", (160, 16), torch.float32),
                ("acc2", (160, 2048), torch.int32), ("expert_y", (160, 2048), torch.float16),
            )
            if name != gate_key or not fused
        }
        self.packed_w13_scale = None
        if packed_swiglu:
            if w13_scale.shape != (128, 1536) or w13_scale.dtype != torch.float32 or w13_scale.device != device:
                raise ValueError("Packed SwiGLU requires FP32[128,1536] native channel scales on the model device")
            packed = torch.zeros((128, 1536, 2), dtype=torch.int32, device=device)
            # Hardware VDEQ16 expects the unchanged FP32 bits in each low word.
            # Prepare once outside capture; never quantize/repack on Decode.
            packed[:, :, 0].copy_(w13_scale.contiguous().view(torch.int32))
            self.packed_w13_scale = packed.reshape(128, 3072)
            w13_scale = self.packed_w13_scale
        shared_w13 = images.get("w13_shared_routes", False)
        shared_w2 = images.get("w2_shared_routes", False)
        shared_routes = shared_w13 or shared_w2
        if fused and not (shared_w13 and shared_w2):
            raise ValueError("Fused experts require shared routing metadata")
        if shared_routes:
            if "route_metadata" not in images:
                raise ValueError("Shared expert routing requires its metadata leaf")
            self.buffers["expert_counts"] = torch.zeros(
                (1, 128), dtype=torch.int32, device=device)
            self.buffers["route_indices"] = torch.full(
                (128, 160), -1, dtype=torch.int32, device=device)
        b = self.buffers
        if not fused:
            w13_meta = (b["expert_counts"], b["route_indices"]) if shared_w13 else ()
            w2_meta = (b["expert_counts"], b["route_indices"]) if shared_w2 else ()
            w13_args = (b["quant"], w13, b["ids"], *w13_meta, b[gate_key])
            if grouped_w13_dequant:
                w13_args = (*w13_args, w13_scale)
            if images.get("paired_w13_swiglu", False):
                if not grouped_w13_dequant:
                    raise ValueError("Paired W13 requires channel-dequantized FP16 output")
                w13_args = (*w13_args, b["scale"])
        bindings = [
            ("quant", (hidden, b["quant"], b["scale"]), 20, ()),
            ("router", (hidden, router, b["logits"]), 8, ()),
            ("topk", (b["logits"], b["ids"], b["weights"]), 20, ()),
        ]
        if shared_routes:
            bindings.append((
                "route_metadata", (b["ids"], b["expert_counts"], b["route_indices"]), 8, (),
            ))
        fused_status = {}
        if fused:
            fused_status = {
                name: torch.full((160, 16), -7, dtype=torch.int32, device=device)
                for name in ("w13", "swiglu", "w2")
            }
            bindings.append((
                "expert",
                (b["quant"], w13, b["ids"], b["expert_counts"], b["route_indices"],
                 w13_scale, b["scale"], fused_status["w13"], b["quant13"], b["scale13"],
                 fused_status["swiglu"], w2, b["acc2"], fused_status["w2"]),
                160, (1, 11),
            ))
        else:
            bindings.extend([
                ("w13", w13_args, 160, (1,)),
                ("swiglu", (b[gate_key], b["scale"], b["ids"], w13_scale, b["quant13"], b["scale13"]), 160, ()),
                ("w2", (b["quant13"], w2, b["ids"], *w2_meta, b["acc2"]), 160, (1,)),
            ])
        bindings.append((
            "combine", (b["acc2"], b["scale13"], b["ids"], w2_scale,
                        b["weights"], b["expert_y"], output), 8, (),
        ))
        self.status = {name: torch.full((rows, 16), -7, dtype=torch.int32, device=device)
                       for name, _, rows, _ in bindings if name != "expert"}
        self.status.update(fused_status)
        try:
            for name, tensors, _, nz in bindings:
                owner = self._owners.enter_context(launch_factory(images[name], device.index))
                arguments = tensors if name == "expert" else (*tensors, self.status[name])
                self._calls.append(owner.prepare(*arguments, blocks=8, native_nz_weights=nz))
        except BaseException:
            self._owners.close()
            raise

    def enqueue(self) -> None:
        """All routing and quantization are device-side in the enclosing graph."""
        for call in self._calls:
            call()

    def close(self) -> None:
        """Close after graphs that reference this object have been destroyed."""
        self._owners.close()
