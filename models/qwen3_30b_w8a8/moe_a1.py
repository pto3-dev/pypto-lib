# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""A1 TP1 MoE components; independent qualification precedes model binding."""

from contextlib import ExitStack
from pathlib import Path

import torch

from .compile_modelslim_a1 import compile_modelslim_source
from .router import compile_a1_router_projection, compile_a1_router_topk
from .weights_a1 import A1ExpertLayerWeights


def compile_a1_route_input(kind: str, pto_isa_root: str | Path) -> bytes:
    """Compile INT8 routing/padding and separate-cache-line FP32 scale replication.

    W13 inputs: quant[1,2048], scale[1,16]; W2 inputs: quant[8,768],
    scale[8,16]. Outputs: quant[128,K], scale[8,16], status[8,16].
    Each of eight blocks writes one active and fifteen zero-padded token rows.
    """
    from simpler_setup import KernelCompiler

    if kind not in ("w13", "w2"):
        raise ValueError(f"Unsupported A1 route input: {kind}")
    source = Path(__file__).with_name(f"a1_route_{kind}.cpp")
    return KernelCompiler("a1").compile_single_launch(str(source), str(pto_isa_root))


def compile_a1_expert_combine(pto_isa_root: str | Path) -> bytes:
    """Compile W2 dequantization and sequential top-8 weighted BF16 combination.

    ABI: INT32 acc[8,2048], FP32 input_scale[8,16], INT32 ids[1,8],
    FP32 weight_scale[128,2048], FP32 route_weights[1,8],
    BF16 expert_y[8,2048], BF16 y[1,2048], INT32 status[8,16].
    Eight output-column blocks; BF16 expert rounding precedes FP32 combination.
    """
    from simpler_setup import KernelCompiler

    source = Path(__file__).with_name("a1_expert_combine.cpp")
    return KernelCompiler("a1").compile_single_launch(str(source), str(pto_isa_root))


def compile_a1_expert_projection(kind: str, pto_isa_root: str | Path) -> bytes:
    """Compile TP1 W13 or W2 INT8 CUBE with device-selected resident ND weights.

    ABI: x INT8[128,K] (eight routes, each sixteen padded rows), weight
    INT8[128*N,K], ids INT32[1,8], acc INT32[8,N], status INT32[8,16].
    Prepare eight blocks. W13: K=2048,N=1536; W2: K=768,N=2048. Each
    route block walks its N16 panels, preserving the qualified A1 grid limit.
    Only the first token row of each route is returned; padded rows are not
    active requests. W2 dequantization and expert reduction are separate gates.
    """
    from simpler_setup import KernelCompiler

    if kind not in ("w13", "w2"):
        raise ValueError(f"Unsupported A1 expert projection: {kind}")
    source = Path(__file__).with_name(f"a1_expert_{kind}.cpp")
    return KernelCompiler("a1").compile_single_launch(str(source), str(pto_isa_root))


def compile_a1_swiglu_quant(pto_isa_root: str | Path) -> bytes:
    """Compile eight device-selected experts' TP1 SwiGLU/INT8 epilogues.

    ABI: INT32 acc[8,1536], FP32 input_scale[8,16], INT32 ids[1,8],
    FP32 weight_scale[128,1536], INT8 quant[8,768], FP32 scale[8,16],
    INT32 status[8,16]. Only column zero of scale/status is written, so each
    block owns a separate cache line. One block per route, no CPU dispatch.
    """
    from simpler_setup import KernelCompiler

    source = Path(__file__).with_name("a1_swiglu_quant.cpp")
    return KernelCompiler("a1").compile_single_launch(str(source), str(pto_isa_root))


def compile_a1_moe_images(
    pto_isa_root: str | Path, *, modelslim: bool = False, build_dir: Path | None = None,
) -> dict[str, bytes]:
    """Compile reusable A1 MoE images for all 48 resident model layers."""
    if modelslim and build_dir is None:
        raise ValueError("ModelSlim MoE compilation requires an explicit build directory")
    images = {
        "route_w13": compile_a1_route_input("w13", pto_isa_root),
        "w13": compile_a1_expert_projection("w13", pto_isa_root),
        "route_w2": compile_a1_route_input("w2", pto_isa_root),
        "w2": compile_a1_expert_projection("w2", pto_isa_root),
    }
    if modelslim:
        assert build_dir is not None
        for name, filename in (
            ("router", "a1_modelslim_router.cpp"), ("topk", "a1_modelslim_topk.cpp"),
            ("swiglu", "a1_swiglu_quant.cpp"), ("combine", "a1_expert_combine.cpp"),
            ("input_quant", "a1_modelslim_dynamic_quant.cpp"),
        ):
            images[name] = compile_modelslim_source(filename, build_dir / name, Path(pto_isa_root))
    else:
        images.update({
            "router": compile_a1_router_projection(pto_isa_root),
            "topk": compile_a1_router_topk(pto_isa_root),
            "swiglu": compile_a1_swiglu_quant(pto_isa_root),
            "combine": compile_a1_expert_combine(pto_isa_root),
        })
    return images


class A1MoeDecode:
    """Resident TP1 B1 MoE launches on the caller's NPU stream.

    Prepare and warm up outside capture; enqueue inside the enclosing decoder
    graph. Destroy every graph referencing this object before close(). Input
    hidden/quant/scale and output addresses remain fixed across replays.
    """

    def __init__(
        self,
        images: dict[str, bytes],
        hidden: torch.Tensor,
        quant: torch.Tensor,
        scale: torch.Tensor,
        router_weight: torch.Tensor,
        weights: A1ExpertLayerWeights,
        output: torch.Tensor,
        *,
        modelslim: bool = False,
    ) -> None:
        from simpler_setup import A1BorrowedLaunch

        storage_dtype = torch.float16 if modelslim else torch.bfloat16
        for name, tensor, shape, dtype in (
            ("hidden", hidden, (1, 2048), storage_dtype),
            ("quant", quant, (1, 2048), torch.int8),
            ("scale", scale, (1, 16), torch.float32),
            ("router", router_weight, (128, 2048), storage_dtype),
            ("w13", weights.w13, (128, 1536, 2048), torch.int8),
            ("w2", weights.w2, (128, 2048, 768), torch.int8),
            ("w13_scale", weights.w13_scale, (128, 1536), torch.float32),
            ("w2_scale", weights.w2_scale, (128, 2048), torch.float32),
            ("output", output, (1, 2048), storage_dtype),
        ):
            if tuple(tensor.shape) != shape or tensor.dtype != dtype or not tensor.is_contiguous():
                raise ValueError(f"A1 MoE {name}: expected contiguous {shape}/{dtype}, got {tensor.shape}/{tensor.dtype}")
        if hidden.device.type != "npu" or hidden.device.index is None:
            raise ValueError(f"A1 MoE requires an NPU input, got {hidden.device}")
        device = hidden.device
        self.output = output
        self._owners = ExitStack()
        self.buffers = {
            "logits": torch.zeros((1, 128), dtype=storage_dtype, device=device),
            "ids": torch.zeros((1, 8), dtype=torch.int32, device=device),
            "weights": torch.zeros((1, 8), dtype=torch.float32, device=device),
            "x13": torch.zeros((128, 2048), dtype=torch.int8, device=device),
            "scale13": torch.zeros((8, 16), dtype=torch.float32, device=device),
            "acc13": torch.zeros((8, 1536), dtype=torch.int32, device=device),
            "quant13": torch.zeros((8, 768), dtype=torch.int8, device=device),
            "swiglu_scale": torch.zeros((8, 16), dtype=torch.float32, device=device),
            "x2": torch.zeros((128, 768), dtype=torch.int8, device=device),
            "scale2": torch.zeros((8, 16), dtype=torch.float32, device=device),
            "acc2": torch.zeros((8, 2048), dtype=torch.int32, device=device),
            "expert_y": torch.zeros((8, 2048), dtype=storage_dtype, device=device),
        }
        self.resident_weights = {
            "router": (router_weight.t().contiguous().to(device) if modelslim else router_weight.to(device)),
            "w13": weights.w13.to(device).reshape(128 * 1536, 2048),
            "w2": weights.w2.to(device).reshape(128 * 2048, 768),
            "w13_scale": weights.w13_scale.to(device),
            "w2_scale": weights.w2_scale.to(device),
        }
        b = self.buffers
        w = self.resident_weights
        bindings = (
            ("router", (hidden, w["router"], b["logits"]), 8),
            ("topk", (b["logits"], b["ids"], b["weights"]), 1),
            ("route_w13", (quant, scale, b["x13"], b["scale13"]), 8),
            ("w13", (b["x13"], w["w13"], b["ids"], b["acc13"]), 8),
            ("swiglu", (b["acc13"], b["scale13"], b["ids"], w["w13_scale"], b["quant13"], b["swiglu_scale"]), 8),
            ("route_w2", (b["quant13"], b["swiglu_scale"], b["x2"], b["scale2"]), 8),
            ("w2", (b["x2"], w["w2"], b["ids"], b["acc2"]), 8),
            ("combine", (b["acc2"], b["scale2"], b["ids"], w["w2_scale"], b["weights"], b["expert_y"], output), 8),
        )
        if modelslim:
            bindings = (("input_quant", (hidden, quant, scale), 1), *bindings)
        self.status = {
            name: torch.full((blocks, 16), -7, dtype=torch.int32, device=device)
            for name, _, blocks in bindings
        }
        self._calls = []
        self._closed = False
        try:
            for name, tensors, blocks in bindings:
                owner = self._owners.enter_context(A1BorrowedLaunch(images[name], device_id=device.index))
                self._calls.append(owner.prepare(*tensors, self.status[name], blocks=blocks))
        except BaseException:
            self._owners.close()
            self._closed = True
            raise

    def enqueue(self) -> None:
        """Enqueue prepared launches without allocation, copies, or synchronization."""
        if self._closed:
            raise RuntimeError("A1 MoE decode session is closed")
        for call in self._calls:
            call()

    def close(self) -> None:
        """Release launch resources after enclosing graph destruction."""
        self._owners.close()
        self._closed = True

    def __enter__(self) -> "A1MoeDecode":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
