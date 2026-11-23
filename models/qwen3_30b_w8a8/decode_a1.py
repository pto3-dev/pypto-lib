# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""A1 TP1 B1 decoder graph composition."""

import re
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import torch

from .attention_a1 import (
    compile_a1_attention_normalize, make_a1_attention_pv,
    make_a1_attention_qk, make_a1_attention_softmax,
)
from .kv_cache import compile_a1_kv_append, compile_a1_kv_gather
from .linear import make_a1_w8a8_projection
from .moe_a1 import A1MoeDecode, compile_a1_moe_images
from .norm_quant_bf16 import make_bf16_add_rms_norm_quant
from .qk_rope import make_a1_head_norm_rope
from .quant import make_a1_projection_quant
from .weights_a1 import A1ExpertLayerWeights


@dataclass(frozen=True)
class A1CompiledKernel:
    """Native image and generated tensor argument order."""

    image: bytes
    order: tuple[str, ...]


@dataclass(frozen=True)
class A1DecoderImages:
    """Shared decoder images, independent of layer weights and tensor addresses."""

    dsl: dict[str, A1CompiledKernel]
    native: dict[str, bytes]


@dataclass(frozen=True)
class A1DecodeMetadata:
    """Fixed device carriers: INT32[1,8] metadata and BF16[512,128] RoPE table."""

    position: torch.Tensor
    slot: torch.Tensor
    block_table: torch.Tensor
    lengths: torch.Tensor
    cos_sin_cache: torch.Tensor


def compile_a1_decoder_images(output_dir: Path, pto_isa_root: Path) -> A1DecoderImages:
    """Compile decoder operators once for the complete 48-layer A1 model."""
    from pypto import backend
    from pypto.backend import BackendType
    from pypto.runtime import RunConfig, a1_runner
    from simpler_setup import KernelCompiler

    programs = {
        "norm": make_bf16_add_rms_norm_quant(1, fused=True),
        "qkv": make_a1_w8a8_projection(5120, 2048),
        "q_norm": make_a1_head_norm_rope(32),
        "k_norm": make_a1_head_norm_rope(4),
        "qk": make_a1_attention_qk(512),
        "softmax": make_a1_attention_softmax(512),
        "pv": make_a1_attention_pv(512, normalize=False),
        "o_quant": make_a1_projection_quant(1, 4096),
        "o": make_a1_w8a8_projection(2048, 4096),
    }
    previous = backend.get_backend_type() if backend.is_backend_configured() else None
    dsl = {}
    try:
        for name, program in programs.items():
            backend.reset_for_testing()
            compiled = program.compile(config=RunConfig(
                arch=BackendType.Ascend310P, save_kernels=True, dump_passes=True,
                save_kernels_dir=str(output_dir / name),
            ))
            work_dir = Path(compiled.output_dir)
            sources = list((work_dir / "kernels").glob("*/*.cpp"))
            if len(sources) != 1:
                raise ValueError(f"A1 {name} requires one static kernel, got {len(sources)}")
            source = sources[0].read_text()
            order = tuple(label.split("__ssa_")[0] for label in re.findall(r"// Unpack tensor: (\w+)", source))
            if not order or len(set(order)) != len(order):
                raise ValueError(f"A1 {name} requires unique named tensor arguments, got {order}")
            image = a1_runner._compile_image(work_dir, sources[0], str(pto_isa_root))
            dsl[name] = A1CompiledKernel(image, order)
    finally:
        backend.reset_for_testing()
        if previous is not None:
            backend.set_backend_type(previous)
    native = compile_a1_moe_images(pto_isa_root)
    native.update({
        "normalize": compile_a1_attention_normalize(pto_isa_root),
        "append": compile_a1_kv_append(pto_isa_root),
        "gather": compile_a1_kv_gather(pto_isa_root),
    })
    compiler = KernelCompiler("a1")
    for name, filename in (("qkv_prepare", "a1_qkv_prepare.cpp"), ("route_o", "a1_route_o.cpp")):
        native[name] = compiler.compile_single_launch(str(Path(__file__).with_name(filename)), str(pto_isa_root))
    return A1DecoderImages(dsl, native)


class A1DecoderLayer:
    """Prepared full decoder layer on the caller's NPU context and stream.

    Input/output, residual and paged-cache addresses remain fixed. Metadata
    contents may change before replay. Reset all enclosing graphs before close.
    """

    def __init__(
        self,
        images: A1DecoderImages,
        hidden: torch.Tensor,
        residual: torch.Tensor,
        metadata: A1DecodeMetadata,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attention_weights: dict[str, torch.Tensor],
        expert_weights: A1ExpertLayerWeights,
        output: torch.Tensor,
        residual_out: torch.Tensor,
    ) -> None:
        from simpler_setup import A1BorrowedLaunch

        if hidden.device.type != "npu" or hidden.device.index is None:
            raise ValueError(f"A1 decoder requires an NPU input, got {hidden.device}")
        device = hidden.device
        self.output = output
        self.residual_out = residual_out
        self._owners = ExitStack()
        self._calls = []
        self._closed = False
        self.status = {}
        self.weights = {name: tensor.to(device) for name, tensor in attention_weights.items()}

        def zeros(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
            return torch.zeros(shape, dtype=dtype, device=device)

        self.buffers = {
            "norm": zeros((1, 2048), torch.bfloat16),
            "skip": zeros((1, 2048), torch.bfloat16),
            "quant": zeros((1, 2048), torch.int8),
            "scale": zeros((1, 16), torch.float32),
            "qkv_x": zeros((128, 2048), torch.int8),
            "qkv_route_scale": zeros((8, 16), torch.float32),
            "qkv_acc": zeros((16, 5120), torch.int32),
            "qkv": zeros((16, 5120), torch.bfloat16),
            "k_input": zeros((8, 128), torch.bfloat16),
            "rope": zeros((1, 128), torch.bfloat16),
            "q_norm": zeros((32, 128), torch.bfloat16),
            "q": zeros((32, 128), torch.bfloat16),
            "k_norm": zeros((8, 128), torch.bfloat16),
            "k": zeros((8, 128), torch.bfloat16),
            "staged_keys": zeros((2048, 128), torch.bfloat16),
            "staged_values": zeros((2048, 128), torch.bfloat16),
            "attention": zeros((32, 128), torch.bfloat16),
            "o_quant": zeros((1, 4096), torch.int8),
            "o_scale": zeros((1, 16), torch.float32),
            "o_x": zeros((128, 4096), torch.int8),
            "o_route_scale": zeros((8, 16), torch.float32),
            "o_acc": zeros((16, 2048), torch.int32),
            "o": zeros((16, 2048), torch.bfloat16),
            "post_norm": zeros((1, 2048), torch.bfloat16),
            "post_quant": zeros((1, 2048), torch.int8),
            "post_scale": zeros((1, 16), torch.float32),
        }
        b, w = self.buffers, self.weights

        def dsl(name: str, tensors: dict[str, torch.Tensor]) -> None:
            kernel = images.dsl[name]
            owner = self._owners.enter_context(A1BorrowedLaunch(kernel.image, device_id=device.index))
            self._calls.append(owner.prepare(*(tensors[label] for label in kernel.order)))

        def native(name: str, tensors: tuple[torch.Tensor, ...], blocks: int, *, status: bool = True) -> None:
            owner = self._owners.enter_context(A1BorrowedLaunch(images.native[name], device_id=device.index))
            if status:
                self.status[name] = torch.full((blocks, 16), -7, dtype=torch.int32, device=device)
                tensors = (*tensors, self.status[name])
            self._calls.append(owner.prepare(*tensors, blocks=blocks))

        try:
            dsl("norm", {
                "x": hidden, "residual": residual, "weight": w["input_layernorm.weight"].reshape(1, 2048),
                "y": b["norm"], "residual_out": b["skip"], "quant": b["quant"], "scale": b["scale"][:, :1],
            })
            native("route_w13", (b["quant"], b["scale"], b["qkv_x"], b["qkv_route_scale"]), 8)
            dsl("qkv", {
                "x": b["qkv_x"][:16], "input_scale": b["scale"], "weight": w["qkv_weight"],
                "weight_scale": w["qkv_scale"], "acc_out": b["qkv_acc"], "y": b["qkv"],
            })
            native("qkv_prepare", (b["qkv"], metadata.position, metadata.cos_sin_cache, b["k_input"], b["rope"]), 1)
            dsl("q_norm", {
                "x": b["qkv"][:1, :4096].reshape(32, 128),
                "weight": w["self_attn.q_norm.weight"].reshape(1, 128), "cos_sin": b["rope"],
                "norm": b["q_norm"], "rotated": b["q"],
            })
            dsl("k_norm", {
                "x": b["k_input"], "weight": w["self_attn.k_norm.weight"].reshape(1, 128),
                "cos_sin": b["rope"], "norm": b["k_norm"], "rotated": b["k"],
            })
            native("append", (
                b["k"][:4], b["qkv"][:1, 4608:].reshape(4, 128), key_cache, value_cache, metadata.slot,
            ), 1)
            native("gather", (
                key_cache, value_cache, metadata.block_table, metadata.lengths, b["staged_keys"], b["staged_values"],
            ), 4)
            self.groups = []
            for group in range(4):
                fields = {
                    "query": b["q"][group * 8 : (group + 1) * 8],
                    "keys": b["staged_keys"][group * 512 : (group + 1) * 512],
                    "values": b["staged_values"][group * 512 : (group + 1) * 512],
                    "lengths": metadata.lengths,
                    "scores": zeros((8, 512), torch.float32),
                    "probabilities": zeros((8, 512), torch.bfloat16),
                    "sums": zeros((1, 8), torch.float32),
                    "out": zeros((8, 128), torch.float32),
                }
                self.groups.append(fields)
                dsl("qk", fields)
                dsl("softmax", fields)
                dsl("pv", fields)
                native("normalize", (fields["out"], fields["sums"], b["attention"][group * 8 : (group + 1) * 8]), 8, status=False)
            dsl("o_quant", {"x": b["attention"].reshape(1, 4096), "quant": b["o_quant"], "scale": b["o_scale"][:, :1]})
            native("route_o", (b["o_quant"], b["o_scale"], b["o_x"], b["o_route_scale"]), 8)
            dsl("o", {
                "x": b["o_x"][:16], "input_scale": b["o_scale"], "weight": w["o_weight"],
                "weight_scale": w["o_scale"], "acc_out": b["o_acc"], "y": b["o"],
            })
            dsl("norm", {
                "x": b["o"][:1], "residual": b["skip"], "weight": w["post_attention_layernorm.weight"].reshape(1, 2048),
                "y": b["post_norm"], "residual_out": residual_out, "quant": b["post_quant"], "scale": b["post_scale"][:, :1],
            })
            self.moe = self._owners.enter_context(A1MoeDecode(
                images.native, b["post_norm"], b["post_quant"], b["post_scale"], w["mlp.gate.weight"], expert_weights, output,
            ))
        except BaseException:
            self._owners.close()
            self._closed = True
            raise

    def enqueue(self) -> None:
        """Enqueue the complete layer without allocating, copying or synchronizing."""
        if self._closed:
            raise RuntimeError("A1 decoder layer is closed")
        for call in self._calls:
            call()
        self.moe.enqueue()

    def close(self) -> None:
        """Release prepared resources after enclosing graphs are destroyed."""
        self._owners.close()
        self._closed = True

    def __enter__(self) -> "A1DecoderLayer":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
