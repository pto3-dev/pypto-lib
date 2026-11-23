# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""Prepared FP16 ModelSlim TP1 decoder; separate from the BF16 checkpoint path."""

import re
from contextlib import ExitStack
from pathlib import Path

import pypto.language as pl
import torch
from pypto import backend
from pypto.backend import BackendType
from pypto.runtime import RunConfig, a1_runner
from simpler_setup import A1BorrowedLaunch, KernelCompiler

from .attention_a1 import compile_a1_attention_normalize, make_a1_attention_qk
from .attention_modelslim_a1 import make_a1_modelslim_softmax_pv
from .decode_a1 import A1CompiledKernel, A1DecodeMetadata, A1DecoderImages
from .kv_cache import compile_a1_kv_append, compile_a1_kv_gather
from .linear_modelslim_a1 import (
    compile_a1_modelslim_dequant,
    decode_a1_modelslim_scale,
    make_a1_modelslim_projection,
)
from .moe_a1 import A1MoeDecode, compile_a1_moe_images
from .norm_modelslim_a1 import make_a1_modelslim_norm
from .qk_rope import make_a1_head_norm_rope
from .quant_modelslim_a1 import make_a1_modelslim_static_quant
from .weights_a1 import A1ExpertLayerWeights


def compile_a1_modelslim_decoder_images(directory: Path, isa: Path) -> A1DecoderImages:
    """Compile only this checkout's sources for the explicit ModelSlim contract."""
    programs = {
        "plain_norm": make_a1_modelslim_norm(fused_add=False),
        "norm": make_a1_modelslim_norm(fused_add=True),
        "qkv_quant": make_a1_modelslim_static_quant(2048),
        "qkv": make_a1_modelslim_projection(5120, 2048),
        "q_norm": make_a1_head_norm_rope(32, storage_dtype=pl.FP16),
        "k_norm": make_a1_head_norm_rope(4, storage_dtype=pl.FP16),
        "qk": make_a1_attention_qk(512, storage_dtype=pl.FP16, position_tiles=8),
        "softmax_pv": make_a1_modelslim_softmax_pv(512),
        "o_quant": make_a1_modelslim_static_quant(4096),
        "o": make_a1_modelslim_projection(2048, 4096),
    }
    previous = backend.get_backend_type() if backend.is_backend_configured() else None
    dsl = {}
    try:
        for name, program in programs.items():
            backend.reset_for_testing()
            compiled = program.compile(
                config=RunConfig(
                    arch=BackendType.Ascend310P, save_kernels=True, save_kernels_dir=str(directory / name)
                )
            )
            work_dir = Path(compiled.output_dir)
            sources = list((work_dir / "kernels").glob("*/*.cpp"))
            if len(sources) != 1:
                raise ValueError(f"ModelSlim {name} requires one static kernel, got {len(sources)}")
            labels = re.findall(r"// Unpack tensor: (\w+)", sources[0].read_text())
            order = tuple(label.split("__ssa_")[0] for label in labels)
            if not order or len(set(order)) != len(order):
                raise ValueError(f"ModelSlim {name}: invalid tensor argument order {order}")
            image = a1_runner._compile_image(work_dir, sources[0], str(isa))
            dsl[name] = A1CompiledKernel(image, order)
    finally:
        backend.reset_for_testing()
        if previous is not None:
            backend.set_backend_type(previous)
    native = compile_a1_moe_images(isa, modelslim=True, build_dir=directory / "moe")
    native.update(
        {
            "normalize": compile_a1_attention_normalize(
                isa, storage_dtype=pl.FP16, build_dir=directory / "normalize"
            ),
            "dequant": compile_a1_modelslim_dequant(isa),
            "append": compile_a1_kv_append(isa),
            "gather": compile_a1_kv_gather(isa),
            "qkv_prepare": KernelCompiler("a1").compile_single_launch(
                str(Path(__file__).with_name("a1_qkv_prepare.cpp")), str(isa)
            ),
            "route_o": KernelCompiler("a1").compile_single_launch(
                str(Path(__file__).with_name("a1_route_o.cpp")), str(isa)
            ),
        }
    )
    return A1DecoderImages(dsl, native)


class A1ModelSlimDecoderLayer:
    """FP16 Norm/bias, static Attention projections and dynamic expert graph.

    Native raw 16-bit copy kernels are shared without reinterpretation of
    arithmetic. Device metadata and all launch addresses stay fixed on replay.
    The enclosing graph must be destroyed before close().
    """

    def __init__(
        self,
        images: A1DecoderImages,
        hidden: torch.Tensor,
        residual: torch.Tensor,
        metadata: A1DecodeMetadata,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        attention_weights: dict[str, torch.Tensor],
        expert_weights: A1ExpertLayerWeights,
        output: torch.Tensor,
        residual_out: torch.Tensor,
        *,
        first_layer: bool = False,
    ) -> None:
        key_cache, value_cache = kv_cache
        device = hidden.device
        if device.type != "npu" or device.index is None or hidden.dtype != torch.float16:
            raise ValueError(
                f"ModelSlim decoder requires indexed FP16 NPU input, got {device}/{hidden.dtype}"
            )
        for tensor in (hidden, residual, output, residual_out):
            if tensor.shape != (1, 2048) or tensor.dtype != torch.float16 or tensor.device != device:
                raise ValueError("ModelSlim hidden/residual tensors must be FP16[1,2048] on one device")
        for tensor in (key_cache, value_cache):
            if tensor.shape != (512, 512) or tensor.dtype != torch.float16 or tensor.device != device:
                raise ValueError("ModelSlim private KV cache must be FP16[512,512] on the model device")
        if "plain_norm" not in images.dsl or "softmax_pv" not in images.dsl:
            raise ValueError("ModelSlim decoder requires explicitly compiled FP16 images")
        self._owners = ExitStack()
        self._calls = []
        self._closed = False
        self.output, self.residual_out = output, residual_out
        self.status = {}
        cpu_weights = dict(attention_weights)
        for name, width in (("qkv", 2048), ("o", 4096)):
            cpu_weights[f"{name}_reciprocal"] = (1 / cpu_weights[f"{name}_input_scale"]).repeat(1, width)
            cpu_weights[f"{name}_offset"] = cpu_weights[f"{name}_input_offset"].half().repeat(1, width)
            cpu_weights[f"{name}_scale"] = decode_a1_modelslim_scale(cpu_weights.pop(f"{name}_deq_scale"))
        self.weights = {name: value.to(device) for name, value in cpu_weights.items()}

        def zeros(shape, dtype=torch.float16):
            return torch.zeros(shape, dtype=dtype, device=device)

        self.buffers = {
            "norm": zeros((1, 2048)),
            "skip": zeros((1, 2048)),
            "quant": zeros((1, 2048), torch.int8),
            "scale": zeros((1, 16), torch.float32),
            "qkv_x": zeros((128, 2048), torch.int8),
            "route_scale": zeros((8, 16), torch.float32),
            "qkv_acc": zeros((16, 5120), torch.int32),
            "qkv": zeros((16, 5120)),
            "k_input": zeros((8, 128)),
            "rope": zeros((1, 128)),
            "q_norm": zeros((32, 128)),
            "q": zeros((32, 128)),
            "k_norm": zeros((8, 128)),
            "k": zeros((8, 128)),
            "staged_keys": zeros((2048, 128)),
            "staged_values": zeros((2048, 128)),
            "attention": zeros((32, 128)),
            "o_quant": zeros((1, 4096), torch.int8),
            "o_x": zeros((128, 4096), torch.int8),
            "o_acc": zeros((16, 2048), torch.int32),
            "o": zeros((1, 2048)),
            "post_norm": zeros((1, 2048)),
            "post_quant": zeros((1, 2048), torch.int8),
            "post_scale": zeros((1, 16), torch.float32),
        }
        b, w = self.buffers, self.weights

        def dsl(name, tensors):
            kernel = images.dsl[name]
            owner = self._owners.enter_context(A1BorrowedLaunch(kernel.image, device_id=device.index))
            blocks = 8 if name == "qk" else 1
            self._calls.append(owner.prepare(*(tensors[label] for label in kernel.order), blocks=blocks))

        def native(name, tensors, blocks, *, image_name=None, status=True):
            owner = self._owners.enter_context(
                A1BorrowedLaunch(images.native[image_name or name], device_id=device.index)
            )
            if status:
                self.status[name] = torch.full((blocks, 16), -7, dtype=torch.int32, device=device)
                tensors = (*tensors, self.status[name])
            self._calls.append(owner.prepare(*tensors, blocks=blocks))

        def norm(name, x, skip, norm_name, y, skip_out):
            dsl(
                name,
                {
                    "x": x,
                    "residual": skip,
                    "weight": w[f"{norm_name}.weight"].reshape(1, 2048),
                    "bias": w[f"{norm_name}.bias"].reshape(1, 2048),
                    "y": y,
                    "residual_out": skip_out,
                },
            )

        try:
            norm(
                "plain_norm" if first_layer else "norm",
                hidden,
                residual,
                "input_layernorm",
                b["norm"],
                b["skip"],
            )
            dsl(
                "qkv_quant",
                {
                    "x": b["norm"],
                    "reciprocal": w["qkv_reciprocal"],
                    "offset": w["qkv_offset"],
                    "quant": b["quant"],
                },
            )
            native("route_w13", (b["quant"], b["scale"], b["qkv_x"], b["route_scale"]), 8)
            dsl("qkv", {"x": b["qkv_x"][:16], "weight": w["qkv_weight"], "acc_out": b["qkv_acc"]})
            native(
                "qkv_dequant",
                (b["qkv_acc"][:1], w["qkv_quant_bias"], w["qkv_scale"], b["qkv"][:1]),
                8,
                image_name="dequant",
            )
            native(
                "qkv_prepare",
                (b["qkv"], metadata.position, metadata.cos_sin_cache, b["k_input"], b["rope"]),
                1,
            )
            for kind, source in (("q", b["qkv"][:1, :4096].reshape(32, 128)), ("k", b["k_input"])):
                dsl(
                    f"{kind}_norm",
                    {
                        "x": source,
                        "weight": w[f"self_attn.{kind}_norm.weight"].reshape(1, 128),
                        "cos_sin": b["rope"],
                        "norm": b[f"{kind}_norm"],
                        "rotated": b[kind],
                    },
                )
            native(
                "append",
                (b["k"][:4], b["qkv"][:1, 4608:].reshape(4, 128), key_cache, value_cache, metadata.slot),
                1,
            )
            native(
                "gather",
                (
                    key_cache,
                    value_cache,
                    metadata.block_table,
                    metadata.lengths,
                    b["staged_keys"],
                    b["staged_values"],
                ),
                4,
            )
            self.groups = []
            for group in range(4):
                fields = {
                    "query": b["q"][group * 8 : (group + 1) * 8],
                    "keys": b["staged_keys"][group * 512 : (group + 1) * 512],
                    "values": b["staged_values"][group * 512 : (group + 1) * 512],
                    "lengths": metadata.lengths,
                    "scores": zeros((8, 512), torch.float32),
                    "sums": zeros((1, 8), torch.float32),
                    "out": zeros((8, 128), torch.float32),
                }
                self.groups.append(fields)
                dsl("qk", fields)
                dsl("softmax_pv", fields)
                native(
                    f"normalize_{group}",
                    (fields["out"], fields["sums"], b["attention"][group * 8 : (group + 1) * 8]),
                    8,
                    image_name="normalize",
                    status=False,
                )
            dsl(
                "o_quant",
                {
                    "x": b["attention"].reshape(1, 4096),
                    "reciprocal": w["o_reciprocal"],
                    "offset": w["o_offset"],
                    "quant": b["o_quant"],
                },
            )
            # Raw INT8 padding only; route scale is not consumed by a static projection.
            native("o_route", (b["o_quant"], b["scale"], b["o_x"], b["route_scale"]), 8, image_name="route_o")
            dsl("o", {"x": b["o_x"][:16], "weight": w["o_weight"], "acc_out": b["o_acc"]})
            native(
                "o_dequant",
                (b["o_acc"][:1], w["o_quant_bias"], w["o_scale"], b["o"]),
                8,
                image_name="dequant",
            )
            norm("norm", b["o"], b["skip"], "post_attention_layernorm", b["post_norm"], residual_out)
            self.moe = self._owners.enter_context(
                A1MoeDecode(
                    images.native,
                    b["post_norm"],
                    b["post_quant"],
                    b["post_scale"],
                    w["mlp.gate.weight"],
                    expert_weights,
                    output,
                    modelslim=True,
                )
            )
        except BaseException:
            self.close()
            raise

    def enqueue(self):
        """Enqueue the fixed-address layer without host synchronization or allocations."""
        if self._closed:
            raise RuntimeError("ModelSlim decoder is closed")
        for call in self._calls:
            call()
        self.moe.enqueue()

    def close(self):
        self._owners.close()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
