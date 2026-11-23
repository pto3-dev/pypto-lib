# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""B20 reuses the qualified PTO3 head arithmetic with immutable per-request bindings."""

from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable

import torch
from simpler_setup import A1BorrowedLaunch, KernelCompiler

from .compile_modelslim_a1 import compile_modelslim_program


def compile_a1_batched_qk_images(build_dir: Path, isa: Path, *, batched_heads: bool = False, direct_head_rows: bool = False, pack_append: bool = False, dsl_direct_heads: bool = False) -> dict:
    """Compile B1-qualified head arithmetic once and the B20 bit-preserving adapters."""
    import pypto.language as pl

    from .qk_rope import make_a1_head_norm_rope
    from .qk_norm_rope_batched_a1 import make_a1_batched_head_norm_rope

    factory = make_a1_batched_head_norm_rope if batched_heads else make_a1_head_norm_rope
    options = {"rows": 20} if batched_heads else {}
    if direct_head_rows and not batched_heads:
        raise ValueError("Direct head row loads require batched heads")
    if dsl_direct_heads:
        if not (batched_heads and direct_head_rows):
            raise ValueError("DSL direct heads require batched direct-row processing")
        options["resident_rows"] = True
    if direct_head_rows and not dsl_direct_heads:
        order = ("x", "weight", "cos_sin", "norm", "rotated")
        images = {kind: (KernelCompiler("a1").compile_single_launch(
            str(Path(__file__).with_name(f"a1_head_norm_rope_direct_{kind}.cpp")), str(isa)), order)
                  for kind in ("q", "k")}
    else:
        images = {kind: compile_modelslim_program(factory(heads, storage_dtype=pl.FP16, **options),
                                                 build_dir / kind, isa)
                  for kind, heads in (("q", 32), ("k", 4))}
    images["batched_heads"] = batched_heads
    images["dsl_direct_heads"] = dsl_direct_heads
    images["pack_append"] = pack_append
    for kind, filename in (("prepare", "a1_qkv_prepare_batched.cpp"), ("pack", "a1_qk_pack_batched.cpp")):
        if kind == "prepare":
            from .qkv_prepare_batched_a1 import prepare_qkv_rows

            image, order = compile_modelslim_program(prepare_qkv_rows, build_dir / "prepare", isa)
            expected = ("qkv", "positions", "table", "queries", "keys", "values", "rope_rows", "status")
            if tuple(order) != expected:
                raise ValueError(f"A1 DSL QKV preparation tensor ABI changed: {order}")
            images[kind] = image
            continue
        if kind == "pack" and pack_append:
            from .qk_pack_append_batched_a1 import pack_qk_append_b20

            image, order = compile_modelslim_program(pack_qk_append_b20, build_dir / "pack", isa)
            expected = ("q", "k", "packed_q", "packed_k", "status", "values",
                        "key_cache", "value_cache", "slots", "append_status")
            if tuple(order) != expected:
                raise ValueError(f"A1 DSL Q/K packing tensor ABI changed: {order}")
            images[kind] = image
            continue
        images[kind] = KernelCompiler("a1").compile_single_launch(str(Path(__file__).with_name(filename)), str(isa))
    return images


class A1BatchedHeadNormRope:
    """No host reads or allocations on replay; Q/K parameters stay borrowed."""

    def __init__(self, images: dict, qkv: torch.Tensor, positions: torch.Tensor, table: torch.Tensor,
                 q_weight: torch.Tensor, k_weight: torch.Tensor, *, cache=None, slots=None, launch_factory: Callable[[bytes, int], Any] = A1BorrowedLaunch):
        if qkv.shape != (20, 5120) or qkv.dtype != torch.float16 or not qkv.is_contiguous():
            raise ValueError("B20 Q/K requires contiguous FP16[20,5120]")
        device = qkv.device
        self._owners = ExitStack()
        self._calls = []
        self.buffers = {
            name: torch.zeros(shape, dtype=torch.float16, device=device)
            for name, shape in (("q_raw", (640, 128)), ("k_raw", (160, 128)),
                                ("q_norm", (640, 128)), ("k_norm", (160, 128)),
                                ("q_rotated", (640, 128)), ("k_rotated", (160, 128)),
                                ("rope", (20, 128)), ("value", (20, 512)),
                                ("query", (80, 16, 128)), ("key", (20, 512)))
        }
        self.status = {name: torch.full((rows, 16), -7, dtype=torch.int32, device=device)
                       for name, rows in (("prepare", 20), ("pack", 80))}
        b = self.buffers
        try:
            owner = self._owners.enter_context(launch_factory(images["prepare"], device.index))
            self._calls.append(owner.prepare(qkv, positions, table, b["q_raw"], b["k_raw"], b["value"],
                                              b["rope"], self.status["prepare"], blocks=8))
            for kind, heads, weight in (("q", 32, q_weight), ("k", 8, k_weight)):
                image, order = images[kind]
                owner = self._owners.enter_context(launch_factory(image, device.index))
                if images.get("batched_heads", False):
                    tensors = {"x": b[f"{kind}_raw"], "weight": weight.reshape(1, 128), "cos_sin": b["rope"],
                               "norm": b[f"{kind}_norm"], "rotated": b[f"{kind}_rotated"]}
                    self._calls.append(owner.prepare(*(tensors[name] for name in order), blocks=8))
                    continue
                for row in range(20):
                    tensors = {"x": b[f"{kind}_raw"][row * heads:(row + 1) * heads],
                               "weight": weight.reshape(1, 128), "cos_sin": b["rope"][row:row + 1],
                               "norm": b[f"{kind}_norm"][row * heads:(row + 1) * heads],
                               "rotated": b[f"{kind}_rotated"][row * heads:(row + 1) * heads]}
                    self._calls.append(owner.prepare(*(tensors[name] for name in order)))
            owner = self._owners.enter_context(launch_factory(images["pack"], device.index))
            if images.get("pack_append", False):
                import torch_npu

                if cache is None or slots is None:
                    raise ValueError("Fused head pack/append requires native cache banks and persistent slots")
                self.status["append"] = torch.full((20, 16), -7, dtype=torch.int32, device=device)
                cache_nz = torch_npu.get_npu_format(cache[0]) == 29
                self._calls.append(owner.prepare(b["q_rotated"], b["k_rotated"], b["query"], b["key"],
                                                  self.status["pack"], b["value"], *cache, slots,
                                                  self.status["append"], blocks=8,
                                                  native_pa_caches=(6, 7) if cache_nz else ()))
            else:
                self._calls.append(owner.prepare(b["q_rotated"], b["k_rotated"], b["query"], b["key"],
                                                  self.status["pack"], blocks=8))
        except BaseException:
            self._owners.close()
            raise

    def enqueue(self) -> None:
        """Enqueue prepared calls into the caller's capturable stream."""
        for call in self._calls:
            call()

    def close(self) -> None:
        """Close only after every graph referring to these bindings is destroyed."""
        self._owners.close()
