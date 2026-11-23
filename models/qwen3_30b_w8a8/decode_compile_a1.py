# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""Startup compilation of the FP16/W8A8 B20 decoder's public DSL leaves."""

import fcntl
import hashlib
import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable

from .compile_modelslim_a1 import compile_modelslim_artifact

_LOG = logging.getLogger(__name__)
_CACHE_SCHEMA = 1


@dataclass(frozen=True)
class A1DecoderCompilation:
    """Compiler output consumed directly by the vLLM model-root assembler."""

    images: dict[str, Any]
    catalogue: dict[str, Any]
    directory: Path
    cache_hit: bool


def image_signature(images: dict[str, Any]) -> str:
    """Include role assignment, ABI, execution options and every ELF byte."""

    def normalize(value):
        if isinstance(value, bytes):
            return {"elf_sha256": hashlib.sha256(value).hexdigest()}
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        return value

    return hashlib.sha256(json.dumps(normalize(images), sort_keys=True).encode()).hexdigest()


def _tree_signature(root: Path, suffixes: tuple[str, ...] | None = None) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if suffixes is None or path.suffix in suffixes:
            digest.update(path.relative_to(root).as_posix().encode() + b"\0")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _build_inputs(physical_pages: int, isa: Path) -> dict[str, Any]:
    import pypto
    import simpler_setup
    from pypto.backend._ptoas_locate import find_ptoas_binary

    ptoas, ccec = find_ptoas_binary(), shutil.which("ccec")
    if not ptoas or not ccec:
        raise ValueError("A1 startup compilation requires PTOAS and CANN ccec")
    tools = {}
    for name, executable in (("ptoas", ptoas), ("ccec", ccec)):
        path = Path(executable).resolve()
        version = subprocess.run([str(path), "--version"], check=True, capture_output=True, text=True)
        tools[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "version": version.stdout + version.stderr,
        }
    model = Path(__file__).parent
    return {
        "schema": _CACHE_SCHEMA,
        "physical_pages": physical_pages,
        "batch": 20,
        "dtype": "FP16",
        "model_sources": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(model.iterdir())
            if p.suffix in (".py", ".cpp", ".h")
        },
        "pypto": _tree_signature(Path(pypto.__file__).parent, (".py", ".so")),
        "simpler_python": _tree_signature(Path(simpler_setup.__file__).parent, (".py",)),
        "simpler_a1": _tree_signature(Path(simpler_setup.__file__).parents[1] / "src/a1", (".h", ".cpp")),
        "isa_root": str(isa),
        "isa_headers": _tree_signature(isa / "include"),
        "tools": tools,
    }


def _program_specs(physical_pages: int) -> list[tuple[str, Callable[[], Any], tuple[str, ...]]]:
    """Describe each DSL leaf and its ABI without constructing the program."""
    import pypto.language as pl

    from .attention_batched_modelslim_a1 import make_attention_b20
    from .combine_modelslim_a1 import (
        resident_expert_combine,
        resident_expert_combine_norm_quant,
    )
    from .expert_w2_batched_a1 import make_w2_b20
    from .expert_w13_batched_a1 import make_w13_b20
    from .linear_batched_modelslim_a1_dsl import make_a1_modelslim_projection
    from .norm_modelslim_a1 import make_a1_modelslim_norm
    from .qk_norm_rope_batched_a1 import make_a1_batched_head_norm_rope
    from .qk_norm_rope_fused_batched_a1 import make_fused_qk_norm_rope
    from .qk_pack_append_batched_a1 import pack_qk_append_b20
    from .qkv_prepare_batched_a1 import prepare_qkv_rows
    from .quant_batched_modelslim_a1_vector import (
        make_a1_batched_modelslim_vector_quant,
    )
    from .quant_dynamic_modelslim_a1_vector import make_a1_modelslim_dynamic_quant
    from .router_batched_a1 import make_router_b20
    from .route_metadata_batched_a1 import route_metadata_b20
    from .swiglu_quant_batched_a1 import swiglu_quant_b20
    from .topk_batched_a1 import topk_b20

    norm = ("x", "residual", "weight", "bias", "y", "residual_out")
    projection = ("x", "weight", "bias", "scales", "output", "status")
    head = ("x", "weight", "cos_sin", "norm", "rotated")
    combine = ("acc", "input_scale", "ids", "weight_scale", "route_weights", "expert_y", "y", "status")
    return [
        ("dsl.plain_norm", partial(make_a1_modelslim_norm, fused_add=False, rows=20, balanced_reduction=True), norm),
        ("dsl.norm", partial(make_a1_modelslim_norm, fused_add=True, rows=20, resident_row=True), norm),
        (
            "dsl.qkv_quant",
            partial(make_a1_batched_modelslim_vector_quant, 2048),
            ("x", "reciprocal", "offset", "quant"),
        ),
        ("dsl.o_quant", partial(make_a1_batched_modelslim_vector_quant, 4096), ("x", "reciprocal", "offset", "quant")),
        ("native.qkv", partial(make_a1_modelslim_projection, "qkv"), projection),
        ("native.o", partial(make_a1_modelslim_projection, "o"), projection),
        (
            "native.combine_norm_quant",
            lambda: resident_expert_combine_norm_quant,
            combine + (
                "residual", "gamma", "beta", "norm_y", "residual_out",
                "reciprocal", "offset", "quant",
            ),
        ),
        (
            "native.attention",
            partial(make_attention_b20, physical_pages=physical_pages),
            ("query", "keys", "values", "lengths", "output", "block_table", "status"),
        ),
        (
            "qk.q",
            partial(make_a1_batched_head_norm_rope, 32, storage_dtype=pl.FP16, rows=20, resident_rows=True),
            head,
        ),
        ("qk.k", partial(make_a1_batched_head_norm_rope, 4, storage_dtype=pl.FP16, rows=20, resident_rows=True), head),
        (
            "qk.fused_qk",
            partial(make_fused_qk_norm_rope),
            (
                "query",
                "query_weight",
                "cos_sin",
                "query_norm",
                "query_rotated",
                "key",
                "key_weight",
                "key_norm",
                "key_rotated",
            ),
        ),
        (
            "qk.prepare",
            lambda: prepare_qkv_rows,
            ("qkv", "positions", "table", "queries", "keys", "values", "rope_rows", "status"),
        ),
        (
            "qk.pack",
            lambda: pack_qk_append_b20,
            (
                "q",
                "k",
                "packed_q",
                "packed_k",
                "status",
                "values",
                "key_cache",
                "value_cache",
                "slots",
                "append_status",
            ),
        ),
        ("moe.router", partial(make_router_b20), ("input", "weight", "output", "status")),
        ("moe.topk", lambda: topk_b20, ("logits", "ids", "weights", "status")),
        (
            "moe.route_metadata", lambda: route_metadata_b20,
            ("ids", "expert_counts", "route_indices", "status"),
        ),
        ("moe.quant", partial(make_a1_modelslim_dynamic_quant, 20, 2048), ("x", "quant", "scale", "status")),
        (
            "moe.swiglu",
            lambda: swiglu_quant_b20,
            ("input", "input_scale", "ids", "weight_scale", "quant", "scale", "status"),
        ),
        ("moe.combine", lambda: resident_expert_combine, combine),
        (
            "moe.w13",
            partial(make_w13_b20, column_partition=True),
            ("input", "weight", "ids", "expert_counts", "route_indices", "output", "channel_scale", "status"),
        ),
        ("moe.w2", partial(make_w2_b20, column_partition=False, columns_per_owner=2), ("input", "weight", "ids", "expert_counts", "route_indices", "output", "status")),
    ]



def _programs(
    physical_pages: int, *, roles: set[str] | None = None
) -> list[tuple[str, Any, tuple[str, ...]]]:
    """Construct the requested leaf programs in the stable catalogue order."""
    return [
        (role, factory(), order)
        for role, factory, order in _program_specs(physical_pages)
        if roles is None or role in roles
    ]

def _layout_metadata(pages, isa, inputs):
    images = {
        "dsl": {},
        "native": {},
        "qk": {"batched_heads": True, "dsl_direct_heads": True, "pack_append": True},
        "moe": {"w13_shared_routes": True, "w2_shared_routes": True},
        "source_hashes": inputs["model_sources"],
        "execution_options": {
            "direct_pages": True,
            "metadata_foreach": True,
            "fused_head_cache": True,
            "packed_projection_scales": True,
            "fused_projection_dequant": True,
            "dsl_projection": True,
            "packed_swiglu_scales": True,
            "grouped_w13_dequant": True,
            "fuse_next_input_norm": True,
            "fuse_next_input_quant": True,
            "dsl_attention": True,
            "attention_physical_pages": pages,
        },
    }
    catalogue = {
        "schema": 4,
        "build_kind": "startup_dsl",
        "source_hashes": inputs["model_sources"],
        "sources": {},
        "isa_source": {"root": str(isa), "include_tree_sha256": inputs["isa_headers"]},
        "execution_scheduler": "aicpu_per_core",
        "execution_scope": "model",
        "per_core_command_version": 2,
        "defer_gate_control": True,
    }
    return images, catalogue


def _validate_physical_pages(physical_pages: int) -> None:
    if type(physical_pages) is not int or not 1 <= physical_pages <= ((1 << 32) - 1) // (32 * 128 * 16):
        raise ValueError(f"A1 physical page extent is invalid: {physical_pages!r}")


def prepare_a1_layer_jit(physical_pages: int, cache_dir: Path) -> A1DecoderCompilation:
    """Build the binding recipe without compiling individual leaf images."""
    from simpler_setup.a1_kernel_plan import A1KernelRole
    from pypto.runtime.a1_runner import _resolve_isa_source

    _validate_physical_pages(physical_pages)
    isa = Path(_resolve_isa_source(os.environ.get("PTO_ISA_ROOT"))).resolve()
    inputs = _build_inputs(physical_pages, isa)
    directory = cache_dir / "direct-layer-jit"
    images, catalogue = _layout_metadata(physical_pages, isa, inputs)
    catalogue["build_kind"] = "direct_layer_jit"
    images["moe"]["expert"] = A1KernelRole("moe.expert")
    for role, _, order in _program_specs(physical_pages):
        if role in {"moe.w13", "moe.swiglu", "moe.w2"}:
            continue
        group, name = role.split(".")
        with_order = group == "dsl" or (group == "qk" and name in ("q", "k", "fused_qk"))
        token = A1KernelRole(role)
        images[group][name] = (token, order) if with_order else token
    images["native"]["attention_o_quant"] = A1KernelRole("native.attention_o_quant")
    return A1DecoderCompilation(images, catalogue, directory, False)


def _compile(directory: Path, isa: Path, inputs: dict[str, Any]) -> tuple[dict, dict, list]:
    from simpler_setup.a1_argument_contract import compiled_tensor_contract
    from simpler_setup.a1_per_core_graph import _logical_entry
    from simpler_setup.elf_parser import allocated_section_fingerprints

    pages = inputs["physical_pages"]
    images, catalogue = _layout_metadata(pages, isa, inputs)
    records = []
    for role, program, expected in _programs(pages):
        _LOG.info("[PTO3 A1 compile] %s", role)
        compiled = compile_modelslim_artifact(program, directory / role, isa)
        if compiled.order != expected:
            raise ValueError(f"A1 {role} tensor ABI changed: got {compiled.order}, expected {expected}")
        _, compiler_files = _logical_entry(compiled.source)
        contract = compiled_tensor_contract(compiled.source)
        if contract is None:
            raise ValueError(f"A1 {role} has no compiler-owned tensor access contract")
        metadata = compiled.source.parent.parent.parent / "compiled_meta.json"
        compiler_files[str(metadata)] = contract["compiled_metadata_sha256"]
        group, name = role.split(".")
        with_order = group == "dsl" or (group == "qk" and name in ("q", "k", "fused_qk"))
        images[group][name] = (compiled.image, compiled.order) if with_order else compiled.image
        image_file = directory / role / "image.bin"
        image_file.write_bytes(compiled.image)
        digest = hashlib.sha256(compiled.image).hexdigest()
        source = compiled.source
        item = {
            "source": str(source),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "include_dirs": [str(source.parent)],
            "allocated_sections": allocated_section_fingerprints(compiled.image),
            "compiler_files": compiler_files,
        }
        binding = compiled.source.with_name(compiled.source.name.removesuffix(".cann.cpp") + ".completion-binding.json")
        if binding.is_file():
            pin = hashlib.sha256(binding.read_bytes()).hexdigest()
            item["a1_completion_binding"] = {"path": str(binding), "sha256": pin}
            compiler_files[str(binding)] = pin
        if role == "native.attention":
            record = directory / role / "attention-compile.json"
            record.write_text(
                json.dumps(
                    {
                        "ccec_compiled": True,
                        "image_sha256": digest,
                        "physical_pages": pages,
                        "order": list(compiled.order),
                    }
                )
            )
            item["attention_specialization"] = {
                "physical_pages": pages,
                "tensor_order": list(compiled.order),
                "blocks": 8,
                "native_pa_caches": [1, 2],
                "compile_record": str(record),
                "compile_record_sha256": hashlib.sha256(record.read_bytes()).hexdigest(),
            }
        catalogue["sources"][digest] = item
        records.append(
            {
                "role": role,
                "file": str(image_file.relative_to(directory)),
                "order": list(compiled.order),
                "with_order": with_order,
                "sha256": digest,
            }
        )
    catalogue["compiled_images_sha256"] = image_signature(images)
    return images, catalogue, records


def compile_a1_decoder(physical_pages: int, cache_dir: Path) -> A1DecoderCompilation:
    """Build on a cache miss before capture; never read a prepackaged model image."""
    from pypto.runtime.a1_runner import _resolve_isa_source

    _validate_physical_pages(physical_pages)
    isa = Path(_resolve_isa_source(os.environ.get("PTO_ISA_ROOT"))).resolve()
    inputs = _build_inputs(physical_pages, isa)
    key = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    cache_dir = Path(cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    directory = cache_dir / key
    with (cache_dir / f"{key}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        directory.mkdir(exist_ok=True)
        manifest = directory / "complete.json"
        hit = manifest.is_file()
        if hit:
            saved = json.loads(manifest.read_text())
            if saved["inputs"] != inputs:
                raise ValueError("A1 compilation cache has a mismatched build identity")
            images, catalogue = saved["images"], saved["catalogue"]
            for record in saved["records"]:
                file = (directory / record["file"]).resolve()
                if not file.is_relative_to(directory):
                    raise ValueError("A1 cached image escapes its compilation directory")
                image = file.read_bytes()
                if hashlib.sha256(image).hexdigest() != record["sha256"]:
                    raise ValueError(f"A1 cached image changed: {record['role']}")
                group, name = record["role"].split(".")
                images[group][name] = (image, tuple(record["order"])) if record["with_order"] else image
            if image_signature(images) != catalogue["compiled_images_sha256"]:
                raise ValueError("A1 cached role/ABI assignment changed")
            for item in catalogue["sources"].values():
                for filename, expected in item["compiler_files"].items():
                    if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
                        raise ValueError(f"A1 cached compiler metadata changed: {filename}")
        else:
            images, catalogue, records = _compile(directory, isa, inputs)
            # Publish only after every DSL leaf has compiled successfully.
            skeleton = {
                key: (dict(value) if isinstance(value, dict) else value) for key, value in images.items()
            }
            for record in records:
                group, name = record["role"].split(".")
                del skeleton[group][name]
            saved = {"inputs": inputs, "images": skeleton, "catalogue": catalogue, "records": records}
            if _build_inputs(physical_pages, isa) != inputs:
                raise ValueError("A1 compiler inputs changed during compilation")
            temporary = directory / "complete.tmp"
            temporary.write_text(json.dumps(saved, indent=2))
            temporary.replace(manifest)
        _LOG.info("[PTO3 A1 compile] %s: %s", "cache hit" if hit else "cold build", directory)
        return A1DecoderCompilation(images, catalogue, directory, hit)
