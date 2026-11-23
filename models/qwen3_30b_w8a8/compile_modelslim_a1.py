# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""Compile this checkout's A1 kernels with explicit FP16 storage specialization."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from simpler_setup import KernelCompiler


def compile_modelslim_source(filename: str, build_dir: Path, pto_isa_root: Path) -> bytes:
    """Generate a compiler input wrapper; never copy prebuilt kernel artifacts."""
    source = Path(__file__).with_name(filename)
    if source.suffix != ".cpp" or not source.is_file():
        raise ValueError(f"Unknown ModelSlim source: {filename}")
    build_dir.mkdir(parents=True, exist_ok=True)
    wrapper = build_dir / f"{source.stem}_fp16.cpp"
    wrapper.write_text(f'#define PYPTO_A1_MODEL_FP16 1\n#include "{source.name}"\n', encoding="utf-8")
    return KernelCompiler("a1").compile_single_launch(
        str(wrapper), str(pto_isa_root), extra_include_dirs=[str(source.parent)]
    )


@dataclass(frozen=True)
class A1KernelCompilation:
    """A DSL leaf and the compiler-owned source used to build its image."""

    image: bytes
    order: tuple[str, ...]
    source: Path


def compile_modelslim_artifact(program: Any, build_dir: Path, pto_isa_root: Path) -> A1KernelCompilation:
    """Compile one DSL leaf, retaining its source and adjacent ABI metadata."""
    from pypto import backend
    from pypto.backend import BackendType
    from pypto.runtime import RunConfig, a1_runner

    previous = backend.get_backend_type() if backend.is_backend_configured() else None
    try:
        backend.reset_for_testing()
        config = RunConfig(arch=BackendType.Ascend310P, save_kernels=True, save_kernels_dir=str(build_dir))
        compiled = program.compile(config=config)
        work_dir = Path(compiled.output_dir)
        sources = list((work_dir / "kernels").glob("*/*.cpp"))
        if len(sources) != 1:
            raise ValueError(f"A1 ModelSlim requires one static kernel, got {len(sources)}")
        labels = re.findall(r"// Unpack tensor: (\w+)", sources[0].read_text())
        order = tuple(label.split("__ssa_")[0] for label in labels)
        if not order or len(order) != len(set(order)):
            raise ValueError(f"A1 ModelSlim requires unique tensor arguments, got {order}")
        image = a1_runner._compile_image(work_dir, sources[0], str(pto_isa_root))
        source = sources[0].with_suffix(".cann.cpp")
        source.write_bytes(
            sources[0].read_bytes()
            + (
                b'\nextern "C" __global__ AICORE void PyptoA1TensorEntry(__gm__ int64_t* args) {\n'
                b"    kernel_entry(args);\n}\n"
            )
        )
        return A1KernelCompilation(image, order, source)
    finally:
        backend.reset_for_testing()
        if previous is not None:
            backend.set_backend_type(previous)


def compile_modelslim_program(
    program: Any, build_dir: Path, pto_isa_root: Path
) -> tuple[bytes, tuple[str, ...]]:
    """Compile one PTO3 InCore program and return its image and tensor argument order."""
    compiled = compile_modelslim_artifact(program, build_dir, pto_isa_root)
    return compiled.image, compiled.order
