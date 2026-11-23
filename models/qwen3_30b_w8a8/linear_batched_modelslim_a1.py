# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
"""B20 projection using explicitly borrowed native physical NZ storage."""

from pathlib import Path
from tempfile import TemporaryDirectory

from simpler_setup import KernelCompiler


def compile_a1_batched_modelslim_projection(
    kind: str,
    pto_isa_root: Path,
    *,
    wide: bool = False,
    fused_dequant: bool = False,
    packed_scale_words: bool = False,
) -> bytes:
    """Build a QKV/O kernel from this checkout, without converting resident weights."""
    if kind not in ("qkv", "o"):
        raise ValueError(f"Expected qkv/o projection, got {kind}")
    if fused_dequant and not wide:
        raise ValueError("A1 fused dequantization requires the wide NZ projection")
    if packed_scale_words and not fused_dequant:
        raise ValueError("A1 native scale words require fused dequantization")
    if packed_scale_words:
        inner, width = (2048, 5120) if kind == "qkv" else (4096, 2048)
        with TemporaryDirectory(prefix="a1-projection-") as temporary:
            source = Path(temporary) / f"a1_{kind}_native_scale.cpp"
            source.write_text(
                '#include "a1_nz_projection_batched_wide.h"\n'
                'extern "C" __global__ AICORE void SingleLaunchEntry(__gm__ int64_t* args) {\n'
                f"  BatchedNZProjectionWide<{inner}, {width}, 256, true, true>(args);\n"
                "}\n",
                encoding="utf-8",
            )
            return KernelCompiler("a1").compile_single_launch(
                str(source),
                str(pto_isa_root),
                extra_include_dirs=[str(Path(__file__).parent)],
            )
    suffix = "_dequant" if fused_dequant else "_wide" if wide else ""
    source = Path(__file__).with_name(f"a1_{kind}_nz_batched{suffix}.cpp")
    return KernelCompiler("a1").compile_single_launch(str(source), str(pto_isa_root))


def compile_a1_batched_modelslim_dequant(pto_isa_root: Path) -> bytes:
    """Build the native VDEQ16 rounding boundary with integer quantization bias."""
    source = Path(__file__).with_name("a1_modelslim_dequant_batched.cpp")
    return KernelCompiler("a1").compile_single_launch(str(source), str(pto_isa_root))
