# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Assemble DSL-generated kernels with the existing Simpler runtime ABI."""

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
from pypto import backend, ir

root = Path(__file__).resolve().parents[1]
backend.set_backend_type(backend.BackendType.Ascend910B)
parser = argparse.ArgumentParser()
parser.add_argument("--softmax-build", default="softmax_expands_compile")
parser.add_argument("--only-softmax", action="store_true")
args = parser.parse_args()
records = []
for name, build in [
    ("softmax", root / "build" / args.softmax_build / "test_production_softmax_compil0"),
    ("qk", root / "build/attention_dsl_compile/test_paged_attention_compiles_0"),
    ("pv", root / "build/attention_dsl_compile/test_paged_attention_compiles_1"),
]:
    if args.only_softmax and name != "softmax":
        continue
    compiled = ir.CompiledProgram.from_dir(build, platform="a2a3")
    callable_ = compiled.chip_callable
    data = ctypes.string_at(callable_.buffer_ptr(), callable_.buffer_size())
    image = build / "callable.image"
    image.write_bytes(data)
    records.append(
        dict(
            name=name,
            build=str(build),
            image=str(image),
            bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
        )
    )
    print("BUILT", records[-1], flush=True)
(root / "evidence" / (args.softmax_build + "_images.json")).write_text(json.dumps(records, indent=2) + "\n")
