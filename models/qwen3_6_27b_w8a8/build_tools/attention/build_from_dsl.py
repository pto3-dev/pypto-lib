# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compile direct DSL sources and materialize Simpler images; never rewrite source."""

import argparse
import ctypes
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from pypto import backend
from pypto.runtime import ExecutionMode, RunConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--only", choices=["all", "softmax"], default="all")
    parser.add_argument("--softmax-mode", choices=["automatic", "explicit"], default="automatic")
    parser.add_argument("--source-root", type=Path, default=ROOT / "src")
    args = parser.parse_args()
    if Path(args.name).name != args.name:
        raise ValueError("Build name must be one path component")
    output = ROOT / "build" / args.name
    output.mkdir(exist_ok=False)
    frozen = output / "source_snapshot"
    shutil.copytree(
        args.source_root / "qwen36_attention_dsl",
        frozen / "qwen36_attention_dsl",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    sys.path.insert(0, str(frozen))
    from qwen36_attention_dsl.paged_attention import compute_scores, compute_partial_values
    from qwen36_attention_dsl.softmax import make_row_cached_softmax

    backend.set_backend_type(backend.BackendType.Ascend910B)
    if args.softmax_mode == "explicit":
        from qwen36_attention_dsl.softmax_explicit import make_explicit_row_cached_softmax

        softmax = make_explicit_row_cached_softmax()
    else:
        softmax = make_row_cached_softmax()
    entries = {"softmax": softmax}
    if args.only == "all":
        entries.update(qk=compute_scores, pv=compute_partial_values)
    records = []
    for label, kernel in entries.items():
        dest = output / label
        compiled = kernel.compile(
            config=RunConfig(
                backend_type=backend.BackendType.Ascend910B,
                execution_mode=ExecutionMode.ONBOARD,
                save_kernels=True,
                save_kernels_dir=str(dest),
                dump_passes=True,
            )
        )
        callable_ = compiled.chip_callable
        data = ctypes.string_at(callable_.buffer_ptr(), callable_.buffer_size())
        image = dest / "callable.image"
        image.write_bytes(data)
        records.append(
            dict(
                name=label,
                build=str(dest),
                image=str(image),
                bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest(),
            )
        )
        print("BUILT", records[-1], flush=True)
    (output / "images.json").write_text(json.dumps(records, indent=2) + "\n")
    paths = sorted(frozen.rglob("*.py"))
    (output / "source_manifest.json").write_text(
        json.dumps({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
