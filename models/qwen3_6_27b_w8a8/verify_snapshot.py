# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Audit the published source checkpoint without importing PyPTO or Torch."""

import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def digest(data):
    return hashlib.sha256(data).hexdigest()


class WithoutImports(ast.NodeTransformer):
    """Ignore unused-import cleanup, preserving all executable/math statements."""

    def visit_Import(self, node):
        return None

    def visit_ImportFrom(self, node):
        return None


def normalized_ast(text):
    tree = WithoutImports().visit(ast.parse(text))
    return digest(ast.dump(tree, include_attributes=False).encode())


def verify(originals=False, artifacts=False):
    entries = json.loads((ROOT / "source_provenance.json").read_text())
    checked = 0
    for relative, entry in entries.items():
        path = ROOT / relative
        data = path.read_bytes()
        if digest(data) != entry["published_sha256"]:
            raise ValueError(f"Published source changed: {relative}")
        if path.suffix == ".py":
            if normalized_ast(data.decode()) != entry["normalized_ast_sha256"]:
                raise ValueError(f"Executable statements changed: {relative}")
        if originals:
            original = Path(entry["source"]).read_bytes()
            if digest(original) != entry["source_sha256"]:
                raise ValueError(f"Original source changed: {entry['source']}")
            if path.suffix == ".py":
                if normalized_ast(original.decode()) != normalized_ast(data.decode()):
                    raise ValueError(f"Published math/body differs: {relative}")
            elif original != data and not (
                entry.get("cpp_empty_line_whitespace_cleaned")
                and b"".join(line if line.strip() else b"\n" for line in original.splitlines(keepends=True))
                == data
            ):
                raise ValueError(f"Non-Python source differs: {relative}")
        checked += 1
    binary_count = 0
    if artifacts:
        for filename, entry in json.loads((ROOT / "evidence/artifacts.json").read_text()).items():
            data = Path(filename).read_bytes()
            if len(data) != entry["bytes"] or digest(data) != entry["sha256"]:
                raise ValueError(f"Tested artifact changed: {filename}")
            binary_count += 1
    return {"source_files": checked, "binary_files": binary_count, "originals": originals, "status": "passed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--originals", action="store_true")
    parser.add_argument("--artifacts", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.originals, args.artifacts), indent=2))


if __name__ == "__main__":
    main()
