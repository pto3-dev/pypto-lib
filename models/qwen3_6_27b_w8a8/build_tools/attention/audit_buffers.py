# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Inspect generated C++ assignments; dynamic addresses remain explicit unknowns."""

from pathlib import Path
import argparse, ast, json, re
from pypto import backend, ir

R = Path(__file__).resolve().parents[1]


def inspect(path):
    text = path.read_text()
    values = {}
    tiles = {}
    assignments = []
    for line in text.splitlines():
        const = re.search(r"(?:const )?(?:int64_t|uint64_t|int32_t) (v\d+) = (.*);", line)
        if const:
            expr = re.sub(r"\((?:u?int(?:32|64)_t)\)", "", const[2]).strip()
            try:
                tree = ast.parse(expr, mode="eval")
                if all(
                    isinstance(
                        n,
                        (
                            ast.Expression,
                            ast.Constant,
                            ast.Name,
                            ast.Load,
                            ast.BinOp,
                            ast.Add,
                            ast.Sub,
                            ast.Mult,
                            ast.Div,
                            ast.FloorDiv,
                            ast.UnaryOp,
                            ast.USub,
                        ),
                    )
                    for n in ast.walk(tree)
                ):
                    values[const[1]] = eval(
                        compile(tree, "<generated-address>", "eval"), {"__builtins__": {}}, values
                    )
            except (SyntaxError, NameError, TypeError, ZeroDivisionError):
                pass
        tile = re.search(
            r"Tile<TileType::(Vec|Mat|Left|Right|Acc), ([^,]+), (\d+), (\d+),.*?> (v\d+) =", line
        )
        if tile:
            tiles[tile[5]] = dict(space=tile[1], dtype=tile[2], shape=[int(tile[3]), int(tile[4])])
        bind = re.search(r"TASSIGN\((v\d+), (v\d+)\)", line)
        if bind and bind[1] in tiles:
            assignments.append(
                dict(
                    tile=bind[1],
                    **tiles[bind[1]],
                    address_bytes=values.get(bind[2]),
                    address_expression=bind[2],
                )
            )
    return dict(
        source=str(path),
        assignments=assignments,
        static_addresses_by_space={
            space: sorted(
                {
                    a["address_bytes"]
                    for a in assignments
                    if a["space"] == space and a["address_bytes"] is not None
                }
            )
            for space in ["Vec", "Mat", "Left", "Right", "Acc"]
        },
        lexical_instructions={
            k: len(re.findall(r"\b" + k + r"\(", text))
            for k in ["TLOAD", "TMOV", "TMATMUL", "TMATMUL_ACC", "TSTORE", "TGATHER", "TROWMAX", "TROWSUM"]
        },
        caveat="Static source inspection only; lexical calls are not dynamic transfer counts, and distinct addresses alone do not prove temporal overlap.",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--softmax", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    device = backend.get_backend_instance(backend.BackendType.Ascend910B)
    limits = {
        s.name: device.get_mem_size(s)
        for s in [
            ir.MemorySpace.Vec,
            ir.MemorySpace.Mat,
            ir.MemorySpace.Left,
            ir.MemorySpace.Right,
            ir.MemorySpace.Acc,
        ]
    }
    files = [
        R / "build/direct_source_v2/qk/ptoas/s2_scores.cpp",
        R / "build/direct_source_v2/pv/ptoas/s2_partial_values.cpp",
        a.softmax / "ptoas/softmax_kernel.cpp",
    ]
    out = dict(backend="Ascend910B", queried_limits_bytes=limits, records=[inspect(p) for p in files])
    a.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({p["source"]: p["static_addresses_by_space"] for p in out["records"]}, indent=2))


if __name__ == "__main__":
    main()
