# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Optional source audit only; AST is read, never rewrites a model/build source."""

import argparse, ast, json
from pathlib import Path

W = Path(__file__).resolve().parents[3]


def text(node):
    return ast.unparse(node)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=W / "evidence/tuning_parameter_inventory_v3.json")
    args = parser.parse_args()
    registry = (W / "src/qwen36/operators/registry.py").read_text()
    tree = ast.parse(registry)
    registered = set()
    for item in tree.body:
        if isinstance(item, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "OPERATORS" for t in item.targets
        ):
            registered = {e.id for e in item.value.elts}
    records = []
    for file in sorted((W / "src/qwen36/operators").glob("*.py")):
        if file.name == "registry.py":
            continue
        tree = ast.parse(file.read_text())
        constants = {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        constants[target.id] = text(node.value)
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            if not any(
                isinstance(d, ast.Call) and text(d.func) == "pl.function" for d in node.decorator_list
            ):
                continue
            loops = []
            loads = []
            transforms = []
            resources = []
            spmd = []
            for child in ast.walk(node):
                if isinstance(child, ast.For):
                    loops.append(dict(index=text(child.target), range=text(child.iter)))
                if not isinstance(child, ast.Call):
                    continue
                name = text(child.func)
                if name in ("pl.load", "pl.tile.load", "pl.slice", "pl.tile.extract"):
                    loads.append(text(child))
                if any(k in name for k in ["reshape", "expand", "set_validshape", "func_attr"]):
                    transforms.append(text(child))
                if name.startswith("pl.system."):
                    resources.append(text(child))
                if name == "pl.spmd":
                    spmd.append(text(child))
            candidate = file.stem in ("collective_lanes", "collective_even_lanes", "down_prefetch")
            records.append(
                dict(
                    name=node.name,
                    module=file.stem,
                    source=str(file),
                    line=node.lineno,
                    selected_by_reference_registry=node.name in registered and not candidate,
                    explicit_opt_in_candidate=candidate,
                    declared_arguments={
                        a.arg: text(a.annotation) if a.annotation else None for a in node.args.args
                    },
                    for_loops=loops,
                    source_loads_and_slices=loads,
                    shape_and_broadcast_operations=transforms,
                    resource_queries=resources,
                    spmd=spmd,
                    module_constants=constants,
                    parameter_status="Inherited reference: only named current-graph experiments establish performance benefit"
                    if not candidate
                    else "Independent candidate; see named micro/full evidence",
                    warning="Static call sites are not dynamic load counts; use generated loops and profile for actual multiplicity. Logical geometry is intentionally fixed; loop/tiling/lane policies require tuning.",
                )
            )
    result = dict(
        scope="All direct reference and explicit opt-in candidate PyPTO source functions",
        functions=records,
        registered_reference_names=len(registered),
        native_source_boundary=str(W / "src/qwen36/native_backend/link_manifest.json"),
        actual_generated_buffers=str(W / "evidence/operator_audit.json"),
    )
    out = args.output
    out.write_text(json.dumps(result, indent=2) + "\n")
    print("TUNING_INVENTORY", len(records), str(out))


if __name__ == "__main__":
    main()
