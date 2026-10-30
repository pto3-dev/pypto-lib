# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Offline evidence extractor, not part of compilation or model execution.

AST is used only to inventory ordinary source. C++ static call counts are not
runtime transfer counts. Unresolved addresses are retained, never assumed safe.
"""

import ast, hashlib, json, re
from pathlib import Path
from pypto import backend, ir

W = Path(__file__).resolve().parents[3]
ROOT = W.parents[1]


def src(node):
    return ast.unparse(node)


def ancestors(node, parents):
    while node in parents:
        node = parents[node]
        yield node


def main():
    backend.set_backend_type(backend.BackendType.Ascend910B)
    b = backend.get_backend_instance(backend.BackendType.Ascend910B)
    capacities = {k: b.get_mem_size(getattr(ir.MemorySpace, k)) for k in ("Mat", "Left", "Right", "Acc")}
    resources = dict(
        backend="Ascend910B",
        compiler_cube_capacity=b.get_core_count(ir.CoreType.CUBE),
        memory_capacity_bytes=capacities,
        physical_cube_count="runtime pl.system.available_cluster_count(); NOT compiler capacity",
        physical_aiv_count="runtime pl.system.available_aiv_count()",
    )
    functions = []
    for file in sorted((W / "src/qwen36/operators").glob("*.py")):
        text = file.read_text()
        tree = ast.parse(text)
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        for f in ast.walk(tree):
            if not isinstance(f, ast.FunctionDef):
                continue
            if not any("pl.function" in src(d) for d in f.decorator_list):
                continue
            own = [
                n
                for n in ast.walk(f)
                if isinstance(n, ast.Call)
                and not any(
                    isinstance(a, ast.FunctionDef) and a is not f
                    for a in list(ancestors(n, parents))[: list(ancestors(n, parents)).index(f)]
                )
            ]
            calls = [src(c) for c in own]
            kernels = [
                src(n)
                for n in ast.walk(f)
                if isinstance(n, ast.With) and any("pl.spmd" in src(x.context_expr) for x in n.items)
            ]
            ancestry = list(ancestors(f, parents))
            parent = next((n.name for n in ancestry if isinstance(n, ast.FunctionDef)), None)
            functions.append(
                dict(
                    name=f.name,
                    owner=parent,
                    source=str(file),
                    line=f.lineno,
                    decorators=[src(d) for d in f.decorator_list],
                    parameters=[
                        dict(name=a.arg, type=src(a.annotation) if a.annotation else None)
                        for a in f.args.args
                    ],
                    returns=src(f.returns) if f.returns else None,
                    pipeline_loops=[c for c in calls if c.startswith("pl.pipeline(")],
                    loads_and_extracts=[
                        c
                        for c in calls
                        if c.startswith(
                            ("pl.load(", "pl.tile.load(", "pl.tile.extract(", "pld.tile.remote_load(")
                        )
                    ],
                    resource_queries=[c for c in calls if c.startswith("pl.system.available_")],
                    spmd_scopes=kernels,
                    provenance="Explicit extraction from validated norm_quant_l1k2560_candidate/sources/rank0/mixed_rank0.py; see reference/migration_manifest.json",
                    current_graph_benefit="Inherited unless a named paired experiment is cited; functionality/IR equality is not performance evidence",
                )
            )
    manifest = json.loads((W / "src/qwen36/native_backend/link_manifest.json").read_text())
    kernels = []
    root = W / "build/behavior_reference/next_levels/rank0"
    for k in manifest["kernels"]:
        name = k["name"].format(rank=0)
        p = (
            W / "src/qwen36/native_backend" / k["native_source"]
            if k["native_source"]
            else root / "kernels" / k["core_type"] / (name + ".cpp")
        )
        text = p.read_text()
        body = text
        if name.endswith("_aic") and name.removesuffix("_aic") + "_aiv" in body:
            body = body.split("static __aicore__ void " + name.removesuffix("_aic") + "_aiv")[0]
        if name.endswith("_aiv") and name.removesuffix("_aiv") + "_aic" in body:
            tag = "static __aicore__ void " + name
            body = body[body.index(tag) :] if tag in body else body
        values = {
            m[1]: int(m[2])
            for m in re.finditer(r"(?:const )?(?:int32_t|int64_t|uint64_t) (v\d+) = (\d+);", body)
        }
        for _ in range(6):
            for m in re.finditer(r"uint64_t (v\d+) = \(uint64_t\) (v\d+);", body):
                if m[2] in values:
                    values[m[1]] = values[m[2]]
        tiles = {}
        sizes = {"bfloat16_t": 2, "half": 2, "float": 4, "int32_t": 4, "int8_t": 1, "uint8_t": 1}
        for m in re.finditer(
            r"Tile<TileType::(Mat|Left|Right|Acc), (bfloat16_t|half|float|int32_t|int8_t|uint8_t), (\d+), (\d+),[^\n]+> (v\d+) =",
            body,
        ):
            space, dtype, rows, cols, var = m.groups()
            tiles[var] = (space, int(rows) * int(cols) * sizes[dtype], dtype, int(rows), int(cols))
        regions = {key: set() for key in capacities}
        unknown = []
        for tile, address in re.findall(r"TASSIGN\((v\d+), (v\d+)\);", body):
            if tile not in tiles:
                continue
            space, nbytes, *_ = tiles[tile]
            if address not in values:
                unknown.append([tile, address])
                continue
            regions[space].add((values[address], nbytes))
        end = {key: max((off + size for off, size in ranges), default=0) for key, ranges in regions.items()}
        if any(end[key] > capacities[key] for key in end):
            raise ValueError((name, end, capacities))
        kernels.append(
            dict(
                **k,
                resolved_name=name,
                source=str(p),
                source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                address_ranges_bytes={key: sorted(val) for key, val in regions.items()},
                maximum_resolved_end_bytes=end,
                unresolved_address_assignments=unknown,
                capacity_check="resolved ranges fit; unresolved assignments require inspection"
                if unknown
                else "all resolved declared Mat/Left/Right/Acc ranges fit",
                syntactic_instruction_count={
                    op: len(re.findall(r"\b" + op + r"\(", body))
                    for op in (
                        "TLOAD",
                        "TEXTRACT",
                        "TMATMUL",
                        "TMATMUL_ACC",
                        "TSTORE",
                        "TPUSH",
                        "TPOP",
                        "TFREE",
                    )
                },
                static_count_warning="Code-site counts only, not dynamic runtime traffic or buffering depth",
            )
        )
    profiles = {}
    for file in [
        ROOT / "evidence/norm_quant_full_profile_analysis.json",
        ROOT / "evidence/l1_candidate_full_profile_analysis.json",
    ]:
        profiles[file.name] = json.loads(file.read_text())
    evidence = dict(
        resources=resources,
        logical_rows=24,
        requests=4,
        mtp_drafts=5,
        source_functions=functions,
        registered_kernels=kernels,
        profiles=profiles,
        buffer_warning="Independent accumulator addresses need not be output-job double buffering; GM C2V slots are not L0C buffers",
        model_scope="64-layer complete target/verifier; native drafter excluded",
    )
    (W / "evidence/operator_audit.json").write_text(json.dumps(evidence, indent=2) + "\n")
    doc = [
        "# 算子形状、参数与实际缓冲审计",
        "",
        f"后端查询：L1/Mat={capacities['Mat']}，L0A/Left={capacities['Left']}，L0B/Right={capacities['Right']}，L0C/Acc={capacities['Acc']} bytes；编译拓扑上限 {resources['compiler_cube_capacity']}，不是物理核数。",
        "",
        "逻辑请求 batch=4；MTP5 target 的编译行数=24（4×6），矩阵 M 补齐至32。INT8 NZ 的内块为16×32，BF16为16×16。下表区分源码函数和实际注册 kernel；没有单独当前整图证据的继承参数均记为未验证收益。",
        "",
        "## 实际执行 kernel",
        "",
        "| ID | kernel | 类型 | Mat / Left / Right / Acc 地址上界 bytes | 原生实现 |",
        "|---|---|---|---|---|",
    ]
    for k in kernels:
        e = k["maximum_resolved_end_bytes"]
        doc.append(
            f"| {k['func_id']} | {k['resolved_name']} | {k['core_type']} | "
            + " / ".join(str(e[x]) for x in capacities)
            + f" | {k['native_source'] or 'PyPTO生成'} |"
        )
    doc += [
        "",
        "地址上界及每个地址区间、未解析项、代码位置和SHA256见 operator_audit.json。上界不是同时存活字节数；CPU静态代码中的TLOAD次数不是运行时次数。不得由本表推断所有Acc均双缓冲。",
        "",
        "## 全部源码算子及接口",
        "",
    ]
    for f in functions:
        doc += [
            f"### {f['name']}",
            "",
            f"路径：{f['source']}:{f['line']}；类型：" + ", ".join(f["decorators"]),
            "",
            "```python",
            f"{f['name']}("
            + ", ".join(a["name"] + ": " + str(a["type"]) for a in f["parameters"])
            + ") -> "
            + str(f["returns"]),
            "```",
            "",
        ]
        if f["pipeline_loops"]:
            doc += ["外层/内层流水调用：`" + "`；`".join(f["pipeline_loops"]) + "`。", ""]
        if f["resource_queries"]:
            doc += ["资源查询：`" + "`；`".join(f["resource_queries"]) + "`。", ""]
        doc += [
            "来源：已测 L1K2560 源码的显式模块化迁移；逐项性能状态见 OPTIMIZATION_STATUS.md，未列入收益的项只确认继承。",
            "",
        ]
    (W / "docs/OPERATOR_AUDIT.md").write_text("\n".join(doc) + "\n")
    print(
        "AUDIT_COMPLETE",
        len(functions),
        "source functions",
        len(kernels),
        "registered kernels",
        resources,
        flush=True,
    )


if __name__ == "__main__":
    main()
