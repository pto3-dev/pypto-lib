# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Assign existing full-grid MLP MIX blocks to unique runtime clusters."""

import ast, json, re, hashlib
from pathlib import Path


def install_fixed_mlp(build):
    build = Path(build)
    reports = []
    for rank in (0, 1):
        root = build / "next_levels" / f"rank{rank}"
        tree = ast.parse((root / "kernel_config.py").read_text())
        assignment = next(
            n
            for n in tree.body
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "KERNELS" for t in n.targets)
        )
        entries = {}
        for node in assignment.value.elts:
            fields = {ast.literal_eval(k): v for k, v in zip(node.keys, node.values)}
            entries[ast.literal_eval(fields["name"])] = ast.literal_eval(fields["func_id"])
        cube = entries["project_mlp_kernel_aic"]
        vec = entries["project_mlp_kernel_aiv"]
        f = root / "orchestration" / f"rank{rank}.cpp"
        text = f.read_text()
        pattern = re.compile(
            r"(MixedKernels mixed_\d+ = \{"
            + str(cube)
            + r", "
            + str(vec)
            + r", "
            + str(vec)
            + r"\};\s*(params_t\d+)\.launch_spec\.set_block_num\([^;]+;)",
            re.M,
        )
        text, count = pattern.subn(
            lambda m: m[1]
            + "\n                                "
            + m[2]
            + ".launch_spec.set_static_block_mapping(true);",
            text,
        )
        if count != 4:
            raise ValueError(f"Expected four MLP sites, found {count}")
        if "set_require_sync_start(true)" in text:
            raise ValueError("Unexpected global sync-start")
        f.write_text(text)
        reports.append(
            dict(
                rank=rank,
                static_sites=4,
                executed_layers=64,
                kernel_ids=[cube, vec, vec],
                no_new_kernels=True,
                no_new_state=True,
                entry_sync_start=False,
                source_sha256=hashlib.sha256(text.encode()).hexdigest(),
            )
        )
    (build / "fixed_mlp_manifest.json").write_text(json.dumps(reports, indent=2) + "\n")
