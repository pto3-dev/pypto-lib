# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Toolkit aliases for standalone collective probes, with their own func-ID map."""

import argparse, json, subprocess
from pathlib import Path
from qwen36.tools.analyze_profiles import W, NODE, prepare


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    folders = []
    for label in ["reference", "candidate"]:
        for rank in range(4):
            folder = a.output / label / f"rank{rank}"
            prepare(a.run / label / f"rank{rank}", folder, rank)
            deps = json.loads((folder / "deps.json").read_text())
            used = {k for t in deps["tasks"] for k in t["kernel_ids"] if k >= 0}
            if used != {0}:
                raise ValueError("Unexpected standalone probe func IDs")
            (folder / "name_map.json").write_text(
                json.dumps(
                    dict(
                        callable_id_to_name={"0": "reduce_add_norm_quant_kernel"},
                        source="explicit standalone compiled root",
                    ),
                    indent=2,
                )
                + "\n"
            )
            folders.append(folder)
    with (a.output / "toolkit.log").open("w") as log:
        subprocess.run(
            [NODE, str(W / "tools/toolkit_analysis.cjs"), *map(str, folders)],
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=180,
            check=True,
        )
    print("COMM_TOOLKIT_DONE", str(a.output), flush=True)


if __name__ == "__main__":
    main()
