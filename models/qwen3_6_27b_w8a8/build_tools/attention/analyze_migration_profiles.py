# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the installed Toolkit on lossless compatibility aliases of native records."""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
GATE_ROOT = ROOT.parent / "attention_dsl_steady_fixture_20261029"
sys.path.insert(0, str(GATE_ROOT / "src"))
from qwen36.tools.analyze_profiles import prepare

folders = []
for label in ["reference", "candidate"]:
    for rank in [0, 3]:
        source = GATE_ROOT / f"evidence/attention_dsl_steady_gate_01/kernel_pair/rank{rank}/profile/{label}"
        dest = ROOT / f"evidence/toolkit_steady/{label}/rank{rank}"
        prepare(source, dest, rank)
        folders.append(dest)
node = "/root/.vscode-server/bin/07f806f999227108933c2e30515b26eecc1fda74/node"
script = ROOT.parent / "direct_w8a8_20261027/tools/toolkit_analysis.cjs"
subprocess.run([node, str(script), *map(str, folders)], check=True, timeout=180)
