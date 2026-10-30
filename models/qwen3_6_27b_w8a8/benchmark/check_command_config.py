# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU-only regression for the benchmark CLI and nested command configuration."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile

P = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("recovery_test", P / "tools/run_fast_fused.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
captured = {}
with tempfile.TemporaryDirectory(prefix="recovery-command-test-") as tmp:
    root = Path(tmp)
    (root / "tools").mkdir()
    (root / "evidence").mkdir()
    for d in ["src", "integration", "bootstrap", "reference", "lib", "vendor_overlay"]:
        (root / d).mkdir()
    (root / "src/qwen38_runtime").mkdir()
    (root / "env.sh").write_text("# command construction test only\n")
    m.P = root
    m.wait_for_idle = lambda *a, **k: None

    def stage(*a, **k):
        cmd = m.runner.command(m.runner.MODEL, k["build"], root, 12345, backend="pypto")
        cfg = json.loads(cmd[cmd.index("--additional-config") + 1])["qwen38_pto_mpmd"]
        assert cfg["compact_inputs"] == "both" and cfg["terminal_profile"] is False
        assert "native_tp_reduce" not in cfg
        assert json.loads(cmd[cmd.index("--speculative-config") + 1])["enforce_eager"] is False
        assert json.loads(cmd[cmd.index("--compilation-config") + 1])["mode"] == 3
        assert k["terminal_validate"] is False
        captured.update(status="passed", gpu_started=False, configuration=cfg)

    m.runner.stage = stage
    sys.argv = [
        "run_fast_fused.py",
        "--name",
        "only_command_test",
        "--devices",
        "1,2,3,4",
        "--skip-terminal-validation",
    ]
    m.main()
(P / "evidence/RECOVERY_COMMAND_TEST.json").write_text(json.dumps(captured, indent=2) + "\n")
print("COMMAND_CONFIGURATION_PASS_NO_GPU")
