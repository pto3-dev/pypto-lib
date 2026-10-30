# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Real-engine prefill/decode, variable lengths and native cache reuse."""

import json
from pathlib import Path
import subprocess
import sys

import pytest


def test_full_pto_native_cache(request, tmp_path):
    model = request.config.getoption("--qwen38-model")
    baseline = request.config.getoption("--qwen38-native-result")
    if not model or not baseline:
        pytest.skip("Provide the local 27B checkpoint and matched native baseline explicitly.")
    result = Path(request.config.getoption("--qwen38-result") or tmp_path / "result.json")
    native = json.loads(Path(baseline).read_text())
    prompts = tmp_path / "prompts.json"
    prompts.write_text(json.dumps([item["prompt"] for item in native["validation_requests"]]))
    root = Path(__file__).resolve().parents[3]
    subprocess.run(
        [
            sys.executable,
            str(root / "examples/pypto_qwen38_benchmark.py"),
            "--model",
            model,
            "--result",
            str(result),
            "--expected-native",
            baseline,
            "--runtime-stats",
            "--validation-prompts",
            str(prompts),
        ],
        check=True,
        timeout=1200,
    )
    report = json.loads(result.read_text())
    assert report["pto_full_model"]
    assert report["verification"]["native_aliases_verified"] == 64
    assert report["verification"]["all_cache_aliases_match"]
    assert report["verification"]["validation_sequences_match_native"]
    assert len([item for item in report["requests"] if not item["warmup"]]) == 3
