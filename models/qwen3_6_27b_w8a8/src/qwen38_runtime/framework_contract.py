# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Fail explicitly when private vLLM extension points change upstream."""

from pathlib import Path
import hashlib, json


def verify_files(entries):
    for entry in entries:
        path = Path(entry["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise RuntimeError(
                "Framework integration contract changed; review before repinning: " + str(path)
            )


def validate_framework():
    root = Path(__file__).resolve().parents[2]
    verify_files(json.loads((root / "reference/framework_contract.json").read_text()))
