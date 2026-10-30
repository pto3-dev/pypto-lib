# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Only this explicit vLLM extension package is replaced; no source rewriting."""

import importlib.abc, importlib.util, os, sys
from pathlib import Path

root = Path(os.environ["QWEN38_RUNTIME_ROOT"])


class IntegrationFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "vllm_ascend.worker.model_runner_v1":
            return importlib.util.spec_from_file_location(
                fullname, root / "vendor_overlay/model_runner_v1.py"
            )
        if fullname == "vllm_ascend.pypto":
            package = root / "integration/vllm_ascend/pypto"
            return importlib.util.spec_from_file_location(
                fullname, package / "__init__.py", submodule_search_locations=[str(package)]
            )
        return None


sys.meta_path.insert(0, IntegrationFinder())
