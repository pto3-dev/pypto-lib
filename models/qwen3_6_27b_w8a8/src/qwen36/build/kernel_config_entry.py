# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Fixed build-metadata loader. No source rewriting or generated expressions."""
from pathlib import Path
from qwen36.build.link_backend import configuration
_config=configuration(Path(__file__).parent)
RUNTIME_CONFIG=_config['RUNTIME_CONFIG']
ORCHESTRATION=_config['ORCHESTRATION']
KERNELS=_config['KERNELS']
