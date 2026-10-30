# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from pathlib import Path
import ctypes, hashlib, json, shutil
from pypto import backend
from pypto.runtime import RunConfig, ExecutionMode
from qwen36_fusion.gate_quant import make_gate_quant

ROOT = Path(__file__).resolve().parents[1]
for label, fused in [("reference", False), ("fused", True)]:
    dest = ROOT / "build" / label
    dest.mkdir(exist_ok=False)
    shutil.copy2(ROOT / "src/qwen36_fusion/gate_quant.py", dest / "source_snapshot.py")
    result = make_gate_quant(fused).compile(
        config=RunConfig(
            backend_type=backend.BackendType.Ascend910B,
            execution_mode=ExecutionMode.ONBOARD,
            save_kernels=True,
            save_kernels_dir=str(dest),
            dump_passes=True,
        )
    )
    chip = result.chip_callable
    data = ctypes.string_at(chip.buffer_ptr(), chip.buffer_size())
    (dest / "callable.image").write_bytes(data)
    (dest / "image.json").write_text(
        json.dumps(
            dict(
                label=label,
                sha256=hashlib.sha256(data).hexdigest(),
                source="qwen36_fusion.gate_quant.make_gate_quant",
                fused=fused,
            ),
            indent=2,
        )
        + "\n"
    )
    print("BUILT", label, flush=True)
