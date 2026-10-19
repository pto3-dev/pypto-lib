# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Selected graph plan storage and generated-ABI rejection checks."""

import importlib
import struct

import pytest
import torch

from models.qwen3_8_27b.device_program_layout import STORAGE_BYTES
from models.qwen3_8_27b.fixed_mlp_adapter import install_fixed_mlp

_device_program = pytest.importorskip("simpler_setup.device_program")
allocate_program_storage = importlib.import_module(
    "models.qwen3_8_27b.attention_program_adapter"
).allocate_program_storage


def test_plan_storage_has_no_timed_tracing_and_zero_events(tmp_path):
    image = bytearray(64)
    struct.pack_into("<16I", image, 0, *([1] * 16))
    for rank in (0, 1):
        folder = tmp_path / "next_levels" / f"rank{rank}"
        folder.mkdir(parents=True)
        (folder / "attention_program.bin").write_bytes(image)

    class Runtime:
        def alloc_stacked_tensor(self, host):
            return host

    result = allocate_program_storage(Runtime(), tmp_path)
    assert result.shape == (2, STORAGE_BYTES)
    assert result.dtype == torch.uint8
    assert bytes(result[0, :60].tolist()) == image[:60]
    assert not result[:, 60:].any()
    assert bytes(result[0].tolist()) == bytes(result[1].tolist())


def test_fixed_mlp_rejects_changed_site_count(tmp_path):
    root = tmp_path / "next_levels/rank0"
    (root / "orchestration").mkdir(parents=True)
    (root / "kernel_config.py").write_text(
        "KERNELS = ["
        "{'name':'project_mlp_kernel_aic','func_id':1},"
        "{'name':'project_mlp_kernel_aiv','func_id':2}]\n"
    )
    cpp = root / "orchestration/rank0.cpp"
    cpp.write_text("MixedKernels mixed_0 = {1, 2, 2};\nparams_t0.launch_spec.set_block_num(grid);\n")
    original = cpp.read_text()
    with pytest.raises(ValueError, match="Expected four MLP sites"):
        install_fixed_mlp(tmp_path)
    assert cpp.read_text() == original
