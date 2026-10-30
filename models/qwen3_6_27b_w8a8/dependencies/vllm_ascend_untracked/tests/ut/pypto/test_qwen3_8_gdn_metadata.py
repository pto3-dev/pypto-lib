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
"""CPU tests of GDN metadata routing and scheduler-owned storage borrowing."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def executor(monkeypatch):
    root = Path(__file__).resolve().parents[3] / "vllm_ascend/pypto"
    for name in ("qwen3_8_cache", "qwen3_8_gdn"):
        spec = importlib.util.spec_from_file_location(f"vllm_ascend.pypto.{name}", root / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
    instance = module.Qwen38GDNExecutor.__new__(module.Qwen38GDNExecutor)
    instance.runner = SimpleNamespace(run=Mock())
    instance.parameters = {}
    instance.cache_bindings = {}
    instance.fused_core = False
    instance.kernels = {name: name for name in ("conv", "split", "qk_norm", "gates", "recurrent")}
    return instance


def batch(flags=None):
    layer = SimpleNamespace(
        prefix="layer",
        conv1d=SimpleNamespace(weight=torch.zeros(10240, 1, 4)),
        A_log=torch.zeros(48),
        dt_bias=torch.zeros(48),
    )
    metadata = SimpleNamespace(
        spec_sequence_masks=None,
        num_actual_tokens=2,
        non_spec_query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        non_spec_state_indices_tensor=torch.tensor([2, 0], dtype=torch.int32),
        has_initial_state=flags,
    )
    cache = (torch.zeros(3, 3, 10240, dtype=torch.bfloat16), torch.zeros(3, 48, 128, 128))
    inputs = (
        torch.zeros(2, 10240, dtype=torch.bfloat16),
        torch.zeros(2, 48, dtype=torch.bfloat16),
        torch.zeros(2, 48, dtype=torch.bfloat16),
        torch.empty(2, 48, 128, dtype=torch.bfloat16),
    )
    return layer, metadata, cache, inputs


def test_decode_missing_flags_continue_native_states(executor):
    layer, metadata, cache, inputs = batch()
    executor.execute(layer, metadata, cache, *inputs)
    conv = executor.runner.run.call_args_list[0].args
    recurrent = executor.runner.run.call_args_list[-1].args
    assert conv[4] is cache[0]
    assert recurrent[7] is cache[1]
    assert torch.equal(conv[6], torch.tensor([2, 0], dtype=torch.int32))
    assert torch.equal(conv[7], torch.ones(2, dtype=torch.int32))
    assert conv[7] is recurrent[10]


def test_prefill_reset_flags_are_preserved(executor):
    layer, metadata, cache, inputs = batch(torch.tensor([1, 0], dtype=torch.int32))
    executor.execute(layer, metadata, cache, *inputs)
    assert torch.equal(executor.runner.run.call_args_list[0].args[7], metadata.has_initial_state)


def test_speculation_is_rejected_before_dispatch(executor):
    layer, metadata, cache, inputs = batch()
    metadata.spec_sequence_masks = torch.tensor([True])
    with pytest.raises(ValueError, match="speculative"):
        executor.execute(layer, metadata, cache, *inputs)
    executor.runner.run.assert_not_called()


def test_inconsistent_boundaries_are_rejected_before_dispatch(executor):
    layer, metadata, cache, inputs = batch()
    metadata.non_spec_query_start_loc = torch.tensor([0, 2], dtype=torch.int32)
    with pytest.raises(ValueError, match="boundaries"):
        executor.execute(layer, metadata, cache, *inputs)
    executor.runner.run.assert_not_called()
