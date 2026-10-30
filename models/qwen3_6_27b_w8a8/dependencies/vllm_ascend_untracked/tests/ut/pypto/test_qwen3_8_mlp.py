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
"""Bridge contracts; actual NPU replay tests live in the benchmark evidence."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from vllm_ascend.pypto import qwen3_8_mlp as bridge


class MLPBridgeContracts(unittest.TestCase):
    def test_native_tp_reduce_runs_once_after_bf16_local_result(self):
        layer = SimpleNamespace(_pto_mlp_prefix="layer7")
        x = torch.ones(1, bridge.HIDDEN, dtype=torch.bfloat16)

        def kernel(value, out, prefix):
            self.assertEqual(prefix, "layer7")
            self.assertIs(value, x)
            out.copy_(value + 3)

        def reduce(value):
            self.assertEqual(value.dtype, torch.bfloat16)
            return value * 2

        with (
            patch.object(torch.ops.vllm, "qwen38_pto_mlp", side_effect=kernel) as op,
            patch.object(bridge, "tensor_model_parallel_all_reduce", side_effect=reduce) as collective,
        ):
            actual = bridge.mlp_forward(layer, x)
        self.assertTrue(torch.equal(actual, torch.full_like(x, 8)))
        op.assert_called_once()
        collective.assert_called_once()

    def test_unprepared_runner_is_rejected_during_capture(self):
        manager = bridge.Qwen38MLPManager.__new__(bridge.Qwen38MLPManager)
        manager.token = None
        with patch.object(torch.npu, "is_current_stream_capturing", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "before graph capture"):
                manager.prepare()

    def test_out_of_capacity_rows_fail_before_device_dispatch(self):
        manager = bridge.Qwen38MLPManager.__new__(bridge.Qwen38MLPManager)
        manager.max_rows = 512
        manager.prepare = Mock()
        for rows in (0, 513):
            value = torch.empty(rows, bridge.HIDDEN, device="meta")
            with self.assertRaisesRegex(ValueError, "prepared capacity"):
                manager.execute(value, value, "layer0")

    def test_initial_pp2_bridge_configuration_is_rejected(self):
        with patch("vllm.distributed.get_tensor_model_parallel_world_size", return_value=1):
            with self.assertRaisesRegex(ValueError, "TP2/PP1"):
                bridge.attach_mlp(None, None, "/unused")

    def test_another_dense_model_is_rejected_before_allocating_npu_storage(self):
        causal = SimpleNamespace(
            model=SimpleNamespace(
                config=SimpleNamespace(hidden_size=4096, intermediate_size=17408, num_hidden_layers=64)
            )
        )
        with (
            patch("vllm.distributed.get_tensor_model_parallel_world_size", return_value=2),
            patch("vllm.distributed.get_pp_group", return_value=SimpleNamespace(world_size=1)),
        ):
            with self.assertRaisesRegex(ValueError, "dense Qwen"):
                bridge.attach_mlp(causal, None, "/unused")


if __name__ == "__main__":
    unittest.main()
