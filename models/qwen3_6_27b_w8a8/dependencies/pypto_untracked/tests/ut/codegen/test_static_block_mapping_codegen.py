# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Static placement requests must survive function attrs into task launch metadata."""

import pypto.language as pl
from _orchestration_codegen_common import _generate_orch_code
from pypto import backend
from pypto.backend import BackendType


def test_static_mapping_is_opt_in_and_preserves_dynamic_tasks():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.AIC)
        def fixed(
            self, x: pl.Tensor[[16, 16], pl.FP32], out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            pl.func_attr({"static_block_mapping": True})
            return out

        @pl.function(type=pl.FunctionType.AIC)
        def dynamic(
            self, x: pl.Tensor[[16, 16], pl.FP32], out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            x: pl.Tensor[[16, 16], pl.FP32],
            a: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            b: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            a = self.fixed(x, a)
            b = self.dynamic(a, b)
            return b

    code = _generate_orch_code(Program)
    assert code.count("launch_spec.set_static_block_mapping(true)") == 1
    assert "params_t0.launch_spec.set_static_block_mapping(true);" in code
    assert "params_t1.launch_spec.set_static_block_mapping" not in code
    assert code.index("params_t0.launch_spec.set_static_block_mapping(true)") < code.index(
        "rt_submit_aic_task(0"
    )
