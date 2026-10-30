# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Capacity-sized plan table; actual resident grid is selected on the worker."""

from pypto import backend, ir

MAX_CLUSTERS = backend.get_backend_instance(backend.BackendType.Ascend910B).get_core_count(ir.CoreType.CUBE)
if MAX_CLUSTERS <= 0:
    raise ValueError("Backend has no Cube topology")
from .geometry import ProductionConfig

SOFTMAX_JOBS = ProductionConfig().target_rows * ProductionConfig().local["q_heads"]
MAX_NODES = 2 * ProductionConfig().target_rows + 4 * MAX_CLUSTERS + 48 + SOFTMAX_JOBS
PLAN_STRIDE = ((64 + 24 * MAX_CLUSTERS + MAX_NODES * (64 + 4 * 40 + 4 * MAX_NODES) + 4095) // 4096) * 4096
FRAMES_OFFSET = PLAN_STRIDE * MAX_CLUSTERS
FRAMES_BYTES = 3 * MAX_CLUSTERS * 512
EVENTS_OFFSET = FRAMES_OFFSET + FRAMES_BYTES
EVENTS_BYTES = MAX_NODES * 128
TRACE_OFFSET = EVENTS_OFFSET + EVENTS_BYTES
TRACE_BYTES_PER_LAYER = MAX_NODES * 128
STORAGE_BYTES = TRACE_OFFSET + 16 * TRACE_BYTES_PER_LAYER

ATTENTION_STORAGE_BYTES = STORAGE_BYTES
MLP_EVENTS_OFFSET = ATTENTION_STORAGE_BYTES
MLP_EVENT_STRIDE = 24 * 128
STORAGE_BYTES = MLP_EVENTS_OFFSET + 64 * MLP_EVENT_STRIDE
