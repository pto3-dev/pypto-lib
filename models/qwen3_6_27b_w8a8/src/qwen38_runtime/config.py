# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Backend, execution policy and checkpoint provenance are separate contracts."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class Backend(str, Enum):
    NATIVE = "native"
    PYPTO = "pypto"


class Execution(str, Enum):
    DIRECT = "direct"
    FULL = "full"


@dataclass(frozen=True)
class RuntimeSettings:
    backend: Backend = Backend.PYPTO
    execution: Execution = Execution.DIRECT
    unsupported_decode: str = "error"
    model_target: str = "Qwen3.8-27B"
    checkpoint_provenance: str = "current mixed W8A8 checkpoint"
    runtime_dir: Path = Path(__file__).resolve().parents[2] / "lib/a2a3"
    weight_strategy: str = "native_nz"
    slots: int = 1

    def __post_init__(self):
        if self.unsupported_decode not in ("error", "native"):
            raise ValueError("unsupported_decode must be error or native")
        if self.weight_strategy not in (
            "native_nz",
            "batched_nd_to_nz",
            "direct_cpu_nz",
            "preloaded_nd_to_nz",
            "device_preloaded_nd_to_nz",
        ):
            raise ValueError("Unsupported weight strategy")
        if self.slots != 1:
            raise ValueError("This ABI owns one mutable workspace; multiple slots require separate bindings")

    @classmethod
    def from_mapping(cls, values):
        if values.get("phase_nz"):
            raise ValueError("Stage ND/NZ switching is not supported by the shared-bank runtime")
        execution = values.get("execution")
        if execution is None:
            execution = "full" if values.get("graph", False) else "direct"
        return cls(
            backend=Backend(values.get("backend", "pypto")),
            execution=Execution(execution),
            unsupported_decode=values.get("unsupported_decode", "error"),
            model_target=values.get("model_target", "Qwen3.8-27B"),
            checkpoint_provenance=values.get("checkpoint_provenance", "current mixed W8A8 checkpoint"),
            runtime_dir=Path(values.get("runtime_dir", cls.__dataclass_fields__["runtime_dir"].default)),
            weight_strategy=values.get("weight_strategy", "native_nz"),
            slots=values.get("slots", 1),
        )
