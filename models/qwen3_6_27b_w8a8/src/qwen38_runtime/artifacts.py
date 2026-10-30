# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Read-only integration boundary with the model/kernel workstream."""

from dataclasses import dataclass
from pathlib import Path
import hashlib, json


@dataclass(frozen=True)
class ArtifactBundle:
    root: Path
    rank: int
    contract: dict
    geometry: dict
    params: tuple
    image: bytes
    hashes: dict
    plans: tuple = ()

    @classmethod
    def load(cls, root, rank):
        root = Path(root).resolve()
        if (root / "REJECTED_DO_NOT_RUN.json").exists():
            raise ValueError("Rejected artifact: " + str(root / "REJECTED_DO_NOT_RUN.json"))
        build = root / "next_levels" / f"rank{rank}"
        contract = json.loads((root / "build_contract.json").read_text())
        if contract.get("compile_smoke"):
            raise ValueError("Smoke artifacts cannot run production requests")
        for field, value in [("tp", 4), ("rows", 24), ("mtp", 5)]:
            if contract.get(field) != value:
                raise ValueError(f"Unsupported artifact {field}")
        geometry = json.loads((root / f"cache_dimensions_rank{rank}.json").read_text())
        meta_bytes = (build / "compiled_meta.json").read_bytes()
        meta = json.loads(meta_bytes)
        params = tuple(meta["params"])
        names = [p["name"] for p in params]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate parameter name")
        for name in ["control_i64", "control_i32", "native_metadata", "global_logits", "hidden_result"]:
            if name not in names:
                raise ValueError("Missing target ABI parameter: " + name)
        image = (build / "callable.image").read_bytes()
        hashes = {
            "callable.image": hashlib.sha256(image).hexdigest(),
            "compiled_meta.json": hashlib.sha256(meta_bytes).hexdigest(),
        }
        plans = []
        if "device_program_storage" in names:
            from .device_plan_layout import MAX_CLUSTERS

            for grid in range(1, MAX_CLUSTERS + 1):
                name = f"production_attention_grid{grid}.bin"
                data = (build / name).read_bytes()
                plans.append(data)
                hashes[name] = hashlib.sha256(data).hexdigest()
        return cls(root, rank, contract, geometry, params, image, hashes, tuple(plans))

    def validate_geometry(self, actual):
        if actual != self.geometry:
            raise ValueError("Native cache geometry differs from artifact ABI")
