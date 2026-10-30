# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Keep explicit candidate runtime sources and ABI registration consistent."""

from pathlib import Path
import json, shutil

N = Path(__file__).resolve().parents[1]
B = N / "build/target_gate_quant_fused"


def main():
    for rank in range(4):
        r = B / f"next_levels/rank{rank}"
        nb = r / "native_backend"
        shutil.copy2(N / f"src/runtime/orchestration_rank{rank}.cpp", nb / "orchestration.cpp")
        ref = nb / "inherited_metadata_reference"
        ref.mkdir()
        for name in ["attention_plan.py", "attention_plan.json"]:
            if not (ref / name).exists():
                shutil.move(nb / name, ref / name)
        (nb / "attention_plan.py").write_text(
            '''"""Rebuild this candidate from the immutable parent plan with explicit fusion."""
    import importlib.util
    from pathlib import Path
    TOOL = Path("'''
            + str(N / "tools/build_fused_target.py")
            + """")
    spec = importlib.util.spec_from_file_location("fusion_builder", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def build(grid):
        parent = module.B / "next_levels" / "rank"""
            + str(rank)
            + """" / f"production_attention_grid{grid}.bin"
        return module.plan(parent.read_bytes())[0]
    """
        )
        specpath = nb / "link_manifest.json"
        spec = json.loads(specpath.read_text())
        for k in spec["kernels"]:
            if k["func_id"] == 30:
                k.update(
                    name="fused_attention_gate_quant",
                    signature=["INOUT", "IN", "IN", "INOUT", "INOUT"],
                    native_source=str(N / "build/fused/kernels/aiv/fused_row.cpp"),
                )
            if k["func_id"] in (39, 40):
                if len(k["signature"]) == 29:
                    k["signature"] = k["signature"][:-1] + ["INOUT", "INOUT", "SCALAR"]
                else:
                    assert len(k["signature"]) == 31
        specpath.write_text(json.dumps(spec, indent=2) + "\n")
        contract = json.loads((B / "build_contract.json").read_text())
        contract["attention_end_fusion"] = (
            "per-token reduce28+BF16 gate+dynamic INT8 quant; standalone quant1536 removed"
        )
        (B / "build_contract.json").write_text(json.dumps(contract, indent=2) + "\n")


if __name__ == "__main__":
    main()
