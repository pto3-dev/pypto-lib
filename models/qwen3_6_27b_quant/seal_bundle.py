# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Seal normal DSL outputs using measured native cache geometry, no fallback."""
import argparse,ast,hashlib,json,time
from pathlib import Path
from qwen38_runtime.sequence_capacity import validate_capacity_parameters,validate_sealed_capacity
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--geometry-dir",type=Path,required=True)
    a=p.parse_args()
    root=a.root.resolve()
    for name in ("build_contract.json","images.json","measured_geometry_provenance.json"):
        if (root/name).exists():raise FileExistsError(root/name)
    images=[];geometries=[];capacities=[];rank_provenance=[];diagnostic_contracts=[]
    for rank in range(4):
        build=root/"next_levels"/f"rank{rank}"
        report=json.loads((build/"clean_build_provenance.json").read_text())
        diagnostic_contracts.append(report.get('diagnostic_contract'))
        if report["runtime"]!="tensormap_and_ringbuffer" or report["variant"]!="row_ready":
            raise ValueError("Only the unchanged row-ready AICPU baseline is in this migration")
        for forbidden in ("foreign_math_sources","post_compile_source_mutation","manual_registry_or_metadata_mutation","legacy_build_invoked"):
            if report.get(forbidden) is not False:raise ValueError(forbidden)
        for name in ("compiled_meta.json","kernel_config.py","callable.image"):
            if report["generated_artifact_hashes"][name]!=sha(build/name):
                raise ValueError("Generated artifact changed: "+name)
        for item in report["registered_generated_sources"]:
            path=Path(item["source"]).resolve()
            if not path.is_relative_to(build.resolve()) or sha(path)!=item["sha256"]:
                raise ValueError("Foreign or changed generated source")
        gp=a.geometry_dir/f"rank{rank}_dimensions.json"
        geometry=json.loads(gp.read_text())
        params=json.loads((build/"compiled_meta.json").read_text())["params"]
        capacity=validate_capacity_parameters(params,geometry)
        if capacity["max_seq"]!=45056:raise ValueError("Expected measured45056 specialization")
        capacities.append(capacity)
        geometries.append((rank,gp,gp.read_bytes()))
        rank_provenance.append(dict(rank=rank,path=str(gp.resolve()),sha256=sha(gp)))
        images.append(dict(rank=rank,bytes=(build/"callable.image").stat().st_size,sha256=sha(build/"callable.image")))
    if any(c!=capacities[0] for c in capacities):raise ValueError("Ranks have inconsistent capacity")
    if any(c!=diagnostic_contracts[0]for c in diagnostic_contracts):raise ValueError('Ranks have inconsistent diagnostic roles')
    declarations=[node.value.value for node in ast.parse((root/'source_snapshot/qwen36/model.py').read_text()).body
                  if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name)and target.id=='DIAGNOSTIC_HIDDEN_TAP'for target in node.targets)
                  and isinstance(node.value,ast.Constant)]
    diagnostic=diagnostic_contracts[0]
    if diagnostic:
        tap=diagnostic.get('diagnostic_hidden_tap')
        if tap not in ('layer3_qkv_input','layer0_boundaries')or diagnostic!={'diagnostic_only':True,'diagnostic_hidden_tap':tap}or declarations!=[tap]:
            raise ValueError('Diagnostic contract does not match the normally compiled source declaration')
    elif declarations:raise ValueError('Unreported diagnostic source declaration')
    policies=[ast.unparse(node) for node in ast.walk(ast.parse((root/"source_snapshot/qwen36/operators/attention.py").read_text()))
              if isinstance(node,ast.Call) and ast.unparse(node.func)=="pl.set_cache_policy"]
    contract=dict(attention_padding_policy="original_positions_v1",model="Qwen3.6 mixed W8A8",tp=4,rows=24,mtp=5,
                  compile_smoke=False,source="normal whole-program PyPTO DSL compilation",entry_abi="logical_cache_v1",
                  native_gdn_layout="VK",ragged_valid_ends="starts5_8_v1",runtime="tensormap_and_ringbuffer",
                  direct_only=False,communication_protocol="row_ready_lines_v1",communication_max_steps=67108864,
                  static_device_program=False,performance_changes=bool(policies),source_scope_cache_policies=policies,
                  sequence_capacity=capacities[0],device_verified=False,post_compile_subkernel_replacement=False,
                  manual_registry_or_metadata_mutation=False)
    if diagnostic:contract.update(diagnostic)
    for rank,gp,payload in geometries:
        dest=root/f"cache_dimensions_rank{rank}.json"
        if dest.exists():raise FileExistsError(dest)
        dest.write_bytes(payload)
    (root/"images.json").write_text(json.dumps(images,indent=2)+"\n")
    (root/"measured_geometry_provenance.json").write_text(json.dumps(dict(ranks=rank_provenance,created_unix=time.time(),historical_fallback=False),indent=2)+"\n")
    (root/"build_contract.json").write_text(json.dumps(contract,indent=2)+"\n")
    for rank in range(4):validate_sealed_capacity(root,rank,45056)
    print("SEALED_MEASURED_DSL_BUNDLE",root,flush=True)
if __name__=="__main__":main()
