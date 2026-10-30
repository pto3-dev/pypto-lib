# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Explicit typed-cache/static-program ABI; never patch generated C++ or AST.

38 leaves are compiled from the direct PyPTO modules. Existing native softmax
and the two device-plan entry kernels remain explicit source files. The fixed
native dispatcher is guarded against changes in frontend graph/argument order.
"""
import argparse,ctypes,hashlib,json,re,shutil
from pathlib import Path
NATIVE=Path(__file__).resolve().parents[1]/'native_backend'
W=Path(__file__).resolve().parents[3]

def normalized_hash(path):
    text=re.sub(r'//[^\n]*|/\*.*?\*/','',path.read_text(),flags=re.S)
    return hashlib.sha256(re.sub(r'\s+','',text).encode()).hexdigest()

def configuration(root):
    from simpler.task_interface import ArgDirection as D
    meta=json.loads((root/'direct_link.json').read_text());rank=meta['rank'];native_sources=Path(meta.get('native_sources',str(NATIVE)))
    spec=json.loads((NATIVE/'link_manifest.json').read_text())
    kernels=[]
    for k in spec['kernels']:
        name=k['name'].format(rank=rank)
        source=native_sources/k['native_source']if k['native_source']else root/'kernels'/k['core_type']/(name+'.cpp')
        if meta.get('device_poll_ticks')==10 and k.get('native_source')in('device_program_aic.cpp','device_program_aiv.cpp'):
            source=native_sources/f'device_program_poll10_{k["core_type"]}.cpp'
        item=dict(func_id=k['func_id'],name=name,source=str(source),core_type=k['core_type'],signature=[getattr(D,s)for s in k['signature']])
        if k.get('extra_include_dirs'):item['extra_include_dirs']=k['extra_include_dirs']
        kernels.append(item)
    return dict(RUNTIME_CONFIG={'runtime':'tensormap_and_ringbuffer','aicpu_thread_num':0},ORCHESTRATION=dict(source=str(native_sources/'orchestration.cpp'),function_name='aicpu_orchestration_entry',signature=[getattr(D,s)for s in spec['orchestration_signature']]),KERNELS=kernels)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--device-poll-ticks',type=int,choices=[0,10],default=0);a=p.parse_args()
    from pypto import backend,ir
    backend.set_backend_type(backend.BackendType.Ascend910B)
    spec=json.loads((NATIVE/'link_manifest.json').read_text());images=[]
    variants={json.loads((a.root/f'direct_rank{rank}.json').read_text()).get('variant','reference')for rank in range(4)}
    if len(variants)!=1:raise ValueError('Mixed rank tuning variants')
    variant=next(iter(variants))
    for rank in range(4):
        root=a.root/'next_levels'/f'rank{rank}'
        if(root/'direct_link.json').exists():raise ValueError('Already linked; use a fresh build')
        raw=root/'orchestration'/f'rank{rank}.cpp'
        if normalized_hash(raw)!=spec['expected_raw_orchestration_sha256']:raise ValueError('Frontend graph changed; update the explicitly owned dispatcher, not a hidden fallback')
        if normalized_hash(root/'kernels/aiv/tiled_softmax.cpp')!=spec['expected_raw_softmax_sha256']:raise ValueError('Softmax reference changed; native optimized implementation must be reviewed')
        if json.loads((root/'compiled_meta.json').read_text())!=json.loads((NATIVE/'logical_meta.json').read_text()):raise ValueError('Logical ABI changed')
        shutil.copy2(root/'compiled_meta.json',root/'logical_compiled_meta.json');shutil.copy2(NATIVE/'physical_meta.json',root/'compiled_meta.json')
        shutil.move(root/'kernel_config.py',root/'kernel_config.generated.py')
        native_build_sources=root/'native_backend'
        shutil.copytree(NATIVE,native_build_sources,ignore=shutil.ignore_patterns('*.o','*.so','__pycache__','*.pyc'))
        (root/'direct_link.json').write_text(json.dumps(dict(rank=rank,native_sources=str(native_build_sources),graph_guard_passed=True,device_poll_ticks=a.device_poll_ticks),indent=2)+'\n')
        shutil.copy2(Path(__file__).with_name('kernel_config_entry.py'),root/'kernel_config.py')
        capacity=backend.get_backend_instance(backend.BackendType.Ascend910B).get_core_count(ir.CoreType.CUBE)
        from qwen36.native_backend.attention_plan import build as build_plan
        for grid in range(1,capacity+1):(root/f'production_attention_grid{grid}.bin').write_bytes(build_plan(grid))
        shutil.copy2(W/'reference'/f'cache_dimensions_rank{rank}.json',a.root/f'cache_dimensions_rank{rank}.json')
        chip=ir.CompiledProgram.from_dir(root,platform='a2a3').chip_callable
        data=ctypes.string_at(chip.buffer_ptr(),chip.buffer_size());(root/'callable.image').write_bytes(data)
        images.append(dict(rank=rank,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()));print('DIRECT_IMAGE',images[-1],flush=True)
    (a.root/'images.json').write_text(json.dumps(images,indent=2)+'\n')
    (a.root/'build_contract.json').write_text(json.dumps(dict(model='Qwen3.6 mixed W8A8',tp=4,rows=24,mtp=5,compile_smoke=False,source='direct qwen36 modules',native_backend='explicit typed-cache/static-plan dispatcher',variant=variant,communication_protocol='lane_frontier_all_v1' if variant=='comm_all_lanes' else ('lane_frontier_v1' if variant=='comm_lanes' else 'group_counter_v1'),communication_max_steps=16000000 if variant in ('comm_lanes','comm_all_lanes') else 67108864,device_poll_ticks=a.device_poll_ticks,performance_changes=variant!='reference' or a.device_poll_ticks!=0),indent=2)+'\n')
if __name__=='__main__':main()
