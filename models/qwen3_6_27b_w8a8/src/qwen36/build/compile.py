# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Direct PyPTO compilation: no source generator, AST or legacy imports."""
import argparse,json
from pathlib import Path
from pypto import backend,ir

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--rank',type=int,required=True,choices=range(4));p.add_argument('--soc',choices=['910b4','910b3'],default='910b4');p.add_argument('--variant',choices=['reference','down_l1k2176','comm_lanes','comm_all_lanes'],default='reference');a=p.parse_args()
    backend.set_backend_type(backend.BackendType.Ascend910B)
    from qwen36.tuning import validate
    resources=validate()
    from qwen36.model import build_program
    program=build_program(a.rank,a.variant)
    target=a.output/'next_levels'/f'rank{a.rank}'
    ir.compile(program,platform='a2a3',output_dir=target,dump_passes=True)
    (a.output/f'direct_rank{a.rank}.json').write_text(json.dumps(dict(rank=a.rank,soc=a.soc,variant=a.variant,source='qwen36.model.build_program',legacy_generator_imported=False,tuning_and_resources=resources,typed_cache_boundary='logical views; optimized backend attachment is a separate explicit step'),indent=2)+'\n')
    print('DIRECT_COMPILED',a.rank,flush=True)
if __name__=='__main__':main()
