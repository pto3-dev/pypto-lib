# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compile the TP4 mixed-W8A8 target with normal PyPTO AICPU lowering."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler-source', type=Path, required=True)
    parser.add_argument('--geometry-dir', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--rank', type=int, choices=range(4))
    parser.add_argument('--soc', choices=['910b4', '910b3'], default='910b4')
    args = parser.parse_args()
    root = args.output_root.resolve()
    source = Path(__file__).resolve().parent
    if args.rank is None:
        for rank in range(4):
            subprocess.run([sys.executable, str(Path(__file__).resolve()),
                            '--compiler-source', str(args.compiler_source),
                            '--geometry-dir', str(args.geometry_dir),
                            '--output-root', str(root), '--soc', args.soc,
                            '--rank', str(rank)], check=True)
        return
    geometry = args.geometry_dir.resolve()/f'rank{args.rank}_dimensions.json'
    if not geometry.is_file():
        raise FileNotFoundError(geometry)
    os.environ['QWEN36_NATIVE_GEOMETRY'] = str(geometry)
    sys.path.insert(0, str(source))
    import pypto
    from pypto import backend, ir, passes
    import pypto.pypto_core as compiler_core
    compiler_source = args.compiler_source.resolve()
    if not Path(pypto.__file__).resolve().is_relative_to(compiler_source):
        raise ImportError('Imported PyPTO is outside --compiler-source')
    if not Path(compiler_core.__file__).resolve().is_relative_to(compiler_source):
        raise ImportError('Compiler extension is outside --compiler-source')
    backend.set_backend_type(backend.BackendType.Ascend910B)
    from qwen36.model import build_program
    from qwen36.tuning import validate
    resources = validate()
    hashes = {str(p.relative_to(source)):sha(p) for p in (source/'qwen36').rglob('*.py')}
    snapshot = root/'source_snapshot/qwen36'
    if not snapshot.exists():
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source/'qwen36', snapshot, ignore=shutil.ignore_patterns('__pycache__'))
    if hashes != {str(p.relative_to(root/'source_snapshot')):sha(p) for p in snapshot.rglob('*.py')}:
        raise ValueError('Build snapshot does not match the current DSL')
    destination = root/'next_levels'/f'rank{args.rank}'
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    program = build_program(args.rank, variant='row_ready', direct_only=False)
    started = time.monotonic()
    compiled = ir.compile(program, backend_type=backend.BackendType.Ascend910B,
                          platform='a2a3', output_dir=str(destination), dump_passes=True,
                          runtime=passes.RuntimeKind.TENSORMAP_AND_RINGBUFFER)
    callable_ = compiled.chip_callable
    (destination/'callable.image').write_bytes(ctypes.string_at(callable_.buffer_ptr(), callable_.buffer_size()))
    config = runpy.run_path(str(destination/'kernel_config.py'))
    generated = []
    for entry in [config['ORCHESTRATION'], *config['KERNELS']]:
        path = Path(entry['source']).resolve()
        if not path.is_relative_to(destination):
            raise ValueError('Compiler registered a source outside its output directory')
        generated.append(dict(name=entry.get('name', 'orchestration'), source=str(path), sha256=sha(path)))
    if hashes != {str(p.relative_to(source)):sha(p) for p in (source/'qwen36').rglob('*.py')}:
        raise ValueError('DSL changed during compilation')
    artifacts = {str(p.relative_to(destination)):sha(p) for p in destination.rglob('*')
                 if p.is_file() and 'cache' not in p.parts}
    provenance = dict(variant='row_ready', runtime='tensormap_and_ringbuffer', rank=args.rank,
                      soc=args.soc, source_hashes=hashes, hardware_capacity=resources,
                      registered_generated_sources=generated, generated_artifact_hashes=artifacts,
                      elapsed_seconds=time.monotonic()-started, foreign_math_sources=False,
                      post_compile_source_mutation=False, manual_registry_or_metadata_mutation=False,
                      legacy_build_invoked=False, status='compiled_not_device_verified')
    (destination/'clean_build_provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print('NORMAL_AICPU_TARGET_COMPILED', destination, flush=True)


if __name__ == '__main__':
    main()
