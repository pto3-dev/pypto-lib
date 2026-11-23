# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run official vLLM bench serve against an existing service; keep raw results."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def command(args, output):
    cmd = [sys.executable, '-m', 'vllm.entrypoints.cli.main', 'bench', 'serve',
           '--backend', 'vllm', '--base-url', args.base_url, '--endpoint', '/v1/completions',
           '--tokenizer', str(args.tokenizer), '--model', args.model,
           '--max-concurrency', '4', '--num-prompts', '4', '--temperature', '0',
           '--seed', str(args.seed), '--ignore-eos', '--ready-check-timeout-sec', '0',
           '--save-result', '--save-detailed', '--result-dir', str(output),
           '--result-filename', 'result.json']
    if args.corpus is not None:
        cmd += ['--dataset-name', 'custom', '--dataset-path', str(args.corpus),
                '--skip-chat-template', '--disable-shuffle', '--output-len', '10240']
    else:
        cmd += ['--dataset-name', 'random', '--random-input-len', '32768',
                '--random-output-len', '10240', '--random-range-ratio', '0', '--random-prefix-len', '0']
    return cmd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--corpus', type=Path, help='Existing official custom JSONL; otherwise official random')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.corpus is not None and not args.corpus.is_file():
        parser.error('--corpus must be an existing file')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cmd = command(args, output)
    metadata = dict(command=cmd, client_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    server_settings_changed=False, requests=4, input_tokens=32768, output_tokens=10240,
                    corpus_sha256=hashlib.sha256(args.corpus.read_bytes()).hexdigest() if args.corpus else None,
                    limitation='Official ITL is a stream-message interval for MTP; no per-token timestamps invented.')
    (output / 'command.json').write_text(json.dumps(metadata, indent=2) + '\n')
    env = os.environ.copy()
    env['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
    with (output / 'bench.log').open('w') as stream:
        completed = subprocess.run(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT, check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)
    result = json.loads((output / 'result.json').read_text())
    if (result.get('completed') != 4 or result.get('failed') != 0
            or result.get('input_lens') != [32768] * 4 or result.get('output_lens') != [10240] * 4):
        raise RuntimeError('Incomplete production workload; inspect unmodified official result.json')
    print(output / 'result.json')


if __name__ == '__main__':
    main()
