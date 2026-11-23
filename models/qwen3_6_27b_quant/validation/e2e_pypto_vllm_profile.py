# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Frozen serving commands, official bench and separate outer CANN profiling.

The model and framework are not edited. All subprocesses stopped by this module
were created by it. Existing-service commands do not own the server lifetime.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
from urllib.parse import urlsplit
import urllib.request

BASE = os.environ.get('BASE_URL', 'http://127.0.0.1:8253').rstrip('/')
HERE = Path(__file__).resolve().parent


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=10) as response:
        return response.read().decode()


def post(path):
    request = urllib.request.Request(BASE + path, data=b'', method='POST')
    with urllib.request.urlopen(request, timeout=180) as response:
        return response.read().decode()


def metric(text, name):
    return sum(float(line.split()[-1]) for line in text.splitlines()
               if line.startswith(name + '{') or line.startswith(name + ' '))


def required_path(name):
    value = os.environ.get(name)
    if not value:
        raise ValueError(f'{name} must be set explicitly')
    return Path(value).resolve()


def frozen_command(root):
    inline = os.environ.get('SERVER_COMMAND_JSON')
    if inline and os.environ.get('SERVER_COMMAND_FILE'):
        raise ValueError('Select SERVER_COMMAND_JSON or SERVER_COMMAND_FILE, not both')
    record = json.loads(inline) if inline else json.loads(required_path('SERVER_COMMAND_FILE').read_text())
    command = list(record['server_command'] if isinstance(record, dict) else record)
    if not command or not all(isinstance(part, str) for part in command):
        raise ValueError('Frozen server command must be a nonempty string argv list')
    def option(name):
        if command.count(name) != 1:
            raise ValueError(f'Frozen command requires exactly one {name}')
        return command[command.index(name) + 1]
    expected = {'--tensor-parallel-size': '4', '--pipeline-parallel-size': '1',
                '--max-model-len': '45056', '--max-num-seqs': '4',
                '--max-num-batched-tokens': '2048', '--gpu-memory-utilization': '0.65',
                '--kv-cache-memory-bytes': '6442450944', '--mamba-cache-mode': 'align'}
    for name, value in expected.items():
        if option(name) != value:
            raise ValueError(f'Frozen protocol mismatch: {name}; expected {value}')
    for flag in ['--async-scheduling', '--enable-chunked-prefill', '--no-enable-prefix-caching']:
        if flag not in command:
            raise ValueError(f'Missing frozen protocol flag {flag}')
    if json.loads(option('--speculative-config')) != dict(method='mtp', num_speculative_tokens=5, enforce_eager=False):
        raise ValueError('Expected the frozen native MTP5 drafter configuration')
    backend = os.environ.get('BACKEND', 'pto')
    if backend not in ('pto', 'native'):
        raise ValueError('BACKEND must be pto or native')
    graph = json.loads(option('--compilation-config'))
    if (graph.get('mode') != (0 if backend == 'pto' else 3)
            or graph.get('cudagraph_capture_sizes') != [6, 12, 18, 24]
            or graph.get('cudagraph_mode') != 'FULL_DECODE_ONLY'):
        raise ValueError('Target graph configuration differs from the frozen protocol')
    if backend == 'pto':
        if not (required_path('PTO_TARGET_SOURCE') / 'src/qwen38_runtime').is_dir():
            raise FileNotFoundError('Explicit frozen PTO_TARGET_SOURCE is missing its runtime package')
        extra = json.loads(option('--additional-config'))
        target = extra['qwen38_pto_mpmd']
        if target.get('execution') != 'direct' or target.get('graph') is not False:
            raise ValueError('Expected AICPU prepared DIRECT target, not HBG or target ACL graph')
        if target.get('same_state_probe_runs') or target.get('all_step_precision'):
            raise ValueError('A mathematical observer cannot be part of a formal benchmark')
        if not Path(target['build_root']).is_dir() or not Path(target['runtime_dir']).is_dir():
            raise FileNotFoundError('Frozen PTO build/runtime directory is missing')
        target['session_dir'] = str(root / 'runtime')
        command[command.index('--additional-config') + 1] = json.dumps(extra)
    model = str(required_path('MODEL'))
    if command[command.index('serve') + 1] != model:
        raise ValueError('MODEL differs from the frozen checkpoint; do not silently switch weights')
    url = urlsplit(BASE)
    if url.scheme != 'http' or url.hostname not in ('127.0.0.1', 'localhost') or url.port is None:
        raise ValueError('Owned service BASE_URL must be an explicit local http host:port')
    for name, value in [('--host', url.hostname), ('--port', str(url.port))]:
        option(name)
        command[command.index(name) + 1] = value
    if '--profiler-config' in command:
        index = command.index('--profiler-config')
        del command[index:index + 2]
    profiler = dict(profiler='torch', torch_profiler_dir=str(root / 'profiler'),
                    torch_profiler_with_stack=False, torch_profiler_with_memory=False,
                    ignore_frontend=True, max_iterations=int(os.environ.get('PROFILE_STEPS', '50')))
    command += ['--profiler-config', json.dumps(profiler)]
    return command


def require_unused_port():
    url = urlsplit(BASE)
    with socket.socket() as sock:
        sock.bind((url.hostname, url.port))


def idle_cards():
    cards = os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '').split(',')
    if len(cards) != 4 or len(set(cards)) != 4 or not all(card.isdecimal() for card in cards):
        raise ValueError('Explicitly select four distinct idle ASCEND_RT_VISIBLE_DEVICES')
    result = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True, timeout=20, check=True)
    if 'Process id' not in result.stdout:
        raise ValueError('Unknown npu-smi process table; cannot confirm idle cards')
    rows = result.stdout.split('Process id', 1)[1]
    owners = re.findall(r'\|\s*(\d+)\s+(\d+)\s*\|\s*(\d+)\s*\|', rows)
    if any(card in cards for card, _, _ in owners):
        raise RuntimeError('Selected cards have device processes; choose another idle group')
    return cards


def check_inputs():
    from transformers import AutoTokenizer
    model, corpus = required_path('MODEL'), required_path('CORPUS')
    tokenizer = AutoTokenizer.from_pretrained(str(model), trust_remote_code=True)
    digest = hashlib.sha256(corpus.read_bytes()).hexdigest()
    manifest = corpus.parent / 'manifest.json'
    if not manifest.is_file() or json.loads(manifest.read_text())['files'].get(corpus.name) != digest:
        raise ValueError('Frozen corpus manifest missing or hash mismatch')
    rows = [json.loads(line) for line in corpus.read_text().splitlines()]
    token_ids = [tokenizer.encode(row['prompt'], add_special_tokens=False) for row in rows]
    if len(rows) != 4 or any(len(ids) != 32768 for ids in token_ids):
        raise ValueError('Expected four distinct corpus entries of exactly32768 tokens')
    return dict(checkpoint=str(model), corpus=str(corpus), input_token_ids=token_ids,
                corpus_sha256=digest, tokenizer_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in model.iterdir() if p.is_file() and ('token' in p.name or p.name == 'config.json')})


def benchmark(root, length):
    root.mkdir(parents=True, exist_ok=False)
    cmd = [sys.executable, '-m', 'vllm.entrypoints.cli.main', 'bench', 'serve',
           '--backend', 'vllm', '--base-url', BASE, '--endpoint', '/v1/completions',
           '--model', str(required_path('MODEL')), '--tokenizer', str(required_path('MODEL')),
           '--served-model-name', 'qwen36-mixed', '--dataset-name', 'custom',
           '--dataset-path', str(required_path('CORPUS')), '--skip-chat-template', '--disable-shuffle',
           '--max-concurrency', '4', '--num-prompts', '4', '--temperature', '0', '--seed', '0',
           '--ignore-eos', '--output-len', str(length), '--extra-body', json.dumps(dict(seed=0, return_token_ids=True)),
           '--save-result', '--save-detailed', '--result-dir', str(root), '--result-filename', 'result.json',
           '--num-warmups', '0', '--ready-check-timeout-sec', '0',
           '--percentile-metrics', 'ttft,tpot,itl,e2el', '--metric-percentiles', '50,90,99']
    save(root / 'command.json', cmd)
    env = os.environ.copy()
    env['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
    with (root / 'bench.log').open('x') as stream:
        return subprocess.Popen(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)


def stop_owned(process):
    if process is None:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            continue
        # Workers can outlive their API parent; the process group is still ours.
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        if sig == signal.SIGTERM:
            time.sleep(.2)
    process.wait(timeout=15)


def wait_benchmark(process, root, length, timeout=2400):
    try:
        if process.wait(timeout=timeout):
            raise RuntimeError(f'Official bench failed: {root / "bench.log"}')
        result = json.loads((root / 'result.json').read_text())
        if (result.get('completed') != 4 or result.get('failed') != 0
                or result.get('input_lens') != [32768] * 4 or result.get('output_lens') != [length] * 4):
            raise ValueError('Official bench did not complete the exact requested workload')
        return result
    finally:
        stop_owned(process)


def capture(root, raw):
    """Follow the GDD full-warmup/full-request capture lifecycle."""
    root.mkdir(parents=True, exist_ok=False)
    if raw.exists():
        raise FileExistsError(f'Archive previous profiler data before capture: {raw}')
    get('/health')
    before = get('/metrics')
    if metric(before, 'vllm:num_requests_running') or metric(before, 'vllm:num_requests_waiting'):
        raise ValueError('Diagnostic capture requires an idle service')
    # GDD capture warms all four requests through the full10240 output tokens.
    warmup = benchmark(root / 'warmup', 10240)
    wait_benchmark(warmup, root / 'warmup', 10240, timeout=3600)
    before = get('/metrics')
    (root / 'metrics_before.prom').write_text(before)
    gen0, prompt0 = (metric(before, 'vllm:' + key) for key in ('generation_tokens_total', 'prompt_tokens_total'))
    client = benchmark(root / 'benchmark', 10240)
    active = False
    try:
        deadline = time.monotonic() + 3600
        while time.monotonic() < deadline:
            if client.poll() is not None:
                raise RuntimeError('Diagnostic client ended before a complete B4 capture window')
            text = get('/metrics')
            if (metric(text, 'vllm:num_requests_running') == 4 and not metric(text, 'vllm:num_requests_waiting')
                    and metric(text, 'vllm:generation_tokens_total') - gen0 >= 256
                    and metric(text, 'vllm:prompt_tokens_total') - prompt0 == 4 * 32768):
                break
            time.sleep(1)
        else:
            raise TimeoutError('No all-prefills-complete B4 window3600s')
        (root / 'metrics_capture_start.prom').write_text(text)
        post('/start_profile')
        active = True
        # max_iterations bounds collection, not the HTTP request's output length.
        # Complete the request instead of cancelling it after a timed sample.
        wait_benchmark(client, root / 'benchmark', 10240, timeout=3600)
    finally:
        try:
            if active:
                post('/stop_profile')
        finally:
            stop_owned(client)
    (root / 'metrics_capture_end.prom').write_text(get('/metrics'))
    if not raw.exists():
        raise FileNotFoundError(f'No profiler data: {raw}')
    destination = root / 'profiler'
    raw.rename(destination)
    save(root / 'COMPLETION.json', dict(diagnostic_only=True, formal_unprofiled_result=False,
        full_10k_completed=True, warmup_output_tokens=10240, requested_output_tokens=10240,
        capture_reference='GDD a62c3c176cc173e2ec84f35054a2e7bb403bc24d',
        diagnostic_request_cancelled_after_profile=False))
    return destination


def analyse(raw, output):
    dirs = sorted(p for p in raw.rglob('*_ascend_pt') if p.is_dir())
    if len(dirs) != 4:
        raise ValueError(f'Expected exactly four rank profiler directories; found {len(dirs)}')
    def export(directory):
        subprocess.run([sys.executable, '-c',
                        'import sys; from torch_npu.profiler.profiler import analyse; analyse(sys.argv[1])',
                        str(directory)], check=True, timeout=600)
        out = directory / 'ASCEND_PROFILER_OUTPUT'
        for name in ['trace_view.json', 'kernel_details.csv', 'op_statistic.csv']:
            if not (out / name).is_file():
                raise FileNotFoundError(out / name)
        with (out / 'kernel_details.csv').open(newline='') as stream:
            count = sum(1 for _ in csv.DictReader(stream))
        if not count:
            raise ValueError(f'Empty device kernel table: {out}')
        return dict(output=str(out), kernel_rows=count)
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(export, dirs))
    with output.open('x') as stream:
        json.dump(rows, stream, indent=2)
        stream.write('\n')
    return rows


def qa(root):
    name = 'long_context_qa.py'
    spec = importlib.util.spec_from_file_location('qwen_qa_client', HERE / name)
    client = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(client)
    kwargs = dict(checkpoint=required_path('MODEL'), corpus=required_path('CORPUS'))
    result = client.run(BASE, root, **kwargs)
    if not result['all_passed']:
        raise RuntimeError('QA field/finish validation failed; preserve qa-result.json')
    return result


def main():
    if sys.argv[1:2] == ['timeline']:
        subprocess.run([sys.executable, str(HERE / 'analyze_outer_trace.py'), *sys.argv[2:]], check=True)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('env', 'check', 'serve', 'qa', 'bench', 'capture', 'analyse'))
    parser.add_argument('--raw', type=Path, help='Explicit existing raw profiler directory for analyse/capture')
    args = parser.parse_args()
    root = required_path('RUN_DIR')
    length = int(os.environ.get('OUTPUT_LEN', '10240'))
    if not 2 <= length <= 12288:
        raise ValueError('OUTPUT_LEN must fit32768 input within45056 context')
    if args.command == 'env':
        print(json.dumps(dict(python=sys.executable, frozen_server_command=frozen_command(root),
                              base_url=BASE, backend=os.environ.get('BACKEND', 'pto')), indent=2))
        return
    if args.command == 'check':
        print(json.dumps(dict(input_gate=check_inputs(), idle_cards=idle_cards(), server_command=frozen_command(root))))
        return
    if args.command == 'serve':
        cards = idle_cards()
        command = frozen_command(root)
        require_unused_port()
        root.mkdir(parents=True, exist_ok=False)
        save(root / 'server_command.json', dict(command=command, cards=cards, pid=os.getpid()))
        if os.environ.get('SERVICE_LATEST_LINK'):
            # Do not resolve an existing symlink: it identifies the pointer to update.
            latest = Path(os.environ['SERVICE_LATEST_LINK']).absolute()
            latest.parent.mkdir(parents=True, exist_ok=True)
            temporary = latest.with_name(latest.name + f'.{os.getpid()}.tmp')
            temporary.symlink_to(root, target_is_directory=True)
            os.replace(temporary, latest)
        os.execvpe(command[0], command, os.environ.copy())
    if args.command == 'qa':
        qa(root)
        return
    if args.command == 'bench':
        gate = check_inputs()
        root.mkdir(parents=True, exist_ok=False)
        save(root / 'INPUT_GATE.json', gate)
        wait_benchmark(benchmark(root / 'benchmark', length), root / 'benchmark', length)
        return
    if args.command == 'analyse':
        if args.raw is None:
            parser.error('analyse requires --raw')
        root.mkdir(parents=True, exist_ok=False)
        analyse(args.raw, root / 'profiler-outputs.json')
        return
    if args.command == 'capture':
        if args.raw is None:
            parser.error('capture requires --raw from the existing service profiler configuration')
        raw = capture(root, args.raw)
        analyse(raw, root / 'profiler-outputs.json')
        return


if __name__ == '__main__':
    main()
