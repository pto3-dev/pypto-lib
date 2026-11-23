# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run a benchmark or a complete outer trace from one saved service session."""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.parse import urlsplit

# The validated environment enables PYTHONSAFEPATH; load this sibling explicitly.
spec = importlib.util.spec_from_file_location(
    'qwen_serving_profile', Path(__file__).with_name('e2e_pypto_vllm_profile.py'))
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


def service_session():
    latest = Path(os.environ['SERVICE_LATEST_LINK'])
    if not latest.is_symlink():
        raise RuntimeError(f'Start the service first: bash serve_vllm_910b4.sh {os.environ["BACKEND"]}')
    session = latest.resolve(strict=True)
    record = json.loads((session / 'server_command.json').read_text())
    command = record['command']
    if record.get('pid') is None:
        raise RuntimeError('Missing service PID; restart using serve_vllm_910b4.sh')
    def option(name):
        return command[command.index(name) + 1]
    url = urlsplit(profile.BASE)
    if option('--host') != url.hostname or option('--port') != str(url.port):
        raise ValueError('Service endpoint differs from serving.local.sh')
    if command[0] != os.environ['VLLM_LAUNCHER']:
        raise ValueError('Selected backend environment differs from the recorded service')
    os.environ.pop('SERVER_COMMAND_FILE', None)
    os.environ['SERVER_COMMAND_JSON'] = json.dumps(command)
    # Reuse all frozen protocol and backend checks; do not rebuild framework imports.
    profile.frozen_command(session)
    deadline = time.monotonic() + 1200
    while time.monotonic() < deadline:
        try:
            os.kill(int(record['pid']), 0)
        except ProcessLookupError as exc:
            raise RuntimeError(f'Service exited; see {session / "server.log"} or its startup terminal') from exc
        try:
            profile.get('/health')
            break
        except OSError:
            time.sleep(2)
    else:
        raise TimeoutError('Service did not become ready in1200s')
    metrics = profile.get('/metrics')
    if (profile.metric(metrics, 'vllm:num_requests_running')
            or profile.metric(metrics, 'vllm:num_requests_waiting')):
        raise RuntimeError('This service has active requests; retry after they finish')
    return session, command


def rank_csvs(outputs):
    """Identify global rank from profiler metadata, never PID or directory order."""
    ranks = {}
    for row in outputs:
        directory = Path(row['output'])
        markers = list(directory.parent.glob('profiler_info_*.json'))
        if len(markers) != 1:
            raise ValueError(f'Expected one rank metadata file: {directory.parent}')
        match = re.fullmatch(r'profiler_info_([0-3])\.json', markers[0].name)
        if match is None or int(match[1]) in ranks:
            raise ValueError(f'Unknown or duplicate profiler rank: {markers[0]}')
        rank = int(match[1])
        csv = directory / 'kernel_details.csv'
        if not csv.is_file() or not (directory / 'trace_view.json').is_file():
            raise FileNotFoundError(f'Incomplete exported device timeline: {directory}')
        ranks[rank] = csv
    if set(ranks) != {0, 1, 2, 3}:
        raise ValueError('Export does not contain exactly rank0 through rank3')
    return [ranks[rank] for rank in range(4)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('qa', 'bench', 'profile'))
    args = parser.parse_args()
    backend = os.environ['BACKEND']
    results = Path(os.environ['RESULTS_ROOT']).resolve() / backend
    results.mkdir(parents=True, exist_ok=True)
    # Serialize our clients only; never stop a server or another user's requests.
    with (results / 'client.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another benchmark/capture for this backend is running') from exc
        session, command = service_session()
        stamp = time.strftime('%Y%m%d_%H%M%S') + f'_{os.getpid()}'
        root = results / f'{args.command}_{os.environ["SCENE"]}_{stamp}'
        os.environ['RUN_DIR'] = str(root)
        if args.command == 'qa':
            profile.qa(root)
            print(f'Long-context QA: {root / "qa-result.json"}')
        elif args.command == 'bench':
            sys.argv = [profile.__file__, 'bench']
            profile.main()
            print(f'Official benchmark: {root / "benchmark/result.json"}')
            print(f'Input IDs and hashes: {root / "INPUT_GATE.json"}')
        else:
            config = json.loads(command[command.index('--profiler-config') + 1])
            raw = Path(config['torch_profiler_dir']).resolve()
            if raw != session / 'profiler':
                raise ValueError('Profiler directory is outside the recorded service session')
            gate = profile.check_inputs()
            # capture includes offline export; it is separate from formal benchmarking.
            sys.argv = [profile.__file__, 'capture', '--raw', str(raw)]
            profile.main()
            profile.save(root / 'INPUT_GATE.json', gate)
            outputs = json.loads((root / 'profiler-outputs.json').read_text())
            csvs = rank_csvs(outputs)
            profile.save(root / 'rank-files.json', {str(rank): str(csv) for rank, csv in enumerate(csvs)})
            timeline = root / 'timeline.json'
            subprocess.run([sys.executable, str(profile.HERE / 'analyze_outer_trace.py'),
                            '--backend', backend, '--rank-csv', *map(str, csvs),
                            '--output', str(timeline)], check=True)
            data = json.loads(timeline.read_text())
            lines = ['| Boundary | P50 (ms) | Mean (ms) |', '|---|---:|---:|']
            for label, key in [('Verify (full target)', 'matched_full_verify'),
                               ('Verify (GDD marker adapter)', 'repository_verify'),
                               ('Decode (GDD rejection start)', 'repository_decode'),
                               ('MTP start interval (additional)', 'mtp_start_alternate')]:
                values = data['summary'][key]
                lines.append(f'| {label} | {values["p50_ms"]:.6f} | {values["mean_ms"]:.6f} |')
            report = '\n'.join(lines) + '\n'
            (root / 'RESULTS.md').write_text(report)
            print(report)
            print(f'Full rank traces and tables: {root / "profiler-outputs.json"}')
            print(f'Verify/Decode cycles: {timeline}')
        profile.save(root / 'service-session.json', dict(session=str(session), backend=backend,
                                                       command=command, scene=os.environ['SCENE']))
        print(f'Results: {root}')


if __name__ == '__main__':
    main()
