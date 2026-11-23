# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Repository B4 boundaries plus a separately labelled full-target comparison.

The full CANN traces are primary artifacts. This script never replaces them
with a folded trace, and every interval retains original kernel CSV rows.
"""
import argparse
from collections import Counter
import csv
from decimal import Decimal as D
import hashlib
import json
from pathlib import Path
import statistics

INVALID = '4294967295'


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def stats(values):
    a = sorted(values)
    if not a:
        return dict(count=0)
    def pct(p):
        pos = (len(a) - 1) * p
        lo = int(pos)
        return a[lo] + (a[min(lo+1, len(a)-1)] - a[lo]) * (pos-lo)
    return dict(count=len(a), p50_ms=pct(.5), mean_ms=statistics.mean(a),
                p90_ms=pct(.9), p99_ms=pct(.99), maximum_ms=a[-1])


def rank_result(path, backend):
    with path.open(newline='') as f:
        rows = []
        for number, r in enumerate(csv.DictReader(f), start=2):
            start = D(r['Start Time(us)'].strip())
            rows.append(dict(row=number, start=start, end=start+D(r['Duration(us)']),
                name=r['Name'], type=r['Type'], graph=r['Model ID'], task=r['Task ID'],
                stream=r['Stream ID'], shape=r['Output Shapes'].strip('"'), inputs=r['Input Shapes']))
    rows.sort(key=lambda r: r['start'])
    samplers = [i for i, r in enumerate(rows) if r['name'] == 'rejection_greedy_sample_triton']
    cycles = []
    rejected = Counter()
    for i, j in zip(samplers, samplers[1:]):
        if rows[i]['shape'] != '4,6' or rows[j]['shape'] != '4,6':
            rejected['not_B4_6positions'] += 1
            continue
        left, right = rows[i]['start'], rows[j]['start']
        step = rows[i:j]
        if backend == 'native':
            counts = Counter(r['graph'] for r in step if r['name'] == 'FusedInferAttentionScore')
            targets = [g for g, n in counts.items() if n == 16]
            if len(targets) != 1:
                rejected['ambiguous_native_target_16FIA'] += 1
                continue
            target = [r for r in step if r['graph'] == targets[0]]
            starts = [r for r in target if r['name'] == 'GemmaRmsNorm']
            ends = [r for r in target if r['name'] == 'AddRmsNormBias']
            if len(starts) != 1 or not ends:
                rejected['missing_repository_norm_boundary'] += 1
                continue
            body_start = starts[0]
            body_end = max(ends, key=lambda r: r['end'])
            full_start = min(target, key=lambda r: r['start'])
            assert full_start['task'] == '0', 'Incomplete native graph boundary'
            graph_end = max(r['end'] for r in target)
            after = [r for r in step if r['start'] >= graph_end and r['graph'] == INVALID]
            heads = [r for r in after if r['type'] in ('MatMul', 'MatMulV2', 'MatMulV3')
                     and '62080' in r['inputs'] and '24,62080' in r['shape']]
            if len(heads) != 1:
                rejected['missing_full_target_head'] += 1
                continue
            gathers = [r for r in after if r['start'] >= heads[0]['end']
                       and r['type'] == 'Transpose' and '24,4,62080' in r['shape']]
            if len(gathers) != 1:
                rejected['missing_full_target_logits_gather'] += 1
                continue
            full_end = gathers[0]
            proof = dict(target_graph=targets[0], native_FIA_count=16,
                repository_first_norm_row=body_start['row'], repository_last_addnorm_row=body_end['row'],
                full_first_task_row=full_start['row'], head_row=heads[0]['row'],
                full_logits_gather_row=full_end['row'])
        else:
            targets = [r for r in step if r['name'] in ('aicore_kernel_0', 'aicore_kernel_0_mix_aic')]
            if len(targets) != 1:
                rejected['ambiguous_PTO_persistent_target'] += 1
                continue
            core = targets[0]
            parents = [r for r in step if r['name'].startswith('simpler_aicpu_exec_')]
            if len(parents) != 1:
                rejected['ambiguous_PTO_runtime_envelope'] += 1
                continue
            cpu = parents[0]
            # The runtime launches two concurrent streams and joins BOTH.
            # AICore may begin before AICPU; containment is not a pairing rule.
            full_start = min((core, cpu), key=lambda r: r['start'])
            full_end = max((core, cpu), key=lambda r: r['end'])
            body_start = body_end = core
            proof = dict(persistent_target_row=targets[0]['row'], kernel=targets[0]['name'],
                runtime_envelope_row=parents[0]['row'], runtime_kernel=parents[0]['name'],
                aicore_task_id=core['task'], aicpu_task_id=cpu['task'],
                aicore_stream=core['stream'], aicpu_stream=cpu['stream'],
                aicpu_start_us=str(cpu['start']), aicpu_end_us=str(cpu['end']),
                aicore_start_us=str(core['start']), aicore_end_us=str(core['end']),
                aicpu_contains_aicore=cpu['start']<=core['start'] and core['end']<=cpu['end'],
                pairing='exactly one runtime AICPU and one persistent AICore in the same complete B4 sampling period; retain task IDs/streams as evidence',
                repository_name_adapter='Donor PyptoRun name is absent in Simpler. Full target uses the union span of concurrent AICPU/AICore tasks; raw CSV untouched')
        graph_counts = Counter(r['graph'] for r in step if r['type'] == 'ArgMaxV2')
        draft_ids = [g for g, count in graph_counts.items() if count == 5 and g != INVALID]
        if len(draft_ids) != 1:
            rejected['not_one_complete_native_5step_draft_graph'] += 1
            continue
        draft = [r for r in step if r['graph'] == draft_ids[0]]
        draft_start = min(draft, key=lambda r: r['start'])
        draft_end = max(draft, key=lambda r: r['end'])
        if not left <= full_start['start'] < full_end['end'] <= right:
            rejected['target_outside_step'] += 1
            continue
        proof.update(draft_graph=draft_ids[0], draft_argmax_count=5,
                     draft_first_row=draft_start['row'], draft_last_row=draft_end['row'])
        cycles.append(dict(index=len(cycles), left_sampler_row=rows[i]['row'],
            right_sampler_row=rows[j]['row'], left_us=str(left), right_us=str(right),
            repository_decode_ms=float((right-left)/1000),
            repository_verify_ms=float((body_end['end']-body_start['start'])/1000),
            matched_full_verify_ms=float((full_end['end']-full_start['start'])/1000),
            draft_ms=float((draft_end['end']-draft_start['start'])/1000),
            draft_start_us=str(draft_start['start']), target_start_us=str(full_start['start']),
            target_end_us=str(full_end['end']), proof=proof))
    if len(cycles) < 8:
        raise ValueError(('Too few verified B4 cycles', path, len(cycles), rejected))
    if any(count for reason, count in rejected.items() if reason != 'not_B4_6positions'):
        raise ValueError(('Unresolved B4 boundary; do not silently discard it', path, rejected))
    alternate = [float((D(b['draft_start_us'])-D(a['draft_start_us']))/1000)
                 for a, b in zip(cycles, cycles[1:]) if a['right_sampler_row'] == b['left_sampler_row']]
    trace = path.parent / 'trace_view.json'
    return dict(kernel_details=str(path), kernel_details_sha256=sha(path),
        full_raw_trace=str(trace), full_raw_trace_sha256=sha(trace), full_raw_trace_bytes=trace.stat().st_size,
        kernel_rows=len(rows), rejected=dict(rejected), cycles=cycles,
        repository_decode=stats([c['repository_decode_ms'] for c in cycles]),
        repository_verify=stats([c['repository_verify_ms'] for c in cycles]),
        matched_full_verify=stats([c['matched_full_verify_ms'] for c in cycles]),
        draft=stats([c['draft_ms'] for c in cycles]), mtp_start_alternate=stats(alternate))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('native', 'pto'), required=True)
    parser.add_argument('--rank-csv', type=Path, nargs=4, required=True,
                        help='Rank0 through rank3 kernel_details.csv; trace_view.json must be beside each CSV')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    ranks = [dict(rank=rank, **rank_result(path, args.backend)) for rank, path in enumerate(args.rank_csv)]
    keys = ('matched_full_verify', 'mtp_start_alternate', 'repository_decode', 'repository_verify', 'draft')
    summary = {key: dict(p50_ms=statistics.median(r[key]['p50_ms'] for r in ranks),
                         mean_ms=statistics.mean(r[key]['mean_ms'] for r in ranks)) for key in keys}
    result = dict(backend=args.backend, ranks=ranks, summary=summary, diagnostic_only=True,
                  verify='matched_full_verify: full target including embedding, head and logits gather',
                  decode='repository_decode: consecutive B4 rejection_greedy_sample_triton starts, following GDD',
                  gdd_verify='Native first GemmaRmsNorm to last AddRmsNormBias; PTO persistent-core span is an explicit name adapter, not literal PyptoRun',
                  additional_mtp_cycle='mtp_start_alternate: consecutive native five-step drafter graph starts; separate from GDD Decode',
                  aggregation='Median of four rank P50; mean of four rank means. Ranks execute concurrently.',
                  input_reader_sha256=sha(Path(__file__)))
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
