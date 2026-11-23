# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Measure client token arrivals and server MTP counters for four fixed prompts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import statistics
import threading
import time
from urllib.request import Request, urlopen

from e2e_pypto_vllm_profile import BASE, get, metric

# Archive-compatible custom client; raw SSE payloads retain actual token IDs.
# This script does not change the service or invent per-token timestamps.


METRICS = {
    'running': 'num_requests_running', 'waiting': 'num_requests_waiting',
    'prompt': 'prompt_tokens_total', 'generated': 'generation_tokens_total',
    'preemptions': 'num_preemptions_total', 'drafts': 'spec_decode_num_drafts_total',
    'offered': 'spec_decode_num_draft_tokens_total', 'accepted': 'spec_decode_num_accepted_tokens_total',
}


def snapshot():
    before = time.monotonic_ns()
    raw = get('/metrics')
    after = time.monotonic_ns()
    values = {key: metric(raw, 'vllm:' + name) for key, name in METRICS.items()}
    return dict(ns=(before + after)//2, scrape_ns=after-before, **values), raw


def request(prompt, model, length, barrier):
    data = dict(model=model, prompt=prompt, temperature=0, seed=0, max_tokens=length,
                ignore_eos=True, stream=True, return_token_ids=True,
                stream_options={'include_usage': True})
    req = Request(BASE+'/v1/completions', data=json.dumps(data).encode(),
                  headers={'Content-Type': 'application/json'})
    barrier.wait()
    start = time.monotonic_ns()
    events, text, messages = [], [], []
    usage = finish = request_id = None
    with urlopen(req, timeout=900) as response:
        for line in response:
            arrival = time.monotonic_ns()
            if not line.startswith(b'data:') or line.strip()==b'data: [DONE]':
                continue
            msg=json.loads(line[5:])
            messages.append(dict(arrival_ns=arrival,payload=msg))
            request_id = msg['id']
            usage = msg.get('usage') or usage
            for choice in msg['choices']:
                ids = choice.get('token_ids') or []
                if ids:
                    events.append((arrival,len(ids)))
                text.append(choice['text'])
                finish = choice['finish_reason'] or finish
    return dict(id=request_id,start_ns=start,events=events,usage=usage,finish=finish,text=''.join(text),
                request=data,messages=messages)


def delta(a,b):
    return {k:b[k]-a[k] for k in METRICS if k not in ('running','waiting')}


def run(corpus,out,length):
    raw_corpus = corpus.read_bytes()
    digest = hashlib.sha256(raw_corpus).hexdigest()
    manifest = json.loads((corpus.parent / 'manifest.json').read_text())
    if digest != manifest['files'][corpus.name]:
        raise ValueError(f'Frozen corpus hash mismatch: {corpus}')
    out.mkdir(parents=True,exist_ok=False)
    prompts = [json.loads(line)['prompt'] for line in raw_corpus.decode().splitlines()]
    assert len(prompts)==4
    model=json.loads(get('/v1/models'))['data'][0]['id']
    start,raw=snapshot()
    (out/'metrics-before.txt').write_text(raw)
    assert start['running']==start['waiting']==0
    samples=[]
    barrier=threading.Barrier(4)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(request,prompt,model,length,barrier) for prompt in prompts]
        while not all(f.done() for f in futures):
            samples.append(snapshot()[0])
            time.sleep(.5)
        rows=[f.result() for f in futures]
    end,raw=snapshot()
    (out/'metrics-after.txt').write_text(raw)
    (out/'requests.json').write_text(json.dumps(rows,ensure_ascii=False))
    (out/'metrics-samples.json').write_text(json.dumps(samples))
    for row in rows:
        assert row['usage']['prompt_tokens']==32768 and row['usage']['completion_tokens']==length, row['usage']
        assert sum(n for _,n in row['events'])==length and row['finish']=='length',row['id']
    overlap_start=max(r['events'][0][0] for r in rows)
    overlap_end=min(r['events'][-1][0] for r in rows)
    eligible=[r for r in samples if overlap_start < r['ns'] < overlap_end
              and r['running']==4 and r['waiting']==0 and r['prompt']-start['prompt']==131072]
    assert len(eligible)>=2 and end['preemptions']==start['preemptions'], (len(eligible),delta(start,end))
    inside = [sample for sample in samples if eligible[0]['ns'] <= sample['ns'] <= eligible[-1]['ns']]
    assert inside == eligible, '共同窗口内观测到排队、退批或 prefill 未完成'
    a,b=eligible[0],eligible[-1]
    window=delta(a,b)
    elapsed=(b['ns']-a['ns'])/1e6
    window_events = sorted(
        (t, n) for row in rows for t, n in row['events'] if a['ns'] < t <= b['ns']
    )
    received = sum(n for _, n in window_events)
    gaps_ms = [(right - left) / 1e6
               for (left, _), (right, _) in zip(window_events, window_events[1:])
               if right - left >= 200_000_000]
    if min(received, window['generated'], window['offered']) <= 0:
        raise ValueError(f'共同窗口缺少生成或草稿计数：{window}')
    whole=delta(start,end)
    result=dict(completed=4,failed=0,corpus_sha256=digest,mean_tpot_ms=statistics.mean(
        (r['events'][-1][0]-r['events'][0][0])/1e6/(length-1) for r in rows),
        mean_ttft_ms=statistics.mean((r['events'][0][0]-r['start_ns'])/1e6 for r in rows),
        output_throughput=4*length*1e9/(max(r['events'][-1][0] for r in rows)-min(r['start_ns'] for r in rows)),
        common_client_tpot_ms=4*elapsed/received, common_server_tpot_ms=4*elapsed/window['generated'],
        common_ms=elapsed,common_client_tokens=received,common_counters=window,full_counters=whole,
        common_window=[a,b],common_sse_gaps=dict(
            threshold_ms=200, count=len(gaps_ms), total_ms=sum(gaps_ms),
            max_ms=max(gaps_ms, default=0)),
        full_acceptance_pct=100*whole['accepted']/whole['offered'],
        common_acceptance_pct=100*window['accepted']/window['offered'])
    # Counters count draft rounds per request. This is a proxy for batch steps.
    if window['drafts'] > 0 and window['drafts'] % 4 == 0 and window['offered'] == 5 * window['drafts']:
        steps = window['drafts'] // 4
        result.update(common_mtp_steps_estimate=steps,
                      common_wall_ms_per_step_estimate=elapsed/steps,
                      common_client_tokens_per_request_step_estimate=received/(4*steps),
                      common_server_tokens_per_request_step_estimate=window['generated']/(4*steps))
    (out/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('corpus',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--output-len',type=int,default=10240)
    args=parser.parse_args()
    if not 2 <= args.output_len <= 45056 - 32768:
        parser.error('--output-len must be between 2 and 12288')
    run(args.corpus,args.output,args.output_len)
