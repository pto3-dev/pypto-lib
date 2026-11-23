# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Four deterministic, two-turn chat checks against an existing OpenAI server.

Standard-library client only. This module neither imports model/framework code
nor launches a service. It saves the exact messages and complete responses.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.request


CASES = [
    dict(id="ALPHA", planned=12, completed=9, budget=800, spent=510),
    dict(id="BRAVO", planned=20, completed=7, budget=1200, spent=350),
    dict(id="CHARLIE", planned=35, completed=16, budget=2400, spent=975),
    dict(id="DELTA", planned=9, completed=4, budget=600, spent=125),
]
SYSTEM = "你是记录核对助手。仅根据用户给出的记录回答。严格输出要求的JSON对象，不要Markdown，不要解释或添加字段。"


def first_turn(case):
    text = (f"记录编号{case['id']}：计划完成{case['planned']}件，已完成{case['completed']}件，"
            f"预算{case['budget']}元，已花费{case['spent']}元。"
            '请输出JSON，字段严格为record_id、planned、completed、budget、spent；'
            'record_id为字符串，其余为整数。')
    expected = dict(record_id=case['id'], **{k: case[k] for k in ('planned', 'completed', 'budget', 'spent')})
    return text, expected


def second_turn(case):
    return ('根据刚才那条记录计算剩余件数和未花费预算。仅输出JSON，字段严格为'
            'record_id、remaining、unspent；record_id为字符串，其余为整数。',
            dict(record_id=case['id'], remaining=case['planned']-case['completed'],
                 unspent=case['budget']-case['spent']))


def json_api(base_url, endpoint, body=None, timeout=600):
    request = urllib.request.Request(base_url.rstrip('/')+endpoint,
        data=None if body is None else json.dumps(body, ensure_ascii=False).encode(),
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def validate(content, expected):
    try:
        answer = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        answer = None
    schema = isinstance(answer, dict) and answer.keys() == expected.keys()
    correct = sum(isinstance(answer, dict) and type(answer.get(k)) is type(v) and answer[k] == v
                  for k, v in expected.items())
    return dict(parsed=answer, schema_valid=schema, correct=correct, total=len(expected),
                content_passed=bool(schema and correct == len(expected)))


def run(base_url, output, *, model=None, timeout=600):
    output.mkdir(parents=True, exist_ok=False)
    model = model or json_api(base_url, '/v1/models')['data'][0]['id']
    histories = [[dict(role='system', content=SYSTEM)] for _ in CASES]
    rows = []
    for turn, builder in enumerate((first_turn, second_turn), 1):
        def evaluate(index):
            case = CASES[index]
            prompt, expected = builder(case)
            messages = histories[index]+[dict(role='user', content=prompt)]
            body = dict(model=model, messages=messages, temperature=0, seed=0,
                        max_tokens=192, stream=False, return_token_ids=True,
                        chat_template_kwargs=dict(enable_thinking=False))
            started = time.monotonic_ns()
            row = dict(case=case['id'], turn=turn, request=body, expected=expected,
                       started_monotonic_ns=started)
            try:
                payload = json_api(base_url, '/v1/chat/completions', body, timeout)
                choice = payload['choices'][0]
                message = choice['message']
                row.update(response=payload, **validate(message.get('content'), expected))
                row['passed'] = row['content_passed'] and choice['finish_reason'] == 'stop'
                histories[index] = messages+[dict(role='assistant', content=message.get('content') or '')]
            except Exception as error:
                detail = error.read().decode(errors='replace') if isinstance(error, urllib.error.HTTPError) else None
                row.update(error=repr(error), http_error_body=detail, passed=False, correct=0, total=len(expected))
                histories[index] = messages
            row.update(ended_monotonic_ns=time.monotonic_ns())
            row['elapsed_ms'] = (row['ended_monotonic_ns']-started)/1e6
            return row
        # All four first turns finish before the four follow-up requests.
        with ThreadPoolExecutor(max_workers=4) as pool:
            current = list(pool.map(evaluate, range(len(CASES))))
        rows.extend(current)
        for row in current:
            print(json.dumps({k: row[k] for k in ('case', 'turn', 'passed', 'elapsed_ms')}, ensure_ascii=False), flush=True)
        (output/f'turn{turn}.json').write_text(json.dumps(current, ensure_ascii=False, indent=2)+'\n')
    result = dict(base_url=base_url, model=model, endpoint='/v1/chat/completions',
        conversations=4, turns_per_conversation=2, requests=len(rows),
        passed=sum(row['passed'] for row in rows), correct=sum(row['correct'] for row in rows),
        total=sum(row['total'] for row in rows), all_passed=all(row['passed'] for row in rows),
        performance_benchmark=False, thinking_enabled=False, results=rows,
        client_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output/'qa-result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('requests', 'passed', 'correct', 'total', 'all_passed')}, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model')
    parser.add_argument('--timeout', type=int, default=600)
    args = parser.parse_args()
    result = run(args.base_url, args.output, model=args.model, timeout=args.timeout)
    raise SystemExit(0 if result['all_passed'] else 1)


if __name__ == '__main__':
    main()
