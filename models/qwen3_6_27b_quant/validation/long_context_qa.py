# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Four exact32K context-extraction checks against the frozen serving interface.

Select evidence near the beginning, middle and end of the existing diverse QA
corpus. Shorten only its trailing unused context to fit the new question; never
repeat text or fabricate per-token arrival times. Save the exact request IDs.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.request

RECORD = re.compile(
    r'记录 (Z\d+-\d+)：[^\n]+?原计划处理(\d+)件，实际完成(\d+)件。'
    r'预算上限为(\d+)元，已支出(\d+)元。依赖记录 (Z\d+-\d+)，'
)
FIELDS = ('planned', 'completed', 'budget', 'spent', 'dependency')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def prepare_case(prompt, tokenizer, input_tokens=32768):
    context, old_question = prompt.rsplit('\n\n请审阅最前面的80条完整记录', 1)
    suffix = old_question[old_question.index('<|im_end|>'):]
    records = list(RECORD.finditer(context))
    if len(records) < 6:
        raise ValueError('Long QA requires diverse complete source records')
    selected = [records[0], records[len(records) // 2], records[-3]]
    expected = {}
    for match in selected:
        rid, planned, completed, budget, spent, dependency = match.groups()
        expected[rid] = dict(zip(FIELDS, [int(planned), int(completed), int(budget), int(spent), dependency]))
    question = (
        '\n\n请仅依据以上项目档案核对以下三个编号：' + '、'.join(expected) + '。'
        '直接输出一个合法JSON对象，以这三个记录编号为键，不要添加其他编号。'
        '每个编号对应的值必须恰好包含五个字段：planned、completed、budget、spent、dependency。'
        'planned是原计划处理件数，completed是实际完成件数，budget是预算上限，spent是已支出金额，'
        '这四项必须是JSON整数，不要加单位、引号、百分比或计算结果。'
        'dependency必须是档案中原文给出的依赖记录编号，使用JSON字符串。'
        '请逐项提取原始事实，不要修改数字，不要把负责项目编号或负责人代号误当成答案，'
        '不要合计，不要四舍五入，不要推测，不要解释，不要Markdown，不要代码围栏。'
        '答案必须同时覆盖开头、中间及末尾的指定记录，且每个编号包含且仅包含上述五项。'
    )
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    suffix_ids = tokenizer.encode(question + suffix, add_special_tokens=False)
    budget = input_tokens - len(suffix_ids)
    if budget <= 0 or len(context_ids) < budget:
        raise ValueError('Natural source context cannot fill the exact input budget; no synthetic padding is allowed')
    kept_context = tokenizer.decode(context_ids[:budget], skip_special_tokens=False)
    surviving = {match[0]: match for match in RECORD.findall(kept_context)}
    regions = []
    for match in selected:
        record = match.groups()
        if surviving.get(record[0]) != record:
            raise ValueError('Input clipping removed or changed a selected evidence record')
        offset = len(tokenizer.encode(context[:match.start()], add_special_tokens=False))
        regions.append(dict(record_id=record[0], approximate_source_token_offset=offset,
                            context_fraction=offset / budget))
    ids = context_ids[:budget] + suffix_ids
    if len(ids) != input_tokens:
        raise ValueError('Input token count differs from the exact32K contract')
    return dict(input_token_ids=ids, input_tokens=len(ids), context_tokens=budget,
                question=question, expected=expected, selected_regions=regions,
                input_token_ids_sha256=sha(json.dumps(ids, separators=(',', ':')).encode()),
                source_prompt_sha256=sha(prompt.encode()), repeated_padding=False)


def prepare(checkpoint, corpus):
    from transformers import AutoTokenizer
    digest = sha(corpus.read_bytes())
    manifest = json.loads((corpus.parent / 'manifest.json').read_text())
    if manifest['files'].get(corpus.name) != digest:
        raise ValueError('Frozen corpus SHA256 mismatch')
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), trust_remote_code=True)
    rows = [json.loads(line) for line in corpus.read_text().splitlines()]
    if len(rows) != 4 or len({row['prompt'] for row in rows}) != 4:
        raise ValueError('Expected four distinct natural-context prompts')
    cases = [prepare_case(row['prompt'], tokenizer) for row in rows]
    fingerprints = {p.name: sha(p.read_bytes()) for p in checkpoint.iterdir()
                    if p.is_file() and ('token' in p.name or p.name == 'config.json')}
    return cases, dict(checkpoint=str(checkpoint), corpus_sha256=digest, tokenizer_hashes=fingerprints,
                       input_tokens_per_request=32768, requests=4, cases=cases,
                       source_client_sha256=sha(Path(__file__).read_bytes()))


def api(base_url, endpoint, body=None, timeout=600):
    request = urllib.request.Request(base_url.rstrip('/') + endpoint,
        data=None if body is None else json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def validate(content, expected):
    try:
        parsed = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        parsed = None
    schema = isinstance(parsed, dict) and parsed.keys() == expected.keys()
    correct = 0
    for rid, values in expected.items():
        actual = parsed.get(rid) if isinstance(parsed, dict) else None
        schema = schema and isinstance(actual, dict) and actual.keys() == values.keys()
        if isinstance(actual, dict):
            correct += sum(type(actual.get(key)) is type(value) and actual[key] == value
                           for key, value in values.items())
    return dict(parsed=parsed, schema_valid=bool(schema), correct_fields=correct, total_fields=15,
                content_passed=bool(schema and correct == 15))


def run(base_url, output, *, checkpoint, corpus, prepare_only=False, timeout=600):
    output.mkdir(parents=True, exist_ok=False)
    cases, gate = prepare(checkpoint, corpus)
    (output / 'INPUT_GATE.json').write_text(json.dumps(gate, ensure_ascii=False, indent=2) + '\n')
    if prepare_only:
        print(json.dumps(dict(input_tokens=[c['input_tokens'] for c in cases], device_requests=0)))
        return gate
    model = api(base_url, '/v1/models')['data'][0]['id']
    def evaluate(index):
        case = cases[index]
        body = dict(model=model, prompt=case['input_token_ids'], max_tokens=512,
                    temperature=0, seed=0, stream=False, return_token_ids=True)
        row = dict(index=index, request=body, expected=case['expected'], input_tokens=case['input_tokens'],
                   selected_regions=case['selected_regions'], start_ns=time.monotonic_ns())
        try:
            payload = api(base_url, '/v1/completions', body, timeout)
            choice = payload['choices'][0]
            row.update(response=payload, **validate(choice['text'], case['expected']))
            row['passed'] = (row['content_passed'] and choice['finish_reason'] == 'stop'
                             and payload.get('usage', {}).get('prompt_tokens') == 32768)
        except Exception as error:
            detail = error.read().decode(errors='replace') if isinstance(error, urllib.error.HTTPError) else None
            row.update(error=repr(error), http_error_body=detail, passed=False, correct_fields=0, total_fields=15)
        row['end_ns'] = time.monotonic_ns()
        (output / f'request{index}.json').write_text(json.dumps(row, ensure_ascii=False, indent=2) + '\n')
        return row
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(evaluate, range(4)))
    summary = dict(requests=4, passed=sum(r['passed'] for r in results), input_tokens=[32768] * 4,
                   correct_fields=sum(r['correct_fields'] for r in results), total_fields=60,
                   all_passed=all(r['passed'] for r in results), cases=results, performance_benchmark=False)
    (output / 'qa-result.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'cases'}))
    if not summary['all_passed']:
        raise RuntimeError('Long-context QA failed; inspect the actual saved responses')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if not args.prepare_only and not args.base_url:
        parser.error('--base-url is required unless --prepare-only')
    run(args.base_url, args.output, checkpoint=args.checkpoint, corpus=args.corpus, prepare_only=args.prepare_only)


if __name__ == '__main__':
    main()
