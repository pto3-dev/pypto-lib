# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prepare (do not benchmark) four distinct, non-repeated 32K contexts.

Use vLLM's official CustomDataset loader. Preserve source text, licenses,
tokenizer hashes, exact token IDs and the generated text/ID round trip.
"""

import concurrent.futures, hashlib, json, re, urllib.request
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer

W = Path(__file__).resolve().parents[3]
ROOT = W.parents[1]
BOOKS = [
    (1342, "Pride and Prejudice"),
    (2701, "Moby Dick"),
    (1400, "Great Expectations"),
    (84, "Frankenstein"),
]


def main():
    folder = W / "datasets/real_text"
    folder.mkdir(parents=True, exist_ok=True)

    def fetch(item):
        number, title = item
        url = f"https://www.gutenberg.org/cache/epub/{number}/pg{number}.txt"
        p = folder / f"pg{number}.txt"
        if not p.exists():
            with urllib.request.urlopen(url, timeout=30) as response:
                p.write_bytes(response.read())
        return item, url, p

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        downloaded = list(pool.map(fetch, BOOKS))
    tokenizer = AutoTokenizer.from_pretrained(
        ROOT / "checkpoints/public-inner-surrogate", local_files_only=True
    )
    rows = []
    metadata = []
    for (number, title), url, path in downloaded:
        raw = path.read_text(encoding="utf-8-sig")
        content = re.split(r"\*\*\* START OF[^\n]*\*\*\*", raw, maxsplit=1)[-1]
        content = re.split(r"\*\*\* END OF[^\n]*\*\*\*", content, maxsplit=1)[0].strip()

        # A single contiguous excerpt from one book; no tiling or repetition.
        def prompt(n):
            user = (
                f"Read this excerpt from {title}.\n<context>\n"
                + content[:n]
                + "\n</context>\nWrite a detailed critical study covering the narrative, chronology, characters, evidence, literary techniques, and alternative interpretations. Use specific details from this excerpt and avoid repeating earlier points. Continue in depth."
            )
            return tokenizer.apply_chat_template(
                [dict(role="user", content=user)], tokenize=False, add_generation_prompt=True
            )

        lo, hi = 1, len(content)
        while lo < hi:
            mid = (lo + hi) // 2
            n = len(tokenizer.encode(prompt(mid), add_special_tokens=False))
            if n < 32768:
                lo = mid + 1
            else:
                hi = mid
        found = None
        for chars in range(max(1, lo - 16), min(len(content), lo + 17)):
            text = prompt(chars)
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) == 32768:
                found = (chars, text, ids)
                break
        if found is None:
            raise ValueError(f"No exact natural excerpt boundary for {title}")
        chars, text, ids = found
        rows.append(dict(prompt=text, output_tokens=10240))
        grams = Counter(tuple(ids[i : i + 4]) for i in range(len(ids) - 3))
        metadata.append(
            dict(
                book=title,
                url=url,
                source=str(path),
                source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                contiguous_excerpt_characters=chars,
                input_tokens=len(ids),
                unique_tokens=len(set(ids)),
                unique_4gram_ratio=len(grams) / (len(ids) - 3),
                most_common_4gram_count=max(grams.values()),
                input_token_ids=ids,
            )
        )
    file = folder / "custom.jsonl"
    file.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    from vllm.benchmarks.datasets import CustomDataset

    samples = CustomDataset(dataset_path=str(file), random_seed=1024, disable_shuffle=True).sample(
        tokenizer=tokenizer,
        num_requests=4,
        output_len=10240,
        request_id_prefix="real-text-",
        skip_chat_template=True,
    )
    actual = [tokenizer.encode(sample.prompt, add_special_tokens=False) for sample in samples]
    if actual != [x["input_token_ids"] for x in metadata]:
        raise ValueError("Official dataset loader changed prompts/order")
    (folder / "manifest.json").write_text(
        json.dumps(
            dict(
                generator="vllm.benchmarks.datasets.CustomDataset",
                seed=1024,
                shuffle=False,
                request_count=4,
                input_tokens_each=32768,
                output_tokens_each=10240,
                representativeness="Public long-form text supplement, not private production traffic; generated outputs still require repetition inspection",
                requests=metadata,
            ),
            indent=2,
        )
        + "\n"
    )
    request = dict(
        model="qwen36-mixed",
        prompt=actual,
        max_tokens=10240,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        seed=1024,
        ignore_eos=True,
        return_token_ids=True,
        stream=True,
        stream_options=dict(include_usage=True),
    )
    (folder / "replay_10k.json").write_text(json.dumps(request, separators=(",", ":")) + "\n")
    print(
        "REAL_TEXT_READY",
        [(x["book"], x["input_tokens"], x["unique_4gram_ratio"]) for x in metadata],
        flush=True,
    )


if __name__ == "__main__":
    main()
