# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Link DSL-produced attention children into an immutable tested model image.

Other target kernels, their orchestration, cache ABI, and static plans remain
byte-identical to --parent. This tool does not generate or rewrite model code.
Run build_from_dsl.py first to regenerate the three mathematical kernels.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARENT = ROOT.parent / "attention_native_fixture_20261029/build/target_key_value"
EXTRACTOR = ROOT.parent / "kernel_tuning_20261027/tools/extract_callable_children"
REPACKER = ROOT / "tools/repack_dsl_child"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--dsl-build", type=Path, required=True)
    parser.add_argument("--softmax-build", type=Path)
    args = parser.parse_args()
    if Path(args.name).name != args.name:
        raise ValueError("Build name must be one path component")
    parent = args.parent.resolve(strict=True)
    builds = {
        27: args.dsl_build / "qk",
        28: args.softmax_build or args.dsl_build / "softmax",
        29: args.dsl_build / "pv",
    }
    sources = {}
    for child, build in builds.items():
        assert (build / "callable.image").is_file(), build
        config = __import__("runpy").run_path(str(build / "kernel_config.py"))
        assert len(config["KERNELS"]) == 1
        sources[child] = config["KERNELS"][0]["source"]
    dest = ROOT / "build" / args.name
    shutil.copytree(
        parent,
        dest,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "key_policy_before", "key_policy_after"),
    )
    records = []
    for rank in range(4):
        old, new = parent / f"next_levels/rank{rank}", dest / f"next_levels/rank{rank}"
        before, after = new / "dsl_before", new / "dsl_after"
        before.mkdir()
        after.mkdir()
        subprocess.run(
            [str(EXTRACTOR), str(old / "callable.image"), str(before)], check=True, stdout=subprocess.DEVNULL
        )
        for child, build in builds.items():
            tmp = new / "pending.image"
            subprocess.run(
                [
                    str(REPACKER),
                    str(new / "callable.image"),
                    str(build / "callable.image"),
                    str(tmp),
                    str(child),
                ],
                check=True,
            )
            tmp.replace(new / "callable.image")
        subprocess.run(
            [str(EXTRACTOR), str(new / "callable.image"), str(after)], check=True, stdout=subprocess.DEVNULL
        )
        changed = sorted(
            int(p.stem) for p in after.glob("*.bin") if p.read_bytes() != (before / p.name).read_bytes()
        )
        assert changed == [27, 28, 29], changed
        assert (old / "compiled_meta.json").read_bytes() == (new / "compiled_meta.json").read_bytes()
        plans = list(old.glob("production_attention_grid*.bin"))
        assert plans and all(p.read_bytes() == (new / p.name).read_bytes() for p in plans)
        # Select generated DSL wrappers if rebuilding the native child registry.
        with (new / "kernel_config.py").open("a") as stream:
            stream.write("\n# DSL attention bodies; no handwritten mathematical override.\n")
            stream.write("DSL_ATTENTION_SOURCES = " + repr(sources) + "\n")
            stream.write(
                "for kernel in KERNELS:\n    if kernel['func_id'] in DSL_ATTENTION_SOURCES:\n        kernel.update(source=DSL_ATTENTION_SOURCES[kernel['func_id']], external=True)\n"
            )
        link = new / "native_backend/link_manifest.json"
        if link.exists():
            metadata = json.loads(link.read_text())
            for entry in metadata["kernels"]:
                if entry["func_id"] in sources:
                    entry["native_source"] = sources[entry["func_id"]]
            link.write_text(json.dumps(metadata, indent=2) + "\n")
        records.append(
            dict(
                rank=rank,
                callable_sha256=sha(new / "callable.image"),
                changed_children_vs_parent=changed,
                metadata_identical=True,
                static_plans_identical=True,
            )
        )
    (dest / "images.json").write_text(json.dumps(records, indent=2) + "\n")
    manifest = dict(
        parent=str(parent),
        candidate=str(dest),
        ranks=records,
        dsl_builds={str(k): str(v.resolve()) for k, v in builds.items()},
        generated_sources=sources,
        donor_sha256={str(k): sha(v / "callable.image") for k, v in builds.items()},
        numerical_body_source=str(ROOT / "src/qwen36_attention_dsl"),
        runtime_engine="unchanged Simpler static-plan interpreter",
    )
    (dest / "dsl_attention_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    contract = json.loads((dest / "build_contract.json").read_text())
    contract["attention_math_implementation"] = "direct_pypto_dsl_qk_softmax_pv"
    (dest / "build_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    print("BUILT_TARGET", dest, flush=True)


if __name__ == "__main__":
    main()
