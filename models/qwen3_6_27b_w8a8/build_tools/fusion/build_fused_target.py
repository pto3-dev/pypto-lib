# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Link explicit DSL fusion with an owned dispatcher; preserve unrelated math bytes.

Static metadata is decoded from the actual tested images, not stale source templates.
No model AST, numerical source substitution, or generated math C++ editing occurs.
"""

import collections, ctypes, hashlib, json, runpy, shutil, struct, subprocess
from pathlib import Path
from simpler_setup.device_program import Invocation, LeafSignature, compile_device_program
from pypto import ir, backend

N = Path(__file__).resolve().parents[1]
B = N.parent / "attention_dsl_20261029/build/target_dsl_explicit_v1"
DEST = N / "build/target_gate_quant_fused"
EX = N.parent / "kernel_tuning_20261027/tools/extract_callable_children"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def plan(old):
    h = struct.unpack_from("<16I", old)
    assert h[1] == 2
    steps = [struct.unpack_from("<16I", old, h[9] + i * 64) for i in range(h[4])]
    counts = collections.Counter(s[0] for s in steps)
    assert counts[28] == (144 if h[3] >= 4 else 2 * h[3]) and counts[30] == min(h[3] * 2, 48)
    nodes = []
    sigs = {}
    for i, s in enumerate(steps):
        if s[0] == 30:
            continue
        bind = struct.unpack_from("<" + "I" * s[6], old, h[10] + s[5] * 4)
        args = tuple(30 + (v & 0x7FFFFFFF == 144) if v >> 31 else v for v in bind)
        assert all(not (v >> 31) or (v & 0x7FFFFFFF) in (24, 144) for v in bind)
        deps = tuple(
            "old" + str(struct.unpack_from("<I", old, h[11] + j * 4)[0]) for j in range(s[7], s[7] + s[8])
        )
        node = Invocation("old" + str(i), s[0], ("aic", "aiv0", "aiv1")[s[1]], s[2], s[3], s[4], args, deps)
        nodes.append(node)
        sigs[s[0]] = LeafSignature(s[0], "aic" if s[1] == 0 else "aiv", len(args))
    assert not any(steps[int(d[3:])][0] == 30 for n in nodes for d in n.after)
    pv = tuple(n.name for n in nodes if n.func_id == 29)
    for row in range(24):
        nodes.append(
            Invocation(
                "fused" + str(row),
                30,
                "aiv" + str((row // h[3]) % 2),
                row % h[3],
                row,
                24,
                (26, 27, 10, 28, 29),
                pv,
            )
        )
    sigs[30] = LeafSignature(30, "aiv", 5)
    p = compile_device_program(
        nodes,
        list(sigs.values()),
        grid=h[3],
        available_clusters=h[3],
        root_arg_count=33,
        epoch_arg=32,
        max_wait_cycles=h[14],
    )
    image = bytearray(p.image)
    nh = struct.unpack_from("<16I", image)
    waited = {struct.unpack_from("<I", image, nh[11] + i * 4)[0] for i in range(nh[6])}
    for i in range(nh[4]):
        struct.pack_into("<I", image, nh[9] + i * 64 + 40, int(i not in waited))
    for i in range(nh[5]):
        off = nh[10] + i * 4
        v = struct.unpack_from("<I", image, off)[0]
        if v in (30, 31):
            struct.pack_into("<I", image, off, (1 << 31) | (24, 144)[v - 30])
    struct.pack_into("<I", image, 4, 2)
    struct.pack_into("<I", image, 48, 30)
    struct.pack_into("<I", image, 52, 31)
    return bytes(image), dict(
        grid=h[3],
        before=dict(counts),
        after=dict(collections.Counter(n.func_id for n in nodes)),
        root_args=31,
    )


def literal(v):
    if hasattr(v, "name"):
        return v.name
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, dict):
        return {k: literal(x) for k, x in v.items()}
    if isinstance(v, (tuple, list)):
        return [literal(x) for x in v]
    return v


def main():
    backend.set_backend_type(backend.BackendType.Ascend910B)
    shutil.copytree(
        B,
        DEST,
        ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", "dsl_before", "dsl_after", "key_policy_before", "key_policy_after"
        ),
    )
    records = []
    for rank in range(4):
        old = B / f"next_levels/rank{rank}"
        new = DEST / f"next_levels/rank{rank}"
        cfg = runpy.run_path(str(old / "kernel_config.py"))
        ks = literal(cfg["KERNELS"])
        orc = literal(cfg["ORCHESTRATION"])
        orc["source"] = str(N / f"src/runtime/orchestration_rank{rank}.cpp")
        for k in ks:
            if k["func_id"] == 30:
                k.update(
                    name="fused_attention_gate_quant",
                    source=str(N / "build/fused/kernels/aiv/fused_row.cpp"),
                    signature=["INOUT", "IN", "IN", "INOUT", "INOUT"],
                    external=True,
                )
            elif k["func_id"] in (39, 40):
                assert k["signature"][-1] == "SCALAR"
                k["signature"] = k["signature"][:-1] + ["INOUT", "INOUT", "SCALAR"]
        # Freeze all resolved dependencies. No inherited import fallback or source-rewriting build.
        config = "from simpler.task_interface import ArgDirection as D\n"
        config += (
            "RUNTIME_CONFIG = "
            + repr(literal(cfg["RUNTIME_CONFIG"]))
            + "\nKERNELS = "
            + repr(ks)
            + "\nORCHESTRATION = "
            + repr(orc)
            + "\n"
        )
        config += 'for entry in [ORCHESTRATION,*KERNELS]:\n    entry["signature"]=[getattr(D,s)for s in entry["signature"]]\n'
        (new / "kernel_config.py").write_text(config)
        # Remove stale prebuilt manifests; use explicit configuration for this new linked image.
        for name in ("binary_manifest.json", "artifact_manifest.json"):
            p = new / name
            if p.exists():
                p.rename(new / (name + ".inherited"))
        plans = []
        for p in sorted(old.glob("production_attention_grid*.bin")):
            image, entry = plan(p.read_bytes())
            (new / p.name).write_bytes(image)
            plans.append(entry)
        before = new / "fusion_before"
        after = new / "fusion_after"
        before.mkdir()
        after.mkdir()
        subprocess.run(
            [str(EX), str(old / "callable.image"), str(before)], check=True, stdout=subprocess.DEVNULL
        )
        chip = ir.CompiledProgram.from_dir(new, platform="a2a3").chip_callable
        (new / "callable.image").write_bytes(ctypes.string_at(chip.buffer_ptr(), chip.buffer_size()))
        subprocess.run(
            [str(EX), str(new / "callable.image"), str(after)], check=True, stdout=subprocess.DEVNULL
        )
        changed = sorted(
            int(p.stem) for p in after.glob("*.bin") if p.read_bytes() != (before / p.name).read_bytes()
        )
        assert changed == [30], changed
        assert (old / "compiled_meta.json").read_bytes() == (new / "compiled_meta.json").read_bytes()
        records.append(
            dict(
                rank=rank,
                callable_sha256=sha(new / "callable.image"),
                changed_math_children=changed,
                plans=plans,
                config_sha256=sha(new / "kernel_config.py"),
            )
        )
        print("FUSED_TARGET_RANK", rank, flush=True)
    (DEST / "images.json").write_text(json.dumps(records, indent=2) + "\n")
    (DEST / "fusion_manifest.json").write_text(
        json.dumps(
            dict(
                parent=str(B),
                source=str(N / "src/qwen36_fusion/gate_quant.py"),
                runtime_sources=str(N / "src/runtime"),
                records=records,
            ),
            indent=2,
        )
        + "\n"
    )
    runpy.run_path(str(N / "tools/sync_runtime_metadata.py"), run_name="__main__")
    print("BUILT_FUSED_TARGET", DEST, flush=True)


if __name__ == "__main__":
    main()
