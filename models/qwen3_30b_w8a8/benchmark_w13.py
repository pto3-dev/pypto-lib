# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -------------------------------------------------------------------------------------------------------
"""Resident expert projection correctness/timing, not a model/HTTP benchmark."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics

import torch
from golden import TensorSpec, run
from pypto.ir import DistributedConfig

from .config import EXPERTS
from .w13 import resident_w13_nd, resident_w13_nt
from .w2 import resident_w2_nt
from .weights import exact_int32, load_expert_layer, load_w13_bank, reference_w13


def tensor_hash(tensor: torch.Tensor) -> str:
    """Hash logical CPU contents to identify the frozen comparison fixture."""
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--tp-rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--routes", type=int, nargs="+", default=[8, 16, 64, 128])
    parser.add_argument("--layout", choices=("nd", "nt", "both"), default="both")
    parser.add_argument("--projection", choices=("w13", "w2"), default="w13")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--golden-data", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.projection == "w2" and args.layout != "nt":
        parser.error("W2 currently supports only --layout nt")
    if any(r < 1 or r > 128 for r in args.routes):
        parser.error("routes must be in [1,128]")
    if args.golden_data and (len(args.routes) != 1 or args.layout == "both"):
        parser.error("replay requires one route count and one layout (weight shapes differ)")
    torch.set_num_threads(4)
    weight_nt = (
        load_expert_layer(args.checkpoint, args.layer, args.tp_rank).w2
        if args.projection == "w2"
        else load_w13_bank(args.checkpoint, args.layer, args.tp_rank)
    )
    layouts = ("nd", "nt") if args.layout == "both" else (args.layout,)
    report = {
        "scope": f"rank-local resident {args.projection.upper()} INT32 only",
        "arguments": vars(args).copy(),
        "cases": [],
    }
    report["arguments"] = {k: str(v) if isinstance(v, Path) else v for k, v in report["arguments"].items()}
    report["fixture"] = "seeded full-range INT8 inputs, real checkpoint weights, repeated expert IDs"
    report["timing_scope"] = (
        "warm fixed-route resident dispatch; no inter-rank communication or cross-layer rotation"
    )
    report["source_sha256"] = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("config.py", "weights.py", "w13.py", "w2.py", "benchmark_w13.py")
    }
    if not args.golden_data:
        report["weight_nt_sha256"] = tensor_hash(weight_nt)
    fixtures = {}
    trials = {}
    for routes in args.routes:
        if routes not in fixtures:
            generator = torch.Generator().manual_seed(args.seed)
            x = torch.randint(-128, 128, (routes, weight_nt.shape[2]), generator=generator, dtype=torch.int8)
            ids = torch.randint(0, EXPERTS, (1, routes), generator=generator, dtype=torch.int32)
            ids[0, 0] = 0
            if routes > 1:
                ids[0, 1] = EXPERTS - 1
            if routes > 2:
                ids[0, 2] = 0
            # Freeze once per route count, across layouts AND repeated trials.
            expected = reference_w13(x, ids, weight_nt) if not args.golden_data else None
            fixtures[routes] = (x, ids, expected)
        x, ids, expected = fixtures[routes]
        trials[routes] = trials.get(routes, 0) + 1
        for layout in layouts:
            weight = weight_nt if layout == "nt" else weight_nt.transpose(1, 2).contiguous()

            def golden(values: dict[str, torch.Tensor]) -> None:
                values["out"].copy_(expected)

            specs = [
                TensorSpec("x", list(x.shape), torch.int8, init_value=x, resident=0),
                TensorSpec("expert_ids", list(ids.shape), torch.int32, init_value=ids, resident=0),
                TensorSpec("weight", list(weight.shape), torch.int8, init_value=weight, resident=0),
                TensorSpec("out", [routes, weight_nt.shape[1]], torch.int32, resident=0),
            ]
            result = run(
                fn=(
                    resident_w2_nt
                    if args.projection == "w2"
                    else resident_w13_nt
                    if layout == "nt"
                    else resident_w13_nd
                ),
                specs=specs,
                golden_fn=golden,
                golden_data=str(args.golden_data) if args.golden_data else None,
                save_data=args.save_data,
                compare_fn={"out": exact_int32},
                config=dict(
                    platform="a2a3",
                    compile_profiling=True,
                    distributed_config=DistributedConfig(device_ids=[args.device]),
                ),
            )
            case = dict(
                layout=layout,
                routes=routes,
                trial=trials[routes],
                passed=result.passed,
                work_dir=str(result.work_dir),
            )
            if not args.golden_data:
                case.update(
                    x_sha256=tensor_hash(x),
                    ids_sha256=tensor_hash(ids),
                    expected_sha256=tensor_hash(expected),
                )
            case["error"] = result.error
            if result.bench is not None:
                samples = result.bench.per_round("effective")
                case["effective_us"] = samples
                case["effective_us_median"] = statistics.median(samples) if samples else None
                case["device_us"] = result.bench.per_round("device")
                case["host_us"] = result.bench.per_round("host")
            else:
                case["timing_status"] = "not measured or timing unavailable"
            report["cases"].append(case)
            if args.report:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(f"[{args.projection.upper()}] {json.dumps(case)}", flush=True)
            if not result.passed:
                raise SystemExit(f"{args.projection.upper()} correctness gate failed: {result.error}")
    print(f"PYPTO_BENCH={os.environ.get('PYPTO_BENCH', '0')}; component timings are not TPOT.", flush=True)


if __name__ == "__main__":
    main()
