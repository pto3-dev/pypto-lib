# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Ordinary-text native/PTO vLLM measurements with explicit compilation modes."""

import argparse
import json
import os
import time
from pathlib import Path
from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--prompt-token-ids", type=Path)
    parser.add_argument("--max-model-len", type=int, default=128)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--async-scheduling", action="store_true")
    parser.add_argument("--compilation-mode", type=int, choices=[0, 3], default=3)
    parser.add_argument(
        "--aclgraph-only",
        action="store_true",
        help="Diagnostic capture/replay with model TorchDynamo compilation disabled.",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--diagnostics-dir", type=Path)
    parser.add_argument("--runtime-stats", action="store_true")
    parser.add_argument("--validation-prompts", type=Path)
    parser.add_argument("--expected-native", type=Path)
    parser.add_argument("--native-nd-reference", action="store_true")
    parser.add_argument("--pto-mpmd-socket")
    parser.add_argument("--pto-mpmd-phase-nz", action="store_true")
    parser.add_argument("--swimlane-output", type=Path)
    parser.add_argument("--logits-validation-dir", type=Path)
    parser.add_argument("--teacher-validation-reference", type=Path)
    parser.add_argument("--teacher-validation-dir", type=Path)
    parser.add_argument("--pto-mlp-build-root", type=Path)
    parser.add_argument("--disable-unavailable-norm-fusions", action="store_true")
    args = parser.parse_args()
    if args.aclgraph_only and not args.graph:
        parser.error("--aclgraph-only requires --graph")
    if args.tensor_parallel_size * args.pipeline_parallel_size != 2:
        parser.error("This task is limited to two physical NPU cards.")
    tokens = (
        json.loads(args.prompt_token_ids.read_text())
        if args.prompt_token_ids
        else AutoTokenizer.from_pretrained(args.model, local_files_only=True).encode(args.prompt)
    )
    if len(tokens) + args.output_tokens > args.max_model_len:
        raise ValueError("Prompt plus output exceeds max_model_len.")
    if args.runtime_stats:
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    if args.diagnostics_dir:
        os.environ["QWEN38_DIAGNOSTICS_DIR"] = str(args.diagnostics_dir.resolve())
    additional_config = {}
    if args.graph and args.compilation_mode == 0:
        additional_config["ascend_compilation_config"] = {"enable_npugraph_ex": False}
    if args.disable_unavailable_norm_fusions:
        additional_config.setdefault("ascend_compilation_config", {}).update(
            fuse_norm_quant=False, fuse_allreduce_rms=False
        )
    if args.aclgraph_only:
        additional_config["qwen38_aclgraph_only"] = True
    if args.pto_mlp_build_root:
        additional_config["qwen38_pto_mlp_build_root"] = str(args.pto_mlp_build_root.resolve())
    if args.pto_mpmd_socket:
        if args.pto_mlp_build_root:
            parser.error("Full MPMD excludes the partial MLP path.")
        additional_config["qwen38_pto_mpmd"] = {
            "socket": args.pto_mpmd_socket,
            "graph": args.graph,
            "phase_nz": args.pto_mpmd_phase_nz,
        }
    if args.swimlane_output and not args.pto_mpmd_socket:
        parser.error("--swimlane-output requires the full MPMD service.")
    if args.swimlane_output:
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    if args.native_nd_reference:
        if args.graph or args.pto_mpmd_socket:
            parser.error("Native ND reference requires eager execution and no MPMD socket.")
        additional_config["qwen38_pto_mpmd"] = {"native_reference": True}
    if args.logits_validation_dir:
        if "qwen38_pto_mpmd" not in additional_config:
            parser.error("--logits-validation-dir requires MPMD or the shared-ND native oracle.")
        additional_config["qwen38_pto_mpmd"]["logits_validation_dir"] = str(
            args.logits_validation_dir.resolve()
        )
    if bool(args.teacher_validation_reference) != bool(args.teacher_validation_dir):
        parser.error("Same-prefix validation requires both reference and output paths.")
    if args.teacher_validation_reference:
        if "qwen38_pto_mpmd" not in additional_config or args.swimlane_output:
            parser.error("Same-prefix validation requires native ND or MPMD without profiling.")
        if args.warmup != 0 or args.repeats != 1:
            parser.error("Same-prefix validation uses exactly one untimed request.")
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    begin = time.perf_counter()
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        distributed_executor_backend="mp",
        enforce_eager=not args.graph,
        compilation_config=(
            CompilationConfig(
                mode=CompilationMode(args.compilation_mode),
                cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
                cudagraph_capture_sizes=[1],
                max_cudagraph_capture_size=1,
            )
            if args.graph
            else None
        ),
        additional_config=additional_config,
        max_model_len=args.max_model_len,
        max_num_seqs=1,
        max_num_batched_tokens=args.max_model_len,
        enable_chunked_prefill=False,
        gpu_memory_utilization=0.96,
        num_gpu_blocks_override=8,
        enable_prefix_caching=False,
        limit_mm_per_prompt={"image": 0, "video": 0},
        language_model_only=True,
        seed=183,
        disable_log_stats=False,
        async_scheduling=args.async_scheduling,
    )
    startup_seconds = time.perf_counter() - begin
    if args.teacher_validation_reference:
        from functools import partial
        from same_prefix_validation import install_same_prefix_validation

        llm.apply_model(
            partial(
                install_same_prefix_validation,
                reference_path=str(args.teacher_validation_reference.resolve()),
                output=str(args.teacher_validation_dir.resolve()),
            )
        )
    if args.diagnostics_dir:
        from vllm_diagnostics import install_diagnostics

        llm.apply_model(install_diagnostics)
    params = SamplingParams(temperature=0, max_tokens=args.output_tokens, ignore_eos=True)
    requests = []
    previous_jit_misses = None
    for index in range(args.warmup + args.repeats):
        begin = time.perf_counter()
        outputs = llm.generate([{"prompt_token_ids": tokens}], params, use_tqdm=False)
        wall_ms = (time.perf_counter() - begin) * 1000
        output = outputs[0]
        m = output.metrics
        metrics = (
            None
            if m is None
            else {
                key: getattr(m, key, None)
                for key in (
                    "arrival_time",
                    "queued_ts",
                    "scheduled_ts",
                    "first_token_ts",
                    "last_token_ts",
                    "first_token_latency",
                )
            }
        )
        first = None if m is None else m.first_token_latency * 1000
        tpot = (
            None
            if m is None
            else (m.last_token_ts - m.first_token_ts) * 1000 / (len(output.outputs[0].token_ids) - 1)
        )
        requests.append(
            {
                "warmup": index < args.warmup,
                "input_tokens": len(tokens),
                "output_tokens": len(output.outputs[0].token_ids),
                "token_ids": list(output.outputs[0].token_ids),
                "text": output.outputs[0].text,
                "e2e_ms": wall_ms,
                "ttft_ms": first,
                "mean_tpot_ms": tpot,
                "output_tokens_per_second": len(output.outputs[0].token_ids) * 1000 / wall_ms,
                "engine_metrics": metrics,
            }
        )
        if args.runtime_stats:
            from vllm_ascend.pypto.qwen3_8_stats import collect_pto_stats

            # The diagnostic RPC is outside the completed request's timing.
            post_request_stats = llm.apply_model(collect_pto_stats)
            jit_misses = sum(worker.get("jit_misses", 0) for worker in post_request_stats)
            requests[-1]["jit_in_timed_request"] = (
                jit_misses > 0 if previous_jit_misses is None else jit_misses > previous_jit_misses
            )
            requests[-1]["cumulative_pto_jit_misses"] = jit_misses
            previous_jit_misses = jit_misses
        pto_gdn = os.environ.get("VLLM_ASCEND_PYPTO_QWEN38_GDN", "0") == "1"
        pto_model = os.environ.get("VLLM_ASCEND_PYPTO_QWEN38_MODEL", "0") == "1"
        engine = (
            "vLLM-Ascend eager with full ordinary-text PTO model"
            if pto_model
            else "vLLM-Ascend eager with PTO GDN core and native remaining operators"
            if pto_gdn
            else "native vLLM-Ascend eager, native QK/RoPE split fallback for 24x256, split GDN gating/recurrent"
        )
        if args.pto_mpmd_socket:
            engine = "vLLM-Ascend native prefill and full PTO MPMD decode TP2/PP1 with shared ND weights and native cache"
        if args.graph:
            graph_description = (
                "ACLGraph-only diagnostic"
                if args.aclgraph_only
                else f"mode-{args.compilation_mode} + FULL_DECODE_ONLY graph"
            )
            engine = engine.replace("eager", graph_description)
            if args.pto_mpmd_socket:
                engine += " with full caller and two L2 graph replay"
        if args.pto_mpmd_phase_nz:
            engine += "; same weight addresses use ND for native prefill and tile-local NZ for PTO decode"
        if args.native_nd_reference:
            engine = "native vLLM-Ascend eager with shared ND weight banks; correctness oracle"
        if args.pto_mlp_build_root:
            engine = "vLLM-Ascend native TP2 model with selected PyPTO MLP kernels, mode-3 FULL_DECODE_ONLY"
        result = {
            "engine": engine,
            "generation_is_teacher_forced": bool(args.teacher_validation_reference),
            "performance_accepted": False if args.teacher_validation_reference else None,
            "teacher_validation_reference": str(args.teacher_validation_reference)
            if args.teacher_validation_reference
            else None,
            "ascend_compilation_config": additional_config.get("ascend_compilation_config", {}),
            "pto_mlp_kernels": bool(args.pto_mlp_build_root),
            "pto_gdn_core": pto_gdn or pto_model or bool(args.pto_mpmd_socket),
            "pto_full_model": pto_model or bool(args.pto_mpmd_socket),
            "pto_full_mpmd": bool(args.pto_mpmd_socket),
            "correctness_accepted": False if args.pto_mpmd_socket else None,
            "pto_mlp_fusion": os.environ.get("VLLM_ASCEND_PYPTO_QWEN38_MLP_FUSION", "0") == "1",
            "diagnostic_timing": args.diagnostics_dir is not None
            or bool(args.pto_mpmd_socket)
            or args.native_nd_reference,
            "physical_devices": [int(x) for x in os.environ["ASCEND_RT_VISIBLE_DEVICES"].split(",")],
            "parallelism": f"PP={args.pipeline_parallel_size}, TP={args.tensor_parallel_size}",
            "startup_seconds": startup_seconds,
            "native_cache_blocks": 8,
            "hccl_buffer_size_mb": os.environ.get("HCCL_BUFFSIZE"),
            "async_scheduling": args.async_scheduling,
            "max_model_len": args.max_model_len,
            "chunked_prefill": False,
            "language_model_only": True,
            "dummy_capture_uses_token_ids": True,
            "requested_graph_mode": "FULL_DECODE_ONLY" if args.graph else "NONE",
            "enforce_eager": not args.graph,
            "runner_compile_mode": "VLLM_COMPILE" if args.graph and args.compilation_mode == 3 else "NONE",
            "model_torch_dynamo_compile_requested": args.graph
            and args.compilation_mode == 3
            and not args.aclgraph_only,
            "aclgraph_only_model_override": args.aclgraph_only,
            "runner_compile_mode_numeric": args.compilation_mode if args.graph else 0,
            "standard_mode3_candidate": args.graph and args.compilation_mode == 3 and not args.aclgraph_only,
            "native_operator_fallbacks": []
            if args.pto_mpmd_socket
            else ["GDN sigmoid gating/recurrent split"]
            + (["QK/RoPE split for local 24-head shape"] if args.tensor_parallel_size == 1 else []),
            "prompt_token_ids": tokens,
            "tpot_definition": "1000 * (last_token_ts - first_token_ts) / (output_tokens - 1)",
            "measurement_scope": "pretokenized LLM.generate call to CPU tokens, including engine scheduling and sampling",
            "requests": requests,
        }
        args.result.write_text(json.dumps(result, indent=2))
        print(json.dumps(requests[-1]), flush=True)
    if args.validation_prompts:
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        result["validation_requests"] = []
        for prompt in json.loads(args.validation_prompts.read_text()):
            prompt_ids = tokenizer.encode(prompt)
            output = llm.generate([{"prompt_token_ids": prompt_ids}], params, use_tqdm=False)[0]
            result["validation_requests"].append(
                {
                    "prompt": prompt,
                    "input_tokens": len(prompt_ids),
                    "token_ids": list(output.outputs[0].token_ids),
                    "text": output.outputs[0].text,
                }
            )
        args.result.write_text(json.dumps(result, indent=2))
    if args.runtime_stats:
        from vllm_ascend.pypto.qwen3_8_stats import collect_pto_stats

        result["pto_worker_stats"] = llm.apply_model(collect_pto_stats)
        args.result.write_text(json.dumps(result, indent=2))

    if args.expected_native:
        expected = json.loads(args.expected_native.read_text())
        actual_sequences = [item["token_ids"] for item in result["requests"]]
        expected_sequences = [item["token_ids"] for item in expected["requests"]]
        actual_validation = [item["token_ids"] for item in result.get("validation_requests", [])]
        expected_validation = [item["token_ids"] for item in expected.get("validation_requests", [])]
        stats = result.get("pto_worker_stats", [])
        aliases = [value for worker in stats for value in worker.get("cache_aliases", {}).values()]
        verification = {
            "native_reference": str(args.expected_native),
            "measured_sequences_match_native": actual_sequences == expected_sequences,
            "validation_sequences_match_native": actual_validation == expected_validation,
            "native_aliases_verified": len(aliases),
            "all_cache_aliases_match": bool(aliases) and all(value["same_storage"] for value in aliases),
        }
        result["verification"] = verification
        args.result.write_text(json.dumps(result, indent=2))
        if not verification["measured_sequences_match_native"]:
            raise AssertionError("Measured greedy outputs differ from the recorded native baseline.")
        if not verification["validation_sequences_match_native"]:
            raise AssertionError(
                "Variable-length/slot-reuse outputs differ from the recorded native baseline."
            )
        if len(aliases) != 64 or not verification["all_cache_aliases_match"]:
            raise AssertionError("The full PTO path must borrow all 64 native layer cache allocations.")

    if args.swimlane_output:
        from functools import partial
        from vllm_ascend.pypto.qwen3_8_mpmd import arm_profile, profile_status

        result["profile_requests"] = []
        for mode in ("timing", "deps"):
            llm.apply_model(partial(arm_profile, mode=mode))
            for attempt in range(8):
                profile_params = SamplingParams(temperature=0, max_tokens=2, ignore_eos=True)
                output = llm.generate([{"prompt_token_ids": tokens}], profile_params, use_tqdm=False)[0]
                status = llm.apply_model(profile_status)
                result["profile_requests"].append(
                    {
                        "mode": mode,
                        "attempt": attempt,
                        "status": status,
                        "token_ids": list(output.outputs[0].token_ids),
                        "excluded_from_performance": True,
                    }
                )
                if all(item["pending"] is None for item in status):
                    break
            else:
                raise RuntimeError("No matching native physical cache slots for the deps capture.")
        result["toolkit_capture"] = str(args.swimlane_output / "toolkit")
        args.result.write_text(json.dumps(result, indent=2))

    if args.pto_mpmd_socket:
        from vllm_ascend.pypto.qwen3_8_mpmd import close_mpmd

        llm.apply_model(close_mpmd)


if __name__ == "__main__":
    main()
