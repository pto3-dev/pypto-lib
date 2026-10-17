# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Closed attention score/softmax/PV/gate device program over original kernels."""

import ast
import hashlib
import json
from pathlib import Path
import re
import struct
from collections import defaultdict

from simpler_setup.device_program import Invocation, LeafSignature, compile_device_program
from simpler_setup.environment import PROJECT_ROOT

from .device_program_layout import (
    PLAN_BYTES,
    FRAMES_OFFSET,
    FRAMES_BYTES,
    EVENTS_OFFSET,
    EVENTS_BYTES,
    TRACE_OFFSET,
    TRACE_BYTES_PER_LAYER,
    STORAGE_BYTES,
)


def attention_plan(ids, bindings, root_count, *, record_trace):
    normq, append, sc, sv, soft, pv, vv, gate = ids
    nodes = [
        Invocation("normq0", normq, "aiv0", 0, 0, 2, bindings[0]),
        Invocation("normq1", normq, "aiv0", 1, 1, 2, bindings[0]),
        Invocation("append", append, "aiv1", 0, 0, 1, bindings[1]),
    ]
    for block in range(2):
        nodes.append(
            Invocation(
                f"score_c{block}", sc, "aic", block, block, 2, bindings[2], ("normq0", "normq1", "append")
            )
        )
        for lane in range(2):
            nodes.append(Invocation(f"score_v{block}_{lane}", sv, f"aiv{lane}", block, block, 2, bindings[2]))
    for block in range(2):
        predecessors = tuple(
            f"score_v{b}_{lane}" for b in (range(2) if block == 0 else [1]) for lane in range(2)
        )
        nodes.append(Invocation(f"soft{block}", soft, "aiv0", block, block, 2, bindings[3], predecessors))
    for block in range(2):
        predecessors = ("soft0",) if block == 0 else ("soft0", "soft1")
        nodes.append(Invocation(f"pv_c{block}", pv, "aic", block, block, 2, bindings[4], predecessors))
        for lane in range(2):
            nodes.append(
                Invocation(
                    f"pv_v{block}_{lane}",
                    vv,
                    f"aiv{lane}",
                    block,
                    block,
                    2,
                    bindings[4] + (bindings[5][0], bindings[5][2]),
                    predecessors,
                )
            )
    signatures = [
        LeafSignature(fid, lane, argc)
        for fid, lane, argc in [
            (normq, "aiv", 10),
            (append, "aiv", 11),
            (sc, "aic", 8),
            (sv, "aiv", 8),
            (soft, "aiv", 2),
            (pv, "aic", 8),
            (vv, "aiv", 10),
            (gate, "aiv", 3),
        ]
    ]
    compiled = compile_device_program(
        nodes,
        signatures,
        grid=2,
        available_clusters=2,
        root_arg_count=root_count,
        epoch_arg=root_count - 1,
        record_trace=record_trace,
    )
    # Step reserved[0] suppresses publication only for events with no remote reader.
    image = bytearray(compiled.image)
    header = struct.unpack_from("<16I", image, 0)
    waited = {struct.unpack_from("<I", image, header[11] + 4 * i)[0] for i in range(header[6])}
    for i in range(header[4]):
        struct.pack_into("<I", image, header[9] + 64 * i + 40, int(i not in waited))
    from dataclasses import replace

    return replace(compiled, image=bytes(image))


HELPER = r"""
static Tensor program_control_view(const Tensor& storage, uint64_t byte_offset,
                                   uint32_t nbytes, DataType dtype) {
    const auto width = get_element_size(dtype);
    if (width == 0 || nbytes % width != 0) __builtin_trap();
    Tensor view = native_pool_view(storage, byte_offset, dtype, {nbytes / static_cast<uint32_t>(width)});
    view.buffer.addr += view.start_offset * width;
    view.buffer.size = nbytes;
    view.start_offset = 0;
    return view;
}
"""


def install_attention_program(build):
    build = Path(build)
    reports = []
    sdk = PROJECT_ROOT / "src/a2a3/runtime/tensormap_and_ringbuffer"
    for rank in (0, 1):
        root = build / "next_levels" / f"rank{rank}"
        config = root / "kernel_config.py"
        tree = ast.parse(config.read_text())
        assign = next(
            n
            for n in tree.body
            if isinstance(n, ast.Assign)
            and any(isinstance(x, ast.Name) and x.id == "KERNELS" for x in n.targets)
        )
        entries = {}
        for node in assign.value.elts:
            d = {ast.literal_eval(k): v for k, v in zip(node.keys, node.values)}
            entries[ast.literal_eval(d["name"])] = ast.literal_eval(d["func_id"])
        names = [
            "norm_rope_q_kernel",
            "norm_rope_k_append_kernel",
            "compute_scores_kernel_aic",
            "compute_scores_kernel_aiv",
            "softmax_scores_kernel",
            "compute_values_kernel_aic",
            "compute_values_kernel_aiv",
            "gate_attention_kernel",
        ]
        ids = tuple(entries[n] for n in names)
        new_id = max(entries.values()) + 1
        for offset, lane in enumerate(("aic", "aiv")):
            path = root / "kernels" / lane / f"device_attention_program_{lane}.cpp"
            text = (sdk / f"kernels/device_program/device_program_{lane}.cpp").read_text()
            text = text.replace(
                '#include "device_program_body.h"',
                '#include "' + str(Path(__file__).with_name("device_program_body.h")) + '"',
            )
            path.write_text(text)
            expression = (
                "{'func_id':"
                + str(new_id + offset)
                + ",'name':'device_attention_program_"
                + lane
                + "','source':str(_ROOT_DIR/'kernels'/'"
                + lane
                + "'/'device_attention_program_"
                + lane
                + ".cpp'),'core_type':'"
                + lane
                + "','extra_include_dirs':"
                + repr(
                    [
                        str(PROJECT_ROOT / "src/a2a3/platform/onboard/aicore"),
                        str(PROJECT_ROOT / "src/a2a3/platform/include"),
                    ]
                )
                + ",'signature':[_D.IN]+[_D.INOUT]*3+[_D.INOUT]*20+[_D.SCALAR]}"
            )
            # The signature is replaced below with the actual deduplicated packet.
            assign.value.elts.append(ast.parse(expression, mode="eval").body)
        cpp = root / "orchestration" / f"rank{rank}.cpp"
        text = cpp.read_text()
        if "ext_device_program_storage" not in text:
            raise ValueError("Resident plan storage missing")
        pattern = re.compile(
            r"CoreTaskArgs (params_t\d+);.*?TaskOutputTensors (task_\d+_outs) = rt_submit_(task|aiv_task|aic_task)\(([^;]+)\);",
            re.S,
        )
        calls = []
        for m in pattern.finditer(text):
            payload = m[0]
            match = re.search(r"MixedKernels mixed_\d+ = \{(\d+), (\d+), (\d+)\};", payload)
            if match:
                kid = tuple(map(int, match.groups()))
                role = (
                    "score"
                    if kid == (ids[2], ids[3], ids[3])
                    else "pv"
                    if kid == (ids[5], ids[6], ids[6])
                    else None
                )
            else:
                direct = re.match(r"(\d+),", m[4])
                fid = int(direct[1]) if direct else -1
                role = (
                    "normq"
                    if fid == ids[0]
                    else "append"
                    if fid == ids[1]
                    else "soft"
                    if fid == ids[4]
                    else "gate"
                    if fid == ids[7]
                    else None
                )
            if role:
                calls.append((role, m))
        if [role for role, _ in calls] != ["normq", "append", "score", "soft", "pv", "gate"]:
            raise ValueError("Expected one closed attention site in sixteen-group loop")
        packets = []
        root_variables = []
        directions = {}
        for role, m in calls:
            args = re.findall(re.escape(m[1]) + r"\.add_(input|inout)\((\w+)\);", m[0])
            packets.append(args)
            if len(args) != {"normq": 10, "append": 11, "score": 8, "soft": 2, "pv": 8, "gate": 3}[role]:
                raise ValueError("Unexpected leaf ABI")
            for direction, var in args:
                if var not in root_variables:
                    root_variables.append(var)
                directions[var] = (
                    "inout" if direction == "inout" or directions.get(var) == "inout" else "input"
                )
            remainder = text[: m.start()] + text[m.end() :]
            if re.search(r"\b" + m[2] + r"\b", remainder):
                raise ValueError("Internal task handle escapes segment")
        # Data-only leaf wrappers accept equivalent contiguous reshapes.
        # GM pipe descriptors retain their complete shape for per-block slicing.
        for leaf_name in names:
            for leaf_path in (root / "kernels").glob("*/" + leaf_name + ".cpp"):
                entry = leaf_path.read_text().split("// --- Kernel entry point ---")[-1]
                metadata = {
                    (name, field)
                    for name, field in re.findall(r"(\w+_tensor)->(\w+)", entry)
                    if field not in ("buffer", "start_offset")
                }
                if metadata - {("__gm_pipe_buffer_tensor", "shapes")}:
                    raise ValueError(f"Leaf reads non-data Tensor metadata: {leaf_name}: {metadata}")
        assignments = defaultdict(set)
        for target, source in re.findall(
            r"^\s*(?:Tensor )?(\w+)\s*=\s*(\w+)(?:\.reshape\([^;]*\))?;", text, re.M
        ):
            assignments[target].add(source)
        aliases = {}
        for _ in range(len(assignments)):
            changed = False
            for target, sources in assignments.items():
                origins = {aliases.get(value, value) for value in sources}
                if len(origins) == 1:
                    origin = next(iter(origins))
                    if origin != target and aliases.get(target) != origin:
                        aliases[target] = origin
                        changed = True
            if not changed:
                break
        representatives = {}
        original_variables = list(root_variables)
        for var in original_variables:
            representatives.setdefault(aliases.get(var, var), var)
        canonical = {var: representatives[aliases.get(var, var)] for var in original_variables}
        packets = [[(direction, canonical[var]) for direction, var in args] for args in packets]
        for var in original_variables:
            if directions[var] == "inout":
                directions[canonical[var]] = "inout"
        root_variables = list(representatives.values())
        limits = (PROJECT_ROOT / "src/common/task_interface/arg_direction.h").read_text()
        max_tensors = int(re.search(r"CORE_MAX_TENSOR_ARGS = (\d+)", limits)[1])
        if 4 + len(root_variables) > max_tensors:
            raise ValueError(
                f"Device program root exceeds SDK CoreTaskArgs tensor capacity: {4 + len(root_variables)} > {max_tensors}"
            )
        bindings = tuple(tuple(4 + root_variables.index(var) for _, var in args) for args in packets)
        root_count = 4 + len(root_variables) + 1
        plan = attention_plan(ids, bindings, root_count, record_trace=True)
        if len(plan.image) > PLAN_BYTES or len(plan.event_names) != 17:
            raise ValueError("Plan storage capacity mismatch")
        (root / "attention_program.bin").write_bytes(plan.image)
        signature = (
            ["_D.IN", "_D.INOUT", "_D.INOUT", "_D.INOUT"]
            + [("_D.INOUT" if directions[v] == "inout" else "_D.IN") for v in root_variables]
            + ["_D.SCALAR"]
        )
        for node in assign.value.elts[-2:]:
            for k, v in zip(node.keys, node.values):
                if isinstance(k, ast.Constant) and k.value == "signature":
                    v.elts = [ast.parse(x, mode="eval").body for x in signature]
        # Only the vector PV entry has the additional gate input/output bindings.
        for node in assign.value.elts:
            fields = {ast.literal_eval(k): v for k, v in zip(node.keys, node.values)}
            if ast.literal_eval(fields["name"]) == "compute_values_kernel_aiv":
                fields["signature"].elts.extend(ast.parse("[_D.IN,_D.INOUT]", mode="eval").body.elts)
        leaf = root / "kernels/aiv/compute_values_kernel_aiv.cpp"
        source = leaf.read_text()
        source = source.replace(
            "int32_t v10, int32_t v11) {",
            "int32_t v10, int32_t v11, __gm__ bfloat16_t* gates, __gm__ bfloat16_t* gated) {",
        )
        marker = "    TPOP<TPipe<0, Direction::DIR_C2V, 16384, 2, 2, true>, Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, TileSplitAxis::TILE_NO_SPLIT>(v30, v35);"
        if source.count(marker) != 1:
            raise ValueError("Unsupported PV receiver ABI")
        source = source.replace(
            marker,
            "    pv_gate::Arena continuation;\n    pv_gate::prefetch(continuation,gates+v32*256);\n" + marker,
        )
        marker = "    TSTORE(v46, v36);"
        if source.count(marker) != 1:
            raise ValueError("PV store mapping changed")
        source = source.replace(marker, marker + "\n    pv_gate::finish(continuation,gated+v32*256);")
        marker = "    compute_values_kernel_aiv(probability__ssa_v0, cache__ssa_v0, table__ssa_v0, lengths__ssa_v0, out__ssa_v0, cf__ssa_v0, of__ssa_v0, __gm_pipe_buffer, __pypto_spmd_block_idx, __pypto_spmd_block_num, __pypto_spmd_subblock_idx);"
        if source.count(marker) != 1:
            raise ValueError("PV entry forwarding changed")
        source = source.replace(
            marker,
            "    auto* gate_descriptor=reinterpret_cast<__gm__ Tensor*>(args[8]);\n    auto* out_descriptor=reinterpret_cast<__gm__ Tensor*>(args[9]);\n    auto* gate_data=reinterpret_cast<__gm__ bfloat16_t*>(gate_descriptor->buffer.addr)+gate_descriptor->start_offset;\n    auto* gated_data=reinterpret_cast<__gm__ bfloat16_t*>(out_descriptor->buffer.addr)+out_descriptor->start_offset;\n"
            + marker[:-2]
            + ",gate_data,gated_data);",
        )
        source = source.replace(
            "// --- ptoas-generated code ---", '#include "pv_gate_arena.h"\n// --- ptoas-generated code ---'
        )
        leaf.write_text(source)
        import shutil

        shutil.copyfile(Path(__file__).with_name("pv_gate_arena.h"), leaf.with_name("pv_gate_arena.h"))
        config.write_text(ast.unparse(ast.fix_missing_locations(tree)) + "\n")
        # Keep view construction and both original GM pipe allocations in scope.
        for _, m in reversed(calls):
            text = text[: m.start()] + text[m.end() :]
        insert = calls[-1][1].end() - sum(m.end() - m.start() for _, m in calls)
        program = f"""\n                        const int program_available = rt_available_cluster_count();
                        if (program_available < 2) __builtin_trap();
                        Tensor attention_program_image = program_control_view(ext_device_program_storage, 0, {PLAN_BYTES}, DataType::UINT8);
                        Tensor attention_program_frames = program_control_view(ext_device_program_storage, {FRAMES_OFFSET}, {FRAMES_BYTES}, DataType::UINT8);
                        Tensor attention_program_events = program_control_view(ext_device_program_storage, {EVENTS_OFFSET}, {EVENTS_BYTES}, DataType::INT64);
                        Tensor attention_program_trace = program_control_view(ext_device_program_storage, {TRACE_OFFSET}+group_index*{TRACE_BYTES_PER_LAYER}, {TRACE_BYTES_PER_LAYER}, DataType::INT64);
                        uint32_t program_epoch_position[1]={{1}};
                        int64_t program_epoch = static_cast<int64_t>(attention_token_epoch)*16+group_index+1;
                        CoreTaskArgs attention_program_args;
                        attention_program_args.add_input(attention_program_image);
                        attention_program_args.add_inout(attention_program_frames);
                        attention_program_args.add_inout(attention_program_events);
                        attention_program_args.add_inout(attention_program_trace);
"""
        for var in root_variables:
            program += f"                        attention_program_args.add_{directions[var]}({var});\n"
        program += f"""                        attention_program_args.add_scalar(program_epoch);
                        attention_program_args.launch_spec.set_block_num(2);
                        attention_program_args.launch_spec.set_static_block_mapping(true);
                        attention_program_args.set_allow_early_resolve(true);
                        MixedKernels attention_program_kernels={{{new_id},{new_id + 1},{new_id + 1}}};
                        TaskOutputTensors attention_program_result=rt_submit_task(attention_program_kernels,attention_program_args);
"""
        text = text[:insert] + program + text[insert:]
        text = text.replace('extern "C" {', HELPER + '\nextern "C" {', 1)
        marker = "        if ((static_cast<int64_t>(t) == 0)) {"
        if text.count(marker) != 1:
            raise ValueError("Expected one graph-stop guard")
        text = text.replace(
            marker,
            marker
            + "\n            uint32_t attention_epoch_index[1]={1};\n            const int64_t attention_token_epoch=get_tensor_data<int32_t>(ext_control_i32,1,attention_epoch_index);",
            1,
        )
        if "set_require_sync_start(true)" in text:
            raise ValueError("No global cohort/drain admission allowed")
        cpp.write_text(text)
        reports.append(
            dict(
                rank=rank,
                original_kernel_ids=list(ids),
                new_kernel_ids=[new_id, new_id + 1],
                nodes_per_layer=17,
                event_names=list(plan.event_names),
                bindings=bindings,
                root_variables=root_variables,
                root_arg_count=root_count,
                grid=2,
                executed_layers=16,
                logical_task_reduction=80,
                physical_dispatch_reduction=320,
                no_global_drain=True,
                original_leaf_sources_unchanged=False,
                pv_gate_ub_continuation=True,
                image_sha256=hashlib.sha256(plan.image).hexdigest(),
            )
        )
    (build / "attention_program_manifest.json").write_text(json.dumps(reports, indent=2) + "\n")


def allocate_program_storage(runtime, build):
    import torch

    host = torch.zeros(2, STORAGE_BYTES, dtype=torch.uint8)
    for rank in (0, 1):
        image = bytearray(
            (Path(build) / "next_levels" / f"rank{rank}" / "attention_program.bin").read_bytes()
        )
        struct.pack_into("<I", image, 60, 0)
        host[rank, : len(image)].copy_(torch.tensor(list(image), dtype=torch.uint8))
    return runtime.alloc_stacked_tensor(host)
