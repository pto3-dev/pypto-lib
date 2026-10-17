# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Combine TP reduce/norm and existing MLP MIX without cohort admission."""

import ast
import hashlib
import json
from pathlib import Path
import re
from simpler_setup.static_program import StaticStep, _assemble, _leaf

NORM_BINDINGS = (*range(8), 12, 13, 14)
UP_BINDINGS = (4, 8, 9, 10)
HELPER = r"""
static __aicore__ inline volatile __gm__ int64_t* mlp_event(__gm__ int64_t* args) {
    auto* tensor=reinterpret_cast<__gm__ Tensor*>(args[11]);
    return reinterpret_cast<volatile __gm__ int64_t*>(tensor->buffer.addr)+tensor->start_offset;
}
static __aicore__ inline bool mlp_input_ready(__gm__ int64_t* args) {
    auto* event=mlp_event(args);
    const int64_t epoch=args[15];
    const uint64_t begin=get_sys_cnt_aicore();
    do {
        // SPR read for cancellation, never an LSU access to an MMIO address.
        if (static_cast<uint32_t>(read_reg(RegId::DATA_MAIN_BASE))==AICORE_EXIT_SIGNAL ||
            get_sys_cnt_aicore()-begin>1000000) {
            auto ctx=get_async_ctx(args);
            pto2::detail::defer_error(ctx,SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED);
            pto2::detail::defer_flush(ctx);
            return false;
        }
        dcci((__gm__ void*)event,cache_line_t::SINGLE_CACHE_LINE);
    } while (*event!=epoch);
    return true;
}
"""


def _render(root, lane):
    norm = StaticStep("reduce_norm", root / "kernels/aiv/reduce_add_norm.cpp", NORM_BINDINGS)
    up = StaticStep("gate_up", root / f"kernels/{lane}/project_mlp_kernel_{lane}.cpp", UP_BINDINGS)
    leaves = [_leaf(up, 1)]
    if lane == "aiv":
        leaves.insert(0, _leaf(norm, 0))
        body = """
    if (get_block_idx(args)==0 && get_sub_block_id(args)==0) {
        static_leaf_0::entry_0(args);
        dsb(DSB_DDR);
        auto* event=mlp_event(args);
        *event=args[15];
        dcci((__gm__ void*)event,SINGLE_CACHE_LINE,CACHELINE_OUT);
    }
    static_leaf_1::entry_1(args);
"""
    else:
        source = leaves[0]
        marker = "int64_t v5, int32_t v6, int32_t v7) {"
        if source.count(marker) != 1:
            raise ValueError("MLP Cube ABI changed")
        source = source.replace(marker, "int64_t v5, int32_t v6, int32_t v7, __gm__ int64_t* region_args) {")
        marker = "  TLOAD(v49, v54);\n  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);\n  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);"
        if source.count(marker) != 1:
            raise ValueError("MLP activation load changed")
        source = source.replace(marker, "")
        marker = "    TLOAD(v74, v87);\n    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);"
        if source.count(marker) != 1:
            raise ValueError("MLP weight prefetch boundary changed")
        source = source.replace(
            marker,
            marker
            + """
    if (v55==v42*v44+v46) {
      if (!mlp_input_ready(region_args)) return;
      TLOAD(v49,v54);
      set_flag(PIPE_MTE2,PIPE_MTE1,EVENT_ID0);
      wait_flag(PIPE_MTE2,PIPE_MTE1,EVENT_ID0);
    }
""",
        )
        marker = "__pypto_l2_cache_offset, __pypto_spmd_block_idx, __pypto_spmd_block_num);"
        if source.count(marker) != 1:
            raise ValueError("MLP Cube forwarding changed")
        source = source.replace(marker, marker[:-2] + ", args);")
        leaves[0] = source
        body = "    static_leaf_1::entry_1(args);\n"
    entry = (
        HELPER
        + '\nextern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t* args) {\n'
        + body
        + "\n}\n"
    )
    return '#include "aicore/aicore.h"\n#include "async_kernel_api.h"\n' + _assemble(leaves, entry)


def install_mlp_input(build):
    from .device_program_layout import MLP_EVENTS_OFFSET

    reports = []
    for rank in (0, 1):
        root = Path(build) / "next_levels" / f"rank{rank}"
        cfg = root / "kernel_config.py"
        tree = ast.parse(cfg.read_text())
        assignment = next(
            n
            for n in tree.body
            if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "KERNELS" for t in n.targets)
        )
        entries = {}
        for node in assignment.value.elts:
            fields = {ast.literal_eval(k): v for k, v in zip(node.keys, node.values)}
            entries[ast.literal_eval(fields["name"])] = ast.literal_eval(fields["func_id"])
        up, vec, norm = (
            entries[n] for n in ("project_mlp_kernel_aic", "project_mlp_kernel_aiv", "reduce_add_norm")
        )
        new_id = max(entries.values()) + 1
        tree.body.insert(
            1, ast.parse("from simpler_setup.environment import PROJECT_ROOT as _MLP_SDK_ROOT").body[0]
        )
        for offset, lane in enumerate(("aic", "aiv")):
            name = f"mlp_input_region_{lane}"
            (root / f"kernels/{lane}/{name}.cpp").write_text(_render(root, lane))
            item = (
                "{'func_id':"
                + str(new_id + offset)
                + ",'name':'"
                + name
                + "','source':str(_ROOT_DIR/'kernels'/'"
                + lane
                + "'/'"
                + name
                + ".cpp'),'core_type':'"
                + lane
                + "','extra_include_dirs':[str(_MLP_SDK_ROOT/'src/a2a3/platform/onboard/aicore'),str(_MLP_SDK_ROOT/'src/a2a3/platform/include')],'signature':[_D.IN,_D.IN,_D.INOUT,_D.IN,_D.INOUT,_D.INOUT,_D.INOUT,_D.IN,_D.IN,_D.INOUT,_D.INOUT,_D.INOUT,_D.SCALAR,_D.SCALAR,_D.SCALAR,_D.SCALAR]}"
            )
            assignment.value.elts.append(ast.parse(item, mode="eval").body)
        cpp = root / f"orchestration/rank{rank}.cpp"
        text = cpp.read_text()
        pattern = re.compile(
            r"CoreTaskArgs (params_t\d+);.*?TaskOutputTensors (task_\d+_outs) = rt_submit_(task|aiv_task|aic_task)\(([^;]+)\);",
            re.S,
        )
        calls = list(pattern.finditer(text))
        edits = []
        sites = []
        for i, call in enumerate(calls):
            if not re.search(r"MixedKernels mixed_\d+ = \{" + f"{up}, {vec}, {vec}" + r"\};", call[0]):
                continue
            previous = calls[i - 1]
            if previous[3] != "aiv_task" or previous[4] != f"{norm}, {previous[1]}":
                raise ValueError("Expected preceding post-attention norm")
            n_args = re.findall(re.escape(previous[1]) + r"\.add_(input|inout|scalar)\((\w+)\);", previous[0])
            u_args = re.findall(re.escape(call[1]) + r"\.add_(input|inout)\((\w+)\);", call[0])
            if len(n_args) != 11 or len(u_args) != 4 or n_args[4][1] != u_args[0][1]:
                raise ValueError("Invalid norm-to-MLP bindings")
            if [a[0] for a in n_args[8:]] != ["scalar"] * 3:
                raise ValueError("Unexpected CommContext packet")
            for old in (previous, call):
                if re.search(r"\b" + old[2] + r"\b", text[: old.start()] + text[old.end() :]):
                    raise ValueError("Internal handle escapes")
            site = len(sites)
            lines = [
                f"Tensor mlp_input_state_{site}=program_control_view(ext_device_program_storage,{MLP_EVENTS_OFFSET}+(group_index*4+{site})*128,128,DataType::INT64);",
                f"CoreTaskArgs mlp_input_args_{site};",
            ]
            for direction, var in n_args[:8] + u_args[1:]:
                lines.append(f"mlp_input_args_{site}.add_{direction}({var});")
            lines.append(f"mlp_input_args_{site}.add_inout(mlp_input_state_{site});")
            for _, var in n_args[8:]:
                lines.append(f"mlp_input_args_{site}.add_scalar({var});")
            lines.append(f"mlp_input_args_{site}.add_scalar(attention_token_epoch);")
            grid = re.search(re.escape(call[1]) + r"\.launch_spec.set_block_num\((\w+)\);", call[0])[1]
            lines += [
                f"mlp_input_args_{site}.launch_spec.set_block_num({grid});",
                f"mlp_input_args_{site}.launch_spec.set_static_block_mapping(true);",
                f"mlp_input_args_{site}.set_allow_early_resolve(true);",
                f"MixedKernels mlp_input_kernels_{site}={{{new_id},{new_id + 1},{new_id + 1}}};",
                f"TaskOutputTensors mlp_input_result_{site}=rt_submit_task(mlp_input_kernels_{site},mlp_input_args_{site});",
            ]
            edits += [(previous.start(), previous.end(), ""), (call.start(), call.end(), "\n".join(lines))]
            sites.append(dict(site=site, norm_bindings=n_args, up_bindings=u_args, grid=grid))
        if len(sites) != 4:
            raise ValueError(f"Expected four MLP sites, got {len(sites)}")
        for start, end, replacement in sorted(edits, reverse=True):
            text = text[:start] + replacement + text[end:]
        if "set_require_sync_start(true)" in text:
            raise ValueError("Cohort admission forbidden")
        cpp.write_text(text)
        cfg.write_text(ast.unparse(ast.fix_missing_locations(tree)) + "\n")
        reports.append(
            dict(
                rank=rank,
                sites=sites,
                layers=64,
                external_tasks_saved=64,
                core_invocations_saved=64,
                new_ids=[new_id, new_id + 1],
                no_global_drain=True,
                no_sync_start=True,
                single_writer_event=True,
                first_weight_loads_before_norm_wait=True,
                original_mad_order=True,
                l1_extra_bytes=0,
                sha256=hashlib.sha256(text.encode()).hexdigest(),
            )
        )
    (Path(build) / "mlp_input_manifest.json").write_text(json.dumps(reports, indent=2) + "\n")
