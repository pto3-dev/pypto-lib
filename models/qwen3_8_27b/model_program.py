# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Generate one HOST program dispatching a complete model to both TP ranks."""

import importlib.util
import ast
import os
from math import prod
from pathlib import Path
import sys
import textwrap

from .dense import make_gemm, make_norm, make_pointwise
from .norm_rope import make_norm_rope, make_packed_norm_rope
from .linear_attention import make_conv, make_qk_norm, make_recurrent
from .conv_qk_norm import make_conv_qk_norm
from .selected_gates import make_selected_gates
from .layout import make_gdn_unpack, make_query_gate_unpack
from .attention import make_rope
from .attention_standard import make_standard_attention
from .kv_cache import make_append
from .vocab import make_embedding
from .reusable import reusable_methods
from .probe import static_method
from .device_greedy import make_device_greedy
from .fused_projection import make_packed_silu
from .swiglu_down import make_swiglu_down, make_nd_swiglu_down
from .nz_tiled import make_nz_tiled_gemm
from .tiled_weights import make_tiled_gemm, make_split_tiled_gemm

from .collective_sources import (
    copy_hidden,
    reduction_source,
    gather_logits_source,
    reduction_add_source,
    reduction_add_norm_source,
    debug_source,
)


def build_model(
    spec,
    directory,
    looped=False,
    decode_tiles="baseline",
    decode_tile_map=None,
    adaptive_launch=False,
    refresh_group_views=False,
    cube_bf16_store=False,
    norm_tile=128,
    norm_output_tile=None,
    fuse_reduction_residual=False,
    fuse_reduction_norm=False,
    gdn_decode_views=False,
    early_dispatch=False,
    adaptive_vector_launch=False,
    fuse_swiglu_down=False,
    decode_norm=False,
):
    bypass_weights = spec.tokens == 1 and os.environ.get("PYPTO_QWEN_WEIGHT_CACHE_BYPASS", "0") == "1"
    if decode_tiles not in ("baseline", "wide-k", "balanced"):
        raise ValueError("Unknown decode tile policy")
    defaults = {
        "baseline": {},
        "wide-k": dict(n_tile=64, k_tile=1024),
        "balanced": dict(n_tile=32, k_tile=512),
    }

    def projection(input_dim, output_dim, **kwargs):
        if spec.packed_weight_policy is not None:
            nt, kt, transpose = spec.weight_tile(output_dim, input_dim)
            if spec.weight_nz(output_dim, input_dim):
                if transpose:
                    raise ValueError("NZ weight policy requires K-contiguous logical matrices.")
                return make_nz_tiled_gemm(input_dim, output_dim, nt, kt, **kwargs)
            return make_tiled_gemm(input_dim, output_dim, n_tile=nt, k_tile=kt, transpose=transpose, **kwargs)
        options = {}
        if spec.tokens == 1:
            options = defaults[decode_tiles]
            if decode_tile_map and f"{input_dim}x{output_dim}" in decode_tile_map:
                nt, kt = decode_tile_map[f"{input_dim}x{output_dim}"]
                if nt < 16 or nt % 16 or kt < 256 or kt % 16:
                    raise ValueError("Invalid Cube tile policy")
                options = dict(n_tile=nt, k_tile=kt)
        if spec.tokens == 1 and input_dim == 5120 and output_dim == 17408:
            from .nd_contiguous import make_gemm as make_contiguous_gemm

            return make_contiguous_gemm(
                input_dim,
                output_dim,
                cube_bf16_store=cube_bf16_store,
                weight_cache_bypass=bypass_weights,
                **kwargs,
                **options,
            )
        return make_gemm(
            input_dim,
            output_dim,
            cube_bf16_store=cube_bf16_store,
            weight_cache_bypass=bypass_weights,
            **kwargs,
            **options,
        )

    t = spec.tokens
    fused_upact = fuse_swiglu_down and t == 1 and spec.packed_weight_policy is None
    fused_down = fuse_swiglu_down and t == 1 and not fused_upact
    if fused_down and not spec.fuse_mlp:
        raise ValueError("SwiGLU/down fusion requires fused gate/up projection.")
    bindings = {
        "rows": t,
        "tokens": t,
        "slots": spec.state_slots,
        "requests": 1,
        "boundaries": 2,
        "cache_rows": spec.cache_rows,
        "table_cols": (spec.max_seq + spec.cache_block_size - 1) // spec.cache_block_size,
        "positions": spec.max_seq,
        "attention_rows": t * 12,
    }
    funcs = {
        "norm_plain": make_norm(5120, inline=True, tile=norm_tile, output_tile=norm_output_tile),
        "norm_hidden": make_norm(
            5120, inline=True, native_round=True, tile=norm_tile, output_tile=norm_output_tile
        ),
        "norm_gdn": make_norm(128, weight_offset=0.0, gated=True, inline=True),
        "norm_q": make_norm(256, inline=True),
        "norm_k": make_norm(256, inline=True),
        "norm_head": make_norm(
            5120, inline=True, native_round=True, tile=norm_tile, output_tile=norm_output_tile
        ),
        "add": make_pointwise(5120, "add", inline=True, tile=512),
        "activate": make_pointwise(8704, "silu_mul", inline=True, tile=512),
        "gate_attention": make_pointwise(3072, "sigmoid_mul", inline=True, tile=512),
        "project_gdn": projection(5120, 5120, inline=True),
        "project_z": projection(5120, 3072, inline=True),
        "project_small": make_gemm(
            5120, 48 if spec.replicate_small else 24, inline=True, cube_bf16_store=cube_bf16_store
        ),
        "project_q": projection(5120, 6144, inline=True),
        "project_kv": projection(5120, 512, inline=True),
        "project_attention_out": projection(3072, 5120, inline=True, out_fp32=True),
        "project_up": projection(5120, 8704, inline=True),
        "project_down": projection(8704, 5120, inline=True, out_fp32=True),
        "project_head": projection(5120, 124160, inline=True),
        "convolve": make_conv_qk_norm(inline=True, tokens=t)
        if t == 1
        else make_conv(5120, inline=True, tokens=t),
        "split_gdn": make_gdn_unpack(8, 24, inline=True, tokens=t),
        "normalize_gdn": make_qk_norm(8, inline=True, tokens=t, refine=True),
        "recurrent": make_recurrent(8, 24, "KV", inline=True, tokens=t),
        "split_query": make_query_gate_unpack(12),
        "rope_q": make_rope(12),
        "rope_k": make_rope(2),
        "append": make_append(2),
        "embedding0": make_embedding(0),
        "embedding1": make_embedding(1),
    }
    if decode_norm:
        if t != 1:
            raise ValueError("Physical-row RMSNorm is decode-only.")
        from .vector_norm import make_vector_norm

        funcs.update(
            norm_plain=make_vector_norm(inline=True),
            norm_hidden=make_vector_norm(inline=True, native_round=True),
            norm_head=make_vector_norm(inline=True, native_round=True),
        )
    if spec.native_qk_rope:
        funcs.update(
            norm_rope_q=make_packed_norm_rope(12) if t == 1 else make_norm_rope(12),
            norm_rope_k=make_norm_rope(2),
        )
    funcs.update(
        zip(
            ("compute_scores", "softmax_scores", "compute_values"),
            make_standard_attention(12, 2, spec.max_seq, spec.cache_block_size),
        )
    )
    if spec.fuse_mlp:
        if fused_upact:
            from .nd_gate_up_swiglu import make_gemm as make_upact

            funcs["project_mlp"] = make_upact(
                5120,
                17408,
                inline=True,
                n_tile=16,
                k_tile=5120,
                stage=1,
                inner_k=640,
                inner_stage=2,
                weight_cache_bypass=bypass_weights,
            )
        else:
            funcs["project_mlp"] = projection(5120, 17408, inline=True)
        funcs["activate_packed"] = make_packed_silu(8704, inline=True)
    if fused_down:
        if spec.packed_weight_policy is None:
            nt, kt = (decode_tile_map or {}).get(
                "8704x5120",
                {"baseline": (128, 256), "wide-k": (64, 1024), "balanced": (32, 512)}[decode_tiles],
            )
            if 5120 % nt or 8704 % kt:
                raise ValueError("ND SwiGLU/down tiles must divide the projection dimensions.")
            funcs["project_down"] = make_nd_swiglu_down(nt=nt, kt=kt, inline=True)
        else:
            nt, kt, transpose = spec.weight_tile(5120, 8704)
            if transpose:
                raise ValueError("SwiGLU/down fusion currently uses K-contiguous weight tiles.")
            funcs["project_down"] = make_swiglu_down(nt=nt, kt=kt, inline=True)
    if spec.fuse_gdn_ba:
        if not spec.native_weight_order:
            raise ValueError("BA fusion requires native adjacent B/A weights.")
        from .nd_ba_vector import make_gemm as make_ba_vector

        funcs["project_ba"] = make_ba_vector(5120, 48, inline=True, n_tile=16, k_tile=1024, stage=1)
    if spec.teacher_input:
        funcs["copy_hidden"] = copy_hidden
    if spec.fuse_gdn_qkv_z:
        if spec.packed_weight_policy is None:
            from .nd_split_projection import make_nd_split_gemm

            nt, kt = (decode_tile_map or {}).get("5120x8192", (64, 1024))
            funcs["project_gdn_z"] = make_nd_split_gemm(5120, 5120, 3072, inline=True, n_tile=nt, k_tile=kt)
        else:
            nt, kt, transpose = spec.weight_tile(8192, 5120)
            if transpose:
                raise ValueError("Split GDN projection currently requires untransposed weight tiles.")
            funcs["project_gdn_z"] = make_split_tiled_gemm(
                5120, 5120, 3072, inline=True, n_tile=nt, k_tile=kt
            )
    if spec.fuse_attention_qkv:
        if spec.packed_weight_policy is not None or t != 1:
            raise ValueError("Joint attention is currently shared ND decode only.")
        from .nd_qkv_projection import make_nd_qkv_gemm

        nt, kt = (decode_tile_map or {}).get("5120x7168", (64, 1024))
        funcs["project_attention_qkv"] = make_nd_qkv_gemm(
            5120, 6144, 512, 512, inline=True, n_tile=nt, k_tile=kt, weight_cache_bypass=bypass_weights
        )
    for rank in (0, 1):
        funcs[f"gates{rank}"] = make_selected_gates(
            24,
            48 if spec.replicate_small else 24,
            rank * 24 if spec.replicate_small else 0,
            inline=True,
            tokens=t,
        )
    methods = [
        gather_logits_source(),
        f"""@pl.function(type=pl.FunctionType.InCore)
def publish_hidden(self, x: pl.Tensor[[{t},5120],pl.BF16], out: pl.Out[pl.Tensor[[{spec.max_seq},5120],pl.BF16]]) -> pl.Tensor[[{spec.max_seq},5120],pl.BF16]:
    for row in pl.range({t}):
        value=pl.load(x,[row,0],[1,5120])
        out=pl.store(value,[row,0],out)
    return out
""",
    ]
    for name, fn in funcs.items():
        dims = dict(bindings)
        if name == "norm_gdn":
            dims["rows"] = t * 24
        elif name == "norm_q":
            dims["rows"] = t * 12
        elif name == "norm_k":
            dims["rows"] = t * 2
        elif name == "project_head":
            dims["rows"] = 1
        converted = reusable_methods(name, fn, dims)
        nz_dimensions = {
            "project_mlp": (17408, 5120),
            "project_gdn": (5120, 5120),
            "project_z": (3072, 5120),
            "project_q": (6144, 5120),
            "project_kv": (512, 5120),
            "project_attention_out": (5120, 3072),
            "project_up": (8704, 5120),
            "project_head": (124160, 5120),
        }
        if name in nz_dimensions and spec.weight_nz(*nz_dimensions[name]):
            tree = ast.parse(converted[0])
            weight = next(arg for arg in tree.body[0].args.args if arg.arg == "weight")
            weight.annotation.slice.elts.append(
                ast.Attribute(value=ast.Name(id="pl", ctx=ast.Load()), attr="NZ", ctx=ast.Load())
            )
            converted[0] = ast.unparse(ast.fix_missing_locations(tree))
        if (
            adaptive_launch
            and name.startswith("project_")
            and name not in ("project_gdn_z", "project_attention_qkv", "project_ba")
        ):
            # The specialised wrapper exposes a static output shape.
            for i, method in enumerate(converted):
                tree = ast.parse(method)
                definition = tree.body[0]
                out_arg = next(arg for arg in definition.args.args if arg.arg == "out")
                annotation = out_arg.annotation
                if isinstance(annotation.slice, ast.Subscript):
                    annotation = annotation.slice
                shape = annotation.slice.elts[0]
                columns = ast.literal_eval(shape)[1]
                nt = 128
                if name == "project_ba":
                    nt = 32
                elif spec.packed_weight_policy is not None and name not in ("project_small", "project_ba"):
                    input_dim = ast.literal_eval(
                        next(arg for arg in definition.args.args if arg.arg == "x").annotation.slice.elts[0]
                    )[1]
                    nt = spec.weight_tile(columns, input_dim)[0]
                elif spec.tokens == 1 and name != "project_small":
                    nt = {"baseline": 128, "wide-k": 64, "balanced": 32}[decode_tiles]
                    input_dim = ast.literal_eval(
                        next(arg for arg in definition.args.args if arg.arg == "x").annotation.slice.elts[0]
                    )[1]
                    if decode_tile_map and f"{input_dim}x{columns}" in decode_tile_map:
                        nt = decode_tile_map[f"{input_dim}x{columns}"][0]
                jobs = (dims["rows"] + 15) // 16 * ((columns + nt - 1) // nt)
                changed = False
                for node in ast.walk(tree):
                    if (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "spmd"
                    ):
                        node.args[0] = ast.Name(id="launch_count", ctx=ast.Load())
                        changed = True
                if changed:
                    prefix = ast.parse(
                        f"available_clusters=pl.system.available_cluster_count()\nlaunch_count=pl.min(available_clusters,{jobs})"
                    ).body
                    definition.body = prefix + definition.body
                converted[i] = ast.unparse(ast.fix_missing_locations(tree))
        if adaptive_vector_launch and t == 1 and name == "recurrent":
            kernel = ast.parse(converted[0]).body[0]
            wrapper = ast.parse(converted[1]).body[0]
            grid = next(
                node
                for node in ast.walk(wrapper)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "spmd"
            )
            jobs = ast.unparse(grid.args[0])
            position = next(
                i
                for i, node in enumerate(kernel.body)
                if isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "get_block_idx"
            )
            kernel.body[position].targets = [ast.Name(id="vector_core", ctx=ast.Store())]
            loop = ast.parse(f"for job in pl.range(vector_core,{jobs},pl.tile.get_block_num()): pass").body[0]
            loop.body = kernel.body[position + 1 : -1]
            kernel.body = kernel.body[: position + 1] + [loop, kernel.body[-1]]
            scope_index = next(i for i, node in enumerate(wrapper.body) if isinstance(node, ast.With))
            prefix = ast.parse(
                f"vector_available=pl.system.available_aiv_count()\nvector_launch_count=pl.min(vector_available,{jobs})"
            ).body
            wrapper.body[scope_index:scope_index] = prefix
            grid.args[0] = ast.Name(id="vector_launch_count", ctx=ast.Load())
            converted = [ast.unparse(ast.fix_missing_locations(node)) for node in [kernel, wrapper]]
        methods.extend(converted)
    if spec.device_greedy:
        methods.append(static_method("greedy", make_device_greedy(inline=True), dict(rows=1)))
    if spec.debug_count:
        methods.append(debug_source(spec))
    methods.extend(
        reduction_source(name, spec, dtype)
        for name, dtype in (("reduce_embedding", "BF16"), ("reduce_partial", "FP32"))
    )
    if fuse_reduction_residual:
        methods.append(reduction_add_source(spec))
    if fuse_reduction_norm and t == 1:
        methods.append(reduction_add_norm_source(spec, norm_tile, norm_output_tile))
    data_type = f"pl.InOut[pld.DistributedTensor[[{spec.reductions * t},5120],pl.FP32]]"
    signal_type = f"pl.InOut[pld.DistributedTensor[[{spec.reductions * 2},1],pl.INT32]]"
    for rank in (0, 1):
        lines = ["@pl.function(type=pl.FunctionType.Orchestration)", f"def rank{rank}(self,"]
        for name, shape, dtype, direction in spec.arguments:
            annotation = f"pl.Tensor[{list(shape)!r},pl.{dtype}]"
            if direction:
                annotation = f"pl.{direction}[{annotation}]"
            lines.append(f"    {name}: {annotation},")
        lines += [
            "    logit_data: pl.InOut[pld.DistributedTensor[[1,124160],pl.FP32]],",
            "    logit_signal: pl.InOut[pld.DistributedTensor[[2,1],pl.INT32]],",
            f"    data: {data_type},",
            f"    signal: {signal_type},",
            ") -> pl.Tensor[[1,8],pl.FP32]:" if spec.device_greedy else ") -> pl.Tensor[[1,248320],pl.FP32]:",
        ]
        view_number = 0
        indent = "    "

        def emit(line):
            nonlocal view_number
            statement = ast.parse(line).body[0]
            for call in ast.walk(statement):
                if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
                    continue
                if not isinstance(call.func.value, ast.Name) or call.func.value.id != "self":
                    continue
                for position, argument in enumerate(call.args):
                    if isinstance(argument, (ast.Call, ast.BinOp, ast.UnaryOp)):
                        name = f"argument_view_{view_number}"
                        view_number += 1
                        lines.append(indent + name + "=" + ast.unparse(argument))
                        call.args[position] = ast.Name(id=name, ctx=ast.Load())
            lines.append(textwrap.indent(ast.unparse(statement), indent))

        def view(name, base, off, shape):
            emit(f"{name}=pl.reshape(pl.slice({base},[{prod(shape)}],[{off}]),{list(shape)!r})")

        for name, pool, offset, shape, dtype in spec.native_cache_layout:
            emit(f"{name}={pool}")
        if spec.pack_host_controls:
            view("ids", "control_i64", 0, (t,))
            view("positions", "control_i64", t, (t,))
            view("kv_slots", "control_i64", 2 * t, (t,))
            for name, off, shape in [
                ("metadata", 0, (4,)),
                ("starts", 4, (2,)),
                ("indices", 6, (1,)),
                ("initial", 7, (1,)),
                ("table", 8, (1, (spec.max_seq + spec.cache_block_size - 1) // spec.cache_block_size)),
                ("lengths", 8 + (spec.max_seq + spec.cache_block_size - 1) // spec.cache_block_size, (1,)),
            ]:
                view(name, "control_i32", off, shape)

        def scratch_views():
            for key, (off, shape) in spec.scratch.items():
                if spec.independent_scratch:
                    emit(f"{key}=scratch_{key}")
                else:
                    view(key, "scratch", off, shape)
            for key, (off, shape) in spec.floating.items():
                if spec.independent_scratch:
                    emit(f"{key}=float_{key}")
                else:
                    view(key, "float_scratch", off, shape)

        scratch_views()
        if spec.teacher_input:
            emit("hidden=self.copy_hidden(teacher_hidden,hidden)")
        else:
            emit(f"hidden=self.embedding{rank}(ids,embedding,hidden)")
            emit("hidden,data,signal=self.reduce_embedding(hidden,hidden,data,signal,metadata,0)")
        if looped:
            if len(set(spec.group_counts.values())) != 1:
                raise ValueError("Device group loop requires equal packed group sizes.")
            lines.append(f"    for group_index in pl.range({spec.layers // 4}):")
            indent = "        "
            if refresh_group_views:
                # Recreate views from their resident allocations for each group.
                scratch_views()
            if spec.layers == 4:
                emit("group_weights=weights0")
            else:
                select = []
                for group in range(spec.layers // 4 - 1):
                    select.append(
                        ("if" if group == 0 else "elif")
                        + f" group_index=={group}:\n    group_weights=weights{group}"
                    )
                select.append(f"else:\n    group_weights=weights{spec.layers // 4 - 1}")
                emit("\n".join(select))
        # Select native per-layer views; no stacked cache pool is allocated.
        native_groups = []
        for group in range(spec.layers // 4):
            lines_for_group = []
            for local_index in range(4):
                absolute = group * 4 + local_index
                if local_index == 3:
                    lines_for_group.extend(
                        [f"keys{local_index}=l{absolute}_keys", f"values{local_index}=l{absolute}_values"]
                    )
                else:
                    lines_for_group.extend(
                        [
                            f"conv{local_index}=l{absolute}_conv_state",
                            f"state{local_index}=l{absolute}_recurrent_state",
                        ]
                    )
                width = 3 + (spec.max_seq + spec.cache_block_size - 1) // spec.cache_block_size
                lines_for_group.append(
                    f"meta{local_index}=pl.reshape(pl.slice(native_metadata,[1,{width}],[{absolute},0]),[{width}])"
                )
            if spec.layers == 4:
                for native_line in lines_for_group:
                    emit(native_line)
            else:
                keyword = "if" if group == 0 else "elif" if group < spec.layers // 4 - 1 else "else"
                header = f"{keyword} group_index=={group}:" if keyword != "else" else "else:"
                native_groups.append(header + "\n" + "\n".join("    " + line for line in lines_for_group))
        if native_groups:
            emit("\n".join(native_groups))
        for index in range(4 if looped else spec.layers):
            names = {}
            for (layer, key), (group, off, shape) in spec.weights.items():
                if layer != index:
                    continue
                name = f"w{index}_" + key.replace(".", "_")
                if "layernorm" in key:
                    shape = (1, 5120)
                elif key.endswith("norm.weight"):
                    shape = (1, shape[0])
                elif len(shape) == 2 and shape[0] >= 128:
                    shape = spec.projection_weight_shape(shape)
                view(name, "group_weights" if looped else f"weights{group}", off, shape)
                names[key] = name
            emit(f"indices=pl.slice(meta{index},[1],[0])")
            emit(f"initial=pl.slice(meta{index},[1],[1])")
            emit(f"lengths=pl.slice(meta{index},[1],[2])")
            emit(
                f"table=pl.reshape(pl.slice(meta{index},[{(spec.max_seq + spec.cache_block_size - 1) // spec.cache_block_size}],[3]),[1,{(spec.max_seq + spec.cache_block_size - 1) // spec.cache_block_size}])"
            )
            norm = "norm_plain" if index == 0 and not spec.native_first_norm else "norm_hidden"
            if looped and index == 0 and not spec.native_first_norm:
                emit(
                    f"if group_index==0:\n    norm=self.norm_plain(hidden,{names['input_layernorm.weight']},hidden,norm)\nelse:\n    norm=self.norm_hidden(hidden,{names['input_layernorm.weight']},hidden,norm)"
                )
            elif not (fuse_reduction_norm and t == 1 and looped and index > 0):
                emit(f"norm=self.{norm}(hidden,{names['input_layernorm.weight']},hidden,norm)")
            if index % 4 != 3:
                projections = (
                    []
                    if spec.fuse_gdn_ba
                    else [("a", "project_small", "in_proj_a"), ("b", "project_small", "in_proj_b")]
                )
                if spec.fuse_gdn_qkv_z:
                    group, offset, shape = spec.weights[index, "linear_attn.in_proj_qkv.weight"]
                    if spec.weights[index, "linear_attn.in_proj_z.weight"][1] != offset + prod(shape):
                        raise ValueError("GDN QKV/Z weights must be adjacent.")
                    view(
                        f"packed_gdn_weight{index}",
                        "group_weights" if looped else f"weights{group}",
                        offset,
                        spec.projection_weight_shape((8192, 5120)),
                    )
                    emit(f"mixed,z=self.project_gdn_z(norm,packed_gdn_weight{index},mixed,z)")
                else:
                    # Current scheduler observations dispatch later ready projections first.
                    # Emit GDN after Z to test earlier release of its vector consumers.
                    projections = [
                        ("z", "project_z", "in_proj_z"),
                        ("mixed", "project_gdn", "in_proj_qkv"),
                    ] + projections
                for output, project, key in projections:
                    emit(f"{output}=self.{project}(norm,{names['linear_attn.' + key + '.weight']},{output})")
                if spec.fuse_gdn_ba:
                    group, offset, shape = spec.weights[index, "linear_attn.in_proj_b.weight"]
                    if spec.weights[index, "linear_attn.in_proj_a.weight"][1] != offset + prod(shape):
                        raise ValueError("B/A weights must be adjacent.")
                    view(
                        f"packed_ba_weight{index}",
                        "group_weights" if looped else f"weights{group}",
                        offset,
                        (48, 5120),
                    )
                    emit(f"gdn_ba=self.project_ba(norm,packed_ba_weight{index},gdn_ba)")
                    emit("b=pl.slice(gdn_ba,[1,24],[0,0])")
                    emit("a=pl.slice(gdn_ba,[1,24],[0,24])")
                g = f"(group_index*3+{index})" if looped else index - index // 4
                view(f"a_log{index}", "float_weights", f"{g}*48", (1, 24))
                view(f"dt_bias{index}", "float_weights", f"{g}*48+24", (1, 24))
                if t == 1:
                    emit(
                        f"convolved,qn,kn,conv{index}=self.convolve(mixed,{names['linear_attn.conv1d.weight']},conv{index},starts,indices,initial,convolved,qn,kn)"
                    )
                else:
                    emit(
                        f"convolved,conv{index}=self.convolve(mixed,{names['linear_attn.conv1d.weight']},conv{index},starts,indices,initial,convolved)"
                    )
                if gdn_decode_views and t == 1:
                    emit("q=pl.reshape(pl.slice(pl.reshape(convolved,[5120]),[1024],[0]),[1,8,128])")
                    emit("k=pl.reshape(pl.slice(pl.reshape(convolved,[5120]),[1024],[1024]),[1,8,128])")
                    emit("v=pl.reshape(pl.slice(pl.reshape(convolved,[5120]),[3072],[2048]),[1,24,128])")
                else:
                    emit("q,k,v=self.split_gdn(convolved,q,k,v)")
                if t != 1:
                    emit("qn,kn=self.normalize_gdn(q,k,qn,kn)")
                emit(f"alpha,beta=self.gates{rank}(a,b,a_log{index},dt_bias{index},alpha,beta)")
                emit(
                    f"gdn_core,state{index}=self.recurrent(qn,kn,v,alpha,beta,state{index},starts,indices,initial,gdn_core)"
                )
                emit(
                    f"gdn_normal=self.norm_gdn(pl.reshape(gdn_core,[{t * 24},128]),{names['linear_attn.norm.weight']},pl.reshape(z,[{t * 24},128]),gdn_normal)"
                )
                emit(
                    f"partial=self.project_attention_out(pl.reshape(gdn_normal,[{t},3072]),{names['linear_attn.out_proj.weight']},partial)"
                )
            else:
                if spec.fuse_attention_qkv:
                    group, offset, shape = spec.weights[index, "self_attn.q_proj.weight"]
                    if (
                        spec.weights[index, "self_attn.k_proj.weight"][1] != offset + prod(shape)
                        or spec.weights[index, "self_attn.v_proj.weight"][1]
                        != offset + prod(shape) + 512 * 5120
                    ):
                        raise ValueError("Native attention Q/K/V weights must be adjacent.")
                    view(
                        f"joint_attention_weight{index}",
                        "group_weights" if looped else f"weights{group}",
                        offset,
                        (7168, 5120),
                    )
                    emit(
                        f"packed_q,full_k,full_v=self.project_attention_qkv(norm,joint_attention_weight{index},packed_q,full_k,full_v)"
                    )
                else:
                    emit(f"packed_q=self.project_q(norm,{names['self_attn.q_proj.weight']},packed_q)")
                    emit(f"full_k=self.project_kv(norm,{names['self_attn.k_proj.weight']},full_k)")
                    emit(f"full_v=self.project_kv(norm,{names['self_attn.v_proj.weight']},full_v)")
                if not (spec.native_qk_rope and t == 1):
                    emit("full_q,full_gate=self.split_query(packed_q,full_q,full_gate)")
                if spec.native_qk_rope:
                    if t == 1:
                        emit(
                            f"full_qr,full_gate=self.norm_rope_q(packed_q,{names['self_attn.q_norm.weight']},cos,sin,positions,full_qr,full_gate)"
                        )
                    else:
                        emit(
                            f"full_qr=self.norm_rope_q(full_q,{names['self_attn.q_norm.weight']},cos,sin,positions,full_qr)"
                        )
                    emit(
                        f"full_kr=self.norm_rope_k(pl.reshape(full_k,[{t},2,256]),{names['self_attn.k_norm.weight']},cos,sin,positions,full_kr)"
                    )
                else:
                    emit(
                        f"full_qn=self.norm_q(pl.reshape(full_q,[{t * 12},256]),{names['self_attn.q_norm.weight']},pl.reshape(full_q,[{t * 12},256]),full_qn)"
                    )
                    emit(
                        f"full_kn=self.norm_k(pl.reshape(full_k,[{t * 2},256]),{names['self_attn.k_norm.weight']},pl.reshape(full_k,[{t * 2},256]),full_kn)"
                    )
                    emit(f"full_qr=self.rope_q(pl.reshape(full_qn,[{t},12,256]),cos,sin,positions,full_qr)")
                    emit(f"full_kr=self.rope_k(pl.reshape(full_kn,[{t},2,256]),cos,sin,positions,full_kr)")
                if spec.native_cache_pools:
                    emit(
                        f"keys{index}=self.append(pl.reshape(full_kr,[{t},512]),full_v,keys{index},kv_slots)"
                    )
                    emit(
                        f"attention_scores=self.compute_scores(full_qr,keys{index},table,lengths,attention_scores)"
                    )
                    emit("attention_probability=self.softmax_scores(attention_scores,attention_probability)")
                    emit(
                        f"full_core=self.compute_values(attention_probability,keys{index},table,lengths,full_core)"
                    )
                else:
                    emit(f"native_key_flat{index}=pl.reshape(keys{index},[{spec.cache_rows},512])")
                    emit(f"native_value_flat{index}=pl.reshape(values{index},[{spec.cache_rows},512])")
                    emit(
                        f"key_flat{index},value_flat{index}=self.append(pl.reshape(full_kr,[{t},512]),full_v,native_key_flat{index},native_value_flat{index},kv_slots)"
                    )
                    emit(
                        f"attention_scores=self.compute_scores(full_qr,pl.reshape(key_flat{index},[{spec.cache_rows},2,256]),table,lengths,attention_scores)"
                    )
                    emit("attention_probability=self.softmax_scores(attention_scores,attention_probability)")
                    emit(
                        f"full_core=self.compute_values(attention_probability,pl.reshape(value_flat{index},[{spec.cache_rows},2,256]),table,lengths,full_core)"
                    )
                emit(f"gated=self.gate_attention(full_gate,pl.reshape(full_core,[{t},3072]),gated)")
                emit(f"partial=self.project_attention_out(gated,{names['self_attn.o_proj.weight']},partial)")
            if fuse_reduction_norm and t == 1:
                emit(
                    f"residual,mlp_norm,data,signal=self.reduce_add_norm(partial,hidden,residual,{names['post_attention_layernorm.weight']},mlp_norm,data,signal,metadata,{'group_index*8+' if looped else ''}{1 + 2 * index})"
                )
            elif fuse_reduction_residual:
                emit(
                    f"residual,data,signal=self.reduce_add(partial,hidden,residual,data,signal,metadata,{'group_index*8+' if looped else ''}{1 + 2 * index})"
                )
            else:
                emit(
                    f"attention,data,signal=self.reduce_partial(partial,attention,data,signal,metadata,{'group_index*8+' if looped else ''}{1 + 2 * index})"
                )
                emit("residual=self.add(hidden,attention,residual)")
            if not (fuse_reduction_norm and t == 1):
                emit(
                    f"mlp_norm=self.norm_hidden(residual,{names['post_attention_layernorm.weight']},residual,mlp_norm)"
                )
            if spec.fuse_mlp:
                group, gate_offset, gate_shape = spec.weights[index, "mlp.gate_proj.weight"]
                _, up_offset, _ = spec.weights[index, "mlp.up_proj.weight"]
                if up_offset != gate_offset + prod(gate_shape):
                    raise ValueError("Local gate/up weights must be contiguous")
                view(
                    f"packed_mlp_weight{index}",
                    "group_weights" if looped else f"weights{group}",
                    gate_offset,
                    spec.projection_weight_shape((17408, 5120)),
                )
                if fused_upact:
                    emit(f"mlp_act=self.project_mlp(mlp_norm,packed_mlp_weight{index},mlp_act)")
                else:
                    emit(f"mlp_packed=self.project_mlp(mlp_norm,packed_mlp_weight{index},mlp_packed)")
                if not fused_down and not fused_upact:
                    emit("mlp_act=self.activate_packed(mlp_packed,mlp_act)")
            else:
                emit(f"mlp_gate=self.project_up(mlp_norm,{names['mlp.gate_proj.weight']},mlp_gate)")
                emit(f"mlp_up=self.project_up(mlp_norm,{names['mlp.up_proj.weight']},mlp_up)")
                emit("mlp_act=self.activate(mlp_gate,mlp_up,mlp_act)")
            down_input = "mlp_packed" if fused_down else "mlp_act"
            emit(f"partial=self.project_down({down_input},{names['mlp.down_proj.weight']},partial)")
            if fuse_reduction_norm and t == 1 and looped and index < 3:
                # The next layer's input norm is on this reduction's dependency chain.
                # Both weights belong to the resident four-layer group.
                next_group, next_offset, _ = spec.weights[index + 1, "input_layernorm.weight"]
                view(f"next_input_norm_weight{index}", "group_weights", next_offset, (1, 5120))
                emit(
                    f"hidden,norm,data,signal=self.reduce_add_norm(partial,residual,hidden,next_input_norm_weight{index},norm,data,signal,metadata,{'group_index*8+' if looped else ''}{2 + 2 * index})"
                )
            elif fuse_reduction_residual:
                emit(
                    f"hidden,data,signal=self.reduce_add(partial,residual,hidden,data,signal,metadata,{'group_index*8+' if looped else ''}{2 + 2 * index})"
                )
            else:
                emit(
                    f"mlp_down,data,signal=self.reduce_partial(partial,mlp_down,data,signal,metadata,{'group_index*8+' if looped else ''}{2 + 2 * index})"
                )
                emit("hidden=self.add(residual,mlp_down,hidden)")
            if spec.debug_every_layer:
                emit(
                    f"debug_hidden=self.save_group_hidden(hidden,debug_hidden,{'group_index*4+' if looped else ''}{index})"
                )
            elif spec.debug_groups and index % 4 == 3:
                emit(
                    f"debug_hidden=self.save_group_hidden(hidden,debug_hidden,{'group_index' if looped else index // 4})"
                )
        indent = "    "
        if spec.teacher_input:
            emit("teacher_out=self.copy_hidden(hidden,teacher_out)")
        emit("head_norm=self.norm_head(hidden,final_norm,hidden,head_norm)")
        emit("hidden_result=self.publish_hidden(head_norm,hidden_result)")
        view("last_hidden", f"pl.reshape(head_norm,[{t * 5120}])", (t - 1) * 5120, (1, 5120))
        emit("out=self.project_head(last_hidden,lm_head,out)")
        emit(
            "global_logits,logit_data,logit_signal=self.gather_logits(out,global_logits,logit_data,logit_signal,metadata)"
        )
        if spec.device_greedy:
            emit("greedy=self.greedy(out,greedy)")
        emit("return greedy" if spec.device_greedy else "return global_logits")
        methods.append("\n".join(lines))
    lines = ["@pl.function(level=pl.Level.HOST,role=pl.Role.Orchestrator)", "def host(self,"]
    for name, shape, dtype, direction in spec.arguments:
        annotation = f"pl.Tensor[{[2, *shape]!r},pl.{dtype}]"
        if direction:
            annotation = f"pl.{direction}[{annotation}]"
        lines.append(f"    {name}: {annotation},")
    lines += [
        ") -> pl.Tensor[[2,1,8],pl.FP32]:" if spec.device_greedy else ") -> pl.Tensor[[2,1,248320],pl.FP32]:",
        "    logit_data_buffer=pld.alloc_window_buffer([1,124160],dtype=pl.FP32)",
        "    logit_signal_buffer=pld.alloc_window_buffer([2,1],dtype=pl.INT32)",
        f"    data_buffer=pld.alloc_window_buffer([{spec.reductions * t},5120],dtype=pl.FP32)",
        f"    signal_buffer=pld.alloc_window_buffer([{spec.reductions * 2},1],dtype=pl.INT32)",
    ]
    for rank in (0, 1):
        lines += [
            f"    logit_data{rank}=pld.window(logit_data_buffer,[1,124160],dtype=pl.FP32)",
            f"    logit_signal{rank}=pld.window(logit_signal_buffer,[2,1],dtype=pl.INT32)",
            f"    data{rank}=pld.window(data_buffer,[{spec.reductions * t},5120],dtype=pl.FP32)",
            f"    signal{rank}=pld.window(signal_buffer,[{spec.reductions * 2},1],dtype=pl.INT32)",
        ]
        args = ",".join(f"{name}[{rank}]" for name, _, _, _ in spec.arguments)
        lines.append(
            f"    result{rank}=self.rank{rank}({args},logit_data{rank},logit_signal{rank},data{rank},signal{rank},device={rank})"
        )
    lines.append("    return greedy" if spec.device_greedy else "    return global_logits")
    methods.append("\n".join(lines))
    source = (
        "import pypto.language as pl\nimport pypto.language.distributed as pld\n\n@pl.program\nclass QwenTP2:\n"
        + "\n\n".join(textwrap.indent(m, "    ") for m in methods)
        + "\n"
    )
    if spec.graph_stop_guard:
        if not spec.pack_host_controls or spec.device_greedy:
            raise ValueError("Graph stop guard requires packed controls and gathered logits.")
        tree = ast.parse(source)
        cols = 9 + (spec.max_seq + spec.cache_block_size - 1) // spec.cache_block_size
        fence = ast.parse(f"""@pl.function(type=pl.FunctionType.InCore)
def graph_stop_fence(self, controls: pl.InOut[pl.Tensor[[1,{cols}],pl.INT32]]) -> pl.Tensor[[1,{cols}],pl.INT32]:
    value=pl.load(controls,[0,0],[1,{((cols + 7) // 8) * 8}],[1,{cols}])
    controls=pl.store(value,[0,0],controls)
    return controls
""").body[0]
        cls = tree.body[-1]
        cls.body.insert(0, fence)
        for method in cls.body:
            if isinstance(method, ast.FunctionDef) and method.name in ("rank0", "rank1"):
                body = method.body[:-1]
                guard = ast.parse("pl.read(control_i32,[3])==0", mode="eval").body
                stopped = ast.parse(
                    f"graph_stop_control=pl.reshape(control_i32,[1,{cols}])\ngraph_stop_control=self.graph_stop_fence(graph_stop_control)\ncontrol_i32=pl.reshape(graph_stop_control,[{cols}])"
                ).body
                method.body = [ast.If(test=guard, body=body, orelse=stopped), method.body[-1]]
        source = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
    if early_dispatch:
        tree = ast.parse(source)
        kernel_names = {
            node.name
            for node in tree.body[-1].body
            if isinstance(node, ast.FunctionDef) and node.name.startswith("reduce_")
        }

        class EarlyDispatch(ast.NodeTransformer):
            def visit_Call(self, node):
                self.generic_visit(node)
                if (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "pl"
                    and node.func.attr == "spmd"
                ):
                    node.keywords.append(
                        ast.keyword(arg="allow_early_resolve", value=ast.Constant(value=True))
                    )
                return node

            def visit_Assign(self, node):
                self.generic_visit(node)
                call = node.value
                if (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == "self"
                    and call.func.attr in kernel_names
                ):
                    target = node.targets[0]
                    node.targets = [
                        ast.Tuple(
                            elts=[target, ast.Name(id="early_reduction_task", ctx=ast.Store())],
                            ctx=ast.Store(),
                        )
                    ]
                    node.value = ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id="pl", ctx=ast.Load()), attr="submit", ctx=ast.Load()
                        ),
                        args=[call.func, *call.args],
                        keywords=[ast.keyword(arg="allow_early_resolve", value=ast.Constant(value=True))],
                    )
                return node

        source = ast.unparse(ast.fix_missing_locations(EarlyDispatch().visit(tree))) + "\n"

    from .native_pool_lowering import lower_native_pools

    source = lower_native_pools(source, spec)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "qwen_tp2_program.py"
    path.write_text(source)
    name = f"_qwen_mpmd_{spec.layers}_{t}_{int(spec.replicate_small)}"
    module_spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[name] = module
    module_spec.loader.exec_module(module)
    return module.QwenTP2
