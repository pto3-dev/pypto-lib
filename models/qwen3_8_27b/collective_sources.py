# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Generate one HOST program dispatching a complete model to both TP ranks."""

"""Collective source generators used by the TP2 model composition."""

import ast

import pypto.language as pl

from .vector_norm import make_vector_norm
from .reusable import reusable_methods


ROWS = pl.dynamic("rows")


@pl.jit.inline
def copy_hidden(x: pl.Tensor[[ROWS, 5120], pl.BF16], out: pl.Out[pl.Tensor[[ROWS, 5120], pl.BF16]]):
    rows = pl.tensor.dim(x, 0)
    for row in pl.spmd(rows, name_hint="copy_teacher_hidden"):
        value = pl.slice(x, [1, 5120], [row, 0])
        out = pl.assemble(out, value, [row, 0])
    return out


def reduction_source(name, spec, input_dtype):
    t, n = spec.tokens, 5120
    loaded = (
        "pl.cast(pl.load(x,[row,0],[1,5120]),pl.FP32)"
        if input_dtype == "BF16"
        else "pl.load(x,[row,0],[1,5120])"
    )
    if input_dtype == "FP32" and spec.native_reduce_round:
        loaded = 'pl.cast(pl.cast(pl.load(x,[row,0],[1,5120]),pl.BF16,mode="rint"),pl.FP32)'
    return f"""@pl.function(type=pl.FunctionType.InCore)
def {name}(self,
    x: pl.Tensor[[{t},{n}],pl.{input_dtype}],
    out: pl.Out[pl.Tensor[[{t},{n}],pl.BF16]],
    data: pl.InOut[pld.DistributedTensor[[{spec.reductions * t},{n}],pl.FP32]],
    signal: pl.InOut[pld.DistributedTensor[[{spec.reductions * 2},1],pl.INT32]],
    metadata: pl.Tensor[[4],pl.INT32],
    operation: pl.Scalar[pl.INT32],
) -> tuple[pl.Tensor[[{t},{n}],pl.BF16],pld.DistributedTensor[[{spec.reductions * t},{n}],pl.FP32],pld.DistributedTensor[[{spec.reductions * 2},1],pl.INT32]]:
    ctx=pld.get_comm_ctx(data)
    rank=pld.rank(ctx)
    peer=1-rank
    epoch=pl.read(metadata,[1])
    for row in pl.range({t}):
        local={loaded}
        data=pl.store(local,[operation*{t}+row,0],data)
    pld.system.notify(signal,peer=peer,offsets=[operation*2+rank,0],value=1,op=pld.NotifyOp.AtomicAdd)
    pld.system.wait(signal,offsets=[operation*2+peer,0],expected=epoch,cmp=pld.WaitCmp.Ge)
    for row in pl.range({t}):
        local_value=pl.load(data,[operation*{t}+row,0],[1,{n}])
        peer_value=pld.tile.remote_load(data,peer=peer,offsets=[operation*{t}+row,0],shape=[1,{n}])
        total=pl.add(local_value,peer_value)
        narrowed=pl.cast(total,pl.BF16,mode="rint")
        out=pl.store(narrowed,[row,0],out)
    return out,data,signal
"""


def gather_logits_source():
    return """@pl.function(type=pl.FunctionType.InCore)
def gather_logits(self,
    local: pl.Tensor[[1,124160],pl.BF16],
    out: pl.Out[pl.Tensor[[1,248320],pl.FP32]],
    data: pl.InOut[pld.DistributedTensor[[1,124160],pl.FP32]],
    signal: pl.InOut[pld.DistributedTensor[[2,1],pl.INT32]],
    metadata: pl.Tensor[[4],pl.INT32],
) -> tuple[pl.Tensor[[1,248320],pl.FP32],pld.DistributedTensor[[1,124160],pl.FP32],pld.DistributedTensor[[2,1],pl.INT32]]:
    ctx=pld.get_comm_ctx(data)
    rank=pld.rank(ctx)
    peer=1-rank
    epoch=pl.read(metadata,[1])
    for offset in pl.range(0,124160,1280):
        value=pl.cast(pl.load(local,[0,offset],[1,1280]),pl.FP32)
        data=pl.store(value,[0,offset],data)
    pld.system.notify(signal,peer=peer,offsets=[rank,0],value=1,op=pld.NotifyOp.AtomicAdd)
    pld.system.wait(signal,offsets=[peer,0],expected=epoch,cmp=pld.WaitCmp.Ge)
    for offset in pl.range(0,124160,1280):
        own=pl.load(data,[0,offset],[1,1280])
        other=pld.tile.remote_load(data,peer=peer,offsets=[0,offset],shape=[1,1280])
        out=pl.store(own,[0,rank*124160+offset],out)
        out=pl.store(other,[0,peer*124160+offset],out)
    return out,data,signal
"""


def reduction_add_source(spec):
    """Fuse the BF16 residual addition after the existing FP32 TP reduction."""
    source = reduction_source("reduce_add", spec, "FP32")
    source = source.replace(
        "    out: pl.Out[", f"    residual: pl.Tensor[[{spec.tokens},5120],pl.BF16],\n    out: pl.Out[", 1
    )
    source = source.replace(
        "        out=pl.store(narrowed,[row,0],out)",
        "        saved=pl.cast(pl.load(residual,[row,0],[1,5120]),pl.FP32)\n"
        "        summed=pl.add(saved,pl.cast(narrowed,pl.FP32))\n"
        '        result=pl.cast(summed,pl.BF16,mode="rint")\n'
        "        out=pl.store(result,[row,0],out)",
    )
    return source


def reduction_add_norm_source(spec, norm_tile, norm_output_tile):
    """Reuse the validated norm body after the rounded TP sum and residual."""
    if spec.tokens != 1:
        raise ValueError("Reduction/norm fusion is decode-only.")
    reduction = ast.parse(reduction_add_source(spec)).body[0]
    reduction.name = "reduce_add_norm"
    args = (
        ast.parse(
            "def f(weight: pl.Tensor[[1,5120],pl.BF16],normalized: pl.Out[pl.Tensor[[1,5120],pl.BF16]]): pass"
        )
        .body[0]
        .args.args
    )
    reduction.args.args[4:4] = args
    norm = make_vector_norm(inline=True, native_round=True)
    kernel = ast.parse(reusable_methods("_fused_norm", norm, dict(rows=1))[0]).body[0]

    class Rename(ast.NodeTransformer):
        def visit_Name(self, node):
            name = {"x": "out", "out": "normalized", "z": "out"}.get(
                node.id, node.id if node.id in ("pl", "self", "weight") else "fused_norm_" + node.id
            )
            return ast.copy_location(ast.Name(id=name, ctx=node.ctx), node)

        def visit_Call(self, node):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "get_block_idx":
                return ast.copy_location(ast.Constant(value=0), node)
            return self.generic_visit(node)

    body = [Rename().visit(node) for node in kernel.body[:-1]]
    # The BF16 residual tile is already resident in this InCore function.
    # Reuse that rounded value instead of writing it and loading it again.
    whole = [
        node
        for node in body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "fused_norm_whole" for target in node.targets)
    ]
    if len(whole) != 1:
        raise ValueError("Could not find the vector norm input tile for reduction fusion.")
    whole[0].value = ast.parse("pl.cast(result,pl.FP32)", mode="eval").body
    sums = [
        node
        for node in body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "fused_norm_partials" for target in node.targets
        )
    ]
    if len(sums) != 1:
        raise ValueError("Could not find RMS partial sum for explicit InCore scratch.")
    sums[0].value.args.append(ast.parse("pl.tile.full([40,128],dtype=pl.FP32,value=0.0)", mode="eval").body)
    zero = [
        node
        for node in body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "fused_norm_zero" for target in node.targets)
    ]
    if len(zero) != 1:
        raise ValueError("Could not find ordered RMS fold zero tile.")
    zero[0].value = ast.parse("pl.tile.full([1,128],dtype=pl.FP32,value=0.0)", mode="eval").body
    gammas = [
        node
        for node in body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "fused_norm_gamma" for target in node.targets)
    ]
    if len(gammas) != 2:
        raise ValueError("Expected gamma load and native BF16-rounding assignment.")
    gammas[0].value = ast.parse(
        "pl.add(pl.cast(pl.load(weight,[0,0],[1,5120]),pl.FP32),1.0)", mode="eval"
    ).body
    stores = [
        node
        for node in body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "normalized" for target in node.targets)
    ]
    if (
        len(stores) != 1
        or not isinstance(stores[0].value, ast.Call)
        or stores[0].value.func.attr != "assemble"
    ):
        raise ValueError("Expected fused RMSNorm output assembly.")
    original = stores[0].value.args
    stores[0].value.func.attr = "store"
    stores[0].value.args = [original[1], original[2], original[0]]
    returned = ast.parse("return out,normalized,data,signal").body[0]
    row_loop = reduction.body[-2]
    if not isinstance(row_loop, ast.For):
        raise ValueError("Expected final TP reduction row loop.")
    row_loop.body.extend(body)
    reduction.body = reduction.body[:-1] + [returned]
    annotation = reduction.returns
    annotation.slice.elts.insert(1, ast.parse("pl.Tensor[[1,5120],pl.BF16]", mode="eval").body)
    return ast.unparse(ast.fix_missing_locations(reduction))


def debug_source(spec):
    return f"""@pl.function(type=pl.FunctionType.InCore)
def save_group_hidden(self, hidden: pl.Tensor[[{spec.tokens},5120],pl.BF16],
    out: pl.Out[pl.Tensor[[{spec.debug_count},5120],pl.BF16]],
    group: pl.Scalar[pl.INT32]) -> pl.Tensor[[{spec.debug_count},5120],pl.BF16]:
    value=pl.load(hidden,[{spec.tokens - 1},0],[1,5120])
    out=pl.store(value,[group,0],out)
    return out
"""
