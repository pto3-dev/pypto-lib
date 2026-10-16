# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Specialize inline operator source for the distributed model builder."""

import ast
import copy
import inspect
import textwrap

import pypto.language as pl


def static_method(name, function, bindings=None):
    fn = function._func
    node = ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0]
    constants = inspect.getclosurevars(fn).nonlocals
    shapes = {
        name: [d if isinstance(d, int) else (bindings or {}).get(d.name, 1) for d in ann.shape]
        for name, ann in fn.__annotations__.items()
        if hasattr(ann, "shape")
    }

    class Constants(ast.NodeTransformer):
        def visit_Name(self, value):
            if (
                isinstance(value.ctx, ast.Load)
                and value.id in constants
                and isinstance(constants[value.id], pl.CachePolicy)
            ):
                policy = "BYPASS" if constants[value.id] == pl.CachePolicy.BYPASS else "DEFAULT"
                return ast.copy_location(ast.parse(f"pl.CachePolicy.{policy}", mode="eval").body, value)
            if (
                isinstance(value.ctx, ast.Load)
                and value.id in constants
                and isinstance(constants[value.id], (int, float, bool))
            ):
                return ast.copy_location(ast.Constant(constants[value.id]), value)
            return value

        def visit_If(self, value):
            self.generic_visit(value)
            if isinstance(value.test, ast.Constant) and isinstance(value.test.value, bool):
                return value.body if value.test.value else value.orelse
            return value

        def visit_Call(self, value):
            self.generic_visit(value)
            func = value.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "dim"
                and isinstance(func.value, ast.Attribute)
                and func.value.attr == "tensor"
                and isinstance(func.value.value, ast.Name)
                and func.value.value.id == "pl"
                and len(value.args) == 2
                and isinstance(value.args[0], ast.Name)
                and value.args[0].id in shapes
                and isinstance(value.args[1], ast.Constant)
            ):
                return ast.copy_location(ast.Constant(shapes[value.args[0].id][value.args[1].value]), value)
            return value

    node = Constants().visit(node)
    node.name = name
    node.args.args.insert(0, ast.arg(arg="self"))
    annotations = {}
    for arg in node.args.args[1:]:
        annotation = fn.__annotations__[arg.arg]
        shape = [
            dim if isinstance(dim, int) else (bindings or {}).get(dim.name, 1) for dim in annotation.shape
        ]
        dtype = {"bfloat16": "BF16", "float32": "FP32", "fp32": "FP32", "int32": "INT32", "int64": "INT64"}[
            str(annotation.dtype)
        ]
        arg.annotation = ast.parse(f"pl.Tensor[{shape!r}, pl.{dtype}]", mode="eval").body
        annotations[arg.arg] = arg.annotation
    returned = node.body[-1].value
    if isinstance(returned, ast.Tuple):
        types = [copy.deepcopy(annotations[v.id]) for v in returned.elts]
        node.returns = ast.Subscript(
            value=ast.Name(id="tuple", ctx=ast.Load()),
            slice=ast.Tuple(elts=types, ctx=ast.Load()),
            ctx=ast.Load(),
        )
    else:
        node.returns = copy.deepcopy(annotations[returned.id])
    node.decorator_list = (
        ast.parse("@pl.function(type=pl.FunctionType.Inline)\ndef f(): pass").body[0].decorator_list
    )
    return ast.unparse(ast.fix_missing_locations(node))
