# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Keep one InCore implementation per operator shape across all model layers."""

import ast
import copy

from .probe import static_method


def reusable_methods(name, function, bindings):
    """Separate a static SPMD launch wrapper from its shared device body."""
    wrapper = ast.parse(static_method(name, function, bindings)).body[0]
    grids = [
        (i, statement)
        for i, statement in enumerate(wrapper.body)
        if isinstance(statement, ast.For)
        and isinstance(statement.iter, ast.Call)
        and isinstance(statement.iter.func, ast.Attribute)
        and statement.iter.func.attr == "spmd"
    ]
    if len(grids) != 1:
        raise ValueError(f"{name}: expected exactly one SPMD compute region")
    index, grid = grids[0]
    if name == "convolve":
        # Preserve the validated loop-carried convolution tile allocations.
        return [ast.unparse(wrapper)]
    kernel = copy.deepcopy(wrapper)
    kernel.name = name + "_kernel"
    kernel.decorator_list = (
        ast.parse("@pl.function(type=pl.FunctionType.InCore)\ndef f(): pass").body[0].decorator_list
    )
    block_index = ast.Assign(
        targets=[copy.deepcopy(grid.target)], value=ast.parse("pl.tile.get_block_idx()", mode="eval").body
    )
    scalar_prefix, views, constants = [], [], {}
    annotations = {arg.arg: arg.annotation for arg in wrapper.args.args[1:]}

    class Resolve(ast.NodeTransformer):
        def visit_Name(self, node):
            return copy.deepcopy(constants.get(node.id, node))

    for statement in wrapper.body[:index]:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            return [ast.unparse(wrapper)]
        variable = statement.targets[0].id
        value = statement.value
        if isinstance(value, ast.Call):
            if not isinstance(value.func, ast.Attribute) or value.func.attr != "reshape":
                return [ast.unparse(wrapper)]
            origin = value.args[0].id
            dtype = ast.unparse(annotations[origin].slice.elts[1])
            shape_node = Resolve().visit(copy.deepcopy(value.args[1]))
            # These expressions originate in statically specialised DSL shapes.
            shape = eval(
                compile(ast.Expression(ast.fix_missing_locations(shape_node)), "<static shape>", "eval"),
                {"__builtins__": {}},
            )
            annotation = ast.parse(f"pl.Tensor[{shape!r}, {dtype}]", mode="eval").body
            kernel.args.args.append(ast.arg(arg=variable, annotation=annotation))
            annotations[variable] = annotation
            views.append(variable)
        else:
            constants[variable] = Resolve().visit(copy.deepcopy(value))
            scalar_prefix.append(copy.deepcopy(statement))
    cache_policies = []
    compute_body = []
    for statement in copy.deepcopy(grid.body):
        if (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Attribute)
            and statement.value.func.attr == "set_cache_policy"
        ):
            operand, policy = statement.value.args
            param_index = next(i for i, arg in enumerate(kernel.args.args[1:]) if arg.arg == operand.id)
            policy_name = ast.unparse(policy)
            if policy_name.endswith(".BYPASS"):
                policy_value = 1
            elif policy_name.endswith(".DEFAULT"):
                policy_value = 0
            else:
                raise ValueError(f"{name}: unsupported cache policy {policy_name}")
            cache_policies.append((param_index, policy_value))
        else:
            compute_body.append(statement)
    kernel.body = scalar_prefix + [block_index] + compute_body + copy.deepcopy(wrapper.body[index + 1 :])
    if not name.startswith("_"):
        kernel.body.insert(0, ast.parse('pl.func_attr({"static_block_mapping": True})').body[0])
    if cache_policies:
        # Pass9's supported function attr, consumed by ConvertTensorToTileOps.
        kernel.body.insert(0, ast.parse(f'pl.func_attr({{"cache_policy": {cache_policies!r}}})').body[0])
    call = ast.Call(
        func=ast.Attribute(value=ast.Name(id="self", ctx=ast.Load()), attr=kernel.name, ctx=ast.Load()),
        args=[ast.Name(id=arg.arg, ctx=ast.Load()) for arg in wrapper.args.args[1:]]
        + [ast.Name(id=view, ctx=ast.Load()) for view in views],
        keywords=[],
    )
    returned = wrapper.body[-1].value
    target = copy.deepcopy(returned)
    for node in ast.walk(target):
        if isinstance(node, (ast.Name, ast.Tuple)):
            node.ctx = ast.Store()
    assignment = ast.Assign(targets=[target], value=call)
    scope = ast.With(
        items=[
            ast.withitem(
                context_expr=ast.Call(
                    func=copy.deepcopy(grid.iter.func), args=copy.deepcopy(grid.iter.args), keywords=[]
                )
            )
        ],
        body=[assignment],
    )
    wrapper.body = wrapper.body[:index] + [scope] + wrapper.body[index + 1 :]
    return [ast.unparse(ast.fix_missing_locations(kernel)), ast.unparse(ast.fix_missing_locations(wrapper))]
