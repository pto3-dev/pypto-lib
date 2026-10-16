# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Materialize typed native pool views inside InCore, preserving pool ownership.

The pinned PyPTO backend cannot change a runtime Tensor view's dtype in an
orchestration function. Keep one writable byte-pool argument at L3/L2, then
reinterpret it only inside the generated InCore cache kernels.
"""

import ast
from math import prod


def lower_native_pools(source, spec):
    if not spec.native_cache_pools:
        return source
    sizes = {size for _, size in spec.native_cache_pools}
    if len(sizes) != 1:
        raise ValueError("This model generation requires equally sized native hybrid pools.")
    nbytes = next(iter(sizes))
    views = {}
    for name, _, offset, shape, dtype in spec.native_cache_layout:
        kind = name.split("_", 1)[1]
        value = (offset, tuple(shape), dtype)
        if kind in views and views[kind] != value:
            raise ValueError("Native pool typed layouts differ between layers.")
        views[kind] = value
    byte_annotation = ast.parse(f"pl.Tensor[[{nbytes // 2}],pl.BF16]", mode="eval").body

    def typed(name, kind, flat_name=None, flat_shape=None):
        offset, shape, dtype = views[kind]
        extent = prod(shape) * (4 if dtype == "FP32" else 2)
        if offset % 2 or extent % 2:
            raise ValueError("Native pool views require BF16 element alignment.")
        sliced = f"pl.slice(cache_bytes,[{extent // 2}],[{offset // 2}])"
        text = (
            f"{name}=pl.reshape({sliced},{list(shape)!r})"
            if dtype == "BF16"
            else f"{name}=pl.reinterpret_view({sliced},pl.{dtype},shape={list(shape)!r})"
        )
        if flat_name:
            text += f"\n{flat_name}=pl.reshape({name},{flat_shape!r})"
        return ast.parse(text).body

    tree = ast.parse(source)
    cls = tree.body[-1]
    for method in cls.body:
        if not isinstance(method, ast.FunctionDef):
            continue
        name = method.name
        kind = None
        remove = set()
        preamble = []
        if name == "convolve":
            kind = "conv_state"
            remove = {"flat"}
            preamble = typed("native_state", kind, "flat", [spec.state_slots * 3, 5120])
        elif name in ("recurrent", "recurrent_kernel"):
            kind = "recurrent_state"
            remove = {"s"}
            if name.endswith("_kernel"):
                preamble = typed("native_state", kind, "s", [spec.state_slots * 24 * 128, 128])
        elif name in ("compute_scores", "compute_scores_kernel"):
            kind = "keys"
            remove = {"cf"}
            if name.endswith("_kernel"):
                preamble = typed("native_cache", kind, "cf", [spec.cache_rows, 512])
        elif name in ("compute_values", "compute_values_kernel"):
            kind = "values"
            remove = {"cf"}
            if name.endswith("_kernel"):
                preamble = typed("native_cache", kind, "cf", [spec.cache_rows, 512])
        elif name in ("append", "append_kernel"):
            kind = "keys"
            if name.endswith("_kernel"):
                preamble = typed("native_key", "keys") + typed("native_value", "values")
                preamble += ast.parse(
                    f"key_cache=pl.reshape(native_key,[{spec.cache_rows},512])\nvalue_cache=pl.reshape(native_value,[{spec.cache_rows},512])"
                ).body
        if kind is None:
            continue
        original_args = [a.arg for a in method.args.args]
        cache_name = (
            "state"
            if kind in ("conv_state", "recurrent_state")
            else "key_cache"
            if name.startswith("append")
            else "cache"
        )
        method.args.args = [
            a
            for a in method.args.args
            if a.arg not in remove and not (name.startswith("append") and a.arg == "value_cache")
        ]
        for argument in method.args.args:
            if argument.arg == cache_name:
                argument.arg = "cache_bytes"
                argument.annotation = ast.parse(ast.unparse(byte_annotation), mode="eval").body

        class Rewrite(ast.NodeTransformer):
            def visit_Assign(self, node):
                if (
                    any(isinstance(t, ast.Name) and t.id in remove for t in node.targets)
                    and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Attribute)
                    and node.value.func.attr == "reshape"
                ):
                    return None
                if (
                    name == "append"
                    and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Attribute)
                    and node.value.func.attr == "append_kernel"
                ):
                    node.targets = [ast.Name(id="cache_bytes", ctx=ast.Store())]
                if name == "convolve" and any(
                    isinstance(t, ast.Name) and t.id in {"h0", "h1", "h2"} for t in node.targets
                ):
                    if isinstance(node.value, ast.Name) or (
                        isinstance(node.value, ast.Call)
                        and isinstance(node.value.func, ast.Attribute)
                        and node.value.func.attr == "slice"
                    ):
                        node.value = ast.parse(
                            f'pl.cast(pl.cast({ast.unparse(node.value)},pl.FP32),pl.BF16,mode="rint")',
                            mode="eval",
                        ).body
                return self.generic_visit(node)

            def visit_Call(self, node):
                if isinstance(node.func, ast.Attribute) and node.func.attr == name + "_kernel":
                    removed = remove | ({"value_cache"} if name.startswith("append") else set())
                    node.args = [
                        arg for arg in node.args if not isinstance(arg, ast.Name) or arg.id not in removed
                    ]
                return self.generic_visit(node)

            def visit_Name(self, node):
                if node.id == cache_name and (name != "append_kernel"):
                    return ast.copy_location(ast.Name(id="cache_bytes", ctx=node.ctx), node)
                return node

            def visit_Return(self, node):
                if name.startswith("append"):
                    return ast.copy_location(
                        ast.Return(value=ast.Name(id="cache_bytes", ctx=ast.Load())), node
                    )
                return self.generic_visit(node)

        method.body = [Rewrite().visit(n) for n in method.body]
        method.body = [n for n in method.body if n is not None]
        if kind in ("conv_state", "recurrent_state"):
            if isinstance(method.returns, ast.Subscript) and isinstance(method.returns.slice, ast.Tuple):
                method.returns.slice.elts[-1] = ast.parse(ast.unparse(byte_annotation), mode="eval").body
        if name.startswith("append"):
            method.returns = ast.parse(ast.unparse(byte_annotation), mode="eval").body
        if preamble:
            if name.endswith("_kernel"):
                method.body = preamble + method.body
            else:
                placed = False
                for node in ast.walk(method):
                    is_spmd = (
                        isinstance(node, ast.For)
                        and isinstance(node.iter, ast.Call)
                        and isinstance(node.iter.func, ast.Attribute)
                        and node.iter.func.attr == "spmd"
                    )
                    if isinstance(node, ast.With):
                        is_spmd = any(
                            isinstance(i.context_expr, ast.Call)
                            and isinstance(i.context_expr.func, ast.Attribute)
                            and i.context_expr.func.attr == "spmd"
                            for i in node.items
                        )
                    if is_spmd:
                        node.body = preamble + node.body
                        placed = True
                        break
                if not placed:
                    raise ValueError("Convolution cache reinterpretation has no InCore scope.")
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n"
