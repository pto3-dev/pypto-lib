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
"""TP2 MLP kernels borrowed from the selected resident PyPTO implementation.

The enclosing native vLLM model owns normalization, attention and HCCL. One
Simpler ChipWorker per vLLM rank executes all local MLP layers. This operator
bridge does not create an L3 MPMD worker. Packed weights reuse native storage.
"""

import ast
import atexit
import hashlib
import importlib.util
from pathlib import Path
import sys
import time
import textwrap
from types import MethodType

import torch
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.forward_context import get_forward_context
from vllm.utils.torch_utils import direct_register_custom_op

HIDDEN = 5120
LOCAL_INTERMEDIATE = 8704
LOCAL_GATE_UP = 2 * LOCAL_INTERMEDIATE


def make_program(max_rows, directory, layer_count=64):
    "One callable reads live device controls on every stream-ordered launch."
    from qwen3_8_27b_mpmd.nz_tiled import make_nz_tiled_gemm
    from qwen3_8_27b_mpmd.swiglu_down import make_swiglu_down
    from qwen3_8_27b_mpmd.reusable import reusable_methods

    methods = []
    for name, fn in [
        ("project_up", make_nz_tiled_gemm(HIDDEN, LOCAL_GATE_UP)),
        ("swiglu_down", make_swiglu_down()),
    ]:
        pair = [ast.parse(t).body[0] for t in reusable_methods(name, fn, {"rows": max_rows})]
        kernel, wrapper = pair
        # Keep runtime row bounds with fixed physical argument shapes. The
        # selected arithmetic and per-core job traversal remain unchanged.
        for method in pair:
            method.args.args.append(
                ast.arg(arg="rows", annotation=ast.parse("pl.Scalar[pl.INDEX]", mode="eval").body)
            )
            method.body = [
                st
                for st in method.body
                if not (
                    isinstance(st, ast.Assign)
                    and isinstance(st.targets[0], ast.Name)
                    and st.targets[0].id == "rows"
                )
            ]
        for node in ast.walk(wrapper):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == kernel.name:
                    node.args.append(ast.Name(id="rows", ctx=ast.Load()))
                if node.func.attr == "spmd":
                    node.keywords.append(ast.keyword(arg="allow_early_resolve", value=ast.Constant(True)))
        if name == "project_up":
            arg = next(a for a in kernel.args.args if a.arg == "weight")
            arg.annotation.slice.elts.append(
                ast.Attribute(value=ast.Name(id="pl", ctx=ast.Load()), attr="NZ", ctx=ast.Load())
            )
        methods.extend(ast.unparse(ast.fix_missing_locations(m)) for m in pair)
    source = "import pypto.language as pl\n\n@pl.program\nclass MLPProgram:\n"
    source += "\n\n".join(textwrap.indent(m, "    ") for m in methods) + "\n\n"
    source += f"""    @pl.function(type=pl.FunctionType.Inline)
    def narrow(self, x: pl.Tensor[[{max_rows},{HIDDEN}],pl.FP32], out: pl.Out[pl.Tensor[[{max_rows},{HIDDEN}],pl.BF16]], rows: pl.Scalar[pl.INDEX]) -> pl.Tensor[[{max_rows},{HIDDEN}],pl.BF16]:
        available = pl.system.available_aiv_count()
        launches = pl.min(available,rows*20)
        for core in pl.spmd(launches,allow_early_resolve=True):
            for job in pl.range(core,rows*20,pl.tile.get_block_num()):
                r = job // 20
                c = job % 20 * 256
                value = pl.load(x,[r,c],[1,256])
                out = pl.store(pl.cast(value,pl.BF16,mode="rint"),[r,c],out)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self,
        x: pl.Tensor[[{max_rows},{HIDDEN}],pl.BF16],
        control: pl.Tensor[[2],pl.INT64],
"""
    for index in range(layer_count):
        source += f"        up{index}: pl.Tensor[[1360,256,256],pl.BF16],\n        down{index}: pl.Tensor[[40,34,128,256],pl.BF16],\n"
    source += f"""        packed: pl.Out[pl.Tensor[[{max_rows},{LOCAL_GATE_UP}],pl.BF16]],
        partial: pl.Out[pl.Tensor[[{max_rows},{HIDDEN}],pl.FP32]],
        out: pl.Out[pl.Tensor[[{max_rows},{HIDDEN}],pl.BF16]],
    ) -> pl.Tensor[[{max_rows},{HIDDEN}],pl.BF16]:
        layer_index = pl.tensor.read(control,[0])
        rows = pl.cast(pl.tensor.read(control,[1]),pl.INDEX)
"""

    def select(first, last, indent):
        if last - first == 1:
            return f"{indent}packed = self.project_up(x,up{first},packed,rows)\n{indent}partial = self.swiglu_down(packed,down{first},partial,rows)\n"
        mid = (first + last) // 2
        return (
            f"{indent}if layer_index < {mid}:\n"
            + select(first, mid, indent + "    ")
            + f"{indent}else:\n"
            + select(mid, last, indent + "    ")
        )

    source += select(0, layer_count, "        ")
    source += "        out = self.narrow(partial,out,rows)\n        return out\n"
    digest = hashlib.sha256(source.encode()).hexdigest()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ("mlp_" + digest[:16] + ".py")
    path.write_text(source)
    name = "_qwen38_vllm_mlp_" + digest[:16]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.MLPProgram, digest


class Qwen38MLPManager:
    "One shared graph token per rank; dynamic rows also serve eager prefill."

    def __init__(self, build_root, max_rows=512):
        self.device_id = torch.npu.current_device()
        self.build_root = Path(build_root) / f"rank{self.device_id}" / f"run-{time.time_ns()}"
        self.max_rows = max_rows
        self.worker = None
        self.layers = {}
        self.calls = {}
        self.token = None
        self.closed = False
        atexit.register(self.close)

    def prepare(self):
        if self.token is not None:
            return
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("PTO MLP must prepare before graph capture.")
        from pypto import ir
        from qwen3_8_27b_mpmd.nz_tiled import pack_nz_tiles
        from qwen3_8_27b_mpmd.tiled_weights import pack_tiles
        from simpler.task_interface import ChipWorker, CallConfig
        from simpler_setup.runtime_builder import RuntimeBuilder

        # Load/pack all local native shards once. No additional resident weight
        # pool is retained; every view refers to its original NPU allocation.
        weights = []
        self.indices = {}
        self.original_pointers = {}
        for index, (prefix, layer) in enumerate(self.layers.items()):
            up = layer.gate_up_proj.weight
            down = layer.down_proj.weight
            if tuple(up.shape) != (LOCAL_GATE_UP, HIDDEN) or tuple(down.shape) != (
                HIDDEN,
                LOCAL_INTERMEDIATE,
            ):
                raise ValueError(f"Expected TP2 ND MLP shards, got {up.shape}, {down.shape}.")
            self.indices[prefix] = index
            self.original_pointers[prefix] = [up.data_ptr(), down.data_ptr()]
            for weight, pack in ((up, pack_nz_tiles), (down, pack_tiles)):
                value = pack(weight.detach().cpu())
                with torch.no_grad():
                    weight.copy_(value.reshape(weight.shape).to(weight.device))
            weights.extend([up.view(1360, 256, 256), down.view(40, 34, 128, 256)])
        program, self.digest = make_program(self.max_rows, self.build_root / "source", len(self.layers))
        self.compiled = ir.compile(program, output_dir=str(self.build_root / "build"), platform="a2a3")
        self.worker = ChipWorker()
        self.worker.init(
            device_id=self.device_id,
            bins=RuntimeBuilder(platform="a2a3").get_binaries(self.compiled.runtime_name, build=False),
        )
        self.handle = self.worker.register_callable(self.compiled.chip_callable)
        self.config = CallConfig()
        self.config.runtime_env.ring_task_window = 1024
        self.config.runtime_env.ring_heap = 32 * 1024 * 1024
        self.config.runtime_env.ring_dep_pool = 32768
        device = torch.device(f"npu:{self.device_id}")
        self.x = torch.empty(self.max_rows, HIDDEN, dtype=torch.bfloat16, device=device)
        self.control = torch.tensor([0, 1], dtype=torch.int64, device=device)
        self.packed = torch.empty(self.max_rows, LOCAL_GATE_UP, dtype=torch.bfloat16, device=device)
        self.partial = torch.empty(self.max_rows, HIDDEN, dtype=torch.float32, device=device)
        self.out = torch.empty_like(self.x)
        self.retained = (self.x, self.control, *weights, self.packed, self.partial, self.out)
        self.arguments = self.make_arguments(self.retained)
        torch.npu.synchronize(self.device_id)
        self.token = self.worker.prepare_graph_run(self.handle, self.arguments, self.config)

    def make_arguments(self, tensors):
        from simpler.task_interface import ChipStorageTaskArgs, ChipTensor
        from simpler_setup.torch_interop import torch_dtype_to_datatype

        args = ChipStorageTaskArgs()
        for tensor in tensors:
            if not tensor.is_contiguous():
                raise ValueError("PTO MLP requires contiguous borrowed tensors.")
            args.add_tensor(
                ChipTensor.make(
                    tensor.data_ptr(),
                    tuple(tensor.shape),
                    torch_dtype_to_datatype(tensor.dtype),
                    child_memory=True,
                )
            )
        return args

    def execute(self, x, out, prefix):
        self.prepare()
        rows = x.shape[0]
        if not 0 < rows <= self.max_rows:
            raise ValueError("MLP row count exceeds the prepared capacity.")
        self.x[:rows].copy_(x)
        self.control[0].fill_(self.indices[prefix])
        self.control[1].fill_(rows)
        self.worker.enqueue_graph_run(self.token, torch.npu.current_stream(self.device_id).npu_stream)
        out.copy_(self.out[:rows])
        self.calls[rows] = self.calls.get(rows, 0) + 1

    def stats(self):
        aliases = (
            {
                prefix: self.original_pointers[prefix]
                == [layer.gate_up_proj.weight.data_ptr(), layer.down_proj.weight.data_ptr()]
                for prefix, layer in self.layers.items()
            }
            if self.token is not None
            else {}
        )
        return dict(
            pto_mlp_executor=True,
            simpler_l2_workers=1,
            simpler_l3_workers=0,
            mpmd=False,
            graph_tokens=int(self.token is not None),
            prepared_layers=len(aliases),
            python_calls_by_rows=self.calls,
            source_digest=getattr(self, "digest", None),
            build_root=str(self.build_root),
            max_rows=self.max_rows,
            jit_misses=int(self.token is not None),
            packed_weights_reuse_native_storage=bool(aliases) and all(aliases.values()),
            pto_operators=["NZ gate/up BF16", "SwiGLU/down FP32", "BF16 narrowing"],
            native_operators=["normalization", "attention", "HCCL TP reduction", "logits", "sampling"],
        )

    def close(self):
        if self.closed or self.worker is None:
            return
        torch.npu.synchronize(self.device_id)
        if self.token is not None:
            self.worker.finalize_graph_run(self.token)
        self.worker.unregister_callable(self.handle)
        self.worker.finalize()
        self.closed = True


def mlp_kernel(x: torch.Tensor, out: torch.Tensor, layer_name: str) -> None:
    layer = get_forward_context().no_compile_layers[layer_name]
    layer._pto_mlp_manager.execute(x, out, layer_name)


def mlp_kernel_fake(x: torch.Tensor, out: torch.Tensor, layer_name: str) -> None:
    return


direct_register_custom_op(
    op_name="qwen38_pto_mlp", op_func=mlp_kernel, mutates_args=["out"], fake_impl=mlp_kernel_fake
)


def mlp_forward(layer, x):
    out = torch.empty_like(x)
    torch.ops.vllm.qwen38_pto_mlp(x, out, layer._pto_mlp_prefix)
    return tensor_model_parallel_all_reduce(out)


def attach_mlp(causal_model, vllm_config, build_root):
    from vllm.distributed import get_tensor_model_parallel_world_size, get_pp_group

    if get_tensor_model_parallel_world_size() != 2 or get_pp_group().world_size != 1:
        raise ValueError("The new MLP bridge requires TP2/PP1.")
    config = causal_model.model.config
    if (config.hidden_size, config.intermediate_size, config.num_hidden_layers) != (5120, 17408, 64):
        raise ValueError("The new MLP bridge requires the dense Qwen 27B checkpoint.")
    if vllm_config.quant_config is not None or vllm_config.model_config.dtype != torch.bfloat16:
        raise ValueError("The new MLP bridge requires unquantized BF16 weights.")
    manager = Qwen38MLPManager(build_root, vllm_config.scheduler_config.max_num_batched_tokens)
    causal_model._qwen38_pto_mlp_manager = manager
    static = vllm_config.compilation_config.static_forward_context
    for index, decoder in enumerate(causal_model.model.layers):
        layer = decoder.mlp
        prefix = f"qwen38_pto_mlp_layer_{index}"
        if prefix in static:
            raise ValueError(f"Duplicate PTO MLP context entry: {prefix}")
        static[prefix] = layer
        manager.layers[prefix] = layer
        layer._pto_mlp_manager = manager
        layer._pto_mlp_prefix = prefix
        layer.forward = MethodType(mlp_forward, layer)
