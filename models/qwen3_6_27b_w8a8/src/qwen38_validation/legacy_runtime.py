# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Caller-owned tensor/stream adapter for a prepared Simpler graph execution."""

from __future__ import annotations

import ctypes as C
import os
from dataclasses import dataclass
from pathlib import Path


class CallConfig(C.Structure):
    _pack_ = 1
    _fields_ = [
        ("aicpu_thread_num", C.c_int32),
        ("enable_chip_swimlane", C.c_int32),
        ("enable_dump_args", C.c_int32),
        ("enable_pmu", C.c_int32),
        ("enable_dep_gen", C.c_int32),
        ("enable_scope_stats", C.c_int32),
        ("runtime_env", C.c_uint64 * 12),
        ("output_prefix", C.c_char * 1024),
    ]


class NativeRunProvenance(C.Structure):
    _fields_ = [
        ("owner_instance_id", C.c_uint8 * 8),
        ("chip_rank", C.c_uint32),
        ("capture_index", C.c_uint32),
        ("task_slot", C.c_uint32),
        ("group_index", C.c_uint32),
        ("group_size", C.c_uint32),
        ("reserved", C.c_uint32),
    ]


class NativeRunDescriptor(C.Structure):
    _fields_ = [
        ("pipeline_slot", C.c_uint32),
        ("arena_bank", C.c_uint32),
        ("run_id", C.c_uint64),
        ("generation", C.c_uint64),
        ("dispatch_id", C.c_uint64),
        ("run_epoch", C.c_uint64),
        ("accepted_state", C.POINTER(C.c_int32)),
        ("accepted_value", C.c_int32),
        ("joinable_boundary", C.c_uint32),
        ("provenance", NativeRunProvenance),
    ]


def checked(rc, operation):
    if rc:
        raise RuntimeError(f"{operation}: runtime error {rc}")


@dataclass(frozen=True)
class DeviceTensorBinding:
    tensor: object
    shape: tuple[int, ...]


class BorrowedGraphRuntime:
    """Retain caller buffers; enqueue on its stream without a service process.

    Graph destruction and replay completion must precede ``close``. This object
    never owns the caller's device/context and issues no device reset.
    """

    def __init__(
        self,
        binaries: Path,
        device: int,
        callable_image: bytes,
        *,
        aicpu_threads=2,
        ring_task_window=0,
        ring_heap=0,
        ring_dep_pool=0,
    ):
        self.pid = os.getpid()
        self.device = device
        self.config = CallConfig(aicpu_thread_num=aicpu_threads)
        self.config.runtime_env[:] = [ring_task_window] * 4 + [ring_heap] * 4 + [ring_dep_pool] * 4
        self.lib = C.CDLL(
            str(binaries / "onboard/tensormap_and_ringbuffer/libhost_runtime.so"), mode=C.RTLD_LOCAL
        )
        signatures = {
            "create_device_context": ([], C.c_void_p),
            "destroy_device_context": ([C.c_void_p], None),
            "finalize_device": ([C.c_void_p], C.c_int),
            "get_runtime_size": ([], C.c_size_t),
            "get_runtime_alignment": ([], C.c_size_t),
            "simpler_kernel_mode_init": (
                [
                    C.c_void_p,
                    C.c_int,
                    C.c_char_p,
                    C.c_size_t,
                    C.c_char_p,
                    C.c_size_t,
                    C.c_char_p,
                    C.c_size_t,
                    C.POINTER(CallConfig),
                    C.c_uint64,
                ],
                C.c_int,
            ),
            "simpler_kernel_mode_prepare_callable": (
                [C.c_void_p, C.c_int32, C.c_void_p, C.c_size_t],
                C.c_int,
            ),
            "simpler_prepare_run": (
                [
                    C.c_void_p,
                    C.c_void_p,
                    C.c_int32,
                    C.c_void_p,
                    C.POINTER(CallConfig),
                    C.POINTER(NativeRunDescriptor),
                ],
                C.c_int,
            ),
            "simpler_launch_run_on_stream": ([C.c_void_p] * 4, C.c_int),
            "simpler_prepare_graph_run": ([C.c_void_p] * 2, C.c_int),
            "simpler_enqueue_graph_run": ([C.c_void_p] * 3, C.c_int),
            "simpler_wait_run": ([C.c_void_p] * 2, C.c_int),
            "simpler_finalize_run": ([C.c_void_p] * 2, C.c_int),
        }
        for name, (args, result) in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = args, result
        self.ctx = self.lib.create_device_context()
        if not self.ctx:
            raise RuntimeError("create_device_context returned null")
        self.retained = None
        self.args = None
        self.prepared = self.graph_ready = False
        self.epoch = 0
        self.callable_id = 0
        aicpu = (binaries / "onboard/tensormap_and_ringbuffer/libaicpu_kernel.so").read_bytes()
        aicore = (binaries / "onboard/tensormap_and_ringbuffer/aicore_kernel.o").read_bytes()
        dispatcher = (binaries / "dispatcher/libsimpler_aicpu_dispatcher.so").read_bytes()
        checked(
            self.lib.simpler_kernel_mode_init(
                self.ctx,
                device,
                aicpu,
                len(aicpu),
                aicore,
                len(aicore),
                dispatcher,
                len(dispatcher),
                C.byref(self.config),
                1,
            ),
            "borrowed init",
        )
        checked(
            self.lib.simpler_kernel_mode_prepare_callable(self.ctx, 0, callable_image, len(callable_image)),
            "prepare callable",
        )
        size, alignment = self.lib.get_runtime_size(), self.lib.get_runtime_alignment()
        self.storage = C.create_string_buffer(size + alignment)
        self.run = (C.addressof(self.storage) + alignment - 1) & -alignment

    def register_callable(self, callable_id, image):
        if self.prepared:
            raise RuntimeError("Cannot register while graph is live")
        checked(
            self.lib.simpler_kernel_mode_prepare_callable(self.ctx, callable_id, image, len(image)),
            "prepare extra callable",
        )

    def bind(self, tensors, scalars=(), *, callable_id=0):
        import torch
        from simpler.task_interface import AddressSpace, ChipStorageTaskArgs, ChipTensor, DataType

        if self.prepared:
            raise RuntimeError("Cannot replace a prepared graph argument list")
        self.callable_id = callable_id
        self.retained = tuple(tensors)
        self.args = ChipStorageTaskArgs()
        dtypes = {
            torch.float32: DataType.FLOAT32,
            torch.bfloat16: DataType.BFLOAT16,
            torch.int8: DataType.INT8,
            torch.int32: DataType.INT32,
            torch.int64: DataType.INT64,
            torch.uint8: DataType.UINT8,
        }
        for value in tensors:
            tensor = value.tensor if isinstance(value, DeviceTensorBinding) else value
            shape = value.shape if isinstance(value, DeviceTensorBinding) else tuple(tensor.shape)
            if (
                tensor.device.type != "npu"
                or tensor.device.index != self.device
                or not tensor.is_contiguous()
            ):
                raise ValueError("Expected contiguous caller-owned tensors on the bound NPU")
            self.args.add_tensor(
                ChipTensor.make(
                    tensor.data_ptr(), shape, dtypes[tensor.dtype], address_space=AddressSpace.DEVICE
                )
            )

        for scalar in scalars:
            self.args.add_scalar(C.c_uint64(scalar))

    def prepare(self):
        if os.getpid() != self.pid or self.prepared or self.args is None:
            raise RuntimeError("Invalid process or argument lifecycle")
        self.epoch += 1
        self.descriptor = NativeRunDescriptor(generation=1, run_epoch=self.epoch)
        checked(
            self.lib.simpler_prepare_run(
                self.ctx,
                self.run,
                self.callable_id,
                self.args.__ptr__(),
                C.byref(self.config),
                C.byref(self.descriptor),
            ),
            "prepare run",
        )
        self.prepared = True

    def eager(self, stream):
        self.prepare()
        checked(
            self.lib.simpler_launch_run_on_stream(self.ctx, self.run, int(stream), None),
            "launch on caller stream",
        )

    def prepare_graph(self):
        self.prepare()
        checked(self.lib.simpler_prepare_graph_run(self.ctx, self.run), "prepare borrowed graph")
        self.graph_ready = True

    def enqueue_graph(self, stream):
        if not self.graph_ready or os.getpid() != self.pid:
            raise RuntimeError("Graph is not prepared in this process")
        checked(self.lib.simpler_enqueue_graph_run(self.ctx, self.run, int(stream)), "enqueue borrowed graph")

    def finish_eager(self):
        checked(self.lib.simpler_wait_run(self.ctx, self.run), "wait run")
        checked(self.lib.simpler_finalize_run(self.ctx, self.run), "finalize run")
        self.prepared = False

    def close(self):
        if self.prepared:
            checked(self.lib.simpler_finalize_run(self.ctx, self.run), "finalize prepared run")
            self.prepared = self.graph_ready = False
        checked(self.lib.finalize_device(self.ctx), "close borrowed resources")
        self.lib.destroy_device_context(self.ctx)
        self.ctx = None
        self.retained = self.args = None
