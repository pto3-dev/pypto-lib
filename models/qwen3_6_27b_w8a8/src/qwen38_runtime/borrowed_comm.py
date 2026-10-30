# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""HCCL/VMM windows on a caller-owned context; no ChipWorker subprocess."""

import ctypes as C
from math import prod
import torch, torch_npu
from qwen38_runtime.borrowed_runtime import checked


class BorrowedCommWindows:
    def __init__(self, library, rank, size, stream, rootinfo_path, buffers):
        self.lib = library
        self.handle = None
        self.buffers = {}
        self.contexts = {}
        signatures = {
            "comm_init": ([C.c_int, C.c_int, C.c_void_p, C.c_char_p], C.c_void_p),
            "comm_alloc_windows": ([C.c_void_p, C.c_size_t, C.POINTER(C.c_uint64)], C.c_int),
            "comm_get_local_window_base": ([C.c_void_p, C.POINTER(C.c_uint64)], C.c_int),
            "comm_derive_context": (
                [
                    C.c_void_p,
                    C.POINTER(C.c_uint32),
                    C.c_size_t,
                    C.c_uint32,
                    C.c_size_t,
                    C.c_size_t,
                    C.POINTER(C.c_uint64),
                ],
                C.c_int,
            ),
            "comm_barrier": ([C.c_void_p], C.c_int),
            "comm_destroy": ([C.c_void_p], C.c_int),
        }
        for name, (args, result) in signatures.items():
            fn = getattr(library, name)
            fn.argtypes = args
            fn.restype = result
        offset = 0
        regions = []
        for name, shape, dtype in buffers:
            offset = (offset + 511) // 512 * 512
            nbytes = prod(shape) * torch.empty((), dtype=dtype).element_size()
            regions.append((name, shape, dtype, offset, nbytes))
            offset += nbytes
        self.handle = library.comm_init(rank, size, stream.npu_stream, str(rootinfo_path).encode())
        if not self.handle:
            raise RuntimeError("Borrowed comm_init failed")
        context = C.c_uint64()
        checked(library.comm_alloc_windows(self.handle, offset, C.byref(context)), "comm windows")
        base = C.c_uint64()
        checked(library.comm_get_local_window_base(self.handle, C.byref(base)), "comm base")
        ranks = (C.c_uint32 * size)(*range(size))
        device = torch.device(f"npu:{stream.device_index}")
        acl = C.CDLL("libascendcl.so")
        zero = acl.aclrtMemsetAsync
        zero.argtypes = [C.c_void_p, C.c_size_t, C.c_int, C.c_size_t, C.c_void_p]
        zero.restype = C.c_int
        checked(zero(base.value, offset, 0, offset, stream.npu_stream), "zero comm windows")
        stream.synchronize()
        checked(library.comm_barrier(self.handle), "initial comm barrier")
        for name, shape, dtype, offset, nbytes in regions:
            context = C.c_uint64()
            checked(
                library.comm_derive_context(self.handle, ranks, size, rank, offset, nbytes, C.byref(context)),
                "derive buffer context",
            )
            ptr = base.value + offset
            raw = torch_npu._C._construct_storage_from_data_pointer(ptr, device, nbytes)
            stride = []
            value = 1
            for dim in reversed(shape):
                stride.insert(0, value)
                value *= dim
            metadata = dict(
                size=torch.Size(shape),
                stride=tuple(stride),
                nbytes=nbytes,
                dtype=dtype,
                data_ptr=ptr,
                storage_offset=0,
                device=device,
                npu_format=2,
                layout=torch.strided,
                requires_grad=False,
            )
            self.buffers[name] = torch_npu._C._construct_NPU_Tensor_From_Storage_And_Metadata(metadata, raw)
            self.contexts[name] = context.value

    def close(self):
        if self.handle is not None:
            checked(self.lib.comm_destroy(self.handle), "destroy borrowed comm")
            self.handle = None
            self.buffers.clear()
            self.contexts.clear()
