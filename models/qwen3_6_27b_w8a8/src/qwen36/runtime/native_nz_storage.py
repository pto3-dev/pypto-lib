# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Same-process native NZ storage: BF16 [out,in] or native W8A8 INT8 [in,out].

No numerical conversion occurs here. Initialization copies physical NZ bytes once,
then native Parameters must be rebound to bank.weights before old owners are released.
INT8 uses 32-column fractals; BF16 uses 16-column fractals.
"""

from dataclasses import dataclass
import ctypes as C
import torch
import torch_npu


@dataclass
class NativeNzBank:
    storage: torch.Tensor
    weights: tuple[torch.Tensor, ...]

    @classmethod
    def from_weights(cls, weights, stream=None):
        weights = tuple(weights)
        if not weights:
            raise ValueError("An NZ bank requires at least one native weight")
        first = weights[0]
        if first.ndim != 2 or first.dtype not in (torch.bfloat16, torch.int8) or first.device.type != "npu":
            raise ValueError("Expected native BF16 or INT8 matrix weights")
        n, k = first.shape
        column_block = 32 if first.dtype == torch.int8 else 16
        if n % 16 or k % column_block:
            raise ValueError("Initial NZ bank only supports whole aligned fractals")
        slab_bytes = n * k * first.element_size()
        for weight in weights:
            if (
                weight.shape != first.shape
                or weight.dtype != first.dtype
                or weight.device != first.device
                or torch_npu.get_npu_format(weight) != 29
                or weight.storage_offset() != 0
                or weight.untyped_storage().nbytes() != slab_bytes
            ):
                raise ValueError("NZ banks require independent, identically shaped native NZ matrices")
        storage = torch.empty((len(weights), slab_bytes), dtype=torch.uint8, device=first.device)
        stream = stream or torch.npu.current_stream(first.device)
        # Flush native queued format conversions before issuing raw ACL copies.
        # PyTorch task-queue submission and direct ACL submission are different APIs.
        stream.synchronize()
        acl = C.CDLL("libascendcl.so")
        copy = acl.aclrtMemcpyAsync
        copy.argtypes = [C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t, C.c_int, C.c_void_p]
        copy.restype = C.c_int
        aliases = []
        for i, weight in enumerate(weights):
            ptr = storage.data_ptr() + i * slab_bytes
            rc = copy(ptr, slab_bytes, weight.data_ptr(), slab_bytes, 3, stream.npu_stream)
            if rc:
                raise RuntimeError(f"Native NZ D2D copy failed: {rc}")
            raw = torch_npu._C._construct_storage_from_data_pointer(ptr, first.device, slab_bytes)
            metadata = dict(
                size=first.shape,
                stride=first.stride(),
                nbytes=slab_bytes,
                dtype=first.dtype,
                data_ptr=ptr,
                storage_offset=0,
                device=first.device,
                npu_format=29,
                layout=first.layout,
                requires_grad=False,
            )
            aliases.append(torch_npu._C._construct_NPU_Tensor_From_Storage_And_Metadata(metadata, raw))
        # The source allocations must remain alive until all raw copies finish.
        stream.synchronize()
        return cls(storage, tuple(aliases))

    @property
    def physical_shape(self):
        n, k = self.weights[0].shape
        column_block = 32 if self.weights[0].dtype == torch.int8 else 16
        return len(self.weights), k // column_block, n // 16, 16, column_block
