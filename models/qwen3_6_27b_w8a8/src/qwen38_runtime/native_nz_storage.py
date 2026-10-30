# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Native NZ storage shared by vLLM and PTO, with retained allocation owners.

The device-preloaded path converts checkpoint-order ND directly into the final NZ
bank. Legacy NZ copying and CPU staging experiments remain explicit alternatives.
These are layout permutations only; no weight quantization or rounding occurs.
"""

from dataclasses import dataclass
import ctypes as C
import torch
import torch_npu


@dataclass
class NativeNzBank:
    storage: torch.Tensor
    weights: tuple[torch.Tensor, ...]
    backing: object = None

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

    @classmethod
    def from_nd_weights(cls, weights):
        """One batched conversion preserving each layer as a batch dimension.

        Never flatten layers into the N axis: its NZ byte order would differ.
        No NZ-to-NZ repacking is performed. Raw aliases retain the batch owner.
        """
        weights = tuple(weights)
        if not weights:
            raise ValueError("An NZ bank requires weights")
        first = weights[0]
        n, k = first.shape
        block = 32 if first.dtype == torch.int8 else 16
        if n % 16 or k % block:
            raise ValueError("Batch NZ requires whole aligned fractals")
        for weight in weights:
            if weight.shape != first.shape or weight.dtype != first.dtype or weight.device != first.device:
                raise ValueError("Batch family shape/dtype/device mismatch")
            if not weight.is_contiguous() or torch_npu.get_npu_format(weight) not in (0, 2):
                raise ValueError("W1 requires deferred, contiguous native ND matrices")
        batch = torch.stack(weights, dim=0)
        nz = torch_npu.npu_format_cast(batch, 29)
        torch.npu.current_stream(first.device).synchronize()
        slab_bytes = n * k * first.element_size()
        total = len(weights) * slab_bytes
        if nz.untyped_storage().nbytes() != total:
            raise ValueError("Unexpected NZ padding; refusing unsafe layer aliases")

        def alias(ptr, size, stride, dtype, nbytes, fmt):
            raw = torch_npu._C._construct_storage_from_data_pointer(ptr, first.device, nbytes)
            meta = dict(
                size=torch.Size(size),
                stride=tuple(stride),
                nbytes=nbytes,
                dtype=dtype,
                data_ptr=ptr,
                storage_offset=0,
                device=first.device,
                npu_format=fmt,
                layout=torch.strided,
                requires_grad=False,
            )
            return torch_npu._C._construct_NPU_Tensor_From_Storage_And_Metadata(meta, raw)

        storage = alias(nz.data_ptr(), (len(weights), slab_bytes), (slab_bytes, 1), torch.uint8, total, 2)
        aliases = tuple(
            alias(nz.data_ptr() + i * slab_bytes, first.shape, first.stride(), first.dtype, slab_bytes, 29)
            for i in range(len(weights))
        )
        return cls(storage, aliases, nz)


def validate_batched_physical_bytes(bank, nd_weights):
    """Initialization-only oracle; reference conversions are not W1 workload.

    Compare physical bytes, including the layer-batch layout, before parameters
    are rebound. One reference slab is alive at a time to bound validation memory.
    """
    for index, weight in enumerate(nd_weights):
        reference_input = (
            weight.detach().to(bank.storage.device)
            if weight.device.type == "cpu"
            else weight.detach().clone(memory_format=torch.contiguous_format)
        )
        reference = torch_npu.npu_format_cast(reference_input, 29)
        slab_bytes = reference.untyped_storage().nbytes()
        if slab_bytes != bank.storage.shape[1]:
            raise ValueError(
                f"Reference NZ slab size {slab_bytes} differs from bank layer {bank.storage.shape[1]}, source {tuple(weight.shape)}"
            )
        raw = torch_npu._C._construct_storage_from_data_pointer(
            reference.data_ptr(), reference.device, slab_bytes
        )
        metadata = dict(
            size=torch.Size((slab_bytes,)),
            stride=(1,),
            nbytes=slab_bytes,
            dtype=torch.uint8,
            data_ptr=reference.data_ptr(),
            storage_offset=0,
            device=reference.device,
            npu_format=2,
            layout=torch.strided,
            requires_grad=False,
        )
        physical = torch_npu._C._construct_NPU_Tensor_From_Storage_And_Metadata(metadata, raw)
        # CPU copies deliberately keep this oracle outside startup/perf claims.
        if not torch.equal(physical.cpu(), bank.storage[index].cpu()):
            raise ValueError(f"Batched NZ physical bytes differ for layer {index}")
        del physical, raw, reference, reference_input
    return len(nd_weights)


def pack_cpu_nz(batch):
    """Aligned BF16/INT8 ND [layers,n,k] -> physical NZ bytes on CPU.

    Pure byte permutation, no arithmetic, quantization or dtype conversion.
    Output axes match native [layers,k/C0,n/16,16,C0].
    """
    if batch.device.type != "cpu" or batch.ndim != 3 or not batch.is_contiguous():
        raise ValueError("CPU packing requires one directly loaded contiguous bank")
    if batch.dtype not in (torch.bfloat16, torch.int8):
        raise ValueError("Unsupported NZ dtype")
    layers, n, k = batch.shape
    block = 32 if batch.dtype == torch.int8 else 16
    if n % 16 or k % block:
        raise ValueError("Only whole native fractals are supported")
    return batch.reshape(layers, n // 16, 16, k // block, block).permute(0, 3, 1, 2, 4).contiguous()


def from_cpu_bank(batch, device):
    """One H2D into final NZ storage; zero NPU ND stack/rebank/format casts."""
    packed = pack_cpu_nz(batch)
    layers, n, k = batch.shape
    slab = n * k * batch.element_size()
    storage = packed.view(torch.uint8).reshape(layers, slab).to(device=device)
    torch.npu.current_stream(device).synchronize()
    aliases = []
    for index in range(layers):
        ptr = storage.data_ptr() + index * slab
        raw = torch_npu._C._construct_storage_from_data_pointer(ptr, device, slab)
        metadata = dict(
            size=torch.Size((n, k)),
            stride=(k, 1),
            nbytes=slab,
            dtype=batch.dtype,
            data_ptr=ptr,
            storage_offset=0,
            device=device,
            npu_format=29,
            layout=torch.strided,
            requires_grad=False,
        )
        aliases.append(torch_npu._C._construct_NPU_Tensor_From_Storage_And_Metadata(metadata, raw))
    return NativeNzBank(storage, tuple(aliases))


def from_preloaded_nd_bank(batch, device):
    """Direct-loaded CPU ND family -> one upload -> one NPU NZ conversion.

    No stack, CPU NZ permutation, device ND repacking or NZ-to-bank copy.
    Native aliases refer directly to the format conversion's output owner.
    """
    if batch.device.type != "cpu" or batch.ndim != 3 or not batch.is_contiguous():
        raise ValueError("Expected the contiguous, directly loaded ND bank")
    if batch.dtype not in (torch.bfloat16, torch.int8):
        raise ValueError("Unsupported bank dtype")
    layers, n, k = batch.shape
    block = 32 if batch.dtype == torch.int8 else 16
    if n % 16 or k % block:
        raise ValueError("Only aligned native fractals are supported")
    return from_device_nd_bank(batch.to(device=device))


def from_device_nd_bank(device_batch):
    """Consume the final preallocated device ND bank, optionally transposed."""
    if device_batch.device.type != "npu" or device_batch.ndim != 3:
        raise ValueError("Expected a device ND family")
    device = device_batch.device
    layers, n, k = device_batch.shape
    block = 32 if device_batch.dtype == torch.int8 else 16
    if device_batch.dtype not in (torch.bfloat16, torch.int8) or n % 16 or k % block:
        raise ValueError("Unsupported native NZ shape/dtype")
    nz = torch_npu.npu_format_cast(device_batch, 29)
    torch.npu.current_stream(device).synchronize()
    slab = n * k * device_batch.element_size()
    total = layers * slab
    if torch_npu.get_npu_format(nz) != 29 or nz.untyped_storage().nbytes() != total:
        raise ValueError("Unexpected batched NZ layout or padded allocation")

    def alias(ptr, shape, stride, dtype, nbytes, fmt):
        raw = torch_npu._C._construct_storage_from_data_pointer(ptr, device, nbytes)
        metadata = dict(
            size=torch.Size(shape),
            stride=stride,
            nbytes=nbytes,
            dtype=dtype,
            data_ptr=ptr,
            storage_offset=0,
            device=device,
            npu_format=fmt,
            layout=torch.strided,
            requires_grad=False,
        )
        return torch_npu._C._construct_NPU_Tensor_From_Storage_And_Metadata(metadata, raw)

    storage = alias(nz.data_ptr(), (layers, slab), (slab, 1), torch.uint8, total, 2)
    weights = tuple(
        alias(nz.data_ptr() + i * slab, (n, k), (k, 1), device_batch.dtype, slab, 29) for i in range(layers)
    )
    return NativeNzBank(storage, weights, nz)


def from_checkpoint_device_bank(batch, *, quantized):
    """Checkpoint-order device ND -> final native NZ allocation in one pass.

    INT8 checkpoints are [out,in], but native matmul uses [in,out]. A single
    five-axis device permutation performs transpose AND fractal packing. It
    writes the final NZ bytes directly, without an intermediate transposed ND.
    BF16 uses the native batch format converter without a logical transpose.
    """
    if batch.device.type != "npu" or batch.ndim != 3 or not batch.is_contiguous():
        raise ValueError("Expected the directly loaded device bank")
    if not quantized:
        return from_device_nd_bank(batch)
    if batch.dtype != torch.int8:
        raise ValueError("Quantized checkpoint bank must be INT8")
    layers, out_dim, in_dim = batch.shape
    if out_dim % 32 or in_dim % 16:
        raise ValueError("INT8 native NZ requires whole fractals")
    packed = batch.view(layers, out_dim // 32, 32, in_dim // 16, 16).permute(0, 1, 3, 4, 2).contiguous()
    torch.npu.current_stream(batch.device).synchronize()
    slab = in_dim * out_dim
    device = batch.device
    storage = packed.view(torch.uint8).reshape(layers, slab)
    weights = []
    for index in range(layers):
        ptr = storage.data_ptr() + index * slab
        raw = torch_npu._C._construct_storage_from_data_pointer(ptr, device, slab)
        metadata = dict(
            size=torch.Size((in_dim, out_dim)),
            stride=(out_dim, 1),
            nbytes=slab,
            dtype=torch.int8,
            data_ptr=ptr,
            storage_offset=0,
            device=device,
            npu_format=29,
            layout=torch.strided,
            requires_grad=False,
        )
        weights.append(torch_npu._C._construct_NPU_Tensor_From_Storage_And_Metadata(metadata, raw))
    return NativeNzBank(storage, tuple(weights), packed)
