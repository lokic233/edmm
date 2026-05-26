# SPDX-License-Identifier: Apache-2.0
"""
EDMM VMM-backed KV cache allocator.

Replaces torch.zeros allocation with CUDA Virtual Memory Management (VMM)
backed allocation. The resulting tensor has the same VA, dtype, and shape
as the standard allocation, but each page-sized region is backed by an
independently remappable physical allocation handle.

Usage:
    Set VLLM_EDMM_ENABLE=1 to activate. Falls back to torch.zeros otherwise.
"""
import ctypes
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import torch

logger = logging.getLogger(__name__)

EDMM_ENABLED = os.environ.get("VLLM_EDMM_ENABLE", "0") == "1"

# CUDA driver constants
CU_MEM_ALLOCATION_TYPE_PINNED = 1
CU_MEM_LOCATION_TYPE_DEVICE = 1
CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3


class _CUmemAllocationProp(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleTypes", ctypes.c_int),
        ("location_type", ctypes.c_int),
        ("location_id", ctypes.c_int),
        ("win32_security", ctypes.c_void_p),
        ("_pad", ctypes.c_ubyte * 24),
    ]


class _CUmemAccessDesc(ctypes.Structure):
    _fields_ = [
        ("location_type", ctypes.c_int),
        ("location_id", ctypes.c_int),
        ("flags", ctypes.c_int),
    ]


class _ExternalCUDABuffer:
    """Wraps a raw CUDA VA pointer so torch.as_tensor can consume it."""

    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|i1",
            "data": (ptr, False),
            "version": 3,
        }


def _load_cuda_driver() -> ctypes.CDLL:
    lib = ctypes.CDLL("libcuda.so.1")
    lib.cudart = ctypes.CDLL("libcudart.so")

    _p = ctypes.POINTER
    _u64 = ctypes.c_uint64
    _sz = ctypes.c_size_t
    _i = ctypes.c_int
    _vp = ctypes.c_void_p

    lib.cuMemAddressReserve.argtypes = [_p(_u64), _sz, _sz, _u64, ctypes.c_ulonglong]
    lib.cuMemAddressReserve.restype = _i
    lib.cuMemCreate.argtypes = [_p(_u64), _sz, _vp, ctypes.c_ulonglong]
    lib.cuMemCreate.restype = _i
    lib.cuMemMap.argtypes = [_u64, _sz, _sz, _u64, ctypes.c_ulonglong]
    lib.cuMemMap.restype = _i
    lib.cuMemSetAccess.argtypes = [_u64, _sz, _vp, _sz]
    lib.cuMemSetAccess.restype = _i
    lib.cuMemUnmap.argtypes = [_u64, _sz]
    lib.cuMemUnmap.restype = _i
    lib.cuMemRelease.argtypes = [_u64]
    lib.cuMemRelease.restype = _i
    lib.cuMemAddressFree.argtypes = [_u64, _sz]
    lib.cuMemAddressFree.restype = _i
    lib.cuMemGetAllocationGranularity.argtypes = [_p(_sz), _vp, _i]
    lib.cuMemGetAllocationGranularity.restype = _i

    return lib


@dataclass
class VMMAllocation:
    va_ptr: int
    total_bytes: int
    page_size: int
    handles: list = field(default_factory=list)
    tensor: Optional[torch.Tensor] = None

    @property
    def num_pages(self) -> int:
        return len(self.handles)


_cuda: Optional[ctypes.CDLL] = None
_allocations: dict[int, VMMAllocation] = {}


def _get_cuda() -> ctypes.CDLL:
    global _cuda
    if _cuda is None:
        _cuda = _load_cuda_driver()
        _cuda.cuInit(0)
    return _cuda


def _ensure_driver_context(device_id: int) -> None:
    """Retain and push the primary CUDA context so driver API calls work
    alongside PyTorch's runtime API context."""
    cuda = _get_cuda()
    dev = ctypes.c_int()
    cuda.cuDeviceGet(ctypes.byref(dev), device_id)
    ctx = ctypes.c_void_p()
    cuda.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev)
    cuda.cuCtxPushCurrent_v2(ctx)


def _check(result: int, msg: str) -> None:
    if result != 0:
        raise RuntimeError(f"CUDA driver error {result}: {msg}")


def allocate_vmm_kv_cache(
    size_bytes: int,
    device: torch.device,
) -> torch.Tensor:
    cuda = _get_cuda()
    device_id = device.index if device.index is not None else 0
    _ensure_driver_context(device_id)

    prop = _CUmemAllocationProp()
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location_type = CU_MEM_LOCATION_TYPE_DEVICE
    prop.location_id = device_id

    granularity = ctypes.c_size_t()
    _check(
        cuda.cuMemGetAllocationGranularity(
            ctypes.byref(granularity), ctypes.byref(prop), 0
        ),
        "cuMemGetAllocationGranularity",
    )
    page_size = max(granularity.value, 2 * 1024 * 1024)

    aligned_size = ((size_bytes + page_size - 1) // page_size) * page_size

    va = ctypes.c_uint64()
    _check(
        cuda.cuMemAddressReserve(ctypes.byref(va), aligned_size, 0, 0, 0),
        "cuMemAddressReserve",
    )

    handles = []
    offset = 0
    while offset < aligned_size:
        chunk = min(page_size, aligned_size - offset)
        handle = ctypes.c_uint64()
        _check(
            cuda.cuMemCreate(ctypes.byref(handle), chunk, ctypes.byref(prop), 0),
            f"cuMemCreate at offset {offset}",
        )
        _check(
            cuda.cuMemMap(va.value + offset, chunk, 0, handle, 0),
            f"cuMemMap at offset {offset}",
        )
        handles.append(handle.value)
        offset += chunk

    access = _CUmemAccessDesc()
    access.location_type = CU_MEM_LOCATION_TYPE_DEVICE
    access.location_id = device_id
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    _check(
        cuda.cuMemSetAccess(va.value, aligned_size, ctypes.byref(access), 1),
        "cuMemSetAccess",
    )

    cuda.cudart.cudaMemset(ctypes.c_void_p(va.value), 0, ctypes.c_size_t(aligned_size))
    torch.cuda.synchronize()

    buf = _ExternalCUDABuffer(va.value, size_bytes)
    tensor = torch.as_tensor(buf, device=f"cuda:{device_id}")

    alloc = VMMAllocation(
        va_ptr=va.value,
        total_bytes=aligned_size,
        page_size=page_size,
        handles=handles,
        tensor=tensor,
    )
    _allocations[va.value] = alloc

    logger.info(
        "EDMM: allocated %d bytes (%d pages x %d MB) at VA 0x%x",
        size_bytes,
        len(handles),
        page_size // (1024 * 1024),
        va.value,
    )

    return tensor


def get_vmm_allocation(tensor: torch.Tensor) -> Optional[VMMAllocation]:
    return _allocations.get(tensor.data_ptr())


_remap_lock = __import__("threading").Lock()


def remap_block(
    tensor: torch.Tensor,
    block_id: int,
    new_physical_handle: int,
) -> int:
    """Swap the physical backing of a single block without changing its VA.

    Args:
        tensor: The VMM-backed KV cache tensor.
        block_id: Which page-sized block to remap (0-indexed).
        new_physical_handle: A cuMemGenericAllocationHandle obtained from
            cuMemCreate, already populated with the desired data.

    Returns:
        The old physical handle that was unmapped (caller may cuMemRelease it).

    Raises:
        ValueError: If the tensor is not VMM-backed or block_id is out of range.
    """
    alloc = get_vmm_allocation(tensor)
    if alloc is None:
        raise ValueError("Tensor is not VMM-backed")
    if block_id < 0 or block_id >= alloc.num_pages:
        raise ValueError(f"block_id {block_id} out of range [0, {alloc.num_pages})")

    cuda = _get_cuda()
    block_va = alloc.va_ptr + block_id * alloc.page_size

    access = _CUmemAccessDesc()
    access.location_type = CU_MEM_LOCATION_TYPE_DEVICE
    access.location_id = 0
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE

    with _remap_lock:
        _check(
            cuda.cuMemUnmap(block_va, alloc.page_size),
            f"cuMemUnmap block {block_id}",
        )
        _check(
            cuda.cuMemMap(block_va, alloc.page_size, 0, new_physical_handle, 0),
            f"cuMemMap block {block_id}",
        )
        _check(
            cuda.cuMemSetAccess(block_va, alloc.page_size, ctypes.byref(access), 1),
            f"cuMemSetAccess block {block_id}",
        )
        old_handle = alloc.handles[block_id]
        alloc.handles[block_id] = new_physical_handle

    torch.cuda.synchronize()
    return old_handle


def create_physical_handle(
    size_bytes: int,
    device_id: int = 0,
) -> int:
    """Allocate a new physical memory handle (for use with remap_block)."""
    cuda = _get_cuda()
    _ensure_driver_context(device_id)

    prop = _CUmemAllocationProp()
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location_type = CU_MEM_LOCATION_TYPE_DEVICE
    prop.location_id = device_id

    handle = ctypes.c_uint64()
    _check(
        cuda.cuMemCreate(ctypes.byref(handle), size_bytes, ctypes.byref(prop), 0),
        "cuMemCreate (new handle)",
    )
    return handle.value


def release_physical_handle(handle: int) -> None:
    """Release a physical memory handle."""
    cuda = _get_cuda()
    _check(cuda.cuMemRelease(handle), "cuMemRelease")


def free_vmm_kv_cache(tensor: torch.Tensor) -> None:
    alloc = _allocations.pop(tensor.data_ptr(), None)
    if alloc is None:
        return

    cuda = _get_cuda()
    cuda.cuMemUnmap(alloc.va_ptr, alloc.total_bytes)
    for h in alloc.handles:
        cuda.cuMemRelease(h)
    cuda.cuMemAddressFree(alloc.va_ptr, alloc.total_bytes)

    logger.info("EDMM: freed VMM allocation at VA 0x%x", alloc.va_ptr)
