"""
EDMM Phase 2 validation: Block-level page remapping.

Tests that remap_block swaps the physical backing of a block_id
while keeping the virtual address stable, and that CUDA kernels
see the new data through the same pointer.
"""

import ctypes
import os

import pytest
import torch

os.environ["VLLM_EDMM_ENABLE"] = "1"

from vllm.v1.worker.gpu.edmm_allocator import (
    allocate_vmm_kv_cache,
    create_physical_handle,
    free_vmm_kv_cache,
    get_vmm_allocation,
    release_physical_handle,
    remap_block,
)

DEVICE = torch.device("cuda:0")
PAGE_2MB = 2 * 1024 * 1024


def _fill_physical_handle_via_temp_map(handle: int, value: int, size: int):
    """Map a physical handle temporarily, fill it, then unmap."""
    cuda = ctypes.CDLL("libcuda.so.1")
    va = ctypes.c_uint64()
    cuda.cuMemAddressReserve(ctypes.byref(va), ctypes.c_size_t(size), 0, 0, 0)
    cuda.cuMemMap(va, ctypes.c_size_t(size), 0, ctypes.c_uint64(handle), 0)

    class Acc(ctypes.Structure):
        _fields_ = [("lt", ctypes.c_int), ("li", ctypes.c_int), ("f", ctypes.c_int)]

    acc = Acc(1, 0, 3)
    cuda.cuMemSetAccess(va, ctypes.c_size_t(size), ctypes.byref(acc), 1)

    rt = ctypes.CDLL("libcudart.so")
    rt.cudaMemset(ctypes.c_void_p(va.value), ctypes.c_int(value), ctypes.c_size_t(size))
    torch.cuda.synchronize()

    cuda.cuMemUnmap(va, ctypes.c_size_t(size))
    cuda.cuMemAddressFree(va, ctypes.c_size_t(size))


class TestRemapBlock:

    def test_remap_changes_data(self):
        """Core test: write pattern A, remap to handle with pattern B,
        read back through same VA and see pattern B."""
        t = allocate_vmm_kv_cache(PAGE_2MB, DEVICE)
        alloc = get_vmm_allocation(t)

        # Fill block 0 with 0x41 ('A')
        t.fill_(0x41)
        assert t[0].item() == 0x41
        assert t[PAGE_2MB - 1].item() == 0x41

        # Create new handle, fill with 0x42 ('B')
        new_handle = create_physical_handle(alloc.page_size)
        _fill_physical_handle_via_temp_map(new_handle, 0x42, alloc.page_size)

        # Remap block 0
        old_handle = remap_block(t, 0, new_handle)

        # Same VA, new data
        assert t[0].item() == 0x42
        assert t[PAGE_2MB - 1].item() == 0x42

        release_physical_handle(old_handle)
        free_vmm_kv_cache(t)

    def test_remap_preserves_other_blocks(self):
        """Remap one block, verify adjacent blocks are untouched."""
        size = 4 * PAGE_2MB
        t = allocate_vmm_kv_cache(size, DEVICE)
        alloc = get_vmm_allocation(t)

        # Fill each block with distinct values
        for i in range(4):
            start = i * PAGE_2MB
            t[start : start + PAGE_2MB] = i + 10

        # Remap block 2 to new data (0x55)
        new_handle = create_physical_handle(alloc.page_size)
        _fill_physical_handle_via_temp_map(new_handle, 0x55, alloc.page_size)
        old_handle = remap_block(t, 2, new_handle)

        # Block 2 changed
        assert t[2 * PAGE_2MB].item() == 0x55
        # Blocks 0, 1, 3 unchanged
        assert t[0].item() == 10
        assert t[PAGE_2MB].item() == 11
        assert t[3 * PAGE_2MB].item() == 13

        release_physical_handle(old_handle)
        free_vmm_kv_cache(t)

    def test_remap_returns_old_handle(self):
        """remap_block returns the previous handle for cleanup."""
        t = allocate_vmm_kv_cache(PAGE_2MB, DEVICE)
        alloc = get_vmm_allocation(t)
        original_handle = alloc.handles[0]

        new_handle = create_physical_handle(alloc.page_size)
        old_handle = remap_block(t, 0, new_handle)

        assert old_handle == original_handle
        assert alloc.handles[0] == new_handle

        release_physical_handle(old_handle)
        free_vmm_kv_cache(t)

    def test_remap_invalid_block_id_raises(self):
        t = allocate_vmm_kv_cache(PAGE_2MB, DEVICE)

        with pytest.raises(ValueError, match="out of range"):
            remap_block(t, 5, 0)

        with pytest.raises(ValueError, match="out of range"):
            remap_block(t, -1, 0)

        free_vmm_kv_cache(t)

    def test_remap_non_vmm_tensor_raises(self):
        t = torch.zeros(1024, dtype=torch.int8, device=DEVICE)
        with pytest.raises(ValueError, match="not VMM-backed"):
            remap_block(t, 0, 0)

    def test_remap_va_pointer_unchanged(self):
        """The tensor's data_ptr must not change after remap."""
        t = allocate_vmm_kv_cache(PAGE_2MB, DEVICE)
        ptr_before = t.data_ptr()

        new_handle = create_physical_handle(get_vmm_allocation(t).page_size)
        _fill_physical_handle_via_temp_map(
            new_handle, 0, get_vmm_allocation(t).page_size
        )
        old = remap_block(t, 0, new_handle)

        assert t.data_ptr() == ptr_before

        release_physical_handle(old)
        free_vmm_kv_cache(t)

    def test_remap_torch_operations_work_after(self):
        """Verify standard torch ops work on the tensor after remap."""
        t = allocate_vmm_kv_cache(PAGE_2MB, DEVICE)
        alloc = get_vmm_allocation(t)

        new_handle = create_physical_handle(alloc.page_size)
        _fill_physical_handle_via_temp_map(new_handle, 0, alloc.page_size)
        old = remap_block(t, 0, new_handle)

        # Write, read, reshape, sum
        t[0] = 7
        t[100] = 13
        assert t[0].item() == 7
        reshaped = t.view(-1, 256)
        assert reshaped.shape[0] == PAGE_2MB // 256
        assert t[:1024].float().sum().item() == 20.0

        release_physical_handle(old)
        free_vmm_kv_cache(t)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
