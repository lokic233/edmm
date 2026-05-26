"""
EDMM Phase 1 validation: VMM-backed KV cache allocation.

Tests that allocate_vmm_kv_cache produces a tensor that is
functionally identical to torch.zeros for all operations
vLLM's attention backends perform on KV cache tensors.
"""

import os

import pytest
import torch

os.environ["VLLM_EDMM_ENABLE"] = "1"

from vllm.v1.worker.gpu.edmm_allocator import (
    allocate_vmm_kv_cache,
    free_vmm_kv_cache,
    get_vmm_allocation,
)

DEVICE = torch.device("cuda:0")
PAGE_2MB = 2 * 1024 * 1024


class TestVMMAllocation:

    def test_basic_allocation_and_properties(self):
        size = 4 * PAGE_2MB
        t = allocate_vmm_kv_cache(size, DEVICE)
        assert t.shape == (size,)
        assert t.dtype == torch.int8
        assert t.device.type == "cuda"
        assert t.is_contiguous()
        free_vmm_kv_cache(t)

    def test_pointer_matches_vmm_va(self):
        size = PAGE_2MB
        t = allocate_vmm_kv_cache(size, DEVICE)
        alloc = get_vmm_allocation(t)
        assert alloc is not None
        assert t.data_ptr() == alloc.va_ptr
        free_vmm_kv_cache(t)

    def test_zero_initialized(self):
        size = PAGE_2MB
        t = allocate_vmm_kv_cache(size, DEVICE)
        assert t.sum().item() == 0
        free_vmm_kv_cache(t)

    def test_read_write_roundtrip(self):
        size = PAGE_2MB
        t = allocate_vmm_kv_cache(size, DEVICE)
        t[0] = 42
        t[100] = -7
        t[size - 1] = 127
        assert t[0].item() == 42
        assert t[100].item() == -7
        assert t[size - 1].item() == 127
        free_vmm_kv_cache(t)

    def test_cross_page_boundary_write(self):
        size = 4 * PAGE_2MB
        t = allocate_vmm_kv_cache(size, DEVICE)
        boundary = PAGE_2MB
        t[boundary - 1] = 11
        t[boundary] = 22
        t[boundary + 1] = 33
        assert t[boundary - 1].item() == 11
        assert t[boundary].item() == 22
        assert t[boundary + 1].item() == 33
        free_vmm_kv_cache(t)

    def test_reshape_and_view(self):
        size = PAGE_2MB
        t = allocate_vmm_kv_cache(size, DEVICE)
        reshaped = t.view(-1, 256)
        assert reshaped.shape == (size // 256, 256)
        reshaped[0, 0] = 55
        assert t[0].item() == 55
        free_vmm_kv_cache(t)

    def test_cudamemcpy_host_to_device(self):
        size = PAGE_2MB
        t = allocate_vmm_kv_cache(size, DEVICE)
        host = torch.arange(1024, dtype=torch.int8)
        t[:1024].copy_(host.to(DEVICE))
        result = t[:1024].cpu()
        assert torch.equal(result, host)
        free_vmm_kv_cache(t)

    def test_matches_torch_zeros_output(self):
        """The core equivalence test: VMM tensor and torch.zeros tensor
        produce identical results under the same operations."""
        size = PAGE_2MB
        vmm_t = allocate_vmm_kv_cache(size, DEVICE)
        std_t = torch.zeros(size, dtype=torch.int8, device=DEVICE)

        pattern = torch.arange(1024, dtype=torch.int8, device=DEVICE)
        vmm_t[:1024] = pattern
        std_t[:1024] = pattern

        vmm_float = vmm_t[:1024].float()
        std_float = std_t[:1024].float()
        assert torch.equal(vmm_float, std_float)

        vmm_sum = vmm_float.sum()
        std_sum = std_float.sum()
        assert vmm_sum.item() == std_sum.item()

        free_vmm_kv_cache(vmm_t)

    def test_multiple_allocations_independent(self):
        t1 = allocate_vmm_kv_cache(PAGE_2MB, DEVICE)
        t2 = allocate_vmm_kv_cache(PAGE_2MB, DEVICE)
        assert t1.data_ptr() != t2.data_ptr()
        t1[0] = 1
        t2[0] = 2
        assert t1[0].item() == 1
        assert t2[0].item() == 2
        free_vmm_kv_cache(t1)
        free_vmm_kv_cache(t2)

    def test_free_releases_resources(self):
        t = allocate_vmm_kv_cache(PAGE_2MB, DEVICE)
        ptr = t.data_ptr()
        assert get_vmm_allocation(t) is not None
        free_vmm_kv_cache(t)
        assert get_vmm_allocation(t) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
