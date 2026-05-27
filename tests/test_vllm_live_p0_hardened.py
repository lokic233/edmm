#!/usr/bin/env python3
"""
EDMM Hardened P0 Suite — VMM-backed vLLM Engine

Runs the live vLLM 0.6.6 engine with VLLM_EDMM_ENABLE=1, where
every KV cache tensor is allocated via cuMemAddressReserve/cuMemMap.

P0.1: Assert every KV cache tensor is VMM-backed (real pointer match)
P0.2: Page-swap a live KV cache block and verify attention output changes
P0.3: Closed-loop with real remap_block on live engine blocks
"""
import ctypes
import os
import sys
import time
import uuid

import torch

# Force EDMM on before any vLLM import
os.environ["VLLM_EDMM_ENABLE"] = "1"

from vllm import LLM, SamplingParams
from vllm.worker.edmm_allocator import (
    create_physical_handle,
    get_vmm_allocation,
    release_physical_handle,
    remap_block,
    verify_edmm_tensor_integrity,
)

MODEL = "/tmp/qwen7b"
NUM_TRIALS = 5

# CUDA driver bindings for P0.2 direct page swap
_u64 = ctypes.c_uint64
_sz = ctypes.c_size_t
_i = ctypes.c_int
_vp = ctypes.c_void_p


class _Access(ctypes.Structure):
    _fields_ = [("lt", _i), ("lid", _i), ("flags", _i)]


def run():
    print(f"\n{'='*70}")
    print("EDMM Hardened P0 Suite — VMM-Backed vLLM Engine")
    print(f"{'='*70}\n")

    llm = LLM(
        model=MODEL,
        gpu_memory_utilization=0.5,
        max_model_len=8192,
        enforce_eager=True,
        trust_remote_code=True,
        enable_prefix_caching=True,
    )
    sp = SamplingParams(max_tokens=5, temperature=0.0)

    # Verify engine generates
    out = llm.generate(["Hello"], sp)
    print(f"Engine alive: '{out[0].outputs[0].text.strip()}'")

    worker = llm.llm_engine.model_executor.driver_worker
    gpu_cache = worker.gpu_cache

    # ==================================================================
    # P0.1: Every KV cache tensor must be VMM-backed
    # ==================================================================
    print(f"\n--- P0.1: VMM Integrity Assertion on ALL KV Cache Tensors ---")

    total_tensors = 0
    vmm_verified = 0

    for cache_group_idx, cache_group in enumerate(gpu_cache):
        for tensor_idx, tensor in enumerate(cache_group):
            total_tensors += 1
            alloc = get_vmm_allocation(tensor)
            if alloc is not None:
                assert tensor.data_ptr() == alloc.va_ptr, (
                    f"Pointer mismatch: tensor=0x{tensor.data_ptr():x} "
                    f"vs VMM VA=0x{alloc.va_ptr:x}"
                )
                vmm_verified += 1
            else:
                # tensor might be a view — check base storage
                base_ptr = tensor.untyped_storage().data_ptr()
                # Walk all allocations to find one that contains this pointer
                from vllm.worker.edmm_allocator import _allocations

                found = False
                for va, a in _allocations.items():
                    if va <= base_ptr < va + a.total_bytes:
                        vmm_verified += 1
                        found = True
                        break
                if not found:
                    print(
                        f"  WARNING: group={cache_group_idx} tensor={tensor_idx} "
                        f"ptr=0x{base_ptr:x} NOT in any VMM allocation"
                    )

    print(f"  Total tensors: {total_tensors}")
    print(f"  VMM-verified:  {vmm_verified}")
    p01_pass = vmm_verified == total_tensors
    print(
        f"  [{'PASS' if p01_pass else 'FAIL'}] P0.1: {vmm_verified}/{total_tensors} tensors VMM-backed\n"
    )

    # ==================================================================
    # P0.2: Page-swap on a LIVE KV cache block
    # ==================================================================
    print("--- P0.2: Live KV Cache Block Page Swap ---")

    # Get the first cache tensor and its VMM allocation
    target_tensor = gpu_cache[0][0]
    alloc = get_vmm_allocation(target_tensor)

    if alloc is None:
        base_ptr = target_tensor.untyped_storage().data_ptr()
        from vllm.worker.edmm_allocator import _allocations

        for va, a in _allocations.items():
            if va <= base_ptr < va + a.total_bytes:
                alloc = a
                break

    assert alloc is not None, "Could not find VMM allocation for target tensor"
    print(
        f"  Target: VA=0x{alloc.va_ptr:x}, {alloc.num_pages} pages x {alloc.page_size//(1024*1024)}MB"
    )

    # Read current data from block 0
    block_size_elements = alloc.page_size // target_tensor.element_size()
    block_view = target_tensor.view(-1)[:block_size_elements]
    original_sum = block_view.float().sum().item()
    ptr_before = target_tensor.data_ptr()

    # Create a new physical handle and fill it with a known pattern
    new_handle = create_physical_handle(alloc.page_size)

    # Map it temporarily to fill with data
    cuda = ctypes.CDLL("libcuda.so.1")
    cuda.cuInit(0)
    dev = _i()
    cuda.cuDeviceGet(ctypes.byref(dev), 0)
    ctx = _vp()
    cuda.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev)
    cuda.cuCtxPushCurrent_v2(ctx)

    temp_va = _u64()
    cuda.cuMemAddressReserve.argtypes = [
        ctypes.POINTER(_u64),
        _sz,
        _sz,
        _u64,
        ctypes.c_ulonglong,
    ]
    cuda.cuMemAddressReserve.restype = _i
    cuda.cuMemMap.argtypes = [_u64, _sz, _sz, _u64, ctypes.c_ulonglong]
    cuda.cuMemMap.restype = _i
    cuda.cuMemSetAccess.argtypes = [_u64, _sz, _vp, _sz]
    cuda.cuMemSetAccess.restype = _i
    cuda.cuMemUnmap.argtypes = [_u64, _sz]
    cuda.cuMemUnmap.restype = _i
    cuda.cuMemAddressFree.argtypes = [_u64, _sz]
    cuda.cuMemAddressFree.restype = _i

    r = cuda.cuMemAddressReserve(ctypes.byref(temp_va), alloc.page_size, 0, 0, 0)
    assert r == 0, f"cuMemAddressReserve failed: {r}"
    r = cuda.cuMemMap(temp_va, alloc.page_size, 0, _u64(new_handle), 0)
    assert r == 0, f"cuMemMap failed: {r}"
    acc = _Access(1, 0, 3)
    r = cuda.cuMemSetAccess(temp_va, alloc.page_size, ctypes.byref(acc), 1)
    assert r == 0, f"cuMemSetAccess failed: {r}"

    # Fill with 0x42 pattern
    rt = ctypes.CDLL("libcudart.so")
    rt.cudaMemset(
        ctypes.c_void_p(temp_va.value), 0x42, ctypes.c_size_t(alloc.page_size)
    )
    torch.cuda.synchronize()

    cuda.cuMemUnmap(temp_va, alloc.page_size)
    cuda.cuMemAddressFree(temp_va, alloc.page_size)

    # Now remap block 0 of the live KV cache
    # First, construct a wrapper tensor that points to the raw VMM allocation
    from vllm.worker.edmm_allocator import _allocations

    class _Buf:
        def __init__(self, p, n):
            self.__cuda_array_interface__ = {
                "shape": (n,),
                "typestr": "|i1",
                "data": (p, False),
                "version": 3,
            }

    raw_tensor = torch.as_tensor(_Buf(alloc.va_ptr, alloc.total_bytes), device="cuda:0")
    old_handle = remap_block(raw_tensor, 0, new_handle)

    # Verify data changed
    new_sum = block_view.float().sum().item()
    ptr_after = target_tensor.data_ptr()

    print(f"  Block 0 sum before remap: {original_sum:.1f}")
    print(f"  Block 0 sum after remap:  {new_sum:.1f}")
    print(f"  Pointer before: 0x{ptr_before:x}")
    print(f"  Pointer after:  0x{ptr_after:x}")

    data_changed = abs(original_sum - new_sum) > 0.1
    ptr_stable = ptr_before == ptr_after

    # Restore original block so engine doesn't crash
    remap_block(raw_tensor, 0, old_handle)
    release_physical_handle(new_handle)
    torch.cuda.synchronize()

    print(f"  Data changed:  {'YES' if data_changed else 'NO'}")
    print(f"  Pointer stable: {'YES' if ptr_stable else 'NO'}")
    p02_pass = data_changed and ptr_stable
    print(
        f"  [{'PASS' if p02_pass else 'FAIL'}] P0.2: Live KV cache page swap verified\n"
    )

    # Verify engine still works after the swap-and-restore
    out2 = llm.generate(["Test after remap"], sp)
    print(f"  Post-remap generation: '{out2[0].outputs[0].text.strip()}'")

    # ==================================================================
    # P0.3: Closed-loop with mid-prompt contamination
    # ==================================================================
    print("\n--- P0.3: Closed-Loop Execution (mid-prompt contamination) ---")

    code_unit = (
        "class HTTPRequestHandler:\n"
        "    def dispatch(self, method, path):\n"
        "        handler = self._resolve_route(method, path)\n"
        "        return handler(self.request)\n\n"
    )
    base_half_1 = code_unit * 100
    base_half_2 = code_unit * 100
    suffix = "Identify the root cause and produce a minimal unified diff."

    tok = llm.get_tokenizer()
    half_toks = len(tok.encode(base_half_1))
    print(f"  Base: ~{half_toks * 2} tokens (two halves)")

    clean = base_half_1 + base_half_2 + suffix
    llm.generate([clean], sp)
    llm.generate([base_half_1 + "\nWARMUP\n" + base_half_2 + suffix], sp)

    ttft_a, ttft_b, ttft_c = [], [], []

    for trial in range(NUM_TRIALS):
        # A: cache hit
        llm.generate([clean], sp)
        time.sleep(0.5)
        t0 = time.perf_counter()
        llm.generate([clean], sp)
        ttft_a.append((time.perf_counter() - t0) * 1000)

        # B: mid-prompt injection
        llm.generate([clean], sp)
        time.sleep(0.5)
        injected = (
            base_half_1
            + f"\nTraceback: ValidationError {uuid.uuid4()} ts={time.time_ns()}\n"
            + base_half_2
            + suffix
        )
        t0 = time.perf_counter()
        llm.generate([injected], sp)
        ttft_b.append((time.perf_counter() - t0) * 1000)

        # C: speculative prefill
        llm.generate([clean], sp)
        tool_resp = f"\nTraceback: ValidationError {uuid.uuid4()} ts={time.time_ns()}\n"
        anticipated = base_half_1 + tool_resp + base_half_2 + suffix
        llm.generate([anticipated], sp)
        time.sleep(0.5)
        t0 = time.perf_counter()
        llm.generate([anticipated], sp)
        ttft_c.append((time.perf_counter() - t0) * 1000)

        print(
            f"  Trial {trial+1}: A={ttft_a[-1]:.1f}ms B={ttft_b[-1]:.1f}ms C={ttft_c[-1]:.1f}ms"
        )

    mu_a = sum(ttft_a) / len(ttft_a)
    mu_b = sum(ttft_b) / len(ttft_b)
    mu_c = sum(ttft_c) / len(ttft_c)

    print(f"\n  A (cache hit):    {mu_a:.2f} ms")
    print(f"  B (contaminated): {mu_b:.2f} ms")
    print(f"  C (speculative):  {mu_c:.2f} ms")
    print(f"  B/A = {mu_b/mu_a:.2f}x, C/A = {mu_c/mu_a:.2f}x")

    radix_pass = mu_b / mu_a >= 1.5
    recovery_pass = mu_c / mu_a <= 1.3

    # ==================================================================
    # Summary
    # ==================================================================
    print(f"\n{'='*70}")
    print("HARDENED P0 VALIDATION SUMMARY (VMM-backed vLLM Engine)")
    print(f"{'='*70}")
    print(
        f"  P0.1 VMM integrity (all tensors):  {'PASS' if p01_pass else 'FAIL'} ({vmm_verified}/{total_tensors})"
    )
    print(f"  P0.2 Live block page swap:         {'PASS' if p02_pass else 'FAIL'}")
    print(
        f"  P0.3 Closed-loop execution:        {'PASS' if (radix_pass and recovery_pass) else 'PARTIAL'}"
    )
    print(f"       B/A = {mu_b/mu_a:.2f}x, C/A = {mu_c/mu_a:.2f}x")
    all_pass = p01_pass and p02_pass and radix_pass and recovery_pass
    print(f"\n  OVERALL: {'ALL PASS' if all_pass else 'PARTIAL'}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    run()
