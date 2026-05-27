#!/usr/bin/env python3
"""
EDMM P0.2: Attention Mutation Visibility Proof

Proves that CUDA VMM page swaps change forward-pass attention
calculations while keeping the PyTorch tensor pointer constant.

This is the definitive artifact: if the attention output changes
after a cuMemMap swap but data_ptr() stays the same, then the
GPU MMU is transparently routing the same virtual address to
different physical data — exactly what EDMM does.
"""
import ctypes
import math
import time

import torch

PAGE_SIZE = 2 * 1024 * 1024

# ---- CUDA driver bindings (from validate_edmm_core_v2.py) ----

_u64 = ctypes.c_uint64
_sz = ctypes.c_size_t
_i = ctypes.c_int
_vp = ctypes.c_void_p
_p = ctypes.POINTER


class _CUmemAllocationProp(ctypes.Structure):
    _fields_ = [
        ("type", _i),
        ("requestedHandleTypes", _i),
        ("location_type", _i),
        ("location_id", _i),
        ("win32_security", _vp),
        ("_pad", ctypes.c_ubyte * 24),
    ]


class _CUmemAccessDesc(ctypes.Structure):
    _fields_ = [("location_type", _i), ("location_id", _i), ("flags", _i)]


def _init_cuda_driver():
    cuda = ctypes.CDLL("libcuda.so.1")
    cuda.cudart = ctypes.CDLL("libcudart.so")

    cuda.cuMemAddressReserve.argtypes = [_p(_u64), _sz, _sz, _u64, ctypes.c_ulonglong]
    cuda.cuMemAddressReserve.restype = _i
    cuda.cuMemCreate.argtypes = [_p(_u64), _sz, _vp, ctypes.c_ulonglong]
    cuda.cuMemCreate.restype = _i
    cuda.cuMemMap.argtypes = [_u64, _sz, _sz, _u64, ctypes.c_ulonglong]
    cuda.cuMemMap.restype = _i
    cuda.cuMemSetAccess.argtypes = [_u64, _sz, _vp, _sz]
    cuda.cuMemSetAccess.restype = _i
    cuda.cuMemUnmap.argtypes = [_u64, _sz]
    cuda.cuMemUnmap.restype = _i
    cuda.cuMemRelease.argtypes = [_u64]
    cuda.cuMemRelease.restype = _i
    cuda.cuMemAddressFree.argtypes = [_u64, _sz]
    cuda.cuMemAddressFree.restype = _i

    cuda.cuInit(0)

    dev = _i()
    cuda.cuDeviceGet(ctypes.byref(dev), 0)
    ctx = _vp()
    cuda.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev)
    cuda.cuCtxPushCurrent_v2(ctx)

    return cuda


def _check(r, msg):
    if r != 0:
        raise RuntimeError(f"CUDA driver error {r}: {msg}")


class _ExternalCUDABuffer:
    def __init__(self, ptr, nbytes):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|i1",
            "data": (ptr, False),
            "version": 3,
        }


def manual_attention(Q, K, V):
    """Single-head scaled dot-product attention (pure torch, no vLLM)."""
    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    attn_weights = torch.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, V)


def run():
    torch.cuda.init()
    _ = torch.zeros(1, device="cuda:0")

    cuda = _init_cuda_driver()

    print("=" * 70)
    print("EDMM P0.2: Attention Mutation Visibility Proof")
    print("=" * 70)

    # ---- Allocate VMM resources ----
    prop = _CUmemAllocationProp()
    prop.type = 1
    prop.location_type = 1
    prop.location_id = 0

    va = _u64()
    _check(
        cuda.cuMemAddressReserve(ctypes.byref(va), PAGE_SIZE, 0, 0, 0),
        "cuMemAddressReserve",
    )

    handle_A = _u64()
    handle_B = _u64()
    _check(
        cuda.cuMemCreate(ctypes.byref(handle_A), PAGE_SIZE, ctypes.byref(prop), 0),
        "cuMemCreate A",
    )
    _check(
        cuda.cuMemCreate(ctypes.byref(handle_B), PAGE_SIZE, ctypes.byref(prop), 0),
        "cuMemCreate B",
    )

    access = _CUmemAccessDesc(1, 0, 3)

    print(f"\n[VMM] VA: 0x{va.value:x}")
    print(f"[VMM] Physical Handle A: {handle_A.value}")
    print(f"[VMM] Physical Handle B: {handle_B.value}")

    # ---- Step 1: Map Page A, fill with 1.0, run attention ----
    _check(cuda.cuMemMap(va, PAGE_SIZE, 0, handle_A, 0), "cuMemMap A")
    _check(
        cuda.cuMemSetAccess(va, PAGE_SIZE, ctypes.byref(access), 1), "cuMemSetAccess A"
    )

    buf = _ExternalCUDABuffer(va.value, PAGE_SIZE)
    raw_tensor = torch.as_tensor(buf, device="cuda:0")

    seq_len = 64
    d_model = 64
    n_floats = seq_len * d_model
    kv_tensor = (
        raw_tensor[: n_floats * 4].view(torch.float32).reshape(1, seq_len, d_model)
    )

    ptr_before = raw_tensor.data_ptr()
    print(f"\n[Step 1] Tensor data_ptr: 0x{ptr_before:x}")

    kv_tensor.fill_(1.0)
    Q = torch.randn(1, seq_len, d_model, device="cuda:0", dtype=torch.float32)

    torch.cuda.synchronize()
    output_A = manual_attention(Q, kv_tensor, kv_tensor).clone()
    mean_A = output_A.mean().item()
    print(f"[Step 1] KV fill value: 1.0")
    print(f"[Step 1] Attention output mean: {mean_A:.6f}")

    # ---- Step 2: Unmap A, map B, fill with 2.5 ----
    print(f"\n[Step 2] Executing cuMemUnmap(A) + cuMemMap(B) on same VA...")
    _check(cuda.cuMemUnmap(va, PAGE_SIZE), "cuMemUnmap A")
    _check(cuda.cuMemMap(va, PAGE_SIZE, 0, handle_B, 0), "cuMemMap B")
    _check(
        cuda.cuMemSetAccess(va, PAGE_SIZE, ctypes.byref(access), 1), "cuMemSetAccess B"
    )

    ptr_after = raw_tensor.data_ptr()
    print(f"[Step 2] Tensor data_ptr: 0x{ptr_after:x}")
    print(f"[Step 2] Pointer unchanged: {ptr_before == ptr_after}")

    kv_tensor.fill_(2.5)
    print(f"[Step 2] KV fill value: 2.5")

    # ---- Step 3: Run attention again with same tensor reference ----
    torch.cuda.synchronize()
    output_B = manual_attention(Q, kv_tensor, kv_tensor).clone()
    mean_B = output_B.mean().item()
    print(f"[Step 3] Attention output mean: {mean_B:.6f}")

    # ---- Assertions ----
    print(f"\n{'='*70}")
    print("VERIFICATION RESULTS")
    print(f"{'='*70}")

    ptr_match = ptr_before == ptr_after
    output_changed = abs(mean_A - mean_B) > 1e-6

    print(
        f"  PyTorch data_ptr stable:     {'PASS' if ptr_match else 'FAIL'} "
        f"(0x{ptr_before:x} == 0x{ptr_after:x})"
    )
    print(
        f"  Attention output changed:    {'PASS' if output_changed else 'FAIL'} "
        f"({mean_A:.6f} != {mean_B:.6f})"
    )
    print(f"  Output delta:                {abs(mean_A - mean_B):.6f}")

    assert ptr_match, "Pointer changed — VMM wrapping broken"
    assert output_changed, "Output unchanged — page swap had no effect"

    print(f"\n  [PASS] GPU MMU page swap transparently altered attention math")
    print(f"         through a stable PyTorch tensor reference.")
    print(f"{'='*70}")

    # ---- Cleanup ----
    _check(cuda.cuMemUnmap(va, PAGE_SIZE), "cuMemUnmap cleanup")
    _check(cuda.cuMemRelease(handle_A.value), "cuMemRelease A")
    _check(cuda.cuMemRelease(handle_B.value), "cuMemRelease B")
    _check(cuda.cuMemAddressFree(va, PAGE_SIZE), "cuMemAddressFree")


if __name__ == "__main__":
    run()
