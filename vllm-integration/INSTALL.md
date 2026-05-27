# EDMM vLLM Runtime Override

## How to apply (vLLM 0.6.6 site-packages injection)

```bash
# 1. Find your vLLM install path
VLLM_DIR=$(python -c "import vllm, os; print(os.path.dirname(vllm.__file__))")

# 2. Copy the EDMM allocator module
cp vllm_v1_worker_gpu/edmm_allocator.py $VLLM_DIR/worker/edmm_allocator.py

# 3. Apply the cache_engine patch (or copy the pre-patched file)
cp cache_engine_patched.py $VLLM_DIR/worker/cache_engine.py

# 4. Run with EDMM enabled
VLLM_EDMM_ENABLE=1 CUDA_VISIBLE_DEVICES=7 python your_script.py
```

## What the patch changes

In `vllm/worker/cache_engine.py`, the `_allocate_kv_cache()` method's
`torch.zeros()` call is wrapped in a conditional:

- `VLLM_EDMM_ENABLE=0` (default): unchanged behavior, `torch.zeros()`
- `VLLM_EDMM_ENABLE=1`: allocates via `allocate_vmm_kv_cache()` using
  CUDA VMM API (`cuMemAddressReserve` + `cuMemCreate` + `cuMemMap`)

The resulting tensor has the same shape, dtype, and device — but each
2MB page is backed by an independently remappable physical allocation.

## Files

| File | Description |
|------|-------------|
| `edmm_allocator.py` | VMM allocator + remap_block + verify_edmm_tensor_integrity |
| `cache_engine_patched.py` | Patched vLLM 0.6.6 cache_engine.py with EDMM conditional |
| `edmm_hook.py` | Scheduler state machine (WAITING_FOR_TOOL_CALL) |
| `request.py` | RequestStatus enum with WAITING_FOR_TOOL_CALL added |
| `scheduler.py` | Patched scheduler with blocked-state promotion |
