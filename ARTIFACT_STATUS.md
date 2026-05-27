# EDMM Artifact Status Disclosure

## Primary Implementation

**vLLM v0.6.6 (V0 Backend)** via site-packages interception.

- `vllm/worker/cache_engine.py`: Patched `_allocate_kv_cache()` to conditionally use `allocate_vmm_kv_cache()` when `VLLM_EDMM_ENABLE=1`.
- `vllm/worker/edmm_allocator.py`: VMM allocator using ctypes bindings to `libcuda.so` (`cuMemAddressReserve`, `cuMemCreate`, `cuMemMap`, `cuMemSetAccess`). Wraps VA as PyTorch tensor via `__cuda_array_interface__`.
- **Activation:** `VLLM_EDMM_ENABLE=0` (default) uses standard `torch.zeros` with zero overhead. `VLLM_EDMM_ENABLE=1` allocates all KV cache layers via CUDA VMM.
- **Verified:** 28/28 KV cache layers VMM-backed on Qwen2.5-1.5B-Instruct and Qwen2.5-7B-Instruct.

## Secondary Prototype

**vLLM V1 architectural fork** files (not runnable on current host due to V1 IPC deadlock on multi-tenant H100).

- `vllm/v1/worker/gpu/attn_utils.py`: Same conditional hook (4 lines).
- `vllm/v1/core/sched/edmm_hook.py`: `EdmmSchedulerHook` with `WAITING_FOR_TOOL_CALL` state machine, CPU workload gating, VRAM leak protection.
- `vllm/v1/core/sched/scheduler.py`: 9 lines added for blocked-state promotion.
- `vllm/v1/request.py`: `WAITING_FOR_TOOL_CALL` enum value.
- **Status:** 31 unit tests pass (allocation, remap, scheduler lifecycle). Not validated through live V1 engine due to IPC deadlock.

## Proven Facts

### P0.1: VMM Allocation (PROVEN)
All 28 KV cache tensors allocated via `cuMemAddressReserve` + `cuMemCreate` + `cuMemMap`. Tensor `data_ptr()` matches VMM VA pointer. Verified by `traces/p0_1_vmm_allocation.jsonl`.

### P0.2: GPU MMU Attention Visibility (PROVEN)
`cuMemUnmap` + `cuMemMap` on a live KV cache block changed attention forward-pass output (sum 3,381 → 50,855,936) while `data_ptr()` remained stable at `0x302000000`. Engine generates correctly after restore. Standalone proof: filling KV tensor with 1.0 produces attention mean 1.000000; swapping physical page to 2.5 produces mean 2.500000 through same pointer.

### P0.5: Token Determinism (PROVEN)
3 prompt types (short, 6K clean, 6K contaminated) × 3 runs each. Token IDs and decoded text match 100% across all runs on VMM-backed engine. No numerical drift from VMM allocation.

### P1.1: Prompt Layout Sensitivity (PROVEN)
4 layouts profiled with unique UUID salt per trial (preventing cache bleeding):
- B1 (appended): 1.01x — prefix preserved
- B2 (mid-prompt): 4.57x — hash chain broken
- B3 (reordered): 7.93x — all prefix destroyed
- Cache hit reference: 1.00x

### P1.3: Context-Length Scaling (PROVEN — penalty characterization)
B2 penalty scales superlinearly across both frameworks:
- vLLM: 1.38x (4K) → 5.41x (32K)
- SGLang RadixAttention: 1.61x (4K) → 5.24x (32K)

## Oracle Upper Bounds

### B4: Oracle Context-Aliasing Upper Bound
**Methodology:** After warming the base prefix in the engine's KV cache, we compute the contaminated prompt's block hashes using `PrefixCachingBlock.hash_block_tokens()` and inject them into `PrefixCachingBlockAllocator._cached_blocks` pointing to the base prefix's physical block IDs. The engine's `find_cached_blocks_prefix()` finds these entries and skips the full prefill.

**What this measures:** The maximum theoretical recovery if a runtime system could predict the contaminated prompt's token sequence and pre-compute its KV cache during the tool-call idle window, then update the prefix cache registry to reflect the new sequence.

**What this does NOT measure:** The cost of prediction, the speculation hit rate, or the runtime overhead of the scheduling decision.

**Semantic correctness:** Verified at 16K context (Seed=42, Temperature=0.0, 64 tokens). B4 output matches B2 output token-for-token. Report: `traces/b4_semantic_correctness_report.json`.

**Results:**
| Context | B2 Miss (ms) | B4 Oracle (ms) | B2/B4 Speedup |
|---------|-------------|---------------|---------------|
| 4,096 | 28.3 | 18.1 | 1.56x |
| 8,192 | 57.9 | 24.7 | 2.34x |
| 16,384 | 153.8 | 43.5 | 3.53x |
| 32,768 | 427.5 | 74.1 | 5.77x |

## Not Yet Proven

### Online Execution-Driven Scheduling
The `EdmmSchedulerHook` state machine is tested in isolation (14 tests pass) but not wired into vLLM's live token output processing loop. Intercepting the `<|call_tool|>` sentinel mid-generation and triggering speculative prefill on a background CUDA stream requires modifications to the engine's output processing path beyond site-packages patching.

### Multi-Tenant TLB Shootdown Scaling
`cuMemMap` page swaps are verified at 50 μs on a single GPU. The impact of concurrent `cuMemMap` calls across multiple requests, and potential TLB shootdown costs under multi-tenant GPU sharing, have not been measured.

### Speculation Hit Rate
All B4 measurements assume perfect prediction (the anticipated prompt matches the actual tool response). Real-world hit rates depend on the tool-call distribution and prediction strategy. No miss-rate analysis has been conducted.

### Cross-GPU / Multi-Node
All measurements are single-GPU (H100 SXM5 96GB). Tensor-parallel and pipeline-parallel configurations have not been tested.

## Hardware

| Platform | GPU | CUDA | Driver | Status |
|----------|-----|------|--------|--------|
| devgpu014.eag3 | 8x NVIDIA H100 SXM5 96GB | 12.8 (nvcc V12.8.93) | 580.82.07 | Primary validation |
