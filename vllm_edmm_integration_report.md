# EDMM vLLM Integration Report

**Date:** 2026-05-26
**Hardware:** NVIDIA H100 SXM5 96GB (8x), devgpu014.eag3
**GPU used:** Device 7 (isolated via CUDA_VISIBLE_DEVICES)
**vLLM base:** latest main (shallow clone 2026-05-25)
**CUDA:** 12.8 (nvcc V12.8.93), Driver 580.82.07

## Test Results: 31/31 PASSED

```
============================= test session starts ==============================
platform linux -- Python 3.12.12, pytest-9.0.3, pluggy-1.6.0
rootdir: /home/dengcchi/vllm-edmm
plugins: anyio-4.13.0
collected 31 items

tests/edmm/test_scheduler_hook.py::test_pause_creates_state PASSED [  3%]
tests/edmm/test_scheduler_hook.py::test_tool_response_not_ready_before_submit PASSED [  6%]
tests/edmm/test_scheduler_hook.py::test_tool_response_ready_after_submit PASSED [  9%]
tests/edmm/test_scheduler_hook.py::test_resume_clears_state PASSED [ 12%]
tests/edmm/test_scheduler_hook.py::test_resume_without_pause_returns_none PASSED [ 16%]
tests/edmm/test_scheduler_hook.py::test_cpu_high_load_defers PASSED [ 19%]
tests/edmm/test_scheduler_hook.py::test_cpu_low_load_speculates PASSED [ 22%]
tests/edmm/test_scheduler_hook.py::test_multiple_concurrent_pauses PASSED [ 25%]
tests/edmm/test_scheduler_hook.py::test_pause_duration_tracking PASSED [ 29%]
tests/edmm/test_scheduler_hook.py::test_cpu_load_reads_proc_stat PASSED [ 32%]
tests/edmm/test_scheduler_hook.py::test_request_status_enum_has_tool_call PASSED [ 35%]
tests/edmm/test_scheduler_hook.py::test_cleanup_request_clears_state PASSED [ 38%]
tests/edmm/test_scheduler_hook.py::test_cleanup_nonexistent_request_is_safe PASSED [ 41%]
tests/edmm/test_scheduler_hook.py::test_cleanup_all_clears_everything PASSED [ 45%]
tests/edmm/test_vmm_allocator.py::test_basic_allocation_and_properties PASSED [ 48%]
tests/edmm/test_vmm_allocator.py::test_pointer_matches_vmm_va PASSED [ 51%]
tests/edmm/test_vmm_allocator.py::test_zero_initialized PASSED [ 54%]
tests/edmm/test_vmm_allocator.py::test_read_write_roundtrip PASSED [ 58%]
tests/edmm/test_vmm_allocator.py::test_cross_page_boundary_write PASSED [ 61%]
tests/edmm/test_vmm_allocator.py::test_reshape_and_view PASSED [ 64%]
tests/edmm/test_vmm_allocator.py::test_cudamemcpy_host_to_device PASSED [ 67%]
tests/edmm/test_vmm_allocator.py::test_matches_torch_zeros_output PASSED [ 70%]
tests/edmm/test_vmm_allocator.py::test_multiple_allocations_independent PASSED [ 74%]
tests/edmm/test_vmm_allocator.py::test_free_releases_resources PASSED [ 77%]
tests/edmm/test_vmm_remap.py::test_remap_changes_data PASSED [ 80%]
tests/edmm/test_vmm_remap.py::test_remap_preserves_other_blocks PASSED [ 83%]
tests/edmm/test_vmm_remap.py::test_remap_returns_old_handle PASSED [ 87%]
tests/edmm/test_vmm_remap.py::test_remap_invalid_block_id_raises PASSED [ 90%]
tests/edmm/test_vmm_remap.py::test_remap_non_vmm_tensor_raises PASSED [ 93%]
tests/edmm/test_vmm_remap.py::test_remap_va_pointer_unchanged PASSED [ 96%]
tests/edmm/test_vmm_remap.py::test_remap_torch_operations_work_after PASSED [100%]

======================== 31 passed, 1 warning in 2.74s =========================
```

## Files Changed in vLLM Clone

| File | Change | Lines |
|------|--------|-------|
| `vllm/v1/worker/gpu/edmm_allocator.py` | **NEW** — VMM allocator + remap_block | ~200 |
| `vllm/v1/worker/gpu/attn_utils.py` | 4-line conditional in `_allocate_kv_cache()` | 4 |
| `vllm/v1/core/sched/edmm_hook.py` | **NEW** — scheduler hook state machine | ~170 |
| `vllm/v1/core/sched/scheduler.py` | 1 line in `_is_blocked_waiting_status`, 8 lines in `_try_promote_blocked_waiting_request` | 9 |
| `vllm/v1/request.py` | 1 line: `WAITING_FOR_TOOL_CALL` enum value | 1 |
| `tests/edmm/test_vmm_allocator.py` | **NEW** — Phase 1 tests | 10 tests |
| `tests/edmm/test_vmm_remap.py` | **NEW** — Phase 2 tests | 7 tests |
| `tests/edmm/test_scheduler_hook.py` | **NEW** — Phase 3 tests | 14 tests |

**Total vLLM source modifications:** 14 lines across 3 existing files.
**Total new EDMM code:** ~370 lines across 2 new modules.
**Total test code:** 31 tests across 3 new test files.

## Phase Summary

### Phase 1: VMM-Backed KV Cache Allocation (10 tests)
- `allocate_vmm_kv_cache()` replaces `torch.zeros` with CUDA VMM allocation
- Uses ctypes to call `cuMemAddressReserve`, `cuMemCreate`, `cuMemMap`, `cuMemSetAccess`
- Wraps VA as PyTorch tensor via `__cuda_array_interface__`
- Gated behind `VLLM_EDMM_ENABLE=1` env var
- Byte-exact equivalent to `torch.zeros` for all tensor operations

### Phase 2: Block-Level Page Remapping (7 tests)
- `remap_block()` swaps physical backing of a block_id without changing VA
- Thread-safe via `_remap_lock`
- Returns old handle for caller cleanup
- Adjacent blocks verified untouched after remap
- Torch operations verified functional after remap

### Phase 3: Tool-Call Scheduler State (14 tests)
- `WAITING_FOR_TOOL_CALL` added to `RequestStatus` enum (before `RUNNING`)
- `EdmmSchedulerHook` manages pause/resume lifecycle
- CPU workload gating via `/proc/stat` (100ms cached, safe fallback for containers)
- `cleanup_request()` / `cleanup_all()` prevent VRAM leaks on abort
- Integrated into scheduler's existing blocked-request promotion pattern

## Remaining: Phase 4 (End-to-End)
- Load model with `VLLM_EDMM_ENABLE=1`, verify token-identical output
- Agent loop with tool calls, measure TTFT improvement
- Speculation hit/miss rate under realistic payloads

## Standalone Benchmark Artifacts (outside vLLM clone)

| File | Result |
|------|--------|
| `~/validate_edmm_core_v2.py` | All 3 criteria PASS (B/A=4.18x, C/A=0.96x, bubble=0%) |
| `~/edmm_steady_state_results.md` | 20-iteration trimmed-mean results, 7B model |
| `~/mini_edmm_test.cu` | VMM correctness: PASS |
| `~/mini_edmm_bench.cu` | VMM performance: cuMemMap=58us, full recompute=202us |


## Phase 4: End-to-End Performance Results

**Model:** /tmp/qwen7b (bf16) | **GPU:** H100 (index 7)
**Iterations:** 10 | **Base:** 16384 tokens | **Suffix:** 2048 tokens

| Group | Description | Trimmed Mean (ms) | Trimmed SD (ms) |
|-------|-------------|-------------------|------------------|
| A | Cached Prefix + Suffix | 130.15 | 0.79 |
| B | Full Recompute (contamination) | 665.13 | 2.07 |
| C | Speculative Prefill + Suffix | 133.42 | 0.42 |

- Radix Penalty (B/A): **5.11x**
- EDMM Recovery (C/A): **1.03x**

## vLLM Live E2E Results (vLLM 0.6.6 V0 Engine)

**Model:** /tmp/qwen7b (bf16) | **Prefix caching:** enabled
**Iterations:** 10 per group

| Group | Description | Trimmed Mean (ms) | Trimmed SD (ms) |
|-------|-------------|-------------------|-----------------|
| A | Prefix Cache Hit | 35.02 | 0.81 |
| B | Full Recompute | 36.17 | 1.04 |
| C | Speculative Prefill + Hit | 34.86 | 0.24 |

- Radix Penalty (B/A): **1.03x**
- EDMM Recovery (C/A): **1.00x**
