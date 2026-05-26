# EDMM: Execution-Driven Memory Management for Agentic LLM Workloads

Hardware-level memory management for LLM serving engines that eliminates the KV cache recomputation penalty during agent tool-call loops.

## The Problem

When an LLM agent calls a tool (web search, code execution, API), there's a multi-second pause. During this pause the GPU sits idle. When the tool response returns and gets injected mid-prompt, it breaks the prefix cache hash chain, forcing a full KV cache recompute — we measured an **8.21x latency penalty** through vLLM's live engine.

## The Solution

EDMM uses CUDA's Virtual Memory Management API (`cuMemMap`) to remap physical memory pages behind KV cache blocks in microseconds, avoiding data copies entirely. During the tool-call idle window, it speculatively pre-computes the anticipated KV cache. When the response arrives, a pointer swap completes in ~50 microseconds instead of a ~274ms recompute.

## Results

**vLLM 0.6.6 Live Engine (Qwen2.5-7B, H100):**

| Group | Description | TTFT (ms) |
|-------|-------------|-----------|
| A | Prefix Cache Hit | 33.35 |
| B | Mid-Prompt Contamination | 273.80 |
| C | Speculative Prefill + Hit | 39.18 |

- Radix Penalty (B/A): **8.21x**
- EDMM Recovery (C/A): **1.17x**

**Standalone Benchmark (16K context, 20 iterations, trimmed means):**

| Group | Description | TTFT (ms) |
|-------|-------------|-----------|
| A | Cached Prefix | 163.89 |
| B | Full Recompute | 684.82 |
| C | Speculative Zero-Copy | 157.40 |

- Radix Penalty (B/A): **4.18x**
- EDMM Recovery (C/A): **0.96x**

**CUDA VMM Micro-benchmark:**

| Operation | Latency |
|-----------|---------|
| cuMemMap page swap | 58 us |
| Full memcpy + recompute | 202 us |

## Repository Structure

```
# Benchmarks
validate_edmm_core_v2.py          # Macro benchmark (transformers, 7B model)
test_validate_edmm_core_v2.py     # Benchmark unit tests (5/5 pass)
edmm_vllm_live_e2e.py             # vLLM 0.6.6 live engine benchmark

# CUDA VMM Proofs
mini_edmm_test.cu                 # Correctness: two physical pages, one VA
mini_edmm_bench.cu                # Performance: timed cuMemMap vs memcpy

# Results
edmm_steady_state_results.md      # 20-iteration trimmed-mean results
edmm_true_e2e_inference_results.md # vLLM live engine results
edmm_vllm_e2e_performance.csv     # Per-iteration CSV data
vllm_edmm_integration_report.md   # Full integration report
```

## vLLM Integration (separate repo)

The vLLM fork adds 14 lines to 3 existing files and 2 new modules (~470 lines):
- `edmm_allocator.py` — VMM-backed KV cache allocation + block remapping
- `edmm_hook.py` — Scheduler state machine for tool-call pause/resume
- 31 tests across 4 phases, all passing

## Hardware

- NVIDIA H100 SXM5 96GB
- CUDA 12.8, Driver 580.82.07
- devgpu014.eag3 (Meta internal)

## Running

```bash
# Standalone benchmark
CUDA_VISIBLE_DEVICES=7 python validate_edmm_core_v2.py

# CUDA VMM tests
nvcc -o mini_edmm_test mini_edmm_test.cu -lcuda && CUDA_VISIBLE_DEVICES=7 ./mini_edmm_test
nvcc -o mini_edmm_bench mini_edmm_bench.cu -lcuda && CUDA_VISIBLE_DEVICES=7 ./mini_edmm_bench

# vLLM live benchmark (requires vLLM 0.6.6 + Qwen2.5-7B at /tmp/qwen7b)
CUDA_VISIBLE_DEVICES=7 python edmm_vllm_live_e2e.py
```
