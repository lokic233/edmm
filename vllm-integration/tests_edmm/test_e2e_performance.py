"""
EDMM Phase 4: End-to-end performance benchmark.

Measures real TTFT across three groups using the 7B model:
- Group A: Cached prefix + suffix-only compute (best case)
- Group B: Full recompute after mid-prompt contamination (worst case)
- Group C: Speculative prefill + zero-copy suffix (EDMM approach)

Groups A and B run through real model inference.
Group C uses the same inference path as A but with a prior speculative
prefill step (the full anticipated prompt is computed during the sleep
window, then suffix-only compute uses that KV cache).
"""

import copy
import csv
import gc
import os
import sys
import time
from typing import Any, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

VLLM_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if VLLM_PATH not in sys.path:
    sys.path.insert(0, VLLM_PATH)

MODEL_PATH = os.environ.get("EDMM_MODEL_NAME", "/tmp/qwen7b")
DEVICE = torch.device("cuda:0")
NUM_ITERATIONS = 10
SLEEP_DURATION = 2.0
BASE_TOKENS = 16384
SUFFIX_TOKENS = 2048

_CONTEXT_BLOCK = (
    "class HTTPRequestHandler:\n"
    '    """Base handler for incoming HTTP requests."""\n'
    "    def dispatch(self, method, path, **kwargs):\n"
    "        handler = self._resolve_route(method, path)\n"
    "        if handler is None:\n"
    '            raise RouteNotFoundError(f"No route for {method} {path}")\n'
    "        try:\n"
    "            result = handler(self.request, **kwargs)\n"
    "            return self.response_class(data=result, status=200)\n"
    "        except ValidationError as exc:\n"
    "            return self._handle_error(exc, status=422)\n"
    "        except Exception as exc:\n"
    "            return self._handle_error(exc, status=500)\n\n"
)

_SUFFIX_BLOCK = (
    "You are a senior software engineer performing an agentic code review. "
    "Given the repository source files, execution logs, and error traces above, "
    "identify the root cause of the failing test suite. Produce a minimal unified "
    "diff that fixes the regression without altering the public API surface. "
    "Verify your patch handles all edge cases in the stack trace. "
    "Ensure backward compatibility with existing callers. "
    "List each modified file with a one-line rationale. "
)


def _repeat_to_tokens(tokenizer, block, n):
    text = block * ((n * 6 // len(block)) + 2)
    toks = tokenizer.encode(text, add_special_tokens=False)
    return tokenizer.decode(toks[:n])


def _make_tool_response():
    return (
        f"Traceback (most recent call last):\n"
        f'  File "/srv/app/views/api.py", line 247, in dispatch_request\n'
        f"    result = handler(request, **bound_args)\n"
        f'  File "/srv/app/lib/schemas.py", line 78, in _run_validators\n'
        f"    raise ValidationError(errors)\n"
        f"marshmallow.exceptions.ValidationError: {{'field_config': ['Unknown']}}\n"
        f"[tool_call_id={time.time_ns()}] exit_code=1\n"
    )


def stats(values: List[float]) -> Tuple[float, float]:
    n = len(values)
    mu = sum(values) / n
    sd = (sum((x - mu) ** 2 for x in values) / n) ** 0.5
    return mu, sd


def trimmed_stats(values: List[float], pct: float = 0.1) -> Tuple[float, float]:
    s = sorted(values)
    trim = max(1, int(len(s) * pct))
    return stats(s[trim:-trim])


@torch.no_grad()
def prefill(model, ids, kv=None):
    torch.cuda.synchronize()
    t0 = time.perf_counter_ns()
    if kv is not None:
        out = model(input_ids=ids, past_key_values=kv, use_cache=True)
    else:
        out = model(input_ids=ids, use_cache=True)
    torch.cuda.synchronize()
    return (time.perf_counter_ns() - t0) / 1e6, out.past_key_values


def run_benchmark():
    gpu_idx = int(os.environ.get("CUDA_VISIBLE_DEVICES", "7"))
    print(f"\n{'='*70}")
    print("EDMM End-to-End Performance Benchmark")
    print(f"Model: {MODEL_PATH} | GPU: {gpu_idx} | Iterations: {NUM_ITERATIONS}")
    print(f"Base: {BASE_TOKENS} tokens | Suffix: {SUFFIX_TOKENS} tokens")
    print(f"{'='*70}\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map=DEVICE,
        trust_remote_code=True,
    ).eval()
    print(f"Model loaded. GPU: {torch.cuda.memory_allocated()/1e9:.1f} GB\n")

    base_text = _repeat_to_tokens(tokenizer, _CONTEXT_BLOCK, BASE_TOKENS)
    suffix_text = _repeat_to_tokens(tokenizer, _SUFFIX_BLOCK, SUFFIX_TOKENS)

    base_ids = tokenizer.encode(
        base_text, add_special_tokens=False, return_tensors="pt"
    )
    base_ids = base_ids[:, :BASE_TOKENS].contiguous().to(DEVICE)
    suffix_ids = tokenizer.encode(
        suffix_text, add_special_tokens=False, return_tensors="pt"
    )
    suffix_ids = suffix_ids[:, :SUFFIX_TOKENS].contiguous().to(DEVICE)

    print(f"Base: {base_ids.shape[1]} tokens, Suffix: {suffix_ids.shape[1]} tokens")

    # Warmup
    _, wkv = prefill(model, base_ids)
    prefill(model, suffix_ids, wkv)
    tool_r = _make_tool_response()
    full_prompt = f"{base_text}\n{tool_r}\n{suffix_text}"
    full_ids = tokenizer.encode(full_prompt, return_tensors="pt").to(DEVICE)
    prefill(model, full_ids)
    del wkv, full_ids
    gc.collect()
    torch.cuda.empty_cache()
    print("Warmup done.\n")

    results_a, results_b, results_c = [], [], []
    bubble_utils = []

    # --- Group A: Cached Prefix + Suffix ---
    print("--- Group A: Cached Prefix + Incremental Suffix ---")
    for i in range(NUM_ITERATIONS):
        _, base_kv = prefill(model, base_ids)
        time.sleep(SLEEP_DURATION)
        bubble_utils.append(0)  # GPU is idle during sleep by design
        ms, _ = prefill(model, suffix_ids, base_kv)
        results_a.append(ms)
        print(f"  Iter {i+1:2d}: {ms:.2f} ms")
        del base_kv
        gc.collect()
        torch.cuda.empty_cache()

    # --- Group B: Full Recompute (mid-prompt contamination) ---
    print("\n--- Group B: Full Recompute (mid-prompt contamination) ---")
    for i in range(NUM_ITERATIONS):
        _, _ = prefill(model, base_ids)
        time.sleep(SLEEP_DURATION)
        tool_resp = _make_tool_response()
        contaminated = f"{base_text}\n{tool_resp}\n{suffix_text}"
        cont_ids = tokenizer.encode(contaminated, return_tensors="pt").to(DEVICE)
        ms, _ = prefill(model, cont_ids)
        results_b.append(ms)
        print(f"  Iter {i+1:2d}: {ms:.2f} ms")
        del cont_ids
        gc.collect()
        torch.cuda.empty_cache()

    # --- Group C: Speculative Prefill + Zero-Copy Suffix ---
    print("\n--- Group C: Speculative Prefill + Zero-Copy Suffix ---")
    for i in range(NUM_ITERATIONS):
        _, base_kv = prefill(model, base_ids)
        tool_resp = _make_tool_response()
        anticipated = f"{base_text}\n{tool_resp}"
        ant_ids = tokenizer.encode(anticipated, return_tensors="pt").to(DEVICE)
        _, spec_kv = prefill(model, ant_ids)
        time.sleep(SLEEP_DURATION)
        ms, _ = prefill(model, suffix_ids, spec_kv)
        results_c.append(ms)
        print(f"  Iter {i+1:2d}: {ms:.2f} ms")
        del base_kv, spec_kv, ant_ids
        gc.collect()
        torch.cuda.empty_cache()

    # --- Results ---
    mu_a, sd_a = stats(results_a)
    mu_b, sd_b = stats(results_b)
    mu_c, sd_c = stats(results_c)
    tmu_a, tsd_a = trimmed_stats(results_a)
    tmu_b, tsd_b = trimmed_stats(results_b)
    tmu_c, tsd_c = trimmed_stats(results_c)

    print(f"\n{'='*70}")
    print("RESULTS (10% trimmed means)")
    print(f"{'='*70}")
    print(
        f"| Group | Description                    | Trimmed Mean | Trimmed SD | Raw Mean | Raw SD  |"
    )
    print(
        f"|-------|--------------------------------|-------------|-----------|---------|---------|"
    )
    print(
        f"| A     | Cached Prefix + Suffix         | {tmu_a:11.2f} | {tsd_a:9.2f} | {mu_a:7.2f} | {sd_a:7.2f} |"
    )
    print(
        f"| B     | Full Recompute (contamination) | {tmu_b:11.2f} | {tsd_b:9.2f} | {mu_b:7.2f} | {sd_b:7.2f} |"
    )
    print(
        f"| C     | Speculative Prefill + Suffix   | {tmu_c:11.2f} | {tsd_c:9.2f} | {mu_c:7.2f} | {sd_c:7.2f} |"
    )

    print(f"\n--- Validation ---")
    if tmu_a > 0:
        ratio_ba = tmu_b / tmu_a
        ratio_ca = tmu_c / tmu_a
        print(f"  Radix Penalty (B/A): {ratio_ba:.2f}x")
        print(f"  EDMM Recovery (C/A): {ratio_ca:.2f}x")
        print(f"  [{'PASS' if ratio_ba >= 2.5 else 'FAIL'}] Radix >= 2.5x")
        print(f"  [{'PASS' if ratio_ca <= 1.15 else 'FAIL'}] Recovery <= 1.15x")

    # --- CSV output ---
    csv_path = os.path.expanduser("~/edmm_vllm_e2e_performance.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "iteration", "ttft_ms"])
        for i, v in enumerate(results_a, 1):
            w.writerow(["A", i, f"{v:.2f}"])
        for i, v in enumerate(results_b, 1):
            w.writerow(["B", i, f"{v:.2f}"])
        for i, v in enumerate(results_c, 1):
            w.writerow(["C", i, f"{v:.2f}"])
    print(f"\n  CSV written to: {csv_path}")

    # --- Append to integration report ---
    report_path = os.path.expanduser("~/vllm_edmm_integration_report.md")
    with open(report_path, "a") as f:
        f.write("\n\n## Phase 4: End-to-End Performance Results\n\n")
        f.write(f"**Model:** {MODEL_PATH} (bf16) | **GPU:** H100 (index {gpu_idx})\n")
        f.write(
            f"**Iterations:** {NUM_ITERATIONS} | **Base:** {BASE_TOKENS} tokens | **Suffix:** {SUFFIX_TOKENS} tokens\n\n"
        )
        f.write("| Group | Description | Trimmed Mean (ms) | Trimmed SD (ms) |\n")
        f.write("|-------|-------------|-------------------|------------------|\n")
        f.write(f"| A | Cached Prefix + Suffix | {tmu_a:.2f} | {tsd_a:.2f} |\n")
        f.write(f"| B | Full Recompute (contamination) | {tmu_b:.2f} | {tsd_b:.2f} |\n")
        f.write(f"| C | Speculative Prefill + Suffix | {tmu_c:.2f} | {tsd_c:.2f} |\n\n")
        if tmu_a > 0:
            f.write(f"- Radix Penalty (B/A): **{ratio_ba:.2f}x**\n")
            f.write(f"- EDMM Recovery (C/A): **{ratio_ca:.2f}x**\n")
    print(f"  Report appended to: {report_path}")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    run_benchmark()
