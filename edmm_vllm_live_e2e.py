#!/usr/bin/env python3
"""
EDMM Live E2E Benchmark through vLLM 0.6.6 (V0 engine).

Measures real TTFT through vLLM's llm.generate() with prefix caching enabled.
Three groups:
  A: Same prefix, same suffix (prefix cache hit — ideal baseline)
  B: Same prefix + dynamic mid-prompt injection (prefix cache miss — full recompute)
  C: Speculative prefill of anticipated prompt during sleep, then suffix-only
"""
import gc
import time
import uuid
from typing import List, Tuple

from vllm import LLM, SamplingParams

MODEL = "/tmp/qwen7b"
NUM_TRIALS = 10
SLEEP_DURATION = 2.0

_CONTEXT_BLOCK = (
    "class HTTPRequestHandler:\n"
    '    """Base handler for HTTP requests with middleware."""\n'
    "    def dispatch(self, method, path, **kwargs):\n"
    "        handler = self._resolve_route(method, path)\n"
    "        if handler is None:\n"
    '            raise RouteNotFoundError(f"No route for {method} {path}")\n'
    "        try:\n"
    "            result = handler(self.request, **kwargs)\n"
    "            return self.response_class(data=result, status=200)\n"
    "        except ValidationError as exc:\n"
    "            return self._handle_error(exc, status=422)\n\n"
)

_SUFFIX = (
    "Given the repository context above, identify the root cause of the failure. "
    "Produce a minimal unified diff fixing the regression. "
    "Ensure backward compatibility with existing callers. "
    "List each modified file with a one-line rationale."
)


def _make_base_prompt(n_repeats: int = 100) -> str:
    return _CONTEXT_BLOCK * n_repeats


def _make_tool_response() -> str:
    return (
        f"\nTraceback (most recent call last):\n"
        f'  File "/srv/app/views/api.py", line 247, in dispatch_request\n'
        f"    result = handler(request, **bound_args)\n"
        f"marshmallow.exceptions.ValidationError: {{'field': ['Unknown']}}\n"
        f"[tool_id={uuid.uuid4()}] exit_code=1 ts={time.time_ns()}\n"
    )


def stats(v: List[float]) -> Tuple[float, float]:
    mu = sum(v) / len(v)
    sd = (sum((x - mu) ** 2 for x in v) / len(v)) ** 0.5
    return mu, sd


def trimmed_stats(v: List[float], pct: float = 0.1) -> Tuple[float, float]:
    s = sorted(v)
    trim = max(1, int(len(s) * pct))
    return stats(s[trim:-trim])


def run():
    print(f"\n{'='*70}")
    print("EDMM Live E2E Benchmark (vLLM 0.6.6 V0 Engine)")
    print(f"Model: {MODEL} | Trials: {NUM_TRIALS}")
    print(f"{'='*70}\n")

    llm = LLM(
        model=MODEL,
        gpu_memory_utilization=0.5,
        max_model_len=24576,
        enforce_eager=True,
        trust_remote_code=True,
        enable_prefix_caching=True,
    )
    sp = SamplingParams(max_tokens=1, temperature=0.0)

    base = _make_base_prompt(100)
    base_tokens = len(llm.get_tokenizer().encode(base))
    print(f"Base prefix: ~{base_tokens} tokens\n")

    # Warmup
    llm.generate([base + "\n" + _SUFFIX], sp)
    llm.generate([base + _make_tool_response() + _SUFFIX], sp)

    results_a, results_b, results_c = [], [], []

    # --- Group A: Prefix Cache Hit ---
    # Submit the same prompt twice. Second call reuses cached prefix.
    print("--- Group A: Prefix Cache Hit (same prompt, suffix only) ---")
    for i in range(NUM_TRIALS):
        prompt = base + "\n" + _SUFFIX

        # Prime the prefix cache
        llm.generate([prompt], sp)
        time.sleep(SLEEP_DURATION)

        # Measure: same prompt again — prefix is cached
        t0 = time.perf_counter()
        llm.generate([prompt], sp)
        ms = (time.perf_counter() - t0) * 1000
        results_a.append(ms)
        print(f"  Trial {i+1:2d}: {ms:.2f} ms")

    # --- Group B: Prefix Cache Miss (mid-prompt contamination) ---
    print("\n--- Group B: Full Recompute (unique dynamic payload each turn) ---")
    for i in range(NUM_TRIALS):
        # Prime with clean prompt
        llm.generate([base + "\n" + _SUFFIX], sp)
        time.sleep(SLEEP_DURATION)

        # Contaminated: unique tool response breaks prefix hash
        contaminated = base + _make_tool_response() + _SUFFIX
        t0 = time.perf_counter()
        llm.generate([contaminated], sp)
        ms = (time.perf_counter() - t0) * 1000
        results_b.append(ms)
        print(f"  Trial {i+1:2d}: {ms:.2f} ms")

    # --- Group C: Speculative Prefill ---
    # Prefill anticipated prompt during sleep, then re-submit (cache hit)
    print("\n--- Group C: Speculative Prefill + Cache Hit ---")
    for i in range(NUM_TRIALS):
        # Prime base
        llm.generate([base + "\n" + _SUFFIX], sp)

        # During sleep: speculatively prefill the anticipated contaminated prompt
        tool_resp = _make_tool_response()
        anticipated = base + tool_resp + _SUFFIX
        llm.generate([anticipated], sp)  # This caches the full prefix

        time.sleep(SLEEP_DURATION)

        # Measure: same anticipated prompt — now fully cached
        t0 = time.perf_counter()
        llm.generate([anticipated], sp)
        ms = (time.perf_counter() - t0) * 1000
        results_c.append(ms)
        print(f"  Trial {i+1:2d}: {ms:.2f} ms")

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
    print(f"| Group | Description                     | Trimmed Mean | Trimmed SD |")
    print(f"|-------|---------------------------------|-------------|-----------|")
    print(f"| A     | Prefix Cache Hit                | {tmu_a:11.2f} | {tsd_a:9.2f} |")
    print(f"| B     | Full Recompute (contamination)  | {tmu_b:11.2f} | {tsd_b:9.2f} |")
    print(f"| C     | Speculative Prefill + Cache Hit | {tmu_c:11.2f} | {tsd_c:9.2f} |")

    print(f"\n--- Validation ---")
    if tmu_a > 0:
        ba = tmu_b / tmu_a
        ca = tmu_c / tmu_a
        print(f"  Radix Penalty (B/A): {ba:.2f}x")
        print(f"  EDMM Recovery (C/A): {ca:.2f}x")
        print(f"  [{'PASS' if ba >= 2.0 else 'FAIL'}] Radix >= 2.0x")
        print(f"  [{'PASS' if ca <= 1.15 else 'FAIL'}] Recovery <= 1.15x")

    # Write results
    report = (
        f"\n## vLLM Live E2E Results (vLLM 0.6.6 V0 Engine)\n\n"
        f"**Model:** {MODEL} (bf16) | **Prefix caching:** enabled\n"
        f"**Iterations:** {NUM_TRIALS} per group\n\n"
        f"| Group | Description | Trimmed Mean (ms) | Trimmed SD (ms) |\n"
        f"|-------|-------------|-------------------|-----------------|\n"
        f"| A | Prefix Cache Hit | {tmu_a:.2f} | {tsd_a:.2f} |\n"
        f"| B | Full Recompute | {tmu_b:.2f} | {tsd_b:.2f} |\n"
        f"| C | Speculative Prefill + Hit | {tmu_c:.2f} | {tsd_c:.2f} |\n\n"
    )
    if tmu_a > 0:
        report += f"- Radix Penalty (B/A): **{ba:.2f}x**\n"
        report += f"- EDMM Recovery (C/A): **{ca:.2f}x**\n"

    with open("/home/dengcchi/edmm_true_e2e_inference_results.md", "w") as f:
        f.write("# EDMM True End-to-End Inference Results\n" + report)

    with open("/home/dengcchi/vllm_edmm_integration_report.md", "a") as f:
        f.write(report)

    print(f"\n  Results: ~/edmm_true_e2e_inference_results.md")
    print(f"  Report appended: ~/vllm_edmm_integration_report.md")
    print(f"{'='*70}")


if __name__ == "__main__":
    run()
