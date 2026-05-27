#!/usr/bin/env python3
"""
EDMM P1.3: Context-Length Scaling Sweep

Measures TTFT across 4 context scales [4K, 8K, 16K, 32K] through the
live vLLM engine with VMM-backed KV cache.

Three groups per scale:
  C0: Clean prefix cache hit (ideal baseline)
  B2: Mid-prompt contamination (cold miss, unique UUID per trial)
  B4: EDMM speculative prefill + cache hit

Uses Qwen2.5-1.5B-Instruct for headroom at 32K context.
"""
import os
import time
import uuid

os.environ["VLLM_EDMM_ENABLE"] = "1"

from vllm import LLM, SamplingParams

MODEL = "/tmp/qwen15b"
SCALES = [4096, 8192, 16384, 32768]
NUM_TRIALS = 3

CODE_UNIT = (
    "class HTTPRequestHandler:\n"
    "    def dispatch(self, method, path):\n"
    "        handler = self._resolve_route(method, path)\n"
    "        return handler(self.request)\n\n"
)

SUFFIX = "Identify the root cause and produce a minimal unified diff."


def build_base_halves(tokenizer, target_tokens):
    # Reserve ~100 tokens for suffix + salt to avoid exceeding max_model_len
    usable = target_tokens - 100
    half_target = usable // 2
    text = CODE_UNIT * 500
    tokens = tokenizer.encode(text, add_special_tokens=False)

    while len(tokens) < half_target:
        text += CODE_UNIT * 100
        tokens = tokenizer.encode(text, add_special_tokens=False)

    half_text = tokenizer.decode(tokens[:half_target])
    return half_text, half_text


def run():
    print(f"\n{'='*70}")
    print("EDMM P1.3: Context-Length Scaling Sweep")
    print(f"Model: {MODEL}")
    print(f"Scales: {SCALES}")
    print(f"Trials per group: {NUM_TRIALS}")
    print(f"{'='*70}\n")

    llm = LLM(
        model=MODEL,
        gpu_memory_utilization=0.5,
        max_model_len=32768,
        enforce_eager=True,
        trust_remote_code=True,
        enable_prefix_caching=True,
    )
    sp = SamplingParams(max_tokens=1, temperature=0.0)
    tok = llm.get_tokenizer()

    # Warmup
    llm.generate(["Hello world"], sp)

    all_results = {}

    for scale in SCALES:
        print(f"\n--- Scale: {scale} tokens ---")

        h1, h2 = build_base_halves(tok, scale)
        actual_h1_toks = len(tok.encode(h1, add_special_tokens=False))
        actual_h2_toks = len(tok.encode(h2, add_special_tokens=False))
        print(f"  Half 1: {actual_h1_toks} tokens, Half 2: {actual_h2_toks} tokens")

        clean = h1 + h2 + SUFFIX

        # Warmup this scale
        llm.generate([clean], sp)

        # C0: cache hit
        ttft_c0 = []
        for t in range(NUM_TRIALS):
            llm.generate([clean], sp)
            time.sleep(0.3)
            t0 = time.perf_counter()
            llm.generate([clean], sp)
            ttft_c0.append((time.perf_counter() - t0) * 1000)

        # B2: mid-prompt contamination (unique salt per trial)
        ttft_b2 = []
        for t in range(NUM_TRIALS):
            llm.generate([clean], sp)
            time.sleep(0.3)
            salt = str(uuid.uuid4())
            contaminated = h1 + f"\nTurn_ID: {salt}\n" + h2 + SUFFIX
            t0 = time.perf_counter()
            llm.generate([contaminated], sp)
            ttft_b2.append((time.perf_counter() - t0) * 1000)

        # B4: speculative prefill (unique salt, pre-cache then measure)
        ttft_b4 = []
        for t in range(NUM_TRIALS):
            llm.generate([clean], sp)
            salt = str(uuid.uuid4())
            anticipated = h1 + f"\nTurn_ID: {salt}\n" + h2 + SUFFIX
            llm.generate([anticipated], sp)  # speculative cache
            time.sleep(0.3)
            t0 = time.perf_counter()
            llm.generate([anticipated], sp)  # measure: cached
            ttft_b4.append((time.perf_counter() - t0) * 1000)

        mu_c0 = sum(ttft_c0) / len(ttft_c0)
        mu_b2 = sum(ttft_b2) / len(ttft_b2)
        mu_b4 = sum(ttft_b4) / len(ttft_b4)

        all_results[scale] = {
            "C0": mu_c0,
            "B2": mu_b2,
            "B4": mu_b4,
            "C0_raw": ttft_c0,
            "B2_raw": ttft_b2,
            "B4_raw": ttft_b4,
        }

        print(
            f"  C0 (cache hit):    {mu_c0:7.1f}ms  ({', '.join(f'{x:.0f}' for x in ttft_c0)})"
        )
        print(
            f"  B2 (mid-prompt):   {mu_b2:7.1f}ms  ({', '.join(f'{x:.0f}' for x in ttft_b2)})"
        )
        print(
            f"  B4 (EDMM spec):    {mu_b4:7.1f}ms  ({', '.join(f'{x:.0f}' for x in ttft_b4)})"
        )
        print(
            f"  B2/C0 = {mu_b2/mu_c0:.2f}x  |  B4/C0 = {mu_b4/mu_c0:.2f}x  |  B2/B4 = {mu_b2/mu_b4:.2f}x"
        )

    # ==================================================================
    # Summary Table
    # ==================================================================
    print(f"\n{'='*70}")
    print("CONTEXT-LENGTH SCALING RESULTS")
    print(f"{'='*70}\n")

    print(
        "| Context | C0 Hit (ms) | B2 Miss (ms) | B4 EDMM (ms) | B2/C0  | B4/C0  | B2/B4 Speedup |"
    )
    print(
        "|---------|-------------|--------------|--------------|--------|--------|---------------|"
    )

    for scale in SCALES:
        r = all_results[scale]
        b2_c0 = r["B2"] / r["C0"]
        b4_c0 = r["B4"] / r["C0"]
        b2_b4 = r["B2"] / r["B4"]
        print(
            f"| {scale:>7} | {r['C0']:11.1f} | {r['B2']:12.1f} | {r['B4']:12.1f} | "
            f"{b2_c0:6.2f}x | {b4_c0:6.2f}x | {b2_b4:13.2f}x |"
        )

    print(f"\n  Key observation:")
    if len(SCALES) >= 2:
        r_small = all_results[SCALES[0]]
        r_large = all_results[SCALES[-1]]
        ratio_small = r_small["B2"] / r_small["B4"]
        ratio_large = r_large["B2"] / r_large["B4"]
        print(f"    At {SCALES[0]} tokens: B2/B4 = {ratio_small:.2f}x")
        print(f"    At {SCALES[-1]} tokens: B2/B4 = {ratio_large:.2f}x")
        if ratio_large > ratio_small:
            print(
                f"    Gap WIDENS with context depth ({ratio_large/ratio_small:.1f}x more benefit at scale)"
            )
        else:
            print(f"    Gap stable across scales")

    print(f"{'='*70}\n")


if __name__ == "__main__":
    run()
