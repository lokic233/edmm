#!/usr/bin/env python3
"""
EDMM P1.3 Cross-Framework Baseline: SGLang RadixAttention

Measures TTFT across 4 context scales [4K, 8K, 16K, 32K] through
SGLang's RadixAttention prefix caching to serve as the paper's
primary baseline comparison.

Two groups per scale:
  C0: Clean prefix cache hit (RadixAttention best case)
  B2: Mid-prompt contamination (cold miss, unique UUID per trial)

Run with the sglang-env venv:
  CUDA_VISIBLE_DEVICES=7 ~/sglang-env/bin/python tests/test_sglang_baseline.py
"""
import os
import time
import uuid

from sglang import Engine

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
    usable = target_tokens - 100
    half_target = usable // 2
    text = CODE_UNIT * 500
    tokens = tokenizer.encode(text)

    while len(tokens) < half_target:
        text += CODE_UNIT * 100
        tokens = tokenizer.encode(text)

    half_text = tokenizer.decode(tokens[:half_target])
    return half_text, half_text


def run():
    print(f"\n{'='*70}")
    print("SGLang RadixAttention Baseline — Context Scaling Sweep")
    print(f"Model: {MODEL}")
    print(f"Scales: {SCALES}")
    print(f"Trials: {NUM_TRIALS}")
    print(f"{'='*70}\n")

    engine = Engine(model_path=MODEL, mem_fraction_static=0.5)
    print("Engine created.\n")

    # Get tokenizer
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    sp = {"max_new_tokens": 1, "temperature": 0.0}

    # Warmup
    engine.generate("Hello world", sp)

    all_results = {}

    for scale in SCALES:
        print(f"--- Scale: {scale} tokens ---")

        h1, h2 = build_base_halves(tok, scale)
        actual_toks = len(tok.encode(h1)) + len(tok.encode(h2))
        print(f"  Total base: ~{actual_toks} tokens")

        clean = h1 + h2 + SUFFIX

        # Warmup this scale
        engine.generate(clean, sp)

        # C0: cache hit
        ttft_c0 = []
        for t in range(NUM_TRIALS):
            engine.generate(clean, sp)
            time.sleep(0.3)
            t0 = time.perf_counter()
            engine.generate(clean, sp)
            ttft_c0.append((time.perf_counter() - t0) * 1000)

        # B2: mid-prompt contamination
        ttft_b2 = []
        for t in range(NUM_TRIALS):
            engine.generate(clean, sp)
            time.sleep(0.3)
            salt = str(uuid.uuid4())
            contaminated = h1 + f"\nTurn_ID: {salt}\n" + h2 + SUFFIX
            t0 = time.perf_counter()
            engine.generate(contaminated, sp)
            ttft_b2.append((time.perf_counter() - t0) * 1000)

        mu_c0 = sum(ttft_c0) / len(ttft_c0)
        mu_b2 = sum(ttft_b2) / len(ttft_b2)

        all_results[scale] = {
            "C0": mu_c0,
            "B2": mu_b2,
            "C0_raw": ttft_c0,
            "B2_raw": ttft_b2,
        }

        print(
            f"  C0 (cache hit):  {mu_c0:7.1f}ms  ({', '.join(f'{x:.0f}' for x in ttft_c0)})"
        )
        print(
            f"  B2 (mid-prompt): {mu_b2:7.1f}ms  ({', '.join(f'{x:.0f}' for x in ttft_b2)})"
        )
        print(f"  B2/C0 = {mu_b2/mu_c0:.2f}x")

    # Summary
    print(f"\n{'='*70}")
    print("SGLANG RADIXATTENTION BASELINE RESULTS")
    print(f"{'='*70}\n")

    print("| Context | C0 Hit (ms) | B2 Miss (ms) | B2/C0 Penalty |")
    print("|---------|-------------|--------------|---------------|")

    for scale in SCALES:
        r = all_results[scale]
        ratio = r["B2"] / r["C0"]
        print(f"| {scale:>7} | {r['C0']:11.1f} | {r['B2']:12.1f} | {ratio:13.2f}x |")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    run()
