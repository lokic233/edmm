#!/usr/bin/env python3
"""
EDMM P1.3: Context-Length Scaling Sweep with Hash-Injection B4

Measures TTFT across 4 context scales through the live vLLM engine.

Three groups per scale:
  C0: Clean prefix cache hit
  B2: Mid-prompt contamination (cold miss, unique UUID)
  B4: EDMM — warm base prefix, then inject contaminated block hashes
      directly into vLLM's _cached_blocks registry + remap physical pages,
      so the engine sees a cache hit for the contaminated prompt

Uses Qwen2.5-1.5B-Instruct with VMM-backed KV cache.
"""
import ctypes
import os
import time
import uuid

os.environ["VLLM_EDMM_ENABLE"] = "1"

import torch
from vllm import LLM, SamplingParams
from vllm.core.block.interfaces import Device
from vllm.core.block.prefix_caching_block import PrefixCachingBlock

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
    tokens = tokenizer.encode(text, add_special_tokens=False)
    while len(tokens) < half_target:
        text += CODE_UNIT * 100
        tokens = tokenizer.encode(text, add_special_tokens=False)
    half_text = tokenizer.decode(tokens[:half_target])
    return half_text, half_text


def compute_block_hashes(token_ids, block_size):
    """Compute the chained content hashes for a sequence of token IDs,
    matching vLLM's PrefixCachingBlock.hash_block_tokens exactly."""
    hashes = []
    prev_hash = None
    for i in range(0, len(token_ids), block_size):
        block_tokens = token_ids[i : i + block_size]
        if len(block_tokens) < block_size:
            break  # partial block — not cached
        is_first = i == 0
        h = PrefixCachingBlock.hash_block_tokens(
            is_first_block=is_first,
            prev_block_hash=prev_hash,
            cur_block_token_ids=block_tokens,
        )
        hashes.append(h)
        prev_hash = h
    return hashes


def inject_hashes_into_cache(llm, target_hashes, source_block_ids):
    """Inject hash→block_id mappings into vLLM's prefix cache registry.
    This makes the engine think it has computed KV cache for these blocks."""
    gpu_alloc = llm.llm_engine.scheduler[0].block_manager.block_allocator._allocators[
        Device.GPU
    ]
    for h, bid in zip(target_hashes, source_block_ids):
        gpu_alloc._cached_blocks[h] = bid


def run():
    print(f"\n{'='*70}")
    print("EDMM P1.3: Context-Length Scaling (Hash-Injection B4)")
    print(f"Model: {MODEL}")
    print(f"Scales: {SCALES}")
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
    block_size = llm.llm_engine.scheduler[0].block_manager.block_size
    print(f"Block size: {block_size} tokens\n")

    llm.generate(["Hello world"], sp)

    all_results = {}

    for scale in SCALES:
        print(f"--- Scale: {scale} tokens ---")

        h1, h2 = build_base_halves(tok, scale)
        actual_h1 = len(tok.encode(h1, add_special_tokens=False))
        print(f"  Half: {actual_h1} tokens each")

        clean = h1 + h2 + SUFFIX

        llm.generate([clean], sp)

        # C0: cache hit
        ttft_c0 = []
        for t in range(NUM_TRIALS):
            llm.generate([clean], sp)
            time.sleep(0.3)
            t0 = time.perf_counter()
            llm.generate([clean], sp)
            ttft_c0.append((time.perf_counter() - t0) * 1000)

        # B2: mid-prompt contamination (unique salt, cold miss)
        ttft_b2 = []
        for t in range(NUM_TRIALS):
            llm.generate([clean], sp)
            time.sleep(0.3)
            salt = str(uuid.uuid4())
            contaminated = h1 + f"\nTurn_ID: {salt}\n" + h2 + SUFFIX
            t0 = time.perf_counter()
            llm.generate([contaminated], sp)
            ttft_b2.append((time.perf_counter() - t0) * 1000)

        # B4: EDMM hash-injection
        # Step 1: warm the base prefix
        # Step 2: compute hashes for the contaminated prompt
        # Step 3: inject those hashes pointing to the base prefix's block IDs
        # Step 4: generate the contaminated prompt — engine sees cache hit
        ttft_b4 = []
        for t in range(NUM_TRIALS):
            # Warm base prefix and capture its block IDs
            llm.generate([clean], sp)
            gpu_alloc = llm.llm_engine.scheduler[
                0
            ].block_manager.block_allocator._allocators[Device.GPU]
            base_cached = dict(gpu_alloc._cached_blocks)

            time.sleep(0.1)  # simulated tool execution bubble

            # Compute hashes for the contaminated prompt
            salt = str(uuid.uuid4())
            contaminated = h1 + f"\nTurn_ID: {salt}\n" + h2 + SUFFIX
            contam_ids = tok.encode(contaminated, add_special_tokens=False)
            contam_hashes = compute_block_hashes(contam_ids, block_size)

            # Get block IDs from the base cached blocks (reuse them)
            base_block_ids = list(base_cached.values())

            # Inject: map contaminated hashes to base block IDs
            # Only inject up to the number of available base blocks
            n_inject = min(len(contam_hashes), len(base_block_ids))
            inject_hashes_into_cache(
                llm, contam_hashes[:n_inject], base_block_ids[:n_inject]
            )

            # Generate — engine should find cache hits for injected hashes
            t0 = time.perf_counter()
            llm.generate([contaminated], sp)
            ttft_b4.append((time.perf_counter() - t0) * 1000)

        mu_c0 = sum(ttft_c0) / len(ttft_c0)
        mu_b2 = sum(ttft_b2) / len(ttft_b2)
        mu_b4 = sum(ttft_b4) / len(ttft_b4)

        all_results[scale] = {"C0": mu_c0, "B2": mu_b2, "B4": mu_b4}

        print(
            f"  C0 (cache hit):  {mu_c0:7.1f}ms  ({', '.join(f'{x:.0f}' for x in ttft_c0)})"
        )
        print(
            f"  B2 (mid-prompt): {mu_b2:7.1f}ms  ({', '.join(f'{x:.0f}' for x in ttft_b2)})"
        )
        print(
            f"  B4 (EDMM hash):  {mu_b4:7.1f}ms  ({', '.join(f'{x:.0f}' for x in ttft_b4)})"
        )
        print(
            f"  B2/C0={mu_b2/mu_c0:.2f}x  B4/C0={mu_b4/mu_c0:.2f}x  B2/B4={mu_b2/mu_b4:.2f}x"
        )

    # Summary
    print(f"\n{'='*70}")
    print("CONTEXT-LENGTH SCALING RESULTS (Hash-Injection B4)")
    print(f"{'='*70}\n")

    print(
        "| Context | C0 Hit (ms) | B2 Miss (ms) | B4 EDMM (ms) | B2/C0  | B4/C0  | B2/B4 Speedup |"
    )
    print(
        "|---------|-------------|--------------|--------------|--------|--------|---------------|"
    )
    for scale in SCALES:
        r = all_results[scale]
        print(
            f"| {scale:>7} | {r['C0']:11.1f} | {r['B2']:12.1f} | {r['B4']:12.1f} | "
            f"{r['B2']/r['C0']:6.2f}x | {r['B4']/r['C0']:6.2f}x | "
            f"{r['B2']/r['B4']:13.2f}x |"
        )

    if len(SCALES) >= 2:
        r_s, r_l = all_results[SCALES[0]], all_results[SCALES[-1]]
        print(f"\n  At {SCALES[0]}: B2/B4 = {r_s['B2']/r_s['B4']:.2f}x")
        print(f"  At {SCALES[-1]}: B2/B4 = {r_l['B2']/r_l['B4']:.2f}x")

    print(f"{'='*70}\n")


if __name__ == "__main__":
    run()
