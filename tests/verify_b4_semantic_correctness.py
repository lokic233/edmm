#!/usr/bin/env python3
"""
B4 Semantic Correctness Validation

Compares token-level output between:
  B2: Standard vLLM full prefill on a contaminated prompt
  B4: Hash-injection shortcut on the same contaminated prompt

If B4's hash injection causes the engine to skip prefill for blocks
whose KV cache contains data from a DIFFERENT token sequence, the
generated tokens will diverge. This test detects that divergence.
"""
import json
import os
import uuid

os.environ["VLLM_EDMM_ENABLE"] = "1"

from vllm import LLM, SamplingParams
from vllm.core.block.interfaces import Device
from vllm.core.block.prefix_caching_block import PrefixCachingBlock

MODEL = "/tmp/qwen15b"
CONTEXT_TOKENS = 16384
SEED = 42

CODE_UNIT = (
    "class HTTPRequestHandler:\n"
    "    def dispatch(self, method, path):\n"
    "        handler = self._resolve_route(method, path)\n"
    "        return handler(self.request)\n\n"
)
SUFFIX = "Identify the root cause and produce a minimal unified diff."


def build_base_halves(tokenizer, target):
    usable = target - 100
    half = usable // 2
    text = CODE_UNIT * 500
    tokens = tokenizer.encode(text, add_special_tokens=False)
    while len(tokens) < half:
        text += CODE_UNIT * 100
        tokens = tokenizer.encode(text, add_special_tokens=False)
    return tokenizer.decode(tokens[:half])


def compute_block_hashes(token_ids, block_size):
    hashes = []
    prev_hash = None
    for i in range(0, len(token_ids), block_size):
        block_tokens = token_ids[i : i + block_size]
        if len(block_tokens) < block_size:
            break
        is_first = i == 0
        h = PrefixCachingBlock.hash_block_tokens(
            is_first_block=is_first,
            prev_block_hash=prev_hash,
            cur_block_token_ids=block_tokens,
        )
        hashes.append(h)
        prev_hash = h
    return hashes


def run():
    print(f"\n{'='*70}")
    print("B4 Semantic Correctness Validation")
    print(f"Model: {MODEL} | Context: {CONTEXT_TOKENS} | Seed: {SEED}")
    print(f"{'='*70}\n")

    llm = LLM(
        model=MODEL,
        gpu_memory_utilization=0.5,
        max_model_len=32768,
        enforce_eager=True,
        trust_remote_code=True,
        enable_prefix_caching=True,
        seed=SEED,
    )
    sp = SamplingParams(max_tokens=64, temperature=0.0)
    tok = llm.get_tokenizer()
    block_size = llm.llm_engine.scheduler[0].block_manager.block_size

    half_text = build_base_halves(tok, CONTEXT_TOKENS)
    salt = str(uuid.uuid4())
    contaminated = half_text + f"\nTurn_ID: {salt}\n" + half_text + SUFFIX

    print(f"Prompt length: ~{len(tok.encode(contaminated))} tokens")
    print(f"Salt: {salt}")

    # --- B2: Standard full prefill ---
    print("\n--- B2: Standard full prefill ---")
    out_b2 = llm.generate([contaminated], sp)
    b2_ids = list(out_b2[0].outputs[0].token_ids)
    b2_text = out_b2[0].outputs[0].text
    print(f"  Tokens: {len(b2_ids)}")
    print(f"  Text: {b2_text[:100]}...")

    # --- B4: Hash-injection ---
    print("\n--- B4: Hash-injection (oracle context aliasing) ---")

    # Warm base prefix
    clean = half_text + half_text + SUFFIX
    llm.generate([clean], sp)

    gpu_alloc = llm.llm_engine.scheduler[0].block_manager.block_allocator._allocators[
        Device.GPU
    ]
    base_cached = dict(gpu_alloc._cached_blocks)
    base_block_ids = list(base_cached.values())

    # Compute contaminated hashes and inject
    contam_ids = tok.encode(contaminated, add_special_tokens=False)
    contam_hashes = compute_block_hashes(contam_ids, block_size)
    n_inject = min(len(contam_hashes), len(base_block_ids))
    for h, bid in zip(contam_hashes[:n_inject], base_block_ids[:n_inject]):
        gpu_alloc._cached_blocks[h] = bid

    out_b4 = llm.generate([contaminated], sp)
    b4_ids = list(out_b4[0].outputs[0].token_ids)
    b4_text = out_b4[0].outputs[0].text
    print(f"  Tokens: {len(b4_ids)}")
    print(f"  Text: {b4_text[:100]}...")

    # --- Comparison ---
    print(f"\n{'='*70}")
    print("COMPARISON")
    print(f"{'='*70}")

    ids_match = b2_ids == b4_ids
    text_match = b2_text == b4_text

    first_divergence = None
    if not ids_match:
        for i in range(max(len(b2_ids), len(b4_ids))):
            b2_tok = b2_ids[i] if i < len(b2_ids) else None
            b4_tok = b4_ids[i] if i < len(b4_ids) else None
            if b2_tok != b4_tok:
                first_divergence = i
                break

    print(f"  Token IDs match: {ids_match}")
    print(f"  Text match:      {text_match}")
    if first_divergence is not None:
        print(f"  First divergence at position: {first_divergence}")
        print(
            f"    B2 token: {b2_ids[first_divergence] if first_divergence < len(b2_ids) else 'END'}"
        )
        print(
            f"    B4 token: {b4_ids[first_divergence] if first_divergence < len(b4_ids) else 'END'}"
        )

    status = "MATCH" if ids_match else "DIVERGENT"
    print(f"\n  Result: {status}")

    if not ids_match:
        print(f"\n  WARNING: B4 hash-injection produces DIFFERENT tokens than B2.")
        print(f"  This means the injected cache blocks contain KV data from the")
        print(f"  CLEAN prompt, not the contaminated prompt. The engine computes")
        print(f"  attention over stale KV values, producing semantically different")
        print(f"  output. B4 must be documented as an ORACLE UPPER BOUND, not as")
        print(f"  a semantically correct optimization.")

    print(f"{'='*70}\n")

    # --- Save report ---
    report = {
        "test": "b4_semantic_correctness",
        "model": MODEL,
        "context_tokens": CONTEXT_TOKENS,
        "seed": SEED,
        "salt": salt,
        "b2_token_ids": b2_ids,
        "b4_token_ids": b4_ids,
        "b2_text": b2_text,
        "b4_text": b4_text,
        "ids_match": ids_match,
        "text_match": text_match,
        "first_divergence_position": first_divergence,
        "status": status,
    }

    report_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "traces",
        "b4_semantic_correctness_report.json",
    )
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    run()
