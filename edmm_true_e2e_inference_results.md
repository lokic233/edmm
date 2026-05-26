# EDMM True E2E Results (vLLM 0.6.6 Live Engine)

**Model:** /tmp/qwen7b (bf16) | **Engine:** vLLM 0.6.6 V0
**Prefix caching:** enabled | **Iterations:** 10

| Group | Description | Trimmed Mean (ms) | Trimmed SD (ms) |
|-------|-------------|-------------------|-----------------|
| A | Prefix Cache Hit | 33.35 | 1.08 |
| B | Mid-Prompt Contamination | 273.80 | 1.75 |
| C | Speculative Prefill + Hit | 39.18 | 0.51 |

- Radix Penalty (B/A): **8.21x**
- EDMM Recovery (C/A): **1.17x**
