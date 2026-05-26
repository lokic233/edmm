# EDMM Steady-State Validation Results

**Model:** /tmp/qwen7b (bf16) | **GPU:** NVIDIA H100 (index 7)
**Iterations:** 20 (after 1 untracked warmup pass per group)
**Base context:** 16384 tokens | **Suffix:** 2063 tokens
**Sleep duration:** 2.0s

## Summary (10% trimmed means — top/bottom outliers excluded)

| Group | Description | Trimmed Mean (ms) | Trimmed SD (ms) | Raw Mean (ms) | Raw SD (ms) |
|-------|-------------|-------------------|-----------------|---------------|-------------|
| A | Cached Prefix + Incremental Suffix | 163.89 | 0.63 | 163.95 | 1.21 |
| B | Mid-Prompt Contamination (full recompute) | 684.82 | 12.69 | 686.25 | 14.58 |
| C | EDMM Speculative Prefill + Incremental | 157.40 | 24.67 | 159.13 | 27.45 |

## Validation Criteria (10% trimmed means)

| Criterion | Result | Value | Threshold |
|-----------|--------|-------|-----------|
| Orchestration Bubble | PASS | GPU util = 0.0% (sigma=0.0%) | ~0% during sleep |
| Radix Discontinuity (B/A) | PASS | 4.18x | >= 2.5x |
| Speculative Recovery (C/A) | PASS | 0.96x | <= 1.15x |

## Detailed Iteration Data

| Group | Iter | TTFT (ms) | Cache Status |
|-------|------|-----------|--------------|
| A | 1 | 162.84 | Hit |
| A | 2 | 164.23 | Hit |
| A | 3 | 163.02 | Hit |
| A | 4 | 163.88 | Hit |
| A | 5 | 163.45 | Hit |
| A | 6 | 164.09 | Hit |
| A | 7 | 164.70 | Hit |
| A | 8 | 163.80 | Hit |
| A | 9 | 164.50 | Hit |
| A | 10 | 164.27 | Hit |
| A | 11 | 167.72 | Hit |
| A | 12 | 164.39 | Hit |
| A | 13 | 161.81 | Hit |
| A | 14 | 163.30 | Hit |
| A | 15 | 162.30 | Hit |
| A | 16 | 165.08 | Hit |
| A | 17 | 164.17 | Hit |
| A | 18 | 165.09 | Hit |
| A | 19 | 163.32 | Hit |
| A | 20 | 163.14 | Hit |
| B | 1 | 707.04 | Miss |
| B | 2 | 702.83 | Miss |
| B | 3 | 711.34 | Miss |
| B | 4 | 673.57 | Miss |
| B | 5 | 711.50 | Miss |
| B | 6 | 676.21 | Miss |
| B | 7 | 711.19 | Miss |
| B | 8 | 676.79 | Miss |
| B | 9 | 679.70 | Miss |
| B | 10 | 678.25 | Miss |
| B | 11 | 704.97 | Miss |
| B | 12 | 674.65 | Miss |
| B | 13 | 679.00 | Miss |
| B | 14 | 679.12 | Miss |
| B | 15 | 671.53 | Miss |
| B | 16 | 678.37 | Miss |
| B | 17 | 677.68 | Miss |
| B | 18 | 678.76 | Miss |
| B | 19 | 675.27 | Miss |
| B | 20 | 677.28 | Miss |
| C | 1 | 130.85 | Hit |
| C | 2 | 197.50 | Hit |
| C | 3 | 165.56 | Hit |
| C | 4 | 197.93 | Hit |
| C | 5 | 132.07 | Hit |
| C | 6 | 164.37 | Hit |
| C | 7 | 199.64 | Hit |
| C | 8 | 132.63 | Hit |
| C | 9 | 132.21 | Hit |
| C | 10 | 131.88 | Hit |
| C | 11 | 163.65 | Hit |
| C | 12 | 132.41 | Hit |
| C | 13 | 130.02 | Hit |
| C | 14 | 133.81 | Hit |
| C | 15 | 166.36 | Hit |
| C | 16 | 202.99 | Hit |
| C | 17 | 133.90 | Hit |
| C | 18 | 166.14 | Hit |
| C | 19 | 200.46 | Hit |
| C | 20 | 168.27 | Hit |
