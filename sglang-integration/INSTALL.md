# SGLang Baseline Integration

## Environment

SGLang requires torch 2.11.0 which conflicts with vLLM 0.6.6 (torch 2.5.1).
A separate isolated venv is required.

## Setup

```bash
# Create isolated venv from conda python3.12
/home/dengcchi/.conda/envs/py312conda/bin/python -m venv ~/sglang-env --without-pip

# Bootstrap pip into venv
/home/dengcchi/.conda/envs/py312conda/bin/python -m pip install \
    --target=~/sglang-env/lib/python3.12/site-packages pip

# Install SGLang (pulls torch 2.11.0, flashinfer, triton, etc.)
~/sglang-env/bin/python -m pip install "sglang[srt]"
```

## Running

```bash
# PATH must include venv bin for ninja (required by CUDA graph compilation)
CUDA_VISIBLE_DEVICES=7 PATH=~/sglang-env/bin:$PATH \
    ~/sglang-env/bin/python sglang-integration/test_sglang_baseline.py
```

## Versions Used

- SGLang: 0.5.12.post1
- PyTorch: 2.11.0+cu130
- FlashInfer: 0.6.11.post1
- Model: Qwen2.5-1.5B-Instruct (local at /tmp/qwen15b)

## Results

| Context | C0 Hit (ms) | B2 Miss (ms) | B2/C0 Penalty |
|---------|-------------|--------------|---------------|
|    4096 |        14.6 |         23.6 |          1.61x |
|    8192 |        21.4 |         43.4 |          2.03x |
|   16384 |        30.1 |         97.7 |          3.25x |
|   32768 |        52.1 |        272.9 |          5.24x |

SGLang's RadixAttention shows the same superlinear scaling penalty as
vLLM's prefix caching under mid-prompt contamination. Neither framework
can solve this at the logical hash level — EDMM operates at the physical
memory layer to bypass it entirely.
