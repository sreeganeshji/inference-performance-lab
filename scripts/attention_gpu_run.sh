#!/usr/bin/env bash
# Decode-attention measurements on a fresh GPU machine (written for a Lambda A100 instance).
#
#   scripts/attention_gpu_run.sh setup     venv outside the repo: vLLM 0.33.0, FlashInfer 0.7.2.post1
#   scripts/attention_gpu_run.sh test      correctness tests on this GPU
#   scripts/attention_gpu_run.sh kernels   CUDA-graph-timed kernel grid, plus a split-count sweep
#   scripts/attention_gpu_run.sh check     greedy FLASH_ATTN vs CUSTOM on Qwen2.5-7B-Instruct (~15 GB download)
#   scripts/attention_gpu_run.sh serving   serving A/B, FLASH_ATTN vs CUSTOM (about an hour)
#   scripts/attention_gpu_run.sh all       every step in order
#
# Results go to results/data/attention/<gpu tag>-*.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

venv="${IPL_VENV:-$HOME/.venvs/ipl-vllm}"
python="$venv/bin/python"
export PATH="$venv/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1

gpu_tag="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1 |
    awk -F', ' '{name=tolower($1); gsub(/nvidia |geforce /, "", name); gsub(/[^a-z0-9]+/, "-", name); sub(/-[0-9]+gb$/, "", name);
                 printf "%s-%dgb", name, ($2 + 512) / 1024}')"
stamp="$(date -u +%Y%m%d)"
out_dir="results/data/attention"

setup() {
    command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    uv venv --python 3.12 "$venv"
    uv pip install --python "$python" vllm==0.33.0 flashinfer-python==0.7.2.post1 pytest ninja numpy
    uv pip install --python "$python" --no-deps -e .
    nvidia-smi
    nvcc --version || echo "warning: nvcc not found; the JIT extension build needs a CUDA 13 toolkit"
    "$python" -c "import torch, vllm; print(torch.__version__, torch.version.cuda, vllm.__version__)"
}

run_tests() {
    "$python" -m pytest -p no:cacheprovider -q tests/test_paged_decode_attention.py
}

kernels() {
    mkdir -p "$out_dir"
    "$python" benchmarks/benchmark_paged_decode_attention.py \
        --batch-sizes 1 4 16 64 --context-lens 1024 4096 16384 --trials 9 \
        --output "$out_dir/$gpu_tag-qwen2.5-7b-$stamp.jsonl" | tee "$out_dir/$gpu_tag-qwen2.5-7b-$stamp.md"
    "$python" benchmarks/benchmark_paged_decode_attention.py \
        --batch-sizes 1 4 16 64 --context-lens 4096 --trials 5 \
        --implementations lab-gqa-split-tc --split-sweep-tc 1 2 3 4 6 8 12 16 24 32 \
        --output "$out_dir/$gpu_tag-split-sweep-$stamp.jsonl" | tee "$out_dir/$gpu_tag-split-sweep-$stamp.md"
}

check() {
    "$python" benchmarks/vllm_attention_check.py --model Qwen/Qwen2.5-7B-Instruct --gpu-memory-utilization 0.85 |
        tee "$out_dir/$gpu_tag-vllm-check-$stamp.txt"
}

serving() {
    "$python" benchmarks/serving_attention_ab.py --rounds 3 \
        --cells 4096:256:1 4096:256:8 4096:256:32 4096:256:64 16384:256:8 16384:256:16 \
        --result-dir "$out_dir/serving/$gpu_tag-$stamp"
}

case "${1:-all}" in
    setup) setup ;;
    test) run_tests ;;
    kernels) kernels ;;
    check) check ;;
    serving) serving ;;
    all) setup && run_tests && kernels && check && serving ;;
    *) echo "usage: $0 {setup|test|kernels|check|serving|all}" >&2; exit 2 ;;
esac
