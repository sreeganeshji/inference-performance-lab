#!/usr/bin/env bash
# Decode-attention measurements on a fresh GPU machine (written for a Lambda A100 instance).
#
#   scripts/attention_gpu_run.sh setup     venv outside the repo: vLLM 0.33.0, FlashInfer 0.7.2.post1
#   scripts/attention_gpu_run.sh test      correctness tests on this GPU
#   scripts/attention_gpu_run.sh kernels   CUDA-graph-timed kernel grid, plus a split-count sweep
#   scripts/attention_gpu_run.sh check     greedy FLASH_ATTN vs CUSTOM on Qwen2.5-7B-Instruct (~15 GB download)
#   scripts/attention_gpu_run.sh serving   serving A/B, FLASH_ATTN vs CUSTOM (about an hour)
#   scripts/attention_gpu_run.sh profile   Nsight Systems traces: attention share, scheduler time
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

# torch's cu130 wheels JIT-build the extension with a CUDA 13 nvcc. Prefer a 13.0 toolkit when one
# sits next to an older system CUDA (Lambda Stack ships its own). Drivers older than R580 run
# CUDA 13 through the forward-compatibility libraries, which data-center GPUs support; setup
# installs both when they are missing.
use_cuda13() {
    if [[ -x /usr/local/cuda-13.0/bin/nvcc ]]; then
        export CUDA_HOME=/usr/local/cuda-13.0
        export PATH="$CUDA_HOME/bin:$PATH"
    fi

    if [[ -d /usr/local/cuda-13.0/compat ]]; then
        export LD_LIBRARY_PATH="/usr/local/cuda-13.0/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
}
use_cuda13

stamp="$(date -u +%Y%m%d)"
out_dir="results/data/attention"

# A function rather than a variable: on a bare image nvidia-smi exists only after setup.
gpu_tag() {
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1 |
        awk -F', ' '{name=tolower($1); gsub(/nvidia |geforce /, "", name); gsub(/[^a-z0-9]+/, "-", name); sub(/-[0-9]+gb$/, "", name);
                     printf "%s-%dgb", name, ($2 + 512) / 1024}'
}

cuda_toolkit() {
    . /etc/os-release
    local repo="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${VERSION_ID//./}/x86_64"
    local packages=() new_driver=0

    # Bare cloud images ship without a GPU driver.
    if ! command -v nvidia-smi >/dev/null; then
        packages+=("linux-headers-$(uname -r)" nvidia-open)
        new_driver=1
    fi

    nvcc --version 2>/dev/null | grep -q "release 13\." || packages+=(cuda-toolkit-13-0)

    if ((${#packages[@]})); then
        curl -fsSLo /tmp/cuda-keyring.deb "$repo/cuda-keyring_1.1-1_all.deb"
        sudo dpkg -i /tmp/cuda-keyring.deb
        sudo apt-get update
        sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "${packages[@]}"
    fi

    # A freshly installed driver loads without a reboot unless nouveau already holds the GPU.
    if ((new_driver)); then
        sudo modprobe nvidia
    fi

    local driver_major
    driver_major="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)"

    if ((driver_major < 580)); then
        sudo apt-get install -y cuda-compat-13-0
    fi

    use_cuda13
}

setup() {
    cuda_toolkit
    command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    uv venv --python 3.12 "$venv"

    # Triton compiles a small C launcher against Python.h at first use; system Pythons on bare
    # images ship without headers.
    if ! "$python" -c 'import os, sysconfig; raise SystemExit(not os.path.exists(sysconfig.get_paths()["include"] + "/Python.h"))'; then
        sudo DEBIAN_FRONTEND=noninteractive apt-get install -y python3.12-dev
    fi

    uv pip install --python "$python" vllm==0.33.0 flashinfer-python==0.7.2.post1 pytest ninja numpy
    uv pip install --python "$python" --no-deps -e .
    nvidia-smi
    nvcc --version
    "$python" -c "import torch, vllm; print(torch.__version__, torch.version.cuda, vllm.__version__, torch.cuda.get_device_name())"
}

run_tests() {
    mkdir -p "$out_dir"
    "$python" -m pytest -p no:cacheprovider -q tests/test_paged_decode_attention.py |
        tee "$out_dir/$(gpu_tag)-tests-$stamp.txt"
}

kernels() {
    local prefix
    prefix="$out_dir/$(gpu_tag)"
    mkdir -p "$out_dir"
    "$python" benchmarks/benchmark_paged_decode_attention.py \
        --batch-sizes 1 4 16 64 --context-lens 1024 4096 16384 --trials 9 \
        --output "$prefix-qwen2.5-7b-$stamp.jsonl" | tee "$prefix-qwen2.5-7b-$stamp.md"
    "$python" benchmarks/benchmark_paged_decode_attention.py \
        --batch-sizes 1 4 16 64 --context-lens 4096 --trials 5 \
        --implementations lab-gqa-split-tc --split-sweep-tc 1 2 3 4 6 8 12 16 24 32 \
        --output "$prefix-split-sweep-$stamp.jsonl" | tee "$prefix-split-sweep-$stamp.md"
}

check() {
    mkdir -p "$out_dir"
    "$python" benchmarks/vllm_attention_check.py --model Qwen/Qwen2.5-7B-Instruct \
        --revision a09a35458c702b33eeacc393d103063234e8bc28 --gpu-memory-utilization 0.85 |
        tee "$out_dir/$(gpu_tag)-vllm-check-$stamp.txt"
}

serving() {
    "$python" benchmarks/serving_attention_ab.py --rounds 3 \
        --cells 4096:256:1 4096:256:8 4096:256:32 4096:256:64 16384:256:8 16384:256:16 \
        --result-dir "$out_dir/serving/$(gpu_tag)-$stamp"
}

profile() {
    local dir
    dir="$out_dir/serving-profile/$(gpu_tag)-$stamp"

    # A long-context decode cell, where attention matters most, and a many-sequence cell, where
    # the scheduler does the most work per step.
    for backend in FLASH_ATTN CUSTOM; do
        for cell in 4096:256:32 1024:128:128; do
            "$python" benchmarks/profile_serving_steps.py --backend "$backend" --cell "$cell" \
                --result-dir "$dir/$backend-${cell//:/-}"
        done
    done
}

case "${1:-all}" in
    setup) setup ;;
    test) run_tests ;;
    kernels) kernels ;;
    check) check ;;
    serving) serving ;;
    profile) profile ;;
    all) setup && run_tests && kernels && check && serving && profile ;;
    *) echo "usage: $0 {setup|test|kernels|check|serving|profile|all}" >&2; exit 2 ;;
esac
