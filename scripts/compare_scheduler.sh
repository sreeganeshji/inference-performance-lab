#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python="${VLLM_PYTHON:-$HOME/.venvs/ipl-vllm/bin/python}"

cmake -S scheduler -B build/scheduler-release -G Ninja -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build/scheduler-release --target replay >/dev/null

# The real Scheduler needs no GPU, only its config, and the model is a local config file.
CUDA_VISIBLE_DEVICES='' VLLM_LOGGING_LEVEL=ERROR PYTHONDONTWRITEBYTECODE=1 \
    "$python" scheduler/compare/compare.py
