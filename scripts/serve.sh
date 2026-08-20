#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$repo_root/scripts/env.sh"
cd "$repo_root"

model="Qwen/Qwen2.5-7B-Instruct"
revision="a09a35458c702b33eeacc393d103063234e8bc28"

exec uv run --locked --no-sync vllm serve "$model" \
    --revision "$revision" \
    --served-model-name qwen2.5-7b-instruct \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --no-enable-prefix-caching \
    --generation-config vllm \
    --seed 0 \
    --host 127.0.0.1 \
    --port "${VLLM_PORT:-8000}"