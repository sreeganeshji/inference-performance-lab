#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$repo_root/scripts/env.sh"
cd "$repo_root"

concurrency="${1:-${CONCURRENCY:-1}}"
input_len="${INPUT_LEN:-256}"
output_len="${OUTPUT_LEN:-128}"
num_prompts="${NUM_PROMPTS:-32}"
protocol="${PROTOCOL:-prefix-cache-off}"

if ((input_len + output_len > 4096)); then
    echo "error: input_len + output_len must not exceed 4096" >&2
    exit 1
fi

revision="a09a35458c702b33eeacc393d103063234e8bc28"
snapshot="$HF_HUB_CACHE/models--Qwen--Qwen2.5-7B-Instruct/snapshots/$revision"

artifact_dir="${RESULT_DIR:-artifacts/raw/$protocol}"
mkdir -p "$artifact_dir"

timestamp="$(date -u +%Y%m%dT%H%M%S%3NZ)"
result_stem="qwen-a100-${protocol}-i${input_len}-o${output_len}-c${concurrency}-${timestamp}"
output="$artifact_dir/${result_stem}.txt"

uv run --locked --no-sync vllm bench serve \
    --backend openai \
    --base-url http://127.0.0.1:8000 \
    --endpoint /v1/completions \
    --model qwen2.5-7b-instruct \
    --tokenizer "$snapshot" \
    --dataset-name random \
    --num-prompts "$num_prompts" \
    --random-input-len "$input_len" \
    --random-output-len "$output_len" \
    --random-range-ratio 0 \
    --ignore-eos \
    --request-rate inf \
    --max-concurrency "$concurrency" \
    --seed 0 \
    --temperature 0 \
    --num-warmups 2 \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,95,99 \
    --save-result \
    --save-detailed \
    --result-dir "$artifact_dir" \
    --result-filename "${result_stem}.json" \
    2>&1 | tee "$output"
    