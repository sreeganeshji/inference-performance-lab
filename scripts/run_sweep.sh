#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

repetitions="${REPETITIONS:-3}"
num_prompts="${NUM_PROMPTS:-32}"
protocol="${PROTOCOL:-prefix-cache-off}"

read -r -a concurrencies <<< "${CONCURRENCIES:-1 2 4 8 16 32}"

workloads=(
    "256 128"
    "2048 32"
)

for workload in "${workloads[@]}"; do
    read -r input_len output_len <<< "$workload"

    for concurrency in "${concurrencies[@]}"; do
        for ((repeat = 1; repeat <= repetitions; repeat++)); do
            echo "input=$input_len output=$output_len concurrency=$concurrency repeat=$repeat/$repetitions"

            NUM_PROMPTS="$num_prompts" \
            INPUT_LEN="$input_len" \
            OUTPUT_LEN="$output_len" \
            PROTOCOL="$protocol" \
                ./scripts/benchmark.sh "$concurrency" >/dev/null
        done
    done
done

echo "Sweep complete."