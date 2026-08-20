#!/usr/bin/env python3

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    "results_dir",
    nargs="?",
    type=Path,
    default=Path("results/data"),
)
parser.add_argument("--num-prompts", type=int, default=32)
args = parser.parse_args()

records: dict[tuple[int, int, int], list[dict]] = defaultdict(list)

for path in args.results_dir.glob("*.json"):
    with path.open() as file:
        result = json.load(file)

    if (
        result.get("num_prompts") != args.num_prompts
        or result.get("failed", 0) != 0
    ):
        continue

    completed = result["completed"]
    input_len = result.get(
        "random_input_len",
        result["total_input_tokens"] // completed,
    )
    output_len = result.get(
        "random_output_len",
        result["total_output_tokens"] // completed,
    )

    key = (
        input_len,
        output_len,
        result["max_concurrency"],
    )

    records[key].append(result)

if not records:
    raise SystemExit("No matching benchmark results found")

metrics = [
    "output_throughput",
    "request_throughput",
    "mean_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "p99_itl_ms",
    "mean_e2el_ms",
    "p99_e2el_ms",
]

print("| Input | Output | Concurrency | Samples | Metric | Mean | Stddev | Min | Max |")
print("|---:|---:|---:|---:|---|---:|---:|---:|---:|")

for (input_len, output_len, concurrency), runs in sorted(records.items()):
    for metric in metrics:
        values = [run[metric] for run in runs]
        stddev = statistics.stdev(values) if len(values) > 1 else 0.0

        print(
            f"| {input_len} | {output_len} | {concurrency} | {len(values)} | "
            f"`{metric}` | {statistics.mean(values):.3f} | "
            f"{stddev:.3f} | {min(values):.3f} | {max(values):.3f} |"
        )
        