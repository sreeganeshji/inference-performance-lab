#!/usr/bin/env python3

import json
import statistics
from collections import defaultdict
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "results_dir",
    nargs="?",
    type=Path,
    default=Path("results/data"),
)
args = parser.parse_args()

records: dict[int, list[dict]] = defaultdict(list)

for path in args.results_dir.glob("*.json"):
    with path.open() as file:
        result = json.load(file)

    if result.get("num_prompts") != 32:
        continue

    records[result["max_concurrency"]].append(result)

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

print("| Concurrency | Samples | Metric | Mean | Stddev | Min | Max |")
print("|---:|---:|---|---:|---:|---:|---:|")

for concurrency in sorted(records):
    runs = records[concurrency]

    for metric in metrics:
        values = [run[metric] for run in runs]
        stddev = statistics.stdev(values) if len(values) > 1 else 0.0

        print(
            f"| {concurrency} | {len(values)} | `{metric}` | "
            f"{statistics.mean(values):.3f} | {stddev:.3f} | "
            f"{min(values):.3f} | {max(values):.3f} |"
        )