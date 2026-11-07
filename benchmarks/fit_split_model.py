"""Fit the split-KV cost model's constants to a measured split sweep on this GPU.

Reads split-sweep records (lab-gqa-split-tc-N rows from benchmark_paged_decode_attention.py
--split-sweep-tc) and grid-searches SplitModel's constants. A candidate's score is its mean
regret: for each (batch, context) in the sweep, the model picks the split count it predicts is
cheapest among the counts that were measured, and the regret is that count's measured time over
the fastest measured time, minus 1. Run it on the GPU that produced the sweep; the model needs the
kernel's resident blocks per SM there.

    python benchmarks/fit_split_model.py results/data/attention/a100-sxm4-40gb-split-fit-*.jsonl
"""

from __future__ import annotations

import argparse
import functools
import itertools
import json
import statistics
import sys
from pathlib import Path

import torch

from inference_performance_lab.kernels.extension import (
    SplitModel,
    _multiprocessor_count,
    _resident_blocks,
    modeled_split_cost,
    split_model,
)

BLOCK_SIZE = 16  # the KV block size benchmark_paged_decode_attention.py uses
GRID = {
    "saturating_blocks_per_sm": [x / 10 for x in range(4, 31)],
    "block_overhead_bytes": [0] + [(8 << 10) << i for i in range(8)],  # 0, 8 KiB .. 1 MiB
    "partial_traffic_weight": [0.5, 1, 2, 3, 4, 6, 8, 12],
    "combine_overhead_bytes": [0, 0.5e6, 1e6, 2.5e6, 4e6, 6e6, 1e7],
}


def load_sweep(paths: list[Path], gpu: str) -> dict[tuple[int, int, int, int], dict[int, float]]:
    """(batch, context, query heads, KV heads) -> {splits: median µs}."""

    sweep: dict[tuple[int, int, int, int], dict[int, float]] = {}

    for path in paths:
        for line in path.read_text().splitlines():
            record = json.loads(line)

            if not record["implementation"].startswith("lab-gqa-split-tc-"):
                continue

            if record["env"]["gpu"] != gpu:
                raise SystemExit(f"{path} was measured on {record['env']['gpu']}, not {gpu}")

            key = (
                record["batch_size"],
                record["context_len"],
                record["num_q_heads"],
                record["num_kv_heads"],
            )
            sweep.setdefault(key, {})[record["num_splits"]] = record["median_us"]

    return sweep


def regrets(model: SplitModel, sweep: dict, sm_count: int, capacity: dict[int, int]) -> dict:
    result = {}

    for (batch, context, q_heads, kv_heads), times in sweep.items():
        group = q_heads // kv_heads
        cost = functools.partial(
            modeled_split_cost,
            num_seqs=batch,
            num_kv_heads=kv_heads,
            kv_blocks=-(-context // BLOCK_SIZE),
            block_size=BLOCK_SIZE,
            queries_per_cta=group,
            capacity=capacity[group],
            sm_count=sm_count,
            tensor_cores=True,
            model=model,
        )
        chosen = min(times, key=cost)
        best = min(times, key=times.get)
        result[(batch, context, q_heads, kv_heads)] = {
            "chosen": chosen,
            "best": best,
            "regret": times[chosen] / times[best] - 1,
        }

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("sweeps", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, help="write the fit report here (markdown)")
    args = parser.parse_args()

    device_index = torch.cuda.current_device()
    sweep = load_sweep(args.sweeps, torch.cuda.get_device_name(device_index))
    sm_count = _multiprocessor_count(device_index)
    groups = {q // kv for _, _, q, kv in sweep}
    capacity = {g: _resident_blocks(device_index, g, True) for g in groups}

    def score(model: SplitModel) -> tuple[float, float]:
        values = [r["regret"] for r in regrets(model, sweep, sm_count, capacity).values()]
        return statistics.mean(values), max(values)

    candidates = [SplitModel(*values) for values in itertools.product(*GRID.values())]
    fitted = min(candidates, key=score)
    current = split_model(device_index)

    lines = [
        f"# Split-model fit on {torch.cuda.get_device_name(device_index)}",
        "",
        (
            f"{len(sweep)} (batch, context) points from {', '.join(p.name for p in args.sweeps)}; "
            f"{sm_count} SMs; resident tensor-core blocks {capacity}; "
            f"{len(candidates)} candidates."
        ),
        "",
        "| model | constants | mean regret | max regret |",
        "|---|---|---:|---:|",
    ]

    for label, model in (("current", current), ("fitted", fitted)):
        mean, worst = score(model)
        lines.append(f"| {label} | `{model}` | {mean:.2%} | {worst:.2%} |")

    lines += [
        "",
        "| batch | context | heads | best splits | current picks | regret | fitted picks | regret |",
        "|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    before = regrets(current, sweep, sm_count, capacity)
    after = regrets(fitted, sweep, sm_count, capacity)

    for key in sorted(sweep):
        batch, context, q_heads, kv_heads = key
        lines.append(
            f"| {batch} | {context} | {q_heads}/{kv_heads} | {before[key]['best']} | "
            f"{before[key]['chosen']} | {before[key]['regret']:.1%} | "
            f"{after[key]['chosen']} | {after[key]['regret']:.1%} |"
        )

    report = "\n".join(lines) + "\n"
    print(report)

    if args.output:
        args.output.write_text(report)


if __name__ == "__main__":
    sys.exit(main())
