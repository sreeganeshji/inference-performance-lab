"""Markdown tables for docs/flash-prefill.md from the prefill benchmark's JSONL results.

python benchmarks/summarize_flash_prefill.py steps results/data/attention/prefill/rtx3080ti-steps.jsonl
python benchmarks/summarize_flash_prefill.py grid results/data/attention/prefill/rtx3080ti-qwen2.5-7b-grid.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

LAB = "lab-skip_masked"
BASELINES = ("vllm-fa2", "sdpa-flash")
STEPS = {
    "lab-simt": "0",
    "lab-mma": "1",
    "lab-swizzle": "2",
    "lab-cp_async": "3",
    "lab-fast_exp2": "4a",
    "lab-hoisted": "4b",
    "lab-heavy_first": "5",
    "lab-skip_masked": "6",
}


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def by_shape(rows: list[dict]) -> dict[tuple[int, int], dict[str, dict]]:
    shapes: dict[tuple[int, int], dict[str, dict]] = {}

    for row in rows:
        shapes.setdefault((row["batch_size"], row["seq_len"]), {})[row["implementation"]] = row

    return shapes


def steps_table(rows: list[dict]) -> None:
    shapes = by_shape(rows)
    names = [name for name in shapes[next(iter(shapes))] if name.startswith("lab-")]
    header = " | ".join(f"{b}×{s}" for b, s in shapes)
    print(f"| step | variant | {header} |")
    print("|---|---|" + "---:|" * len(shapes))

    for name in names + list(BASELINES):
        step = STEPS.get(name, "")
        cells = []

        for results in shapes.values():
            row = results[name]
            cells.append(f"{row['median_us']:,.1f} µs ({row['pct_of_peak']:.0f}%)")

        print(f"| {step} | `{name}` | " + " | ".join(cells) + " |")

    print()
    print(
        f"Speedup of `{LAB}` over FA2: "
        + ", ".join(
            f"{b}×{s} {results['vllm-fa2']['median_us'] / results[LAB]['median_us']:.3f}×"
            for (b, s), results in shapes.items()
        )
        + "."
    )


def grid_table(rows: list[dict]) -> None:
    shapes = by_shape(rows)
    print(
        "| batch | seq | lab µs | lab TFLOP/s | lab % of peak | FA2 µs | FA2 TFLOP/s | FA2 % of peak "
        "| SDPA µs | speedup vs FA2 | speedup vs SDPA | SM MHz |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    speedups = []

    for (batch, seq), results in shapes.items():
        lab, fa2, sdpa = results[LAB], results["vllm-fa2"], results["sdpa-flash"]
        speedup = fa2["median_us"] / lab["median_us"]
        speedups.append(speedup)
        print(
            f"| {batch} | {seq} | {lab['median_us']:,.1f} | {lab['tflops']:.1f} | "
            f"{lab['pct_of_peak']:.1f} | {fa2['median_us']:,.1f} | {fa2['tflops']:.1f} | "
            f"{fa2['pct_of_peak']:.1f} | {sdpa['median_us']:,.1f} | **{speedup:.3f}×** | "
            f"{sdpa['median_us'] / lab['median_us']:.3f}× | {lab['sm_clock_mhz']:.0f} |"
        )

    geomean = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
    print()
    print(
        f"`{LAB}` is faster than FA2 in {sum(s > 1 for s in speedups)} of {len(speedups)} shapes: "
        f"{min(speedups):.3f}× to {max(speedups):.3f}×, geometric mean {geomean:.3f}×."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table", choices=("steps", "grid"))
    parser.add_argument("jsonl", type=Path)
    args = parser.parse_args()
    rows = load(args.jsonl)
    steps_table(rows) if args.table == "steps" else grid_table(rows)


if __name__ == "__main__":
    main()
