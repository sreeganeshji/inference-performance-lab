"""Tabulate the key metrics from `ncu --import <report> --page details` text, one row per kernel.

    python benchmarks/summarize_ncu_details.py results/data/attention/ncu/*-details.txt

Each file holds one or more kernels (a decode call can be a split kernel plus a combine kernel).
Durations are under ncu's locked clocks, so compare them with each other, not with benchmark
timings.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

KERNEL_LINE = re.compile(
    r"^  (?P<name>\S.*?) \((?P<grid>[\d, ]+)\)x\((?P<block>[\d, ]+)\), Context"
)
METRIC_LINE = re.compile(r"^    (?P<name>\S.*?\S)\s{2,}(?P<unit>\S+)?\s+(?P<value>[-\d.,]+)$")

COLUMNS = {
    "Duration": "duration",
    "DRAM Throughput": "DRAM %",
    "Compute (SM) Throughput": "SM %",
    "Issue Slots Busy": "issue %",
    "Registers Per Thread": "regs",
    "Theoretical Occupancy": "occ. theory %",
    "Achieved Occupancy": "occ. achieved %",
    "Waves Per SM": "waves",
    "L2 Hit Rate": "L2 hit %",
    "Warp Cycles Per Issued Instruction": "cycles/issue",
}


def short_name(name: str) -> str:
    name = re.sub(r"^void ", "", name).replace("<unnamed>::", "")
    return re.sub(r"<.*", "", name.split("(")[0])


def parse(path: Path) -> list[dict]:
    kernels: list[dict] = []

    for line in path.read_text(errors="replace").splitlines():
        if match := KERNEL_LINE.match(line):
            kernels.append(
                {
                    "file": path.name,
                    "kernel": short_name(match["name"]),
                    "grid": match["grid"].replace(" ", ""),
                }
            )
        elif kernels and (match := METRIC_LINE.match(line)) and match["name"] in COLUMNS:
            value = match["value"].replace(",", "")
            unit = match["unit"] or ""
            # ncu picks the unit per kernel; report every duration in µs.
            scale = {"ms": 1000.0, "us": 1.0, "ns": 0.001}.get(unit, 1.0)
            kernels[-1].setdefault(
                COLUMNS[match["name"]],
                f"{float(value) * scale:.1f}" if match["name"] == "Duration" else value,
            )

    return kernels


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("details", type=Path, nargs="+")
    args = parser.parse_args()

    headers = ["report", "kernel", "grid", *COLUMNS.values()]
    print("| " + " | ".join(headers) + " |")
    print("|" + "---|" * len(headers))

    for path in args.details:
        label = path.name.removesuffix("-details.txt")

        for kernel in parse(path):
            cells = [label, f"`{kernel['kernel']}`", kernel["grid"]]
            cells += [kernel.get(column, "") for column in COLUMNS.values()]
            print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    sys.exit(main())
