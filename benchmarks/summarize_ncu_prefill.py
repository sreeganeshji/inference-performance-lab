"""Side-by-side Nsight Compute evidence for prefill-attention kernels (markdown on stdout).

    python benchmarks/summarize_ncu_prefill.py lab=lab.ncu-rep fa2=fa2.ncu-rep

Prints launch and occupancy figures, pipe utilization, shared-memory bank conflicts, memory
traffic and warp-stall reasons from each report's raw page, then the executed-SASS opcode mix from
its source page. Needs `ncu` on PATH.
"""

from __future__ import annotations

import collections
import csv
import io
import subprocess
import sys

METRICS = [
    ("kernel time (µs, ncu-locked clock)", "gpu__time_duration.sum", 1e-3),
    ("SM clock (MHz)", "sm__cycles_elapsed.avg.per_second", 1e-6),
    ("SMs busy (% of elapsed cycles)", ("sm__cycles_active.avg", "sm__cycles_elapsed.avg"), 100),
    ("registers / thread", "launch__registers_per_thread", 1),
    ("shared memory / block (bytes, incl. 1 KB reserved)", "launch__shared_mem_per_block", 1),
    ("block limit: registers", "launch__occupancy_limit_registers", 1),
    ("block limit: shared memory", "launch__occupancy_limit_shared_mem", 1),
    ("theoretical occupancy (%)", "sm__maximum_warps_per_active_cycle_pct", 1),
    ("achieved occupancy (%)", "sm__warps_active.avg.pct_of_peak_sustained_active", 1),
    (
        "tensor pipe (HMMA) active (% of peak, SM-active cycles)",
        "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active",
        1,
    ),
    (
        "tensor pipe (HMMA) active (% of peak, elapsed cycles)",
        "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed",
        1,
    ),
    ("warp instructions executed", "smsp__inst_executed.sum", 1),
    (
        "XU pipe (MUFU.EX2) (% of peak)",
        "sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active",
        1,
    ),
    ("FMA pipe (% of peak)", "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active", 1),
    ("ALU pipe (% of peak)", "sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active", 1),
    ("shared load bank conflicts", "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum", 1),
    ("shared store bank conflicts", "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum", 1),
    ("shared load wavefronts", "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum", 1),
    ("DRAM read (MB)", "dram__bytes_read.sum", 1e-6),
    ("L2 → SM read (MB)", "lts__t_sectors_srcunit_tex_op_read.sum", 32e-6),
    ("L2 hit rate (%)", "lts__t_sector_hit_rate.pct", 1),
]

STALLS = [
    "math_pipe_throttle",
    "wait",
    "barrier",
    "short_scoreboard",
    "long_scoreboard",
    "mio_throttle",
    "not_selected",
    "dispatch_stall",
    "lg_throttle",
    "no_instruction",
    "branch_resolving",
]


def ncu(path: str, *args: str) -> str:
    return subprocess.run(
        ["ncu", "--import", path, *args], capture_output=True, text=True, check=True
    ).stdout


def raw_metrics(path: str) -> tuple[dict[str, str], str]:
    rows = list(
        csv.reader(io.StringIO(ncu(path, "--page", "raw", "--csv", "--print-units", "base")))
    )
    header, values = rows[0], rows[2]
    return dict(zip(header, values, strict=True)), values[header.index("Kernel Name")]


def opcode_mix(path: str) -> collections.Counter[str]:
    lines = ncu(path, "--page", "source", "--csv", "--print-source", "sass").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith('"Address"'))
    counts: collections.Counter[str] = collections.Counter()

    for row in csv.DictReader(io.StringIO("\n".join(lines[start:]))):
        words = row["Source"].split()

        if not words:
            continue

        opcode = words[1] if words[0].startswith("@") else words[0]
        counts[opcode.split(".")[0]] += int(row["Instructions Executed"].replace(",", ""))

    return counts


def number(raw: dict[str, str], key: str | tuple[str, str]) -> float | None:
    if isinstance(key, tuple):  # a ratio of two metrics
        numerator, denominator = (number(raw, part) for part in key)
        return None if numerator is None or not denominator else numerator / denominator

    value = raw.get(key, "")
    return None if value in ("", "n/a") else float(value.replace(",", ""))


def cell(value: float | None, scale: float = 1.0) -> str:
    if value is None:
        return "n/a"

    value *= scale
    return f"{value:,.0f}" if abs(value) >= 100 else f"{value:.2f}"


def main() -> None:
    reports = [argument.split("=", 1) for argument in sys.argv[1:]]
    loaded = [(name, *raw_metrics(path), opcode_mix(path)) for name, path in reports]
    names = [name for name, *_ in loaded]

    print("| metric | " + " | ".join(names) + " |")
    print("|---|" + "---:|" * len(names))

    for label, key, scale in METRICS:
        print(
            f"| {label} | "
            + " | ".join(cell(number(raw, key), scale) for _, raw, _, _ in loaded)
            + " |"
        )

    for stall in STALLS:
        key = f"smsp__average_warps_issue_stalled_{stall}_per_issue_active.ratio"
        print(
            f"| stall: {stall} (warps per issue) | "
            + " | ".join(cell(number(raw, key)) for _, raw, _, _ in loaded)
            + " |"
        )

    print()
    print(
        "Executed SASS by opcode (warp-level instructions, millions; opcodes above 0.5% of the largest total):"
    )
    print()
    print("| opcode | " + " | ".join(names) + " |")
    print("|---|" + "---:|" * len(names))
    mixes = [mix for *_, mix in loaded]
    opcodes = sorted(set().union(*mixes), key=lambda op: -max(mix[op] for mix in mixes))

    threshold = 0.005 * max(sum(mix.values()) for mix in mixes)

    for opcode in opcodes:
        if max(mix[opcode] for mix in mixes) >= threshold:
            print(f"| {opcode} | " + " | ".join(f"{mix[opcode] / 1e6:.1f}" for mix in mixes) + " |")

    print("| total | " + " | ".join(f"{sum(mix.values()) / 1e6:.1f}" for mix in mixes) + " |")
    print()

    for name, _, kernel, _ in loaded:
        print(f"- {name}: `{kernel}`")


if __name__ == "__main__":
    main()
