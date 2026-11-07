"""Nsight Systems trace of vLLM serving, reduced to the numbers that bound end-to-end gains.

Starts `vllm serve` (Qwen2.5-7B-Instruct) under Nsight Systems with collection paused, warms it with
one `vllm bench serve` run of the cell, records a second identical run, and reduces the trace to:

- GPU time by kernel and the share spent in attention, which bounds what any attention speedup
  can buy end to end;
- per-call CPU time of Scheduler.schedule and update_from_output, and of BlockPool's
  get_new_blocks and free_blocks inside them (NVTX ranges that the lab plugin adds when
  IPL_SCHEDULER_NVTX=1);
- GPU idle time over the recorded window, and how much of the scheduler's time fell while the GPU
  was idle, which bounds what a faster scheduler could save.

    python benchmarks/profile_serving_steps.py --backend CUSTOM --cell 4096:256:32

Both backends load the lab plugin here (for the NVTX ranges); it only registers CUSTOM, so a
FLASH_ATTN run still executes stock vLLM kernels. Tracing adds overhead, so read the results as
shares and bounds, not as serving throughput.
"""

from __future__ import annotations

import argparse
import bisect
import datetime as dt
import json
import os
import platform
import sqlite3
import statistics
import subprocess
import sys
from pathlib import Path

from serving_attention_ab import run_cell, serve_command, stop_server, wait_until_ready

# Kernel-name substrings: this lab's decode kernels, then FlashAttention-2's.
ATTENTION_KERNELS = {
    "lab decode": ("paged_decode_attention", "combine_splits"),
    "FlashAttention-2": ("flash_fwd",),
}
SCHEDULER_RANGES = ("scheduler.schedule", "scheduler.update_from_output")
BLOCK_POOL_RANGES = ("block_pool.get_new_blocks", "block_pool.free_blocks")


def record(args: argparse.Namespace, cell: tuple[int, int, int], result_dir: Path) -> Path:
    session = f"ipl-{os.getpid()}"
    report = args.trace_dir / "trace"
    env = os.environ.copy()
    env.update(
        VLLM_PLUGINS="inference_performance_lab",
        IPL_SCHEDULER_NVTX="1",
        # Flushing trace buffers at exit takes longer than vLLM's default 5 s.
        VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS="300",
    )
    command = [
        "nsys",
        "profile",
        "--start-later=true",
        f"--session-new={session}",
        "--trace=cuda,nvtx",
        "--cuda-graph-trace=node",
        "--sample=none",
        "--cpuctxsw=none",
        "--kill=none",
        "--force-overwrite=true",
        "-o",
        str(report),
        *serve_command(args.backend, args),
    ]
    log = (result_dir / "server.log").open("w")
    server = subprocess.Popen(
        command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    )

    try:
        wait_until_ready(server, args.port)
        run_cell(cell, args, result_dir, "warmup")
        subprocess.run(["nsys", "start", f"--session={session}"], check=True)
        run_cell(cell, args, result_dir, "recorded")
        subprocess.run(["nsys", "stop", f"--session={session}"], check=True)
    finally:
        stop_server(server)

    sqlite_path = report.with_suffix(".sqlite")
    subprocess.run(
        [
            "nsys",
            "export",
            "--type",
            "sqlite",
            "--force-overwrite=true",
            "-o",
            str(sqlite_path),
            str(report.with_suffix(".nsys-rep")),
        ],
        check=True,
    )
    return sqlite_path


def merge_intervals(intervals: list[tuple[int, int]], min_gap_ns: int) -> list[list[int]]:
    """Union of [start, end) intervals sorted by start, closing gaps shorter than min_gap_ns."""

    merged: list[list[int]] = []

    for start, end in intervals:
        if merged and start - merged[-1][1] < min_gap_ns:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    return merged


def busy_within(busy: list[list[int]], starts: list[int], start: int, end: int) -> int:
    i = max(bisect.bisect_right(starts, start) - 1, 0)
    total = 0

    while i < len(busy) and busy[i][0] < end:
        total += max(0, min(end, busy[i][1]) - max(start, busy[i][0]))
        i += 1

    return total


def duration_stats(durations_ns: list[int]) -> dict:
    ordered = sorted(durations_ns)

    def percentile(p: float) -> float:
        return ordered[min(len(ordered) - 1, int(p * len(ordered)))] / 1e3

    return {
        "calls": len(ordered),
        "total_ms": sum(ordered) / 1e6,
        "median_us": statistics.median(ordered) / 1e3,
        "p90_us": percentile(0.90),
        "p99_us": percentile(0.99),
        "max_us": ordered[-1] / 1e3,
    }


def analyze(sqlite_path: Path, min_gap_us: float) -> dict:
    db = sqlite3.connect(sqlite_path)
    kernels = db.execute(
        "SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON k.shortName = s.id ORDER BY k.start"
    ).fetchall()
    ranges = db.execute(
        "SELECT n.start, n.end, COALESCE(n.text, s.value) FROM NVTX_EVENTS n "
        "LEFT JOIN StringIds s ON n.textId = s.id WHERE n.end IS NOT NULL"
    ).fetchall()

    window_start, window_end = kernels[0][0], max(end for _, end, _ in kernels)
    window_ns = window_end - window_start
    busy = merge_intervals([(start, end) for start, end, _ in kernels], int(min_gap_us * 1e3))
    starts = [interval[0] for interval in busy]
    busy_ns = sum(end - start for start, end in busy)

    by_kernel: dict[str, list[int]] = {}

    for start, end, name in kernels:
        entry = by_kernel.setdefault(name, [0, 0])
        entry[0] += end - start
        entry[1] += 1

    kernel_ns = sum(total for total, _ in by_kernel.values())
    attention = {
        label: sum(
            total
            for name, (total, _) in by_kernel.items()
            if any(pattern in name for pattern in patterns)
        )
        / kernel_ns
        for label, patterns in ATTENTION_KERNELS.items()
    }
    top_kernels = [
        {"name": name, "share": total / kernel_ns, "total_ms": total / 1e6, "count": count}
        for name, (total, count) in sorted(by_kernel.items(), key=lambda item: -item[1][0])[:12]
    ]

    in_window = [
        (start, end, name)
        for start, end, name in ranges
        if name in SCHEDULER_RANGES + BLOCK_POOL_RANGES and window_start <= start < window_end
    ]
    scheduler = {
        name: duration_stats([end - start for start, end, label in in_window if label == name])
        for name in SCHEDULER_RANGES + BLOCK_POOL_RANGES
        if any(label == name for _, _, label in in_window)
    }
    # The block-pool ranges nest inside the scheduler ranges, so only the latter are summed here.
    scheduler_ns = sum(end - start for start, end, name in in_window if name in SCHEDULER_RANGES)
    exposed_ns = sum(
        (end - start) - busy_within(busy, starts, start, end)
        for start, end, name in in_window
        if name in SCHEDULER_RANGES
    )

    return {
        "window_ms": window_ns / 1e6,
        "gpu_busy_fraction": busy_ns / window_ns,
        "min_gap_us": min_gap_us,
        "kernel_ms": kernel_ns / 1e6,
        "attention_share_of_kernel_time": attention,
        "top_kernels": top_kernels,
        "scheduler_ranges": scheduler,
        "scheduler_cpu_fraction_of_window": scheduler_ns / window_ns,
        "scheduler_ms_while_gpu_idle": exposed_ns / 1e6,
        "scheduler_while_gpu_idle_fraction_of_window": exposed_ns / window_ns,
    }


def markdown(summary: dict) -> str:
    a = summary["analysis"]
    env = summary["environment"]
    lines = [
        f"# {summary['backend']} serving profile, cell {summary['cell']}",
        "",
        f"{env['gpu']}, driver {env['driver']}, vLLM {env['vllm']}, {env['nsys']}",
        "",
        (
            f"- recorded window: {a['window_ms']:.0f} ms; benchmark output throughput under "
            f"tracing: {summary['benchmark']['output_throughput']:.1f} tok/s"
        ),
        (
            f"- GPU busy {a['gpu_busy_fraction']:.1%} of the window (gaps under "
            f"{a['min_gap_us']:g} µs count as busy)"
        ),
    ]

    for label, share in a["attention_share_of_kernel_time"].items():
        lines.append(f"- {label} kernels: {share:.1%} of GPU kernel time")

    lines += [
        (
            f"- scheduler CPU time: {a['scheduler_cpu_fraction_of_window']:.1%} of the window; "
            f"while the GPU was idle: {a['scheduler_ms_while_gpu_idle']:.1f} ms "
            f"({a['scheduler_while_gpu_idle_fraction_of_window']:.2%} of the window)"
        ),
        "",
        "| range | calls | total ms | median µs | p90 µs | p99 µs | max µs |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    lines += [
        f"| {name} | {s['calls']} | {s['total_ms']:.1f} | {s['median_us']:.1f} | "
        f"{s['p90_us']:.1f} | {s['p99_us']:.1f} | {s['max_us']:.1f} |"
        for name, s in a["scheduler_ranges"].items()
    ]
    lines += ["", "| kernel | share | total ms | count |", "|---|---:|---:|---:|"]
    lines += [
        f"| `{k['name'][:90]}` | {k['share']:.1%} | {k['total_ms']:.1f} | {k['count']} |"
        for k in a["top_kernels"]
    ]
    return "\n".join(lines) + "\n"


def environment() -> dict:
    gpu, driver = (
        subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        .stdout.splitlines()[0]
        .split(", ")
    )
    nsys = subprocess.run(["nsys", "--version"], capture_output=True, text=True, check=True)
    import torch
    import vllm

    return {
        "gpu": gpu,
        "driver": driver,
        "torch": torch.__version__,
        "vllm": vllm.__version__,
        "nsys": nsys.stdout.strip(),
        "python": platform.python_version(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--backend", choices=("FLASH_ATTN", "CUSTOM"), default="CUSTOM")
    parser.add_argument("--cell", default="4096:256:32", help="input_len:output_len:concurrency")
    parser.add_argument("--max-model-len", type=int, default=20480)
    parser.add_argument("--prompts-per-slot", type=int, default=4)
    parser.add_argument("--min-prompts", type=int, default=16)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--min-gap-us", type=float, default=10.0)
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument(
        "--trace-dir", type=Path, help="where the .nsys-rep and .sqlite go (large; not results)"
    )
    parser.add_argument("--analyze-only", type=Path, help="reduce an existing .sqlite trace")
    args = parser.parse_args()

    cell = tuple(int(part) for part in args.cell.split(":"))
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    name = f"{args.backend}-i{cell[0]}-o{cell[1]}-c{cell[2]}"
    result_dir = args.result_dir or Path("results/data/attention/serving-profile") / stamp / name
    result_dir.mkdir(parents=True, exist_ok=True)
    args.trace_dir = args.trace_dir or Path.home() / ".cache" / "ipl-traces" / stamp / name
    args.trace_dir.mkdir(parents=True, exist_ok=True)

    sqlite_path = args.analyze_only or record(args, cell, result_dir)
    benchmark = json.loads((result_dir / "recorded.json").read_text())
    summary = {
        "backend": args.backend,
        "cell": args.cell,
        "trace": str(sqlite_path),
        "environment": environment(),
        "benchmark": {
            key: benchmark[key]
            for key in ("completed", "duration", "output_throughput", "median_tpot_ms")
        },
        "analysis": analyze(sqlite_path, args.min_gap_us),
    }
    (result_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    (result_dir / "summary.md").write_text(markdown(summary))
    print(markdown(summary))


if __name__ == "__main__":
    sys.exit(main())
