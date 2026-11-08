"""Fused add-RMSNorm and SiLU-and-multiply against vLLM's CUDA ops, timed with CUDA-graph replay.

benchmark_fused_add_rms_norm.py and benchmark_silu_and_mul.py time a Python loop of eager launches
between two CUDA events. At a few rows each kernel runs for a few microseconds, so that loop
measures launch overhead as much as the kernel. Here each implementation's calls are captured in
one CUDA graph and replayed, the way vLLM runs decode steps, so the timing is the GPU's alone.

Each graph cycles through enough input sets that one pass reads at least 3x the L2 cache (capped at
64 sets; small shapes stay L2-resident, as they are in serving). Trials are interleaved in a
shuffled order, the first round is discarded, and the table reports medians.

    python benchmarks/benchmark_elementwise_graphs.py --output results/data/elementwise/<gpu>.jsonl
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import random
import statistics
from collections.abc import Callable
from pathlib import Path

import torch
import vllm
from vllm import _custom_ops as vllm_ops

from inference_performance_lab.kernels.extension import load_extension

HIDDEN_SIZE = 3584
INTERMEDIATE_SIZE = 18944
EPSILON = 1e-6
TOKEN_COUNTS = [1, 8, 128, 2048]
MAX_ROTATIONS = 64

# Minimum traffic per row: RMSNorm reads x, residual and weight and writes x and residual
# (weight counted once per row, as the kernels read it); SiLU reads gate and up, writes one output.
BYTES_PER_ROW = {
    "fused_add_rms_norm": HIDDEN_SIZE * 2 * 5,
    "silu_and_mul": INTERMEDIATE_SIZE * 2 * 3,
}


def rms_norm_problem(rows: int, rotations: int):
    weight = torch.randn(HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16) * 0.1 + 1.0
    sets = [
        (
            torch.randn(rows, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16),
            torch.randn(rows, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16),
        )
        for _ in range(rotations)
    ]
    implementations = {
        "vllm": lambda i: vllm_ops.fused_add_rms_norm(*sets[i], weight, EPSILON),
        "lab": lambda i: torch.ops.inference_performance_lab.fused_add_rms_norm(
            *sets[i], weight, EPSILON
        ),
    }
    return implementations, sets, weight


def silu_problem(rows: int, rotations: int):
    sets = [
        (
            torch.empty(rows, INTERMEDIATE_SIZE, device="cuda", dtype=torch.bfloat16),
            torch.randn(rows, 2 * INTERMEDIATE_SIZE, device="cuda", dtype=torch.bfloat16),
        )
        for _ in range(rotations)
    ]
    implementations = {
        "vllm": lambda i: torch.ops._C.silu_and_mul(*sets[i]),
        "lab": lambda i: torch.ops.inference_performance_lab.silu_and_mul_packed(*sets[i]),
    }
    return implementations, sets, None


def check_rms_norm(rows: int) -> None:
    x = torch.randn(rows, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = torch.randn(HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16) * 0.1 + 1.0
    lab_x, lab_residual = x.clone(), residual.clone()
    vllm_ops.fused_add_rms_norm(x, residual, weight, EPSILON)
    torch.ops.inference_performance_lab.fused_add_rms_norm(lab_x, lab_residual, weight, EPSILON)
    torch.testing.assert_close(lab_residual, residual, rtol=0, atol=0)
    torch.testing.assert_close(lab_x, x, rtol=1.6e-2, atol=1e-2)


def check_silu(rows: int) -> None:
    x = torch.randn(rows, 2 * INTERMEDIATE_SIZE, device="cuda", dtype=torch.bfloat16)
    expected = torch.empty(rows, INTERMEDIATE_SIZE, device="cuda", dtype=torch.bfloat16)
    actual = torch.empty_like(expected)
    torch.ops._C.silu_and_mul(expected, x)
    torch.ops.inference_performance_lab.silu_and_mul_packed(actual, x)
    torch.testing.assert_close(actual, expected, rtol=1.6e-2, atol=1e-2)


def capture(run: Callable[[int], None], rotations: int, target_ms: float) -> tuple:
    for rotation in range(rotations):
        run(rotation)

    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()

    for rotation in range(rotations):
        run(rotation)

    end.record()
    end.synchronize()

    estimate_us = start.elapsed_time(end) * 1000.0 / rotations
    repeats = max(
        1, min(4096 // rotations, math.ceil(target_ms * 1000.0 / (estimate_us * rotations)))
    )
    graph, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(stream):
        run(0)

        with torch.cuda.graph(graph, stream=stream):
            for _ in range(repeats):
                for rotation in range(rotations):
                    run(rotation)

    torch.cuda.current_stream().wait_stream(stream)
    graph.replay()
    torch.cuda.synchronize()
    return graph, repeats * rotations


def time_graph(graph: torch.cuda.CUDAGraph) -> float:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0


@torch.inference_mode()
def benchmark(op: str, rows: int, trials: int, target_ms: float) -> list[dict]:
    l2_bytes = torch.cuda.get_device_properties(0).L2_cache_size
    rotations = min(MAX_ROTATIONS, max(1, math.ceil(3 * l2_bytes / (rows * BYTES_PER_ROW[op]))))
    (check_rms_norm if op == "fused_add_rms_norm" else check_silu)(rows)
    problem = rms_norm_problem if op == "fused_add_rms_norm" else silu_problem
    implementations, _, _ = problem(rows, rotations)
    graphs = {name: capture(run, rotations, target_ms) for name, run in implementations.items()}
    samples: dict[str, list[float]] = {name: [] for name in graphs}
    order = list(graphs)
    rng = random.Random(0)

    for trial in range(trials + 1):
        rng.shuffle(order)

        for name in order:
            graph, calls = graphs[name]
            elapsed_us = time_graph(graph) / calls

            if trial > 0:
                samples[name].append(elapsed_us)

    return [
        {
            "op": op,
            "rows": rows,
            "implementation": name,
            "median_us": statistics.median(values),
            "min_us": min(values),
            "max_us": max(values),
            "trials": trials,
            "rotations": rotations,
            "calls_per_trial": graphs[name][1],
            "gbps": rows * BYTES_PER_ROW[op] / (statistics.median(values) * 1e-6) / 1e9,
        }
        for name, values in samples.items()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rows", type=int, nargs="+", default=TOKEN_COUNTS)
    parser.add_argument("--trials", type=int, default=9)
    parser.add_argument("--target-ms", type=float, default=20.0)
    parser.add_argument("--output", type=Path, help="append one JSON line per result")
    args = parser.parse_args()

    load_extension()
    torch.manual_seed(0)
    props = torch.cuda.get_device_properties(0)
    env = {
        "date": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "gpu": props.name,
        "sm_count": props.multi_processor_count,
        "l2_bytes": props.L2_cache_size,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "vllm": vllm.__version__,
    }
    print(f"# CUDA-graph replay timing; {', '.join(f'{k}={v}' for k, v in env.items())}\n")
    print("| op | rows | vLLM µs | lab µs | vLLM / lab | vLLM GB/s | lab GB/s |")
    print("|---|---:|---:|---:|---:|---:|---:|")

    for op in ("fused_add_rms_norm", "silu_and_mul"):
        for rows in args.rows:
            results = {
                r["implementation"]: r for r in benchmark(op, rows, args.trials, args.target_ms)
            }
            base, lab = results["vllm"], results["lab"]
            print(
                f"| {op} | {rows} | {base['median_us']:.2f} | {lab['median_us']:.2f} | "
                f"{base['median_us'] / lab['median_us']:.3f}x | {base['gbps']:.0f} | "
                f"{lab['gbps']:.0f} |",
                flush=True,
            )

            if args.output:
                with args.output.open("a") as file:
                    for result in results.values():
                        file.write(json.dumps({**result, "env": env}) + "\n")


if __name__ == "__main__":
    main()
