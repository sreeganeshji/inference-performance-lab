"""Measured BF16 tensor-core throughput of this GPU: cuBLAS GEMMs with FP32 accumulation.

The prefill benchmark reports attention as a percentage of this number rather than of a spec
sheet. The GEMMs use BF16 inputs and outputs with FP32 accumulation (CUBLAS_COMPUTE_32F), the same
precision as the attention kernels' mma.sync.m16n8k16.f32.bf16.bf16.f32. Each size is captured in
one CUDA graph and replayed; the reported figure is the best median across sizes. The SM clock is
sampled through NVML while the GEMMs run, because GeForce boost clocks move with power and
temperature and the peak moves with them.
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
from pathlib import Path
from typing import Self

import torch
from benchmark_paged_decode_attention import environment

SIZES = ((4096, 4096, 4096), (8192, 8192, 8192), (16384, 8192, 8192), (8192, 8192, 16384))


class ClockSampler:
    """Samples the SM clock (MHz) every 50 ms on a background thread, if NVML is available."""

    def __init__(self) -> None:
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
        except Exception:  # noqa: BLE001 - NVML is optional
            self._nvml = None

    def __enter__(self) -> Self:
        if self._nvml is not None:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()

        if self._thread is not None:
            self._thread.join()

    def _run(self) -> None:
        while not self._stop.wait(0.05):
            self.samples.append(
                self._nvml.nvmlDeviceGetClockInfo(self._handle, self._nvml.NVML_CLOCK_SM)
            )

    def summary(self) -> dict[str, float] | None:
        if not self.samples:
            return None

        return {
            "sm_clock_mhz_median": statistics.median(self.samples),
            "sm_clock_mhz_min": min(self.samples),
            "sm_clock_mhz_max": max(self.samples),
        }


def time_gemm(m: int, n: int, k: int, *, trials: int, target_ms: float) -> list[float]:
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((k, n), device="cuda", dtype=torch.bfloat16)
    c = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)

    for _ in range(3):
        torch.matmul(a, b, out=c)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    torch.matmul(a, b, out=c)
    end.record()
    end.synchronize()
    repeats = max(1, round(target_ms / start.elapsed_time(end)))

    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        for _ in range(repeats):
            torch.matmul(a, b, out=c)

    torch.cuda.current_stream().wait_stream(stream)
    graph.replay()
    torch.cuda.synchronize()

    tflops = []

    for _ in range(trials):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        seconds = start.elapsed_time(end) / 1000.0 / repeats
        tflops.append(2.0 * m * n * k / seconds / 1e12)

    return tflops


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--target-ms", type=float, default=200.0)
    parser.add_argument("--output", type=Path, help="write the result as JSON")
    parser.add_argument(
        "--profile-once",
        type=int,
        nargs=3,
        metavar=("M", "N", "K"),
        help="run three untimed GEMMs of this size and exit (for Nsight Compute)",
    )
    args = parser.parse_args()

    # Keep cuBLAS on FP32 accumulation; reduced-precision split-K reductions are a different op.
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    if args.profile_once:
        m, n, k = args.profile_once
        a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
        b = torch.randn((k, n), device="cuda", dtype=torch.bfloat16)

        for _ in range(3):
            torch.matmul(a, b)

        torch.cuda.synchronize()
        return

    info = environment()
    rows = []

    print("# " + ", ".join(f"{key}={value}" for key, value in info.items()))
    print()
    print("| M | N | K | median TFLOP/s | min–max TFLOP/s | SM clock MHz (median, min–max) |")
    print("|---:|---:|---:|---:|---:|---|")

    for m, n, k in SIZES:
        with ClockSampler() as clocks:
            samples = time_gemm(m, n, k, trials=args.trials, target_ms=args.target_ms)

        clock = clocks.summary()
        row = {
            "m": m,
            "n": n,
            "k": k,
            "median_tflops": statistics.median(samples),
            "min_tflops": min(samples),
            "max_tflops": max(samples),
            "clock": clock,
        }
        rows.append(row)
        clock_text = (
            f"{clock['sm_clock_mhz_median']:.0f} ({clock['sm_clock_mhz_min']}–"
            f"{clock['sm_clock_mhz_max']})"
            if clock
            else "n/a"
        )
        print(
            f"| {m} | {n} | {k} | {row['median_tflops']:.1f} | "
            f"{row['min_tflops']:.1f}–{row['max_tflops']:.1f} | {clock_text} |"
        )

    best = max(rows, key=lambda row: row["median_tflops"])
    print()
    print(f"peak_tflops = {best['median_tflops']:.1f} (cuBLAS {best['m']}x{best['n']}x{best['k']})")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"peak_tflops": best["median_tflops"], "gemms": rows, "env": info}, indent=2)
            + "\n"
        )


if __name__ == "__main__":
    main()
