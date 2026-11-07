"""Causal prefill attention over packed sequences: this lab's kernels against vLLM's FlashAttention-2
(flash_attn_varlen_func) and PyTorch SDPA's flash backend, timed with CUDA-graph replay.

Method (the same as benchmark_paged_decode_attention.py):
- Every batch holds batch_size sequences of seq_len tokens, packed as vLLM's attention forward
  receives a fresh prompt: Q [tokens, 28, 128], K and V [tokens, 4, 128] with int32 cu_seqlens.
  SDPA reads the same memory through [batch, heads, seq, 128] views.
- Calls rotate across independent input sets until a graph replay touches at least three times the
  L2 capacity, so no call reads data an earlier call left in L2.
- Each implementation's calls are captured in one CUDA graph; trials replay the graphs in a
  shuffled order, and the reported time is the median per-call time across trials.
- Outputs are checked against FlashAttention-2 before any timing.
- FLOPs count the causal work exactly: 4 * head_dim * num_q_heads * L(L + 1)/2 per sequence (two
  GEMMs, two FLOPs per multiply-add, and only the unmasked half of the score matrix).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from benchmark_paged_decode_attention import MODELS, environment, time_graph
from measure_bf16_tensor_peak import ClockSampler
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from inference_performance_lab.kernels.extension import PREFILL_VARIANTS, load_extension

HEAD_DIM = 128

IMPLEMENTATIONS = tuple(f"lab-{variant}" for variant in PREFILL_VARIANTS) + (
    "vllm-fa2",
    "sdpa-flash",
)


@dataclass
class Result:
    implementation: str
    batch_size: int
    seq_len: int
    num_q_heads: int
    num_kv_heads: int
    median_us: float
    min_us: float
    max_us: float
    trials: int
    calls_per_trial: int
    rotations: int
    flops: int
    tflops: float
    pct_of_peak: float | None
    sm_clock_mhz: float | None


def causal_flops(batch_size: int, seq_len: int, num_q_heads: int) -> int:
    return 4 * HEAD_DIM * num_q_heads * batch_size * seq_len * (seq_len + 1) // 2


def input_bytes(batch_size: int, seq_len: int, num_q_heads: int, num_kv_heads: int) -> int:
    # Q, K and V read once and the output written once.
    return batch_size * seq_len * HEAD_DIM * 2 * (2 * num_q_heads + 2 * num_kv_heads)


def make_problem(
    batch_size: int, seq_len: int, num_q_heads: int, num_kv_heads: int, rotations: int
):
    torch.manual_seed(0)
    tokens = batch_size * seq_len

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(shape, device="cuda", dtype=torch.bfloat16)

    queries = [randn(tokens, num_q_heads, HEAD_DIM) for _ in range(rotations)]
    keys = [randn(tokens, num_kv_heads, HEAD_DIM) for _ in range(rotations)]
    values = [randn(tokens, num_kv_heads, HEAD_DIM) for _ in range(rotations)]
    cu_seqlens = torch.arange(0, tokens + 1, seq_len, dtype=torch.int32, device="cuda")

    return queries, keys, values, cu_seqlens


def build_implementation(
    name: str, problem, batch_size: int, seq_len: int, scale: float, out: torch.Tensor
) -> Callable[[int], None]:
    """A call that runs `name` on input set `rotation` and leaves its result in `out`."""

    queries, keys, values, cu_seqlens = problem
    num_q_heads = queries[0].shape[1]

    if name.startswith("lab-"):
        load_extension()
        variant = PREFILL_VARIANTS.index(name.removeprefix("lab-"))

        def run(rotation: int) -> None:
            torch.ops.inference_performance_lab.flash_prefill_attention(
                out,
                queries[rotation],
                keys[rotation],
                values[rotation],
                cu_seqlens,
                seq_len,
                scale,
                True,
                variant,
            )

        return run

    if name == "vllm-fa2":
        from vllm.vllm_flash_attn import flash_attn_varlen_func

        def run(rotation: int) -> None:
            # The arguments vLLM 0.33's FlashAttentionImpl would pass for a fresh prompt on Ampere,
            # with K and V taken from forward's key/value tensors rather than the paged cache.
            flash_attn_varlen_func(
                q=queries[rotation],
                k=keys[rotation],
                v=values[rotation],
                out=out,
                cu_seqlens_q=cu_seqlens,
                max_seqlen_q=seq_len,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_k=seq_len,
                softmax_scale=scale,
                causal=True,
                fa_version=2,
            )

        return run

    if name == "sdpa-flash":
        out_view = out.view(batch_size, seq_len, num_q_heads, HEAD_DIM)

        def bhsd(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch_size, seq_len, -1, HEAD_DIM).transpose(1, 2)

        def attention(rotation: int) -> torch.Tensor:
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                return F.scaled_dot_product_attention(
                    bhsd(queries[rotation]),
                    bhsd(keys[rotation]),
                    bhsd(values[rotation]),
                    is_causal=True,
                    scale=scale,
                    enable_gqa=True,
                )

        def run(rotation: int) -> None:
            # SDPA has no out= argument. Its result already sits in [batch, seq, heads, dim] order
            # in memory, so the timed call leaves out the copy into `out` (see main()).
            attention(rotation)

        def run_with_copy(rotation: int) -> None:
            out_view.copy_(attention(rotation).transpose(1, 2))

        run.check = run_with_copy  # type: ignore[attr-defined]
        return run

    raise ValueError(f"unknown implementation {name}")


@torch.inference_mode()
def benchmark_config(
    implementations: list[str],
    batch_size: int,
    seq_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    *,
    trials: int,
    target_ms: float,
    peak_tflops: float | None,
) -> list[Result]:
    l2_bytes = torch.cuda.get_device_properties(0).L2_cache_size
    per_call = input_bytes(batch_size, seq_len, num_q_heads, num_kv_heads)
    rotations = max(1, math.ceil(3 * l2_bytes / per_call))
    problem = make_problem(batch_size, seq_len, num_q_heads, num_kv_heads, rotations)
    scale = HEAD_DIM**-0.5

    # One output buffer for every implementation: at 8 x 16K tokens it is 940 MB.
    out = torch.empty_like(problem[0][0])
    runs = {
        name: build_implementation(name, problem, batch_size, seq_len, scale, out)
        for name in implementations
    }

    # Correctness gate: every implementation must agree with FlashAttention-2 on rotation 0.
    build_implementation("vllm-fa2", problem, batch_size, seq_len, scale, out)(0)
    reference_out = out.clone()

    for name, run in runs.items():
        out.fill_(float("nan"))
        getattr(run, "check", run)(0)
        torch.testing.assert_close(
            out, reference_out, rtol=1.6e-2, atol=1e-2, msg=lambda m, n=name: f"{n}: {m}"
        )

    del reference_out

    graphs: dict[str, tuple[torch.cuda.CUDAGraph, int]] = {}
    stream = torch.cuda.Stream()

    for name, run in runs.items():
        for rotation in range(rotations):
            run(rotation)

        torch.cuda.synchronize()

        # Size the graph from a short eager estimate so each replay lasts about target_ms.
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        for rotation in range(rotations):
            run(rotation)

        end.record()
        end.synchronize()

        estimate_us = start.elapsed_time(end) * 1000.0 / rotations
        repeats = max(
            1, min(4096 // rotations, math.ceil(target_ms * 1000.0 / (estimate_us * rotations)))
        )
        calls = repeats * rotations

        graph = torch.cuda.CUDAGraph()
        stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(stream):
            for rotation in range(rotations):
                run(rotation)

            with torch.cuda.graph(graph, stream=stream):
                for _ in range(repeats):
                    for rotation in range(rotations):
                        run(rotation)

        torch.cuda.current_stream().wait_stream(stream)
        graph.replay()
        torch.cuda.synchronize()
        graphs[name] = (graph, calls)

    samples: dict[str, list[float]] = {name: [] for name in runs}
    order = list(runs)
    rng = random.Random(0)

    # One discarded round lets clocks settle after capture.
    with ClockSampler() as clocks:
        for trial in range(trials + 1):
            rng.shuffle(order)

            for name in order:
                graph, calls = graphs[name]
                elapsed_us = time_graph(graph) / calls

                if trial > 0:
                    samples[name].append(elapsed_us)

    clock = clocks.summary()

    flops = causal_flops(batch_size, seq_len, num_q_heads)
    results = []

    for name in runs:
        median_us = statistics.median(samples[name])
        tflops = flops / (median_us * 1e-6) / 1e12
        results.append(
            Result(
                implementation=name,
                batch_size=batch_size,
                seq_len=seq_len,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                median_us=median_us,
                min_us=min(samples[name]),
                max_us=max(samples[name]),
                trials=trials,
                calls_per_trial=graphs[name][1],
                rotations=rotations,
                flops=flops,
                tflops=tflops,
                pct_of_peak=100.0 * tflops / peak_tflops if peak_tflops else None,
                sm_clock_mhz=clock["sm_clock_mhz_median"] if clock else None,
            )
        )

    del graphs, runs, problem, out
    torch.cuda.empty_cache()

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", choices=MODELS, default="qwen2.5-7b")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument(
        "--seq-lens", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192, 16384]
    )
    parser.add_argument(
        "--implementations", nargs="+", choices=IMPLEMENTATIONS, default=list(IMPLEMENTATIONS)
    )
    parser.add_argument("--trials", type=int, default=9)
    parser.add_argument("--target-ms", type=float, default=50.0)
    parser.add_argument(
        "--peak", type=Path, help="JSON from measure_bf16_tensor_peak.py, for the % of peak column"
    )
    parser.add_argument("--output", type=Path, help="append one JSON line per result")
    args = parser.parse_args()

    num_q_heads, num_kv_heads = MODELS[args.model]
    info = environment()
    peak_tflops = json.loads(args.peak.read_text())["peak_tflops"] if args.peak else None

    print(
        f"# {args.model}: {num_q_heads} query heads, {num_kv_heads} KV heads, head_dim {HEAD_DIM}, "
        f"causal, packed varlen"
    )
    print("# " + ", ".join(f"{key}={value}" for key, value in info.items()))

    if peak_tflops:
        print(f"# measured BF16 tensor peak (cuBLAS, FP32 accumulate): {peak_tflops:.1f} TFLOP/s")

    # SDPA's flash backend returns a fresh [batch, heads, seq, dim] view; its time excludes a
    # copy into a caller's buffer, which favours it slightly.
    print()
    print(
        "| batch | seq | implementation | median µs | min–max µs | TFLOP/s | % of peak | SM MHz |"
    )
    print("|---:|---:|---|---:|---:|---:|---:|---:|")

    for batch_size in args.batch_sizes:
        for seq_len in args.seq_lens:
            results = benchmark_config(
                args.implementations,
                batch_size,
                seq_len,
                num_q_heads,
                num_kv_heads,
                trials=args.trials,
                target_ms=args.target_ms,
                peak_tflops=peak_tflops,
            )

            for result in results:
                pct = "" if result.pct_of_peak is None else f"{result.pct_of_peak:.1f}"
                clock = "" if result.sm_clock_mhz is None else f"{result.sm_clock_mhz:.0f}"
                print(
                    f"| {batch_size} | {seq_len} | {result.implementation} | "
                    f"{result.median_us:.1f} | {result.min_us:.1f}–{result.max_us:.1f} | "
                    f"{result.tflops:.2f} | {pct} | {clock} |",
                    flush=True,
                )

                if args.output:
                    args.output.parent.mkdir(parents=True, exist_ok=True)

                    with args.output.open("a") as file:
                        file.write(
                            json.dumps({"model": args.model, **asdict(result), "env": info}) + "\n"
                        )


if __name__ == "__main__":
    main()
