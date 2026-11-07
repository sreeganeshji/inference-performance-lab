"""Decode attention over a paged KV cache: this lab's kernels against vLLM's FlashAttention-2 and
FlashInfer, timed with CUDA-graph replay so host launch overhead stays out of the numbers.

Method:
- Every implementation reads the same scattered paged cache (vLLM's FLASH_ATTN layout).
- Calls rotate across independent KV pools until a graph replay touches at least three times the
  L2 capacity, so no call reads data an earlier call left in L2.
- Each implementation's calls are captured in one CUDA graph; trials replay the graphs in a
  shuffled order, and the reported time is the median per-call time across trials.
- Outputs are checked against this lab's GQA kernel before any timing.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import platform
import random
import statistics
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from inference_performance_lab.kernels.extension import choose_num_splits, load_extension

HEAD_DIM = 128
BLOCK_SIZE = 16

MODELS = {
    "qwen2.5-7b": (28, 4),
    "qwen2.5-1.5b": (12, 2),
    "qwen2.5-3b": (16, 2),
    "qwen3-4b": (32, 8),
}

IMPLEMENTATIONS = (
    "lab-per-query-head",
    "lab-gqa",
    "lab-gqa-split",
    "lab-gqa-split-tc",
    "vllm-fa2",
    "flashinfer",
    "flashinfer-tensor-cores",
)


@dataclass
class Result:
    implementation: str
    batch_size: int
    context_len: int
    num_q_heads: int
    num_kv_heads: int
    median_us: float
    min_us: float
    max_us: float
    trials: int
    calls_per_trial: int
    rotations: int
    num_splits: int | None
    min_bytes: int
    gbps: float


def kv_bytes(batch_size: int, context_len: int, num_kv_heads: int) -> int:
    return batch_size * context_len * num_kv_heads * HEAD_DIM * 2 * 2


def minimum_bytes(batch_size: int, context_len: int, num_q_heads: int, num_kv_heads: int) -> int:
    # K and V once, plus the query read and the output write.
    return (
        kv_bytes(batch_size, context_len, num_kv_heads)
        + 2 * batch_size * num_q_heads * HEAD_DIM * 2
    )


def make_problem(
    batch_size: int, context_len: int, num_q_heads: int, num_kv_heads: int, rotations: int
):
    torch.manual_seed(0)
    blocks_per_seq = math.ceil(context_len / BLOCK_SIZE)
    num_blocks = batch_size * blocks_per_seq

    key_pools = torch.randn(
        (rotations, num_blocks, BLOCK_SIZE, num_kv_heads, HEAD_DIM),
        device="cuda",
        dtype=torch.bfloat16,
    )
    value_pools = torch.randn_like(key_pools)

    # Scatter each sequence's blocks across its pool, as a long-running block allocator would.
    block_table = (
        torch.randperm(num_blocks, device="cuda").to(torch.int32).view(batch_size, blocks_per_seq)
    )
    seq_lens = torch.full((batch_size,), context_len, dtype=torch.int32, device="cuda")
    query = torch.randn((batch_size, num_q_heads, HEAD_DIM), device="cuda", dtype=torch.bfloat16)

    return query, key_pools, value_pools, block_table, seq_lens


def build_implementation(
    name: str, problem, scale: float
) -> tuple[Callable[[int], None], torch.Tensor, int | None]:
    query, key_pools, value_pools, block_table, seq_lens = problem
    batch_size, num_q_heads, _ = query.shape
    num_kv_heads = key_pools.shape[3]
    group_size = num_q_heads // num_kv_heads
    context_len = int(seq_lens.max())
    out = torch.empty_like(query)

    if name.startswith("lab-"):
        load_extension()
        tensor_cores = name.startswith("lab-gqa-split-tc")

        if name == "lab-per-query-head":
            queries_per_cta, splits = 1, 1
        elif name == "lab-gqa":
            queries_per_cta, splits = group_size, 1
        elif name in ("lab-gqa-split", "lab-gqa-split-tc"):
            queries_per_cta = group_size
            splits = choose_num_splits(
                batch_size,
                num_kv_heads,
                device=query.device,
                queries_per_cta=queries_per_cta,
                tensor_cores=tensor_cores,
                max_seq_len=context_len,
                block_size=BLOCK_SIZE,
            )
        else:
            # A fixed split count from --split-sweep, e.g. lab-gqa-split-15 or lab-gqa-split-tc-15.
            queries_per_cta, splits = group_size, int(name.rsplit("-", 1)[1])

        partial_out = torch.empty(
            (batch_size, num_q_heads, max(splits, 1), HEAD_DIM), device="cuda"
        )
        partial_lse = torch.empty((batch_size, num_q_heads, max(splits, 1)), device="cuda")

        def run(rotation: int) -> None:
            torch.ops.inference_performance_lab.paged_decode_attention(
                out,
                partial_out,
                partial_lse,
                query,
                key_pools[rotation],
                value_pools[rotation],
                block_table,
                seq_lens,
                scale,
                queries_per_cta,
                splits,
                tensor_cores,
            )

        return run, out, splits

    if name == "vllm-fa2":
        from vllm.vllm_flash_attn import flash_attn_varlen_func

        cu_seqlens_q = torch.arange(batch_size + 1, dtype=torch.int32, device="cuda")

        def run(rotation: int) -> None:
            # The arguments vLLM 0.33's FlashAttentionImpl passes on Ampere (FA2, splits by heuristic).
            flash_attn_varlen_func(
                q=query,
                k=key_pools[rotation],
                v=value_pools[rotation],
                out=out,
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=1,
                seqused_k=seq_lens,
                max_seqlen_k=context_len,
                softmax_scale=scale,
                causal=True,
                block_table=block_table,
                fa_version=2,
            )

        return run, out, None

    if name.startswith("flashinfer"):
        import flashinfer

        blocks_per_seq = block_table.shape[1]
        indptr = torch.arange(batch_size + 1, dtype=torch.int32, device="cuda") * blocks_per_seq
        indices = block_table.reshape(-1).contiguous()
        last_page_len = torch.full(
            (batch_size,), (context_len - 1) % BLOCK_SIZE + 1, dtype=torch.int32, device="cuda"
        )
        workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace,
            "NHD",
            use_cuda_graph=True,
            use_tensor_cores=name == "flashinfer-tensor-cores",
            paged_kv_indptr_buffer=indptr,
            paged_kv_indices_buffer=indices,
            paged_kv_last_page_len_buffer=last_page_len,
        )
        wrapper.plan(
            indptr,
            indices,
            last_page_len,
            num_q_heads,
            num_kv_heads,
            HEAD_DIM,
            BLOCK_SIZE,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
            sm_scale=scale,
        )

        def run(rotation: int) -> None:
            wrapper.run(query, (key_pools[rotation], value_pools[rotation]), out=out)

        return run, out, None

    raise ValueError(f"unknown implementation {name}")


def time_graph(graph: torch.cuda.CUDAGraph) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0


@torch.inference_mode()
def benchmark_config(
    implementations: list[str],
    batch_size: int,
    context_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    *,
    trials: int,
    target_ms: float,
) -> list[Result]:
    l2_bytes = torch.cuda.get_device_properties(0).L2_cache_size
    per_call = kv_bytes(batch_size, context_len, num_kv_heads)
    rotations = max(1, math.ceil(3 * l2_bytes / per_call))
    problem = make_problem(batch_size, context_len, num_q_heads, num_kv_heads, rotations)
    scale = HEAD_DIM**-0.5

    runs: dict[str, tuple[Callable[[int], None], torch.Tensor, int | None]] = {}

    for name in implementations:
        try:
            runs[name] = build_implementation(name, problem, scale)
        except RuntimeError as error:
            # FlashInfer's CUDA-core decode kernel only instantiates GQA group sizes 1-4 and 8.
            if not name.startswith("flashinfer"):
                raise

            print(f"# skipped {name}: {str(error).splitlines()[-1][-120:]}")

    # Correctness gate: every implementation must agree with the lab's GQA kernel on rotation 0.
    reference_run, reference_out, _ = build_implementation("lab-gqa", problem, scale)
    reference_run(0)

    for name, (run, out, _) in runs.items():
        run(0)
        torch.testing.assert_close(
            out, reference_out, rtol=1.6e-2, atol=1e-2, msg=lambda m, n=name: f"{n}: {m}"
        )

    graphs: dict[str, tuple[torch.cuda.CUDAGraph, int]] = {}
    stream = torch.cuda.Stream()

    for name, (run, _, _) in runs.items():
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
    for trial in range(trials + 1):
        rng.shuffle(order)

        for name in order:
            graph, calls = graphs[name]
            elapsed_us = time_graph(graph) / calls

            if trial > 0:
                samples[name].append(elapsed_us)

    bytes_needed = minimum_bytes(batch_size, context_len, num_q_heads, num_kv_heads)
    results = []

    for name in runs:
        median_us = statistics.median(samples[name])
        results.append(
            Result(
                implementation=name,
                batch_size=batch_size,
                context_len=context_len,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                median_us=median_us,
                min_us=min(samples[name]),
                max_us=max(samples[name]),
                trials=trials,
                calls_per_trial=graphs[name][1],
                rotations=rotations,
                num_splits=runs[name][2],
                min_bytes=bytes_needed,
                gbps=bytes_needed / (median_us * 1e-6) / 1e9,
            )
        )

    del graphs, runs, problem
    torch.cuda.empty_cache()

    return results


def environment() -> dict[str, str]:
    properties = torch.cuda.get_device_properties(0)
    info = {
        "date": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "gpu": properties.name,
        "sm_count": str(properties.multi_processor_count),
        "l2_bytes": str(properties.L2_cache_size),
        "torch": torch.__version__,
        "cuda": str(torch.version.cuda),
        "python": platform.python_version(),
    }

    for package in ("vllm", "flashinfer"):
        try:
            info[package] = __import__(package).__version__
        except ImportError:
            info[package] = "not installed"

    try:
        info["driver"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        info["driver"] = "unknown"

    return info


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", choices=MODELS, default="qwen2.5-7b")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16, 64])
    parser.add_argument("--context-lens", type=int, nargs="+", default=[1024, 4096, 16384])
    parser.add_argument(
        "--implementations", nargs="+", choices=IMPLEMENTATIONS, default=list(IMPLEMENTATIONS)
    )
    parser.add_argument(
        "--split-sweep",
        type=int,
        nargs="*",
        default=[],
        help="also time lab-gqa-split at these split counts",
    )
    parser.add_argument(
        "--split-sweep-tc",
        type=int,
        nargs="*",
        default=[],
        help="also time lab-gqa-split-tc at these split counts",
    )
    parser.add_argument("--trials", type=int, default=9)
    parser.add_argument("--target-ms", type=float, default=20.0)
    parser.add_argument("--output", type=Path, help="append one JSON line per result")
    args = parser.parse_args()

    num_q_heads, num_kv_heads = MODELS[args.model]
    info = environment()

    print(
        f"# {args.model}: {num_q_heads} query heads, {num_kv_heads} KV heads, head_dim {HEAD_DIM}"
    )
    print("# " + ", ".join(f"{key}={value}" for key, value in info.items()))
    print()
    print("| batch | context | implementation | splits | median µs | min–max µs | GB/s |")
    print("|---:|---:|---|---:|---:|---:|---:|")

    for batch_size in args.batch_sizes:
        for context_len in args.context_lens:
            results = benchmark_config(
                args.implementations
                + [f"lab-gqa-split-{splits}" for splits in args.split_sweep]
                + [f"lab-gqa-split-tc-{splits}" for splits in args.split_sweep_tc],
                batch_size,
                context_len,
                num_q_heads,
                num_kv_heads,
                trials=args.trials,
                target_ms=args.target_ms,
            )

            for result in results:
                splits = "" if result.num_splits is None else str(result.num_splits)
                print(
                    f"| {batch_size} | {context_len} | {result.implementation} | {splits} | "
                    f"{result.median_us:.2f} | {result.min_us:.2f}–{result.max_us:.2f} | {result.gbps:.0f} |"
                )

                if args.output:
                    args.output.parent.mkdir(parents=True, exist_ok=True)

                    with args.output.open("a") as file:
                        file.write(
                            json.dumps({"model": args.model, **asdict(result), "env": info}) + "\n"
                        )


if __name__ == "__main__":
    main()
