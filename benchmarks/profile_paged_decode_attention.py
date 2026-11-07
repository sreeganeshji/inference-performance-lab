"""Run one decode-attention configuration a few times, for Nsight Compute or Nsight Systems.

    ncu --set full -k regex:paged_decode -c 1 -o <out> \\
        python benchmarks/profile_paged_decode_attention.py --batch-size 4 --context-len 4096
"""

from __future__ import annotations

import argparse

import torch
from benchmark_paged_decode_attention import (
    IMPLEMENTATIONS,
    MODELS,
    build_implementation,
    make_problem,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", choices=MODELS, default="qwen2.5-7b")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-len", type=int, default=4096)
    parser.add_argument(
        "--implementation",
        default="lab-gqa-split",
        help=f"one of {IMPLEMENTATIONS} or lab-gqa-split-N",
    )
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()

    num_q_heads, num_kv_heads = MODELS[args.model]

    with torch.inference_mode():
        problem = make_problem(
            args.batch_size, args.context_len, num_q_heads, num_kv_heads, rotations=1
        )
        run, _, splits = build_implementation(args.implementation, problem, 128**-0.5)

        for _ in range(args.iterations):
            run(0)

        torch.cuda.synchronize()

    print(f"{args.implementation}: splits={splits}")


if __name__ == "__main__":
    main()
