"""Run one prefill-attention configuration a few times, for Nsight Compute or Nsight Systems.

    ncu --set full -k regex:"flash_prefill_mma|flash_fwd_kernel" -c 1 -o <out> \\
        python benchmarks/profile_flash_prefill_attention.py --batch-size 1 --seq-len 8192 \\
        --implementation lab-mma_swizzle_cp_async
"""

from __future__ import annotations

import argparse

import torch
from benchmark_flash_prefill_attention import (
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
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--implementation", choices=IMPLEMENTATIONS, required=True)
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()

    num_q_heads, num_kv_heads = MODELS[args.model]

    with torch.inference_mode():
        problem = make_problem(
            args.batch_size, args.seq_len, num_q_heads, num_kv_heads, rotations=1
        )
        out = torch.empty_like(problem[0][0])
        run = build_implementation(
            args.implementation, problem, args.batch_size, args.seq_len, 128**-0.5, out
        )

        for _ in range(args.iterations):
            run(0)

        torch.cuda.synchronize()

    print(f"{args.implementation}: batch {args.batch_size}, seq {args.seq_len}")


if __name__ == "__main__":
    main()
