"""Compare every pair of greedy outputs that vllm_attention_check.py saved, one per backend.

vllm_attention_check.py writes each backend's tokens and logprobs to check-<BACKEND>.json. Comparing
all pairs, not only each candidate against FLASH_ATTN, shows how far correct BF16 backends drift
apart from one another, which is the scale to read the lab backend's divergence against.

    python benchmarks/compare_check_outputs.py ~/.cache/ipl-models/check-*.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path


def compare(reference: list[dict], candidate: list[dict]) -> tuple[int, int, list[int], float]:
    matching = total = 0
    first_divergence = []
    max_logprob_delta = 0.0

    for ref, other in zip(reference, candidate):
        prefix = 0

        while (
            prefix < min(len(ref["tokens"]), len(other["tokens"]))
            and ref["tokens"][prefix] == other["tokens"][prefix]
        ):
            prefix += 1

        first_divergence.append(prefix)
        matching += prefix
        total += len(ref["tokens"])

        for a, b in zip(ref["logprobs"][:prefix], other["logprobs"][:prefix]):
            if math.isfinite(a) and math.isfinite(b):
                max_logprob_delta = max(max_logprob_delta, abs(a - b))

    return matching, total, first_divergence, max_logprob_delta


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("outputs", type=Path, nargs="+", help="check-<BACKEND>.json files")
    args = parser.parse_args()

    runs = {path.stem.removeprefix("check-"): json.loads(path.read_text()) for path in args.outputs}
    print("| pair | identical greedy prefix | first divergence per prompt | max abs logprob diff |")
    print("|---|---:|---|---:|")

    for a, b in itertools.combinations(sorted(runs), 2):
        matching, total, first_divergence, delta = compare(runs[a], runs[b])
        print(f"| {a} vs {b} | {matching}/{total} | {first_divergence} | {delta:.3g} |")


if __name__ == "__main__":
    sys.exit(main())
