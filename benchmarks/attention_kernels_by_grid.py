"""Attention kernel time in an Nsight Systems trace, grouped by kernel and launch grid.

vLLM's FlashAttention backend runs a batch's decode and prefill rows in one varlen launch. Its
grid shows what that launch was: decode-only steps pack each GQA group into one block (gridZ is
the number of KV heads), while a step with a prefill chunk sizes gridX by the chunk's query tiles
and runs one block per query head (gridZ is the number of query heads). Grouping by grid therefore
splits attention time into decode-only and mixed steps, which profile_serving_steps.py's totals
cannot do.

    python benchmarks/attention_kernels_by_grid.py ~/.cache/ipl-traces/*/FLASH_ATTN-*/trace.sqlite
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

QUERY = """
SELECT s.value, k.gridX, MIN(k.gridY), MAX(k.gridY), k.gridZ, COUNT(*), SUM(k.end - k.start)
FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName = s.id
WHERE s.value LIKE 'flash_fwd%' OR s.value LIKE 'paged_decode%' OR s.value LIKE 'combine_splits%'
GROUP BY s.value, k.gridX, k.gridZ
ORDER BY 7 DESC
LIMIT ?
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("traces", type=Path, nargs="+", help="trace.sqlite files (nsys export)")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    for trace in args.traces:
        rows = sqlite3.connect(trace).execute(QUERY, (args.top,)).fetchall()
        print(f"## {trace.parent.name}\n")
        print("| kernel | gridX | gridY (sequences) | gridZ | calls | total s | mean µs |")
        print("|---|---:|---|---:|---:|---:|---:|")

        for name, grid_x, y_min, y_max, grid_z, calls, total in rows:
            print(
                f"| `{name[:40]}` | {grid_x} | {y_min}–{y_max} | {grid_z} | {calls} | "
                f"{total / 1e9:.2f} | {total / calls / 1e3:.1f} |"
            )

        print()


if __name__ == "__main__":
    sys.exit(main())
