#!/usr/bin/env python3

import argparse
import html
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL_RESULTS = ROOT / "results" / "kernels"
SERVING_RESULTS = ROOT / "results" / "data" / "prefix-cache-off"
ASSET_DIR = ROOT / "docs" / "assets"
SUMMARY_PATH = ROOT / "results" / "generated-summary.md"

TOKEN_COUNTS = [1, 8, 128, 2048]
CONCURRENCIES = [1, 2, 4, 8, 16, 32]
WORKLOADS = [(256, 128), (2048, 32)]

BLUE = "#2563eb"
ORANGE = "#ea580c"
INK = "#172033"
MUTED = "#64748b"
GRID = "#dbe3ef"
BACKGROUND = "#ffffff"


def read_speedups(pattern: str, speedup_column: int) -> dict[int, list[float]]:
    values: dict[int, list[float]] = defaultdict(list)
    paths = sorted(KERNEL_RESULTS.glob(pattern))

    if len(paths) != 5:
        raise SystemExit(f"Expected five kernel runs for {pattern!r}, found {len(paths)}")

    for path in paths:
        seen = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.startswith("|") or line.startswith("|---"):
                continue

            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if not cells[0].isdigit():
                continue

            token_count = int(cells[0])
            if token_count in seen:
                raise SystemExit(f"Duplicate token count {token_count} in {path}")
            seen.add(token_count)
            speedup = float(cells[speedup_column].removesuffix("x"))
            if not math.isfinite(speedup) or speedup <= 0:
                raise SystemExit(f"Invalid speedup in {path}: {speedup}")
            values[token_count].append(speedup)

        if seen != set(TOKEN_COUNTS):
            raise SystemExit(f"Incomplete or unexpected token counts in {path}: {sorted(seen)}")

    if sorted(values) != TOKEN_COUNTS:
        raise SystemExit(f"Unexpected token counts for {pattern!r}: {sorted(values)}")

    return dict(values)


def read_serving_results() -> dict[tuple[int, int, int], dict[str, float]]:
    records: dict[tuple[int, int, int], list[dict[str, float]]] = defaultdict(list)

    for path in sorted(SERVING_RESULTS.glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        if (
            result.get("failed", 0) != 0
            or result.get("num_prompts") != 32
            or result.get("completed") != 32
        ):
            raise SystemExit(f"Expected 32 successful baseline requests in {path}")

        completed = int(result["completed"])
        input_len = int(result.get("random_input_len", result["total_input_tokens"] // completed))
        output_len = int(
            result.get("random_output_len", result["total_output_tokens"] // completed)
        )
        key = (input_len, output_len, int(result["max_concurrency"]))
        for metric in ["output_throughput", "mean_ttft_ms"]:
            value = float(result[metric])
            if not math.isfinite(value) or value <= 0:
                raise SystemExit(f"Invalid {metric} in {path}: {value}")
        records[key].append(result)

    expected = {
        (input_len, output_len, concurrency)
        for input_len, output_len in WORKLOADS
        for concurrency in CONCURRENCIES
    }
    if set(records) != expected:
        missing = sorted(expected - set(records))
        unexpected = sorted(set(records) - expected)
        raise SystemExit(
            f"Unexpected serving result set; missing={missing}, unexpected={unexpected}"
        )

    for key, runs in records.items():
        if len(runs) != 3:
            raise SystemExit(f"Expected three serving runs for {key}, found {len(runs)}")

    return {
        key: {
            "samples": float(len(runs)),
            "output_throughput": statistics.mean(float(run["output_throughput"]) for run in runs),
            "mean_ttft_ms": statistics.mean(float(run["mean_ttft_ms"]) for run in runs),
        }
        for key, runs in records.items()
    }


def text_element(
    x: float,
    y: float,
    value: str,
    *,
    size: int = 14,
    weight: int = 400,
    fill: str = INK,
    anchor: str = "start",
    transform: str | None = None,
) -> str:
    transform_attribute = f' transform="{transform}"' if transform else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" '
        f'fill="{fill}" text-anchor="{anchor}"{transform_attribute}>{html.escape(value)}</text>'
    )


def svg_document(width: int, height: int, elements: list[str]) -> str:
    body = "\n  ".join(elements)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">\n'
        "  <style>text { font-family: Inter, Segoe UI, Arial, sans-serif; }</style>\n"
        f"  {body}\n"
        "</svg>\n"
    )


def render_kernel_chart(series: list[tuple[str, str, dict[int, list[float]]]]) -> str:
    width, height = 1120, 600
    left, right, top, bottom = 85, 35, 115, 80
    plot_width = width - left - right
    plot_height = height - top - bottom
    y_min, y_max = 0.8, 1.9
    y_ticks = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8]

    def x_position(index: int) -> float:
        horizontal_padding = 28
        return (
            left
            + horizontal_padding
            + index * (plot_width - 2 * horizontal_padding) / (len(TOKEN_COUNTS) - 1)
        )

    def y_position(value: float) -> float:
        return top + (y_max - value) * plot_height / (y_max - y_min)

    elements = [
        f'<rect width="{width}" height="{height}" fill="{BACKGROUND}" rx="12"/>',
        text_element(left, 42, "Standalone CUDA kernel speedup vs vLLM", size=25, weight=700),
        text_element(
            left,
            70,
            "Median across five A100 runs; min–max bars retain every measurement, including outliers",
            size=14,
            fill=MUTED,
        ),
    ]

    for tick in y_ticks:
        y = y_position(tick)
        elements.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="{GRID}"/>'
        )
        elements.append(text_element(left - 12, y + 5, f"{tick:.1f}×", fill=MUTED, anchor="end"))

    parity_y = y_position(1.0)
    elements.append(
        f'<line x1="{left}" y1="{parity_y:.1f}" x2="{width - right}" y2="{parity_y:.1f}" '
        f'stroke="{MUTED}" stroke-width="1.5" stroke-dasharray="7 6"/>'
    )

    for index, token_count in enumerate(TOKEN_COUNTS):
        x = x_position(index)
        elements.append(text_element(x, height - 42, str(token_count), fill=MUTED, anchor="middle"))

    elements.append(text_element(width / 2, height - 12, "Token rows", fill=MUTED, anchor="middle"))
    elements.append(
        text_element(
            23,
            top + plot_height / 2,
            "Speedup (higher is better)",
            fill=MUTED,
            anchor="middle",
            transform=f"rotate(-90 23 {top + plot_height / 2:.1f})",
        )
    )

    offsets = [-9, 9]
    for series_index, (name, color, values) in enumerate(series):
        points: list[str] = []
        for index, token_count in enumerate(TOKEN_COUNTS):
            samples = values[token_count]
            median = statistics.median(samples)
            low, high = min(samples), max(samples)
            x = x_position(index) + offsets[series_index]
            y = y_position(median)
            low_y, high_y = y_position(low), y_position(high)
            label_y = y - 13
            if series_index > 0:
                previous_median = statistics.median(series[series_index - 1][2][token_count])
                if abs(median - previous_median) < 0.08:
                    label_y = y + 24
            points.append(f"{x:.1f},{y:.1f}")
            elements.extend(
                [
                    (
                        f'<line x1="{x:.1f}" y1="{high_y:.1f}" x2="{x:.1f}" y2="{low_y:.1f}" '
                        f'stroke="{color}" stroke-width="2"/>'
                    ),
                    (
                        f'<line x1="{x - 5:.1f}" y1="{high_y:.1f}" x2="{x + 5:.1f}" y2="{high_y:.1f}" '
                        f'stroke="{color}" stroke-width="2"/>'
                    ),
                    (
                        f'<line x1="{x - 5:.1f}" y1="{low_y:.1f}" x2="{x + 5:.1f}" y2="{low_y:.1f}" '
                        f'stroke="{color}" stroke-width="2"/>'
                    ),
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}" stroke="white" stroke-width="2"/>',
                    text_element(
                        x,
                        label_y,
                        f"{median:.2f}×",
                        size=12,
                        weight=600,
                        fill=color,
                        anchor="middle",
                    ),
                ]
            )
        elements.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="3"/>'
        )

    legend_x = width - 400
    for index, (name, color, _) in enumerate(series):
        x = legend_x + index * 205
        elements.append(
            f'<line x1="{x}" y1="43" x2="{x + 26}" y2="43" stroke="{color}" stroke-width="4"/>'
        )
        elements.append(text_element(x + 34, 48, name, size=13, weight=600))

    return svg_document(width, height, elements)


def render_serving_chart(serving: dict[tuple[int, int, int], dict[str, float]]) -> str:
    width, height = 1200, 630
    top, bottom = 120, 85
    panel_width, gap = 500, 90
    first_left = 80
    plot_height = height - top - bottom
    panels = [
        ("Output throughput", "output_throughput", "tokens/s", 2000.0, 500.0),
        ("Mean time to first token", "mean_ttft_ms", "ms", 2500.0, 500.0),
    ]
    workload_series = [
        ((256, 128), BLUE, "256 input / 128 output"),
        ((2048, 32), ORANGE, "2048 input / 32 output"),
    ]

    elements = [
        f'<rect width="{width}" height="{height}" fill="{BACKGROUND}" rx="12"/>',
        text_element(
            first_left, 42, "A100 serving scales throughput at a latency cost", size=25, weight=700
        ),
        text_element(
            first_left,
            70,
            "Qwen2.5-7B, BF16, prefix cache disabled; points are means across three runs",
            size=14,
            fill=MUTED,
        ),
    ]

    for panel_index, (title, metric, unit, y_max, tick_step) in enumerate(panels):
        left = first_left + panel_index * (panel_width + gap)

        def x_position(index: int, panel_left: float = left) -> float:
            return panel_left + index * panel_width / (len(CONCURRENCIES) - 1)

        def y_position(value: float, panel_y_max: float = y_max) -> float:
            return top + (panel_y_max - value) * plot_height / panel_y_max

        elements.append(text_element(left, top - 25, title, size=18, weight=700))
        for tick in range(math.floor(y_max / tick_step) + 1):
            value = tick * tick_step
            y = y_position(value)
            elements.append(
                f'<line x1="{left}" y1="{y:.1f}" x2="{left + panel_width}" y2="{y:.1f}" stroke="{GRID}"/>'
            )
            elements.append(
                text_element(left - 12, y + 5, f"{value:.0f}", fill=MUTED, anchor="end")
            )

        for index, concurrency in enumerate(CONCURRENCIES):
            x = x_position(index)
            elements.append(
                text_element(x, height - 47, str(concurrency), fill=MUTED, anchor="middle")
            )

        elements.append(
            text_element(
                left + panel_width / 2, height - 17, "Concurrency", fill=MUTED, anchor="middle"
            )
        )
        elements.append(
            text_element(
                left - 55,
                top + plot_height / 2,
                unit,
                fill=MUTED,
                anchor="middle",
                transform=f"rotate(-90 {left - 55} {top + plot_height / 2:.1f})",
            )
        )

        for workload, color, _ in workload_series:
            points = []
            for index, concurrency in enumerate(CONCURRENCIES):
                value = serving[(*workload, concurrency)][metric]
                x, y = x_position(index), y_position(value)
                points.append(f"{x:.1f},{y:.1f}")
                elements.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}" stroke="white" stroke-width="2"/>'
                )
            elements.append(
                f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="3"/>'
            )

    legend_x = width - 485
    for index, (_, color, name) in enumerate(workload_series):
        x = legend_x + index * 235
        elements.append(
            f'<line x1="{x}" y1="69" x2="{x + 26}" y2="69" stroke="{color}" stroke-width="4"/>'
        )
        elements.append(text_element(x + 34, 74, name, size=13, weight=600))

    return svg_document(width, height, elements)


def render_summary(
    rmsnorm: dict[int, list[float]],
    silu: dict[int, list[float]],
    serving: dict[tuple[int, int, int], dict[str, float]],
) -> str:
    lines = [
        "# Generated benchmark summary",
        "",
        "This file is generated by `scripts/generate_readme_assets.py` from the committed raw result files.",
        "Do not edit it manually.",
        "",
        "## Standalone CUDA kernels",
        "",
        "Speedups are relative to vLLM on an NVIDIA A100-SXM4-40GB. The median, minimum, and maximum use five",
        "repeated benchmark runs.",
        "",
        "| Kernel | Tokens | Runs | Median speedup | Minimum | Maximum |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for name, values in [("Fused RMSNorm", rmsnorm), ("Packed SiLU-and-multiply", silu)]:
        for token_count in TOKEN_COUNTS:
            samples = values[token_count]
            lines.append(
                f"| {name} | {token_count} | {len(samples)} | {statistics.median(samples):.3f}x "
                f"| {min(samples):.3f}x | {max(samples):.3f}x |"
            )

    lines.extend(
        [
            "",
            "## Baseline serving sweep",
            "",
            "Qwen2.5-7B-Instruct, BF16, prefix caching disabled, 32 requests per run, three runs per point.",
            "",
            "| Input/output | Concurrency | Samples | Output throughput | Mean TTFT |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for workload in WORKLOADS:
        for concurrency in CONCURRENCIES:
            result = serving[(*workload, concurrency)]
            lines.append(
                f"| {workload[0]}/{workload[1]} | {concurrency} | {int(result['samples'])} "
                f"| {result['output_throughput']:.3f} tok/s | {result['mean_ttft_ms']:.3f} ms |"
            )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate portfolio charts from historical A100 results."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail on missing or stale artifacts without writing files.",
    )
    args = parser.parse_args()

    rmsnorm = read_speedups("fused-add-rms-norm-packed-cache*.txt", 3)
    silu = read_speedups("silu-and-mul-packed*.txt", 5)
    serving = read_serving_results()

    artifacts = {
        ASSET_DIR / "kernel-speedups.svg": render_kernel_chart(
            [("Fused RMSNorm", BLUE, rmsnorm), ("Packed SiLU", ORANGE, silu)]
        ),
        ASSET_DIR / "serving-scaling.svg": render_serving_chart(serving),
        SUMMARY_PATH: render_summary(rmsnorm, silu, serving),
    }
    if args.check:
        stale = [
            str(path.relative_to(ROOT))
            for path, content in artifacts.items()
            if not path.exists() or path.read_bytes() != content.encode("utf-8")
        ]
        if stale:
            raise SystemExit("Missing or stale artifacts: " + ", ".join(stale))
        print("All generated artifacts match the recorded results.")
        return

    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    for path, content in artifacts.items():
        path.write_text(content, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
