"""Serving A/B: vLLM's FLASH_ATTN backend against this lab's CUSTOM decode-attention backend.

For each round, start `vllm serve` with one backend, run every workload cell with
`vllm bench serve`, stop the server, then do the same with the other backend. Rounds alternate
the starting backend (A B, B A, A B, ...) so slow drift in clocks or temperature does not favor
one side. The summary reports, per cell, the median across rounds and the CUSTOM / FLASH_ATTN
ratio of output throughput and median TPOT.

    python benchmarks/serving_attention_ab.py --rounds 3 \\
        --cells 4096:256:1 4096:256:8 4096:256:32 4096:256:64 16384:256:8

Cells are input_len:output_len:concurrency. Prefix caching is off, prompts are random token
sequences of fixed length, and EOS is ignored, so every request does the same work.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

MODEL = "Qwen/Qwen2.5-7B-Instruct"
REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
SERVED_NAME = "qwen2.5-7b-instruct"
BACKENDS = ("FLASH_ATTN", "CUSTOM")


def start_server(backend: str, args: argparse.Namespace, log_path: Path) -> subprocess.Popen:
    env = os.environ.copy()
    # The baseline runs without this lab's plugin, exactly as stock vLLM would.
    env["VLLM_PLUGINS"] = "inference_performance_lab" if backend == "CUSTOM" else ""
    log = log_path.open("w")
    return subprocess.Popen(
        serve_command(backend, args),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def serve_command(backend: str, args: argparse.Namespace) -> list[str]:
    return [
        "vllm",
        "serve",
        MODEL,
        "--revision",
        REVISION,
        "--served-model-name",
        SERVED_NAME,
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        "0.90",
        "--no-enable-prefix-caching",
        "--generation-config",
        "vllm",
        "--seed",
        "0",
        "--attention-backend",
        backend,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
    ]


def wait_until_ready(server: subprocess.Popen, port: int, timeout_s: float = 1200) -> None:
    deadline = time.monotonic() + timeout_s

    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"vllm serve exited with code {server.returncode}")

        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as response:
                if response.status == 200:
                    return
        except OSError:
            pass

        time.sleep(5)

    raise TimeoutError("vllm serve did not become ready")


def stop_server(server: subprocess.Popen) -> None:
    os.killpg(server.pid, signal.SIGINT)

    try:
        server.wait(timeout=120)
    except subprocess.TimeoutExpired:
        os.killpg(server.pid, signal.SIGKILL)
        server.wait()


def run_cell(
    cell: tuple[int, int, int], args: argparse.Namespace, result_dir: Path, stem: str
) -> dict:
    input_len, output_len, concurrency = cell
    num_prompts = max(args.min_prompts, args.prompts_per_slot * concurrency)
    command = [
        "vllm",
        "bench",
        "serve",
        "--backend",
        "openai",
        "--base-url",
        f"http://127.0.0.1:{args.port}",
        "--endpoint",
        "/v1/completions",
        "--model",
        SERVED_NAME,
        "--tokenizer",
        MODEL,
        "--dataset-name",
        "random",
        "--num-prompts",
        str(num_prompts),
        "--random-input-len",
        str(input_len),
        "--random-output-len",
        str(output_len),
        "--random-range-ratio",
        "0",
        "--ignore-eos",
        "--request-rate",
        "inf",
        "--max-concurrency",
        str(concurrency),
        "--seed",
        "0",
        "--temperature",
        "0",
        "--num-warmups",
        "2",
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--metric-percentiles",
        "50,90,99",
        "--save-result",
        "--result-dir",
        str(result_dir),
        "--result-filename",
        f"{stem}.json",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    (result_dir / f"{stem}.log").write_text(completed.stdout + completed.stderr)

    if completed.returncode != 0:
        raise RuntimeError(f"vllm bench serve failed for {stem}; see {stem}.log")

    return json.loads((result_dir / f"{stem}.json").read_text())


def summarize(records: list[dict], cells: list[tuple[int, int, int]]) -> str:
    lines = [
        (
            "| input | output | concurrency | FLASH_ATTN tok/s | CUSTOM tok/s | ratio | "
            "FLASH_ATTN TPOT p50 ms | CUSTOM TPOT p50 ms | TPOT ratio | rounds |"
        ),
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for cell in cells:
        values = {}

        for backend in BACKENDS:
            rows = [r for r in records if r["backend"] == backend and tuple(r["cell"]) == cell]
            values[backend] = (
                statistics.median(r["output_throughput"] for r in rows),
                statistics.median(r["median_tpot_ms"] for r in rows),
                len(rows),
            )

        (base_tps, base_tpot, rounds), (lab_tps, lab_tpot, _) = (
            values["FLASH_ATTN"],
            values["CUSTOM"],
        )
        lines.append(
            f"| {cell[0]} | {cell[1]} | {cell[2]} | {base_tps:.1f} | {lab_tps:.1f} | "
            f"{lab_tps / base_tps:.3f} | {base_tpot:.2f} | {lab_tpot:.2f} | "
            f"{lab_tpot / base_tpot:.3f} | {rounds} |"
        )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--cells", nargs="+", default=["4096:256:1", "4096:256:8", "4096:256:32", "4096:256:64"]
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-model-len", type=int, default=20480)
    parser.add_argument("--prompts-per-slot", type=int, default=4)
    parser.add_argument("--min-prompts", type=int, default=16)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--result-dir", type=Path)
    args = parser.parse_args()

    cells = [tuple(int(part) for part in cell.split(":")) for cell in args.cells]
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    result_dir = args.result_dir or Path("results/data/attention/serving") / stamp
    result_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    for round_index in range(args.rounds):
        order = BACKENDS if round_index % 2 == 0 else BACKENDS[::-1]

        for backend in order:
            print(f"round {round_index + 1}/{args.rounds}: {backend}", flush=True)
            server = start_server(
                backend, args, result_dir / f"server-{backend}-r{round_index}.log"
            )

            try:
                wait_until_ready(server, args.port)

                for cell in cells:
                    stem = f"{backend}-r{round_index}-i{cell[0]}-o{cell[1]}-c{cell[2]}"
                    result = run_cell(cell, args, result_dir, stem)
                    records.append(
                        {"backend": backend, "round": round_index, "cell": cell, **result}
                    )
                    print(
                        f"  {stem}: {result['output_throughput']:.1f} tok/s, "
                        f"TPOT p50 {result['median_tpot_ms']:.2f} ms",
                        flush=True,
                    )
            finally:
                stop_server(server)

    summary = summarize(records, cells)
    (result_dir / "summary.md").write_text(summary + "\n")
    (result_dir / "records.json").write_text(json.dumps(records, indent=1))
    print(summary)


if __name__ == "__main__":
    sys.exit(main())
