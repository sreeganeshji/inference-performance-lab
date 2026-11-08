"""Replay every trace through vLLM and the C++ scheduler and report matching steps."""

import datetime
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import vllm
from make_traces import FAMILIES, write_all_traces
from replay_vllm import replay

REPO = Path(__file__).resolve().parents[2]
REPLAY_BINARY = REPO / "build/scheduler-release/replay"
RESULTS = REPO / "results/scheduler/comparison.json"
FIELDS = (
    "new",
    "hits",
    "running",
    "scheduled",
    "preempted",
    "finished",
    "free_after_schedule",
    "free_after_update",
)


def replay_cpp(path):
    result = subprocess.run(
        [str(REPLAY_BINARY), str(path)], capture_output=True, text=True, check=True
    )
    return [json.loads(line) for line in result.stdout.splitlines()]


def mismatched_fields(vllm_step, cpp_step):
    return [field for field in FIELDS if vllm_step[field] != cpp_step[field]]


def compare_trace(family, seed, path):
    vllm_steps = replay(path)
    cpp_steps = replay_cpp(path)
    common = min(len(vllm_steps), len(cpp_steps))
    matched = 0
    first_mismatch = None

    for index in range(common):
        fields = mismatched_fields(vllm_steps[index], cpp_steps[index])
        if not fields:
            matched += 1
        elif first_mismatch is None:
            first_mismatch = {
                "step": index,
                "fields": fields,
                "vllm": vllm_steps[index],
                "cpp": cpp_steps[index],
            }

    if len(vllm_steps) != len(cpp_steps) and first_mismatch is None:
        first_mismatch = {"step": common, "fields": ["length"]}

    return {
        "family": family,
        "seed": seed,
        "matched": matched,
        "total": max(len(vllm_steps), len(cpp_steps)),
        "vllm_steps": len(vllm_steps),
        "cpp_steps": len(cpp_steps),
        "first_mismatch": first_mismatch,
    }


def git(*args):
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, cwd=REPO, check=True
    ).stdout.strip()


def machine_info():
    cpu = ""
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            cpu = line.split(":", 1)[1].strip()
            break
    return {
        "cpu": cpu,
        "os": platform.platform(),
        "python": platform.python_version(),
        "compiler": subprocess.run(
            ["g++", "--version"], capture_output=True, text=True, check=True
        ).stdout.splitlines()[0],
    }


def summarize(traces):
    families = {}
    for family in FAMILIES:
        members = [t for t in traces if t["family"] == family]
        families[family] = {
            "matched": sum(t["matched"] for t in members),
            "total": sum(t["total"] for t in members),
            "traces": len(members),
        }
    return families


def main():
    with tempfile.TemporaryDirectory() as directory:
        traces = [compare_trace(*trace) for trace in write_all_traces(directory)]

    matched = sum(trace["matched"] for trace in traces)
    total = sum(trace["total"] for trace in traces)
    families = summarize(traces)
    mismatches = [trace for trace in traces if trace["first_mismatch"]]

    report = {
        "matched": matched,
        "total": total,
        "num_traces": len(traces),
        "families": families,
        "vllm_version": vllm.__version__,
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--", "scheduler")),
        "date": datetime.datetime.now(datetime.UTC).date().isoformat(),
        "machine": machine_info(),
        "seeds": {family: list(seeds) for family, (_, seeds) in FAMILIES.items()},
        "traces": traces,
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(report, indent=1) + "\n")

    for family, counts in families.items():
        print(f"{family:8} {counts['matched']:6} of {counts['total']:6} steps")
    for trace in mismatches:
        fields = ", ".join(trace["first_mismatch"]["fields"])
        print(f"mismatch in {trace['family']}-{trace['seed']}: {fields}", file=sys.stderr)
    print(f"matched {matched} of {total} steps across {len(traces)} traces")


if __name__ == "__main__":
    main()
