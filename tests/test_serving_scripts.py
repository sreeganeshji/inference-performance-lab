"""Check benchmark command construction without starting a server or installing vLLM."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="The serving scripts target Linux/WSL"
)


@pytest.fixture
def benchmark_command(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in ["benchmark.sh", "env.sh"]:
        shutil.copyfile(ROOT / "scripts" / name, scripts / name)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv = bin_dir / "uv"
    uv.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['ARGUMENT_CAPTURE']).write_text(json.dumps(sys.argv[1:]))\n"
        "print('benchmark output')\n"
        "sys.exit(int(os.environ.get('BENCHMARK_EXIT_CODE', '0')))\n"
    )
    uv.chmod(0o755)
    env = os.environ.copy()
    env.update(
        PATH=str(bin_dir) + os.pathsep + env["PATH"],
        ARGUMENT_CAPTURE=str(tmp_path / "arguments.json"),
        HF_HUB_CACHE=str(tmp_path / "hf-hub"),
        HF_XET_CACHE=str(tmp_path / "hf-xet"),
        INPUT_LEN="2048",
        OUTPUT_LEN="32",
        NUM_PROMPTS="64",
    )
    for variable in ["RESULT_DIR", "PROTOCOL", "VLLM_PORT", "BENCHMARK_EXIT_CODE"]:
        env.pop(variable, None)
    return scripts / "benchmark.sh", env


@pytest.mark.parametrize(
    ("protocol", "port"),
    [(None, None), ("rmsnorm/vllm-c", "8123"), ("rmsnorm/inference-lab", "9001")],
)
def test_benchmark_handles_protocol_directories_and_server_port(benchmark_command, protocol, port):
    script, env = benchmark_command
    if protocol is not None:
        env["PROTOCOL"] = protocol
    if port is not None:
        env["VLLM_PORT"] = port
    result = subprocess.run(
        ["bash", str(script), "8"], env=env, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    arguments = json.loads(Path(env["ARGUMENT_CAPTURE"]).read_text())

    def argument(flag):
        return arguments[arguments.index(flag) + 1]

    assert argument("--base-url") == f"http://127.0.0.1:{port or '8000'}"
    assert argument("--num-prompts") == "64"
    assert argument("--max-concurrency") == "8"
    result_dir = Path(argument("--result-dir"))
    assert result_dir == Path("artifacts/raw") / (protocol or "prefix-cache-off")
    filename = argument("--result-filename")
    assert Path(filename).name == filename
    assert (protocol or "prefix-cache-off").replace("/", "-") in filename
    log_path = script.parent.parent / result_dir / Path(filename).with_suffix(".txt")
    assert log_path.read_text() == "benchmark output\n"


def test_benchmark_propagates_failure_through_tee(benchmark_command):
    script, env = benchmark_command
    env.update(PROTOCOL="rmsnorm/inference-lab", BENCHMARK_EXIT_CODE="7")
    result = subprocess.run(
        ["bash", str(script), "8"], env=env, capture_output=True, text=True, check=False
    )
    assert result.returncode == 7
