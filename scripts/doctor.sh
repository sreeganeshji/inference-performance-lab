#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

echo "Repository: $repo_root"
uv --version

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "nvidia-smi: unavailable"
fi

if command -v nvcc >/dev/null 2>&1; then
    nvcc --version | tail -n 1
else
    echo "error: nvcc is unavailable; install the CUDA toolkit and add its bin directory to PATH" >&2
    exit 1
fi

uv run --locked --no-sync python - <<'PY'
import sys
from importlib.util import find_spec

import torch

print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("error: PyTorch cannot access a CUDA device")

device = torch.cuda.current_device()
properties = torch.cuda.get_device_properties(device)
major, minor = torch.cuda.get_device_capability(device)

print("GPU:", properties.name)
print("Compute capability:", f"{major}.{minor}")
print("VRAM:", f"{properties.total_memory / 1024**3:.1f} GiB")

if find_spec("vllm") is None:
    print("vLLM: not installed (kernel-only profile)")
else:
    import vllm

    print("vLLM:", vllm.__version__)
PY
