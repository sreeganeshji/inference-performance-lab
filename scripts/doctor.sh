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
    echo "nvcc: unavailable"
fi

uv run --locked --no-sync python - <<'PY'
import sys
import torch
import vllm

print("Python:", sys.version.split()[0])
print("vLLM:", vllm.__version__)
print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
