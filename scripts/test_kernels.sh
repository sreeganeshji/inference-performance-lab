#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ -n "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
    echo "CUDA architecture override: $TORCH_CUDA_ARCH_LIST"
else
    uv run --locked --no-sync python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("error: PyTorch cannot access a CUDA device")

major, minor = torch.cuda.get_device_capability()
print(f"CUDA architecture: {major}.{minor} (detected by PyTorch)")
PY
fi

uv run --locked --no-sync python -c \
    "from inference_performance_lab.kernels.extension import load_extension; load_extension(verbose=True)"

uv run --locked --no-sync pytest -m "gpu and not vllm" \
    tests/test_fused_add_rms_norm.py \
    tests/test_silu_and_mul.py
