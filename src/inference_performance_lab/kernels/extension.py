from __future__ import annotations

from functools import cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


@cache
def load_extension(*, verbose: bool = False) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    build_directory = (
        repo_root
        / ".cache"
        / "torch_extensions"
        / "fused_add_rms_norm"
    )
    build_directory.mkdir(parents=True, exist_ok=True)

    load(
        name="inference_performance_lab_cuda",
        sources=[
            str(repo_root / "csrc" / "fused_add_rms_norm.cpp"),
            str(
                repo_root
                / "csrc"
                / "fused_add_rms_norm_kernel.cu"
            ),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        build_directory=str(build_directory),
        is_python_module=False,
        verbose=verbose,
    )


def fused_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> None:
    load_extension()

    torch.ops.inference_performance_lab.fused_add_rms_norm(
        x,
        residual,
        weight,
        epsilon,
    )