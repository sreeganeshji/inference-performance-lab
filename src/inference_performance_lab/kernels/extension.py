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
            str(repo_root / "csrc" / "torch_ops.cpp"),
            str(repo_root / "csrc"/ "fused_add_rms_norm_kernel.cu"),
            str(repo_root / "csrc" / "silu_and_mul_kernel.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            "-O3",
            "-lineinfo",
            "--ptxas-options=-v",
        ],
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

def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    load_extension()

    if x.ndim == 0 or x.shape[-1] % 2 != 0:
        raise ValueError("the final tensor dimension must be even")

    output_shape = x.shape[:-1] + (x.shape[-1] // 2,)
    output = torch.empty(
        output_shape,
        dtype=x.dtype,
        device=x.device,
    )

    torch.ops.inference_performance_lab.silu_and_mul(
        output,
        x,
    )

    return output

def silu_and_mul_packed(
        x: torch.Tensor,
    ) -> torch.Tensor:
    load_extension()

    if x.ndim == 0 or x.shape[-1] % 2 != 0:
        raise ValueError("the final tensor dimension must be even")

    output_shape = x.shape[:-1] + (x.shape[-1] // 2,)
    output = torch.empty(
        output_shape,
        dtype=x.dtype,
        device=x.device,
    )

    torch.ops.inference_performance_lab.silu_and_mul_packed(
        output,
        x,
    )

    return output