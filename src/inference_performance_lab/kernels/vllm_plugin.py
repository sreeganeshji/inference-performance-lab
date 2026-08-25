from __future__ import annotations

import torch
from torch import Tensor

from inference_performance_lab.kernels.extension import (
    fused_add_rms_norm as custom_fused_add_rms_norm,
    load_extension,
)

PROVIDER = "inference_lab"
HIDDEN_SIZE = 3584


def supports_fused_add_rms_norm(
    x: Tensor,
    x_residual: Tensor,
    weight: Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> bool:
    del epsilon

    return (
        variance_size is None
        and weight is not None
        and x.device.type == "cuda"
        and x_residual.device == x.device
        and weight.device == x.device
        and x.dtype == torch.bfloat16
        and x_residual.dtype == x.dtype
        and weight.dtype == x.dtype
        and x.ndim == 2
        and x.shape == x_residual.shape
        and x.shape[-1] == HIDDEN_SIZE
        and weight.shape == (HIDDEN_SIZE,)
        and x.is_contiguous()
        and x_residual.is_contiguous()
        and weight.is_contiguous()
    )


def register() -> None:
    from vllm import ir

    op = ir.ops.fused_add_rms_norm
    if PROVIDER in op.impls:
        return

    load_extension()

    @op.register_impl(
        PROVIDER,
        supported=torch.cuda.is_available(),
        supports_args=supports_fused_add_rms_norm,
        inplace=True,
    )
    def inference_lab_fused_add_rms_norm(
        x: Tensor,
        x_residual: Tensor,
        weight: Tensor | None,
        epsilon: float,
        variance_size: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        assert weight is not None
        assert variance_size is None

        custom_fused_add_rms_norm(
            x,
            x_residual,
            weight,
            epsilon,
        )
        return x, x_residual