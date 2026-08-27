from __future__ import annotations

import torch
from torch.nn import functional as F


def fused_add_rms_norm_reference(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 oracle for fused residual addition and RMSNorm."""

    if x.shape != residual.shape:
        raise ValueError("x and residual must have identical shapes")

    if weight.ndim != 1 or weight.shape[0] != x.shape[-1]:
        raise ValueError("weight must match the final tensor dimension")

    residual_out_fp32 = x.float() + residual.float()
    inverse_rms = torch.rsqrt(
        residual_out_fp32.square().mean(dim=-1, keepdim=True) + epsilon
    )

    output = residual_out_fp32 * inverse_rms * weight.float()

    return output.to(x.dtype), residual_out_fp32.to(residual.dtype)

def silu_and_mul_reference(x: torch.Tensor) -> torch.Tensor:
    """FP32 oracle for the fused SwiGLU activation."""

    if x.ndim == 0 or x.shape[-1] % 2 != 0:
        raise ValueError("the final tensor dimension must be even")

    gate, up = x.float().chunk(2, dim=-1)
    output = F.silu(gate) * up

    return output.to(x.dtype)