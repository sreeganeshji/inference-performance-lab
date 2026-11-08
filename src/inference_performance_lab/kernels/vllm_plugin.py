from __future__ import annotations

import functools
import os

import torch
from torch import Tensor

from inference_performance_lab.kernels.extension import (
    fused_add_rms_norm as custom_fused_add_rms_norm,
)
from inference_performance_lab.kernels.extension import load_extension

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
    register_attention_backend()
    register_fused_add_rms_norm()

    if os.environ.get("IPL_SCHEDULER_NVTX") == "1":
        annotate_scheduler()


def annotate_scheduler() -> None:
    """Wrap the V1 scheduler's per-step work in NVTX ranges for Nsight Systems.

    Enabled with IPL_SCHEDULER_NVTX=1. The ranges are on the engine-core thread, so a trace shows
    how long each schedule() and update_from_output() call takes, how much of that is block
    allocation and freeing, and whether the GPU sits idle meanwhile.
    """

    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.sched.scheduler import Scheduler

    targets = [
        (Scheduler, "schedule", "scheduler.schedule"),
        (Scheduler, "update_from_output", "scheduler.update_from_output"),
        (BlockPool, "get_new_blocks", "block_pool.get_new_blocks"),
        (BlockPool, "free_blocks", "block_pool.free_blocks"),
    ]

    for cls, name, label in targets:
        method = getattr(cls, name)

        if not getattr(method, "_ipl_nvtx", False):
            setattr(cls, name, _nvtx_range(method, label))


def _nvtx_range(method, label: str):
    @functools.wraps(method)
    def wrapper(*args, **kwargs):
        torch.cuda.nvtx.range_push(label)

        try:
            return method(*args, **kwargs)
        finally:
            torch.cuda.nvtx.range_pop()

    wrapper._ipl_nvtx = True
    return wrapper


def register_attention_backend() -> None:
    """Make the decode-attention backend selectable with --attention-backend CUSTOM."""

    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

    register_backend(
        AttentionBackendEnum.CUSTOM,
        "inference_performance_lab.kernels.vllm_attention.LabDecodeAttentionBackend",
    )


def register_fused_add_rms_norm() -> None:
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

        custom_fused_add_rms_norm(x, x_residual, weight, epsilon)
        return x, x_residual
