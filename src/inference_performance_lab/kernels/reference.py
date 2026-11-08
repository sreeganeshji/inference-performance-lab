from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def fused_add_rms_norm_reference(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 oracle for fused residual addition and RMSNorm."""

    if x.shape != residual.shape:
        raise ValueError("x and residual must have identical shapes")

    if weight.ndim != 1 or weight.shape[0] != x.shape[-1]:
        raise ValueError("weight must match the final tensor dimension")

    residual_out_fp32 = x.float() + residual.float()
    inverse_rms = torch.rsqrt(residual_out_fp32.square().mean(dim=-1, keepdim=True) + epsilon)

    output = residual_out_fp32 * inverse_rms * weight.float()

    return output.to(x.dtype), residual_out_fp32.to(residual.dtype)


def silu_and_mul_reference(x: torch.Tensor) -> torch.Tensor:
    """FP32 oracle for the fused SwiGLU activation."""

    if x.ndim == 0 or x.shape[-1] % 2 != 0:
        raise ValueError("the final tensor dimension must be even")

    gate, up = x.float().chunk(2, dim=-1)
    output = F.silu(gate) * up

    return output.to(x.dtype)


def paged_decode_attention_reference(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """FP32 oracle: each sequence's single decode query over its paged KV cache (GQA aware)."""

    num_seqs, num_q_heads, head_dim = query.shape
    _, block_size, num_kv_heads, _ = key_cache.shape
    group_size = num_q_heads // num_kv_heads

    output = torch.zeros(
        (num_seqs, num_q_heads, head_dim), dtype=torch.float32, device=query.device
    )

    for seq in range(num_seqs):
        seq_len = int(seq_lens[seq])

        if seq_len == 0:
            continue

        blocks = block_table[seq, : math.ceil(seq_len / block_size)].long()
        keys = key_cache[blocks].reshape(-1, num_kv_heads, head_dim)[:seq_len].float()
        values = value_cache[blocks].reshape(-1, num_kv_heads, head_dim)[:seq_len].float()
        queries = query[seq].float().view(num_kv_heads, group_size, head_dim)

        scores = torch.einsum("hgd,lhd->hgl", queries, keys) * scale
        probabilities = scores.softmax(dim=-1)
        output[seq] = torch.einsum("hgl,lhd->hgd", probabilities, values).reshape(
            num_q_heads, head_dim
        )

    return output.to(query.dtype)
