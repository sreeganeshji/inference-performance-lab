"""Routing fresh prefills to the lab's prefill kernel from the CUSTOM vLLM backend.

Documented and tested here, but not wired into vllm_attention.py (docs/flash-prefill.md,
"Integration", has the design). A fresh prefill is a request whose whole prompt is in this step:
query_len == seq_len, so nothing of it is in the paged KV cache yet (no prefix-cache hit, not a
later chunk of a chunked prefill). Its attention then needs only the step's own K and V, which
vLLM hands to AttentionImpl.forward as `key` and `value` ([num_tokens, num_kv_heads, head_size]
views of the fused QKV projection). The KV-cache write is a separate step (do_kv_cache_update),
so reading K and V from forward's arguments skips nothing.
"""

from __future__ import annotations

import torch

from inference_performance_lab.kernels.extension import (
    PREFILL_DEFAULT_VARIANT,
    PREFILL_VARIANTS,
    load_extension,
)


def is_fresh_prefill(
    query_start_loc_cpu: torch.Tensor, seq_lens_cpu: torch.Tensor, first_prefill: int
) -> bool:
    """Whether every request from first_prefill on computes its whole sequence in this step.

    Call it in the metadata builder with CommonAttentionMetadata.query_start_loc_cpu and
    seq_lens_cpu_upper_bound (exact for prefill rows), so the decision needs no device sync.
    """

    starts = query_start_loc_cpu[first_prefill:]
    query_lens = starts[1:] - starts[:-1]
    return query_lens.numel() > 0 and bool(torch.equal(query_lens, seq_lens_cpu[first_prefill:]))


def fresh_prefill_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    scale: float,
    *,
    variant: str = PREFILL_DEFAULT_VARIANT,
) -> None:
    """Causal attention for fresh-prefill rows, read from forward's query/key/value.

    query, key, value and output are the prefill rows only (forward's tensors sliced from the first
    prefill token); cu_seqlens is int32 [num_prefills + 1] relative to that token (the prefill
    metadata's query_start_loc) and max_seqlen any upper bound on the prefill query lengths.
    """

    load_extension()
    torch.ops.inference_performance_lab.flash_prefill_attention(
        output,
        query,
        key,
        value,
        cu_seqlens,
        max_seqlen,
        scale,
        True,
        PREFILL_VARIANTS.index(variant),
    )
