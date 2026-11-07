import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from inference_performance_lab.kernels.extension import DECODE_VARIANTS, paged_decode_attention
from inference_performance_lab.kernels.reference import paged_decode_attention_reference

HEAD_DIM = 128

# (num_q_heads, num_kv_heads): Qwen2.5-7B, Qwen2.5-1.5B, Qwen2.5-3B, Qwen3-4B, and plain MHA.
HEAD_LAYOUTS = [(28, 4), (12, 2), (16, 2), (32, 8), (8, 8)]

SEQ_LEN_CASES = {
    "one-token": [1],
    "block-edges": [15, 16, 17, 32, 33],
    "ragged": [1, 47, 300, 1000, 2049],
    "padded-slot": [0, 64, 0, 129],
}

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")


def make_inputs(
    seq_lens: list[int],
    num_q_heads: int,
    num_kv_heads: int,
    block_size: int = 16,
    *,
    strided_query: bool = False,
    seed: int = 0,
):
    """Random queries and a paged cache whose blocks are scattered across the pool."""

    generator = torch.Generator(device="cuda").manual_seed(seed)
    num_seqs = len(seq_lens)
    blocks_per_seq = [math.ceil(seq_len / block_size) for seq_len in seq_lens]
    max_blocks = max(max(blocks_per_seq), 1)
    num_blocks = sum(blocks_per_seq) + 7

    key_cache = torch.randn(
        (num_blocks, block_size, num_kv_heads, HEAD_DIM),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    value_cache = torch.randn_like(key_cache)

    permutation = torch.randperm(num_blocks, generator=generator, device="cuda").to(torch.int32)
    block_table = torch.zeros((num_seqs, max_blocks), dtype=torch.int32, device="cuda")
    next_block = 0

    for seq, count in enumerate(blocks_per_seq):
        block_table[seq, :count] = permutation[next_block : next_block + count]
        next_block += count

    if strided_query:
        # Decode queries are a slice of the fused QKV projection in vLLM, so token rows are strided.
        qkv_width = (num_q_heads + 2 * num_kv_heads) * HEAD_DIM
        qkv = torch.randn(
            (num_seqs, qkv_width), generator=generator, device="cuda", dtype=torch.bfloat16
        )
        query = qkv[:, : num_q_heads * HEAD_DIM].view(num_seqs, num_q_heads, HEAD_DIM)
    else:
        query = torch.randn(
            (num_seqs, num_q_heads, HEAD_DIM),
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )

    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")

    return query, key_cache, value_cache, block_table, seq_lens_tensor


def assert_matches_reference(inputs, **kwargs) -> None:
    query, key_cache, value_cache, block_table, seq_lens = inputs
    scale = HEAD_DIM**-0.5

    expected = paged_decode_attention_reference(
        query, key_cache, value_cache, block_table, seq_lens, scale
    )
    actual = paged_decode_attention(
        query, key_cache, value_cache, block_table, seq_lens, scale, **kwargs
    )

    torch.testing.assert_close(actual, expected, rtol=1.6e-2, atol=1e-2)


@pytest.mark.gpu
@requires_cuda
@pytest.mark.parametrize("variant", DECODE_VARIANTS)
@pytest.mark.parametrize("heads", HEAD_LAYOUTS, ids=lambda h: f"q{h[0]}-kv{h[1]}")
@pytest.mark.parametrize("case", SEQ_LEN_CASES)
@torch.inference_mode()
def test_matches_reference(variant: str, heads: tuple[int, int], case: str) -> None:
    inputs = make_inputs(SEQ_LEN_CASES[case], *heads)
    assert_matches_reference(inputs, variant=variant)


@pytest.mark.gpu
@requires_cuda
@pytest.mark.parametrize("num_splits", [2, 3, 7, 64])
@pytest.mark.parametrize("case", ["one-token", "ragged", "padded-slot"])
@torch.inference_mode()
def test_forced_split_counts(num_splits: int, case: str) -> None:
    # 64 splits leaves most splits empty for short sequences; the combine pass must ignore them.
    inputs = make_inputs(SEQ_LEN_CASES[case], 28, 4)
    assert_matches_reference(inputs, variant="gqa_split", num_splits=num_splits)


@pytest.mark.gpu
@requires_cuda
@pytest.mark.parametrize("block_size", [16, 32])
@pytest.mark.parametrize("variant", DECODE_VARIANTS)
@torch.inference_mode()
def test_block_sizes(block_size: int, variant: str) -> None:
    inputs = make_inputs(SEQ_LEN_CASES["ragged"], 28, 4, block_size)
    assert_matches_reference(inputs, variant=variant)


@pytest.mark.gpu
@requires_cuda
@pytest.mark.parametrize("variant", DECODE_VARIANTS)
@torch.inference_mode()
def test_strided_query(variant: str) -> None:
    inputs = make_inputs(SEQ_LEN_CASES["ragged"], 28, 4, strided_query=True)
    assert_matches_reference(inputs, variant=variant)


@pytest.mark.gpu
@requires_cuda
@pytest.mark.parametrize("variant", DECODE_VARIANTS)
@torch.inference_mode()
def test_long_context(variant: str) -> None:
    inputs = make_inputs([32768, 5000], 28, 4)
    assert_matches_reference(inputs, variant=variant)


@pytest.mark.gpu
@requires_cuda
@torch.inference_mode()
def test_padded_slots_write_zeros() -> None:
    query, key_cache, value_cache, block_table, seq_lens = make_inputs([0, 33, 0], 28, 4)

    for variant in DECODE_VARIANTS:
        out = torch.full_like(query, float("nan"))
        paged_decode_attention(
            query, key_cache, value_cache, block_table, seq_lens, 0.1, variant=variant, out=out
        )

        assert torch.count_nonzero(out[0]) == 0 and torch.count_nonzero(out[2]) == 0
        assert torch.isfinite(out).all()


@pytest.mark.gpu
@requires_cuda
@pytest.mark.parametrize(
    ("variant", "num_splits"),
    [(variant, None) for variant in DECODE_VARIANTS] + [("gqa_split", 7), ("gqa_split_tc", 7)],
)
@torch.inference_mode()
def test_stale_cache_slots_are_ignored(variant: str, num_splits: int | None) -> None:
    # vLLM never clears freed blocks, so a slot no valid token owns can hold anything, NaN
    # included, and 0 * NaN is NaN: the kernel must keep those values out of both products.
    seq_lens = [1, 15, 17, 300, 2049]
    query, key_cache, value_cache, block_table, seq_lens_tensor = make_inputs(seq_lens, 28, 4)
    block_size = key_cache.shape[1]
    owned = torch.zeros(key_cache.shape[:2], dtype=torch.bool, device="cuda")

    for seq, seq_len in enumerate(seq_lens):
        positions = torch.arange(seq_len, device="cuda")
        blocks = block_table[seq, positions // block_size].long()
        owned[blocks, positions % block_size] = True

    key_cache[~owned] = float("nan")
    value_cache[~owned] = float("nan")

    assert_matches_reference(
        (query, key_cache, value_cache, block_table, seq_lens_tensor),
        variant=variant,
        num_splits=num_splits,
    )


@pytest.mark.gpu
@requires_cuda
@torch.inference_mode()
def test_large_scores_stay_finite() -> None:
    # A large scale pushes scores far past exp's range; the running max must keep it stable.
    query, key_cache, value_cache, block_table, seq_lens = make_inputs([500, 2000], 28, 4)
    query = query * 30

    expected = paged_decode_attention_reference(
        query, key_cache, value_cache, block_table, seq_lens, 1.0
    )

    for variant in DECODE_VARIANTS:
        actual = paged_decode_attention(
            query, key_cache, value_cache, block_table, seq_lens, 1.0, variant=variant
        )
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=1.6e-2, atol=1e-2)


@pytest.mark.gpu
@requires_cuda
@torch.inference_mode()
def test_rejects_unsupported_inputs() -> None:
    query, key_cache, value_cache, block_table, seq_lens = make_inputs([16], 28, 4)

    with pytest.raises(RuntimeError, match="head_dim must be 128"):
        paged_decode_attention(
            query[..., :64], key_cache[..., :64], value_cache[..., :64], block_table, seq_lens, 1.0
        )

    with pytest.raises(RuntimeError, match="int32"):
        paged_decode_attention(query, key_cache, value_cache, block_table.long(), seq_lens, 1.0)


@pytest.mark.gpu
@pytest.mark.vllm
@requires_cuda
@pytest.mark.parametrize("heads", [(28, 4), (12, 2)], ids=lambda h: f"q{h[0]}-kv{h[1]}")
@torch.inference_mode()
def test_matches_vllm_flash_attention(heads: tuple[int, int]) -> None:
    flash_attn = pytest.importorskip("vllm.vllm_flash_attn")
    query, key_cache, value_cache, block_table, seq_lens = make_inputs(
        SEQ_LEN_CASES["ragged"], *heads
    )
    num_seqs = query.shape[0]
    scale = HEAD_DIM**-0.5

    expected = flash_attn.flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        max_seqlen_q=1,
        cu_seqlens_q=torch.arange(num_seqs + 1, dtype=torch.int32, device="cuda"),
        max_seqlen_k=int(seq_lens.max()),
        seqused_k=seq_lens,
        softmax_scale=scale,
        causal=True,
        block_table=block_table,
        fa_version=2,
    )
    actual = paged_decode_attention(query, key_cache, value_cache, block_table, seq_lens, scale)

    torch.testing.assert_close(actual, expected, rtol=1.6e-2, atol=1e-2)


@pytest.mark.gpu
@pytest.mark.vllm
@requires_cuda
def test_vllm_custom_backend_matches_flash_attention() -> None:
    # Two short vLLM runs in fresh processes (about two minutes): FLASH_ATTN, then CUSTOM.
    pytest.importorskip("vllm")
    script = Path(__file__).resolve().parents[1] / "benchmarks" / "vllm_attention_check.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--enforce-eager", "--max-tokens", "16"],
        capture_output=True,
        text=True,
        timeout=1200,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout[-2000:] + completed.stderr[-2000:]
    assert "Using AttentionBackendEnum.CUSTOM backend" in completed.stdout
    assert "Lab decode kernel ran on a mixed decode + prefill batch" in completed.stdout
    assert "identical greedy prefix: 128/128 tokens" in completed.stdout
