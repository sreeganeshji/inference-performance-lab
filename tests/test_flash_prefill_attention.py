import itertools

import pytest
import torch
from torch.nn import functional as F

from inference_performance_lab.kernels.extension import PREFILL_VARIANTS, flash_prefill_attention
from inference_performance_lab.kernels.reference import flash_prefill_attention_reference
from inference_performance_lab.kernels.vllm_prefill import fresh_prefill_attention, is_fresh_prefill

HEAD_DIM = 128
SCALE = HEAD_DIM**-0.5

# (num_q_heads, num_kv_heads): Qwen2.5-7B, Qwen2.5-1.5B, Qwen2.5-3B, Qwen3-4B, and plain MHA.
HEAD_LAYOUTS = [(28, 4), (12, 2), (16, 2), (32, 8), (8, 8)]

# Single-sequence lengths around the 64-row tile edges, plus long prompts.
LENGTHS = [1, 15, 16, 17, 63, 64, 65, 127, 128, 129, 1000, 4096, 8192]

RAGGED_BATCHES = {
    "tile-edges": [1, 15, 16, 17, 63, 64, 65, 127, 128, 129],
    "mixed": [1000, 1, 4096, 300, 64],
    "empty-sequences": [17, 0, 129, 0, 1],
}

ATOL = 1e-2
RTOL = 1.6e-2

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")


def make_inputs(
    seq_lens: list[int],
    num_q_heads: int,
    num_kv_heads: int,
    *,
    fused_qkv: bool = False,
    seed: int = 0,
):
    """Random packed Q, K, V and cu_seqlens; fused_qkv slices all three from one QKV projection."""

    generator = torch.Generator(device="cuda").manual_seed(seed)
    total_tokens = sum(seq_lens)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16)

    if fused_qkv:
        # vLLM's attention layer receives views of the fused QKV output, so token rows are strided.
        qkv = randn(total_tokens, (num_q_heads + 2 * num_kv_heads) * HEAD_DIM)
        query, key, value = qkv.split(
            [num_q_heads * HEAD_DIM, num_kv_heads * HEAD_DIM, num_kv_heads * HEAD_DIM], dim=-1
        )
        query = query.view(total_tokens, num_q_heads, HEAD_DIM)
        key = key.view(total_tokens, num_kv_heads, HEAD_DIM)
        value = value.view(total_tokens, num_kv_heads, HEAD_DIM)
    else:
        query = randn(total_tokens, num_q_heads, HEAD_DIM)
        key = randn(total_tokens, num_kv_heads, HEAD_DIM)
        value = randn(total_tokens, num_kv_heads, HEAD_DIM)

    cu_seqlens = torch.tensor(
        [0] + torch.tensor(seq_lens).cumsum(0).tolist(), dtype=torch.int32, device="cuda"
    )

    return query, key, value, cu_seqlens, max(seq_lens)


def assert_matches_reference(inputs, variant: str, *, causal: bool = True, scale: float = SCALE):
    query, key, value, cu_seqlens, max_seqlen = inputs
    expected = flash_prefill_attention_reference(
        query, key, value, cu_seqlens, scale, causal=causal
    )
    actual = flash_prefill_attention(
        query, key, value, cu_seqlens, max_seqlen, scale, causal=causal, variant=variant
    )

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), expected, atol=ATOL, rtol=RTOL)


@pytest.mark.gpu
@requires_cuda
@pytest.mark.parametrize("variant", PREFILL_VARIANTS)
@pytest.mark.parametrize("length", LENGTHS)
@torch.inference_mode()
def test_single_sequence_lengths(variant: str, length: int) -> None:
    assert_matches_reference(make_inputs([length], 28, 4), variant)


@pytest.mark.gpu
@requires_cuda
@pytest.mark.parametrize("variant", PREFILL_VARIANTS)
@pytest.mark.parametrize("heads", HEAD_LAYOUTS, ids=lambda h: f"q{h[0]}-kv{h[1]}")
@pytest.mark.parametrize("batch", RAGGED_BATCHES)
@torch.inference_mode()
def test_ragged_batches(variant: str, heads: tuple[int, int], batch: str) -> None:
    assert_matches_reference(make_inputs(RAGGED_BATCHES[batch], *heads), variant)


@pytest.mark.gpu
@requires_cuda
@pytest.mark.parametrize("variant", PREFILL_VARIANTS)
@torch.inference_mode()
def test_non_causal(variant: str) -> None:
    assert_matches_reference(make_inputs(RAGGED_BATCHES["mixed"], 28, 4), variant, causal=False)


@pytest.mark.gpu
@requires_cuda
@pytest.mark.parametrize("variant", PREFILL_VARIANTS)
@torch.inference_mode()
def test_strided_inputs_and_output(variant: str) -> None:
    query, key, value, cu_seqlens, max_seqlen = make_inputs(
        RAGGED_BATCHES["mixed"], 28, 4, fused_qkv=True
    )
    expected = flash_prefill_attention_reference(query, key, value, cu_seqlens, SCALE)

    # The output is a strided view too: every other 256-byte head slot of a wider buffer.
    buffer = torch.full(
        (query.shape[0], 2 * 28, HEAD_DIM), float("nan"), dtype=torch.bfloat16, device="cuda"
    )
    out = buffer[:, ::2]
    flash_prefill_attention(
        query, key, value, cu_seqlens, max_seqlen, SCALE, variant=variant, out=out
    )

    torch.testing.assert_close(out.float(), expected, atol=ATOL, rtol=RTOL)
    assert torch.isnan(buffer[:, 1::2]).all(), "the kernel wrote outside the output view"


@pytest.mark.gpu
@requires_cuda
@pytest.mark.parametrize("variant", PREFILL_VARIANTS)
@torch.inference_mode()
def test_oversized_grid(variant: str) -> None:
    # max_seqlen only sizes the grid; blocks past a sequence's end must do nothing.
    query, key, value, cu_seqlens, max_seqlen = make_inputs([65, 129], 28, 4)
    expected = flash_prefill_attention_reference(query, key, value, cu_seqlens, SCALE)
    actual = flash_prefill_attention(
        query, key, value, cu_seqlens, max_seqlen + 1000, SCALE, variant=variant
    )

    torch.testing.assert_close(actual.float(), expected, atol=ATOL, rtol=RTOL)


@pytest.mark.gpu
@requires_cuda
@pytest.mark.parametrize("variant", PREFILL_VARIANTS)
@torch.inference_mode()
def test_large_scores_stay_finite(variant: str) -> None:
    # Scores near ±1000 overflow exp; the running max must keep every step in range.
    query, key, value, cu_seqlens, max_seqlen = make_inputs([500, 2000, 129], 28, 4)
    inputs = (query * 30, key, value, cu_seqlens, max_seqlen)

    assert_matches_reference(inputs, variant, scale=1.0)


@pytest.mark.gpu
@requires_cuda
@pytest.mark.parametrize("variant", PREFILL_VARIANTS)
@torch.inference_mode()
def test_repeat_runs_are_bitwise_identical(variant: str) -> None:
    query, key, value, cu_seqlens, max_seqlen = make_inputs(RAGGED_BATCHES["mixed"], 28, 4)
    first = flash_prefill_attention(
        query, key, value, cu_seqlens, max_seqlen, SCALE, variant=variant
    )

    for _ in range(3):
        again = flash_prefill_attention(
            query, key, value, cu_seqlens, max_seqlen, SCALE, variant=variant
        )
        assert torch.equal(again, first)


@pytest.mark.gpu
@requires_cuda
@pytest.mark.parametrize("variant", PREFILL_VARIANTS)
@pytest.mark.parametrize("heads", [(28, 4), (12, 2)], ids=lambda h: f"q{h[0]}-kv{h[1]}")
@torch.inference_mode()
def test_matches_sdpa(variant: str, heads: tuple[int, int]) -> None:
    seq_lens = RAGGED_BATCHES["mixed"]
    query, key, value, cu_seqlens, max_seqlen = make_inputs(seq_lens, *heads)
    actual = flash_prefill_attention(
        query, key, value, cu_seqlens, max_seqlen, SCALE, variant=variant
    )
    bounds = cu_seqlens.tolist()

    for start, end in itertools.pairwise(bounds):
        # SDPA wants [batch, heads, tokens, head_dim]; one sequence at a time.
        expected = F.scaled_dot_product_attention(
            query[start:end].transpose(0, 1)[None],
            key[start:end].transpose(0, 1)[None],
            value[start:end].transpose(0, 1)[None],
            is_causal=True,
            scale=SCALE,
            enable_gqa=True,
        )[0].transpose(0, 1)

        torch.testing.assert_close(actual[start:end], expected, atol=ATOL, rtol=RTOL)


@pytest.mark.gpu
@pytest.mark.vllm
@requires_cuda
@pytest.mark.parametrize("variant", PREFILL_VARIANTS)
@pytest.mark.parametrize("heads", HEAD_LAYOUTS, ids=lambda h: f"q{h[0]}-kv{h[1]}")
@torch.inference_mode()
def test_matches_vllm_flash_attention(variant: str, heads: tuple[int, int]) -> None:
    flash_attn = pytest.importorskip("vllm.vllm_flash_attn")
    query, key, value, cu_seqlens, max_seqlen = make_inputs(RAGGED_BATCHES["mixed"], *heads)

    expected = flash_attn.flash_attn_varlen_func(
        q=query,
        k=key,
        v=value,
        max_seqlen_q=max_seqlen,
        cu_seqlens_q=cu_seqlens,
        max_seqlen_k=max_seqlen,
        cu_seqlens_k=cu_seqlens,
        softmax_scale=SCALE,
        causal=True,
        fa_version=2,
    )
    actual = flash_prefill_attention(
        query, key, value, cu_seqlens, max_seqlen, SCALE, variant=variant
    )

    torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)


@pytest.mark.gpu
@requires_cuda
@torch.inference_mode()
def test_fresh_prefill_routing() -> None:
    # A vLLM-ordered batch: three decodes with cached context, then three fresh prompts.
    decode_contexts = [100, 200, 50]
    prompts = [17, 129, 1000]
    query_lens = [1] * len(decode_contexts) + prompts
    query_start_loc = torch.tensor([0] + torch.tensor(query_lens).cumsum(0).tolist())
    seq_lens = torch.tensor(decode_contexts + prompts)
    num_decodes = len(decode_contexts)

    assert is_fresh_prefill(query_start_loc, seq_lens, num_decodes)
    assert not is_fresh_prefill(query_start_loc, seq_lens + 1, num_decodes)  # cached prefixes
    assert not is_fresh_prefill(query_start_loc, seq_lens, len(query_lens))  # no prefills

    query, key, value, _, _ = make_inputs(query_lens, 28, 4, fused_qkv=True)
    output = torch.full_like(query, float("nan"))
    prefill_start_loc = (query_start_loc[num_decodes:] - num_decodes).to("cuda", torch.int32)

    fresh_prefill_attention(
        output[num_decodes:],
        query[num_decodes:],
        key[num_decodes:],
        value[num_decodes:],
        prefill_start_loc,
        max(query_lens),
        SCALE,
    )

    expected = flash_prefill_attention_reference(
        query[num_decodes:], key[num_decodes:], value[num_decodes:], prefill_start_loc, SCALE
    )
    torch.testing.assert_close(output[num_decodes:].float(), expected, atol=ATOL, rtol=RTOL)
    assert torch.isnan(output[:num_decodes]).all(), "decode rows must be left to the decode kernel"


@pytest.mark.gpu
@requires_cuda
@torch.inference_mode()
def test_rejects_unsupported_inputs() -> None:
    query, key, value, cu_seqlens, max_seqlen = make_inputs([16, 33], 28, 4)

    with pytest.raises(RuntimeError, match="head_dim must be 128"):
        flash_prefill_attention(
            query[..., :64], key[..., :64], value[..., :64], cu_seqlens, max_seqlen, SCALE
        )

    with pytest.raises(RuntimeError, match="int32"):
        flash_prefill_attention(query, key, value, cu_seqlens.long(), max_seqlen, SCALE)

    with pytest.raises(RuntimeError, match="same tokens"):
        flash_prefill_attention(query, key[:-1], value[:-1], cu_seqlens, max_seqlen, SCALE)

    with pytest.raises(RuntimeError, match="multiple of num_kv_heads"):
        flash_prefill_attention(query[:, :27], key, value, cu_seqlens, max_seqlen, SCALE)
