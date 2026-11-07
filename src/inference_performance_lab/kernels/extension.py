from __future__ import annotations

import math
from functools import cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

TORCH_TARGET_VERSION = "0x020d000000000000"  # PyTorch 2.13 stable ABI.


@cache
def load_extension(*, verbose: bool = False) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    build_directory = repo_root / ".cache" / "torch_extensions" / "fused_add_rms_norm"
    build_directory.mkdir(parents=True, exist_ok=True)

    load(
        name="inference_performance_lab_cuda",
        sources=[
            str(repo_root / "csrc" / "torch_ops.cpp"),
            str(repo_root / "csrc" / "fused_add_rms_norm_kernel.cu"),
            str(repo_root / "csrc" / "silu_and_mul_kernel.cu"),
            str(repo_root / "csrc" / "paged_decode_attention_kernel.cu"),
        ],
        extra_cflags=["-O3", f"-DTORCH_TARGET_VERSION={TORCH_TARGET_VERSION}"],
        extra_cuda_cflags=[
            "-O3",
            "-lineinfo",
            "--ptxas-options=-v",
            f"-DTORCH_TARGET_VERSION={TORCH_TARGET_VERSION}",
        ],
        build_directory=str(build_directory),
        is_python_module=False,
        verbose=verbose,
    )


def fused_add_rms_norm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> None:
    load_extension()

    torch.ops.inference_performance_lab.fused_add_rms_norm(x, residual, weight, epsilon)


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    load_extension()

    if x.ndim == 0 or x.shape[-1] % 2 != 0:
        raise ValueError("the final tensor dimension must be even")

    output_shape = x.shape[:-1] + (x.shape[-1] // 2,)
    output = torch.empty(output_shape, dtype=x.dtype, device=x.device)

    torch.ops.inference_performance_lab.silu_and_mul(output, x)

    return output


def silu_and_mul_packed(x: torch.Tensor) -> torch.Tensor:
    load_extension()

    if x.ndim == 0 or x.shape[-1] % 2 != 0:
        raise ValueError("the final tensor dimension must be even")

    output_shape = x.shape[:-1] + (x.shape[-1] // 2,)
    output = torch.empty(output_shape, dtype=x.dtype, device=x.device)

    torch.ops.inference_performance_lab.silu_and_mul_packed(output, x)

    return output


DECODE_VARIANTS = ("per_query_head", "gqa", "gqa_split", "gqa_split_tc")

# Split-KV heuristic constants, fitted to split sweeps on the RTX 3080 Ti (see choose_num_splits).
# Each split gets at least MIN_KV_BLOCKS_PER_SPLIT KV blocks. The combine kernel's fixed cost is
# COMBINE_OVERHEAD_BYTES of equivalent traffic (about 3 us at 850 GB/s); a block's prologue and
# epilogue cost BLOCK_OVERHEAD_BYTES of its slot's streaming time; split outputs cost
# PARTIAL_TRAFFIC_WEIGHT times their raw bytes, because the combine pass is far from streaming speed.
MIN_KV_BLOCKS_PER_SPLIT = 4
MAX_SPLITS = 64
COMBINE_OVERHEAD_BYTES = 2.5e6
BLOCK_OVERHEAD_BYTES = 16 * 1024
PARTIAL_TRAFFIC_WEIGHT = 3
TC_SATURATING_BLOCKS_PER_SM = 0.8
KV_BYTES_PER_TOKEN = 2 * 128 * 2  # one K row and one V row of BF16, per KV head


@cache
def _multiprocessor_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


@cache
def _resident_blocks(device_index: int, queries_per_cta: int, tensor_cores: bool) -> int:
    """Thread blocks the whole GPU holds at once for one kernel instantiation (one full wave)."""

    load_extension()
    like = torch.empty(0, device=f"cuda:{device_index}")
    per_sm = torch.ops.inference_performance_lab.paged_decode_attention_blocks_per_sm(
        like, queries_per_cta, tensor_cores
    )

    return per_sm * _multiprocessor_count(device_index)


def choose_num_splits(
    num_seqs: int,
    num_kv_heads: int,
    *,
    device: torch.device,
    queries_per_cta: int,
    tensor_cores: bool = False,
    max_seq_len: int | None = None,
    block_size: int = 16,
) -> int:
    """Split each sequence's KV range so the grid keeps DRAM busy to the end (flash-decoding).

    Model: DRAM saturates once `saturating` blocks run at once. A wave of n concurrent blocks, each
    reading W bytes, then takes W * max(n, saturating) / bandwidth. The tensor-core kernel keeps
    up to 32 KB in flight per block, so about 0.8 blocks per SM saturate DRAM; the CUDA-core kernel
    is issue-bound per block and needs every resident slot. Splits add their FP32 outputs (written,
    then read) and the combine kernel's fixed cost.

    On an 80-SM RTX 3080 Ti holding 240 tensor-core blocks, 64 sequence-heads of 4K tokens run
    best unsplit (64 blocks already saturate), while 256 sequence-heads prefer 3-4 splits: a single
    split would leave a 16-block tail that runs at a fraction of the bandwidth.
    """

    device_index = device.index if device.index is not None else torch.cuda.current_device()
    capacity = _resident_blocks(device_index, queries_per_cta, tensor_cores)
    saturating = (
        int(TC_SATURATING_BLOCKS_PER_SM * _multiprocessor_count(device_index))
        if tensor_cores
        else capacity
    )
    blocks_without_split = max(num_seqs * num_kv_heads, 1)
    kv_blocks = math.ceil((max_seq_len or 32768) / block_size)
    max_splits = max(1, min(MAX_SPLITS, kv_blocks // MIN_KV_BLOCKS_PER_SPLIT))

    # Each split writes, and the combine pass reads, one FP32 output row and one LSE per query head.
    partial_bytes = queries_per_cta * (128 + 1) * 4 * 2

    def modeled_bytes(splits: int) -> float:
        grid = blocks_without_split * splits
        bytes_per_block = math.ceil(kv_blocks / splits) * block_size * KV_BYTES_PER_TOKEN
        full_waves, tail = divmod(grid, capacity)
        busy_slots = full_waves * max(capacity, saturating) + (max(tail, saturating) if tail else 0)
        cost = busy_slots * (bytes_per_block + BLOCK_OVERHEAD_BYTES)

        if splits > 1:
            cost += PARTIAL_TRAFFIC_WEIGHT * grid * partial_bytes + COMBINE_OVERHEAD_BYTES

        return cost

    return min(range(1, max_splits + 1), key=modeled_bytes)


def paged_decode_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    *,
    variant: str = "gqa_split",
    num_splits: int | None = None,
    max_seq_len: int | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode attention over a paged KV cache in vLLM's FLASH_ATTN layout.

    query: [num_seqs, num_q_heads, 128]; key_cache and value_cache: [num_blocks, block_size,
    num_kv_heads, 128]; block_table: int32 [num_seqs, max_blocks]; seq_lens: int32 [num_seqs].
    """

    load_extension()

    if variant not in DECODE_VARIANTS:
        raise ValueError(f"variant must be one of {DECODE_VARIANTS}")

    num_seqs, num_q_heads, head_dim = query.shape
    num_kv_heads = key_cache.shape[2]
    group_size = num_q_heads // num_kv_heads

    tensor_cores = variant == "gqa_split_tc"

    if variant == "per_query_head":
        queries_per_cta, splits = 1, 1
    elif variant == "gqa":
        queries_per_cta, splits = group_size, 1
    else:
        queries_per_cta = group_size
        splits = num_splits or choose_num_splits(
            num_seqs,
            num_kv_heads,
            device=query.device,
            queries_per_cta=queries_per_cta,
            tensor_cores=tensor_cores,
            max_seq_len=max_seq_len,
            block_size=key_cache.shape[1],
        )

    if out is None:
        out = torch.empty((num_seqs, num_q_heads, head_dim), dtype=query.dtype, device=query.device)

    if splits > 1:
        partial_out = torch.empty(
            (num_seqs, num_q_heads, splits, head_dim), dtype=torch.float32, device=query.device
        )
        partial_lse = torch.empty(
            (num_seqs, num_q_heads, splits), dtype=torch.float32, device=query.device
        )
    else:
        partial_out = torch.empty(0, dtype=torch.float32, device=query.device)
        partial_lse = torch.empty(0, dtype=torch.float32, device=query.device)

    torch.ops.inference_performance_lab.paged_decode_attention(
        out,
        partial_out,
        partial_lse,
        query,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        scale,
        queries_per_cta,
        splits,
        tensor_cores,
    )

    return out
