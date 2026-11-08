"""A vLLM attention backend that runs this lab's tensor-core decode kernel on decode tokens.

The plugin registers it as AttentionBackendEnum.CUSTOM; select it with

    vllm serve <model> --attention-backend CUSTOM

It is vLLM's FlashAttention backend with two changes. The metadata builder asks vLLM to put
decode requests first in the batch (reorder_batch_threshold = 1) and sizes a split-KV workspace
once, so full CUDA graphs can capture the decode kernel. The forward pass sends the decode rows to
paged_decode_attention (variant gqa_split_tc) and any remaining rows (prefill, chunked prefill,
spec-decode verification) to FlashAttention-2. Layers the kernel does not handle, such as head
sizes other than 128, FP8 KV caches, sliding windows, ALiBi, soft-capping, sinks, GQA groups
larger than 8, and decode context parallelism, use FlashAttention-2 for every row.
"""

from __future__ import annotations

import dataclasses

import torch
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills

from inference_performance_lab.kernels.extension import choose_num_splits, load_extension

# Under the vllm logger hierarchy so vLLM's handlers print these lines.
logger = init_logger("vllm.inference_performance_lab.attention")

HEAD_DIM = 128
MAX_QUERIES_PER_KV = 8

# The split workspace holds max_num_seqs * WORKSPACE_SPLITS_PER_SEQ (sequence, split) pairs. The
# heuristic only splits deeply when few sequences are decoding, so this bound rarely binds.
WORKSPACE_SPLITS_PER_SEQ = 8


@dataclasses.dataclass
class LabDecodeMetadata(FlashAttentionMetadata):
    lab_num_splits: int = 1
    lab_partial_out: torch.Tensor | None = None
    lab_partial_lse: torch.Tensor | None = None
    # FlashAttention metadata for the rows after the decodes, or None when every row is a decode.
    lab_prefill: FlashAttentionMetadata | None = None


class LabDecodeMetadataBuilder(FlashAttentionMetadataBuilder):
    def __init__(self, kv_cache_spec, layer_names, vllm_config, device) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        # Decode requests (query length 1) come first in every batch.
        self._init_reorder_batch_threshold(1)

        self.lab_num_kv_heads = kv_cache_spec.num_kv_heads
        self.lab_block_size = kv_cache_spec.block_size
        self.lab_queries_per_kv = self.num_heads_q // self.lab_num_kv_heads
        self.lab_workspace_pairs = (
            vllm_config.scheduler_config.max_num_seqs * WORKSPACE_SPLITS_PER_SEQ
        )
        self.lab_partial_out = torch.empty(
            self.lab_workspace_pairs * self.num_heads_q * HEAD_DIM,
            dtype=torch.float32,
            device=device,
        )
        self.lab_partial_lse = torch.empty(
            self.lab_workspace_pairs * self.num_heads_q, dtype=torch.float32, device=device
        )
        self.lab_splits_by_decodes: dict[int, int] = {}

    def _lab_num_splits(self, num_decodes: int) -> int:
        # Depends only on the decode count, so a captured CUDA graph stays valid for any context
        # length: the kernel reads each sequence's length from device memory.
        splits = self.lab_splits_by_decodes.get(num_decodes)

        if splits is None:
            splits = choose_num_splits(
                num_decodes,
                self.lab_num_kv_heads,
                device=self.device,
                queries_per_cta=self.lab_queries_per_kv,
                tensor_cores=True,
                block_size=self.lab_block_size,
            )
            splits = max(1, min(splits, self.lab_workspace_pairs // num_decodes))
            self.lab_splits_by_decodes[num_decodes] = splits

        return splits

    def build(self, common_prefix_len, common_attn_metadata, fast_build: bool = False):
        base = super().build(common_prefix_len, common_attn_metadata, fast_build)
        fields = {field.name: getattr(base, field.name) for field in dataclasses.fields(base)}

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(common_attn_metadata, decode_threshold=1)
        )
        fields.update(
            num_decode_reqs=num_decodes,
            num_prefill_reqs=num_prefills,
            num_decode_tokens=num_decode_tokens,
            num_prefill_tokens=num_prefill_tokens,
        )

        if num_decodes == 0:
            return LabDecodeMetadata(**fields)

        splits = self._lab_num_splits(num_decodes)
        pairs = num_decodes * splits
        heads = self.num_heads_q
        partial_out = self.lab_partial_out[: pairs * heads * HEAD_DIM].view(
            num_decodes, heads, splits, HEAD_DIM
        )
        partial_lse = self.lab_partial_lse[: pairs * heads].view(num_decodes, heads, splits)

        prefill = None

        if num_prefills > 0:
            prefill = dataclasses.replace(
                base,
                num_actual_tokens=base.num_actual_tokens - num_decode_tokens,
                query_start_loc=base.query_start_loc[num_decodes:] - num_decode_tokens,
                seq_lens=base.seq_lens[num_decodes:],
                block_table=base.block_table[num_decodes:],
                slot_mapping=base.slot_mapping[num_decode_tokens:],
            )

        return LabDecodeMetadata(
            **fields,
            lab_num_splits=splits,
            lab_partial_out=partial_out,
            lab_partial_lse=partial_lse,
            lab_prefill=prefill,
        )


class LabDecodeAttentionImpl(FlashAttentionImpl):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.lab_enabled = (
            self.attn_type == AttentionType.DECODER
            and self.head_size == HEAD_DIM
            and self.kv_cache_dtype == "auto"
            and self.alibi_slopes is None
            and tuple(self.sliding_window) == (-1, -1)
            and not self.logits_soft_cap
            and self.sinks is None
            and self.num_queries_per_kv <= MAX_QUERIES_PER_KV
            and getattr(self, "dcp_world_size", 1) == 1
        )

        if self.lab_enabled:
            load_extension()

        logger.info_once(
            "Lab decode attention %s (heads %d/%d, head_size %d, kv_cache_dtype %s)",
            "enabled" if self.lab_enabled else "disabled; using FlashAttention-2",
            self.num_heads,
            self.num_kv_heads,
            self.head_size,
            self.kv_cache_dtype,
        )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            not self.lab_enabled
            or not isinstance(attn_metadata, LabDecodeMetadata)
            or attn_metadata.num_decode_reqs == 0
            or output_scale is not None
            or output_block_scale is not None
            or query.dtype != torch.bfloat16
        ):
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )

        num_decodes = attn_metadata.num_decode_reqs

        if attn_metadata.lab_prefill is not None:
            logger.info_once("Lab decode kernel ran on a mixed decode + prefill batch")
        else:
            logger.info_once("Lab decode kernel ran on a decode-only batch")

        # (B, H, N, 2*D) -> ((B, N, H, D), (B, N, H, D)), the same views FlashAttention reads.
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)

        torch.ops.inference_performance_lab.paged_decode_attention(
            output[:num_decodes],
            attn_metadata.lab_partial_out,
            attn_metadata.lab_partial_lse,
            query[:num_decodes],
            key_cache,
            value_cache,
            attn_metadata.block_table[:num_decodes],
            attn_metadata.seq_lens[:num_decodes],
            self.scale,
            self.num_queries_per_kv,
            attn_metadata.lab_num_splits,
            True,
        )

        if attn_metadata.lab_prefill is not None:
            super().forward(
                layer,
                query[num_decodes:],
                key[num_decodes:],
                value[num_decodes:],
                kv_cache,
                attn_metadata.lab_prefill,
                output[num_decodes:],
            )

        return output


class LabDecodeAttentionBackend(FlashAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type[LabDecodeAttentionImpl]:
        return LabDecodeAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[LabDecodeMetadataBuilder]:
        return LabDecodeMetadataBuilder
