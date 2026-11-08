"""Accuracy of every prefill implementation against the FP32 reference, shape by shape.

Errors are measured on the BF16 outputs against flash_prefill_attention_reference (FP32 math,
FP32 result):
- max abs: max |out - ref| over the whole output;
- max rel: max abs / max |ref|, the error relative to the output's scale;
- pass: every element satisfies |out - ref| <= atol + rtol * |ref| with the test tolerances
  (atol 1e-2, rtol 1.6e-2). Failing shapes stay in the table.

FlashAttention-2 (vllm_flash_attn) and SDPA's flash backend run on the same inputs, so the lab
kernels' errors can be read against two production kernels with the same BF16 rounding of P.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from benchmark_paged_decode_attention import environment
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from inference_performance_lab.kernels.extension import (
    PREFILL_DEFAULT_VARIANT,
    PREFILL_VARIANTS,
    flash_prefill_attention,
)
from inference_performance_lab.kernels.reference import flash_prefill_attention_reference

HEAD_DIM = 128
ATOL = 1e-2
RTOL = 1.6e-2
LENGTHS = [1, 15, 16, 17, 63, 64, 65, 127, 128, 129, 1000, 4096, 8192]
HEAD_LAYOUTS = [(28, 4), (12, 2), (16, 2), (32, 8), (8, 8)]
RAGGED = [1, 15, 16, 17, 63, 64, 65, 127, 128, 129, 1000, 4096]


@dataclass
class Case:
    name: str
    seq_lens: list[int]
    num_q_heads: int
    num_kv_heads: int
    fused_qkv: bool = False
    causal: bool = True
    query_scale: float = 1.0
    softmax_scale: float = HEAD_DIM**-0.5


@dataclass
class Error:
    case: str
    implementation: str
    max_abs: float
    max_rel: float
    passed: bool


def cases() -> list[Case]:
    result = [Case(f"len {length}, 28/4", [length], 28, 4) for length in LENGTHS]
    result += [Case(f"ragged x{len(RAGGED)}, {q}/{kv}", RAGGED, q, kv) for q, kv in HEAD_LAYOUTS]
    result.append(Case("ragged, fused-QKV strides, 28/4", RAGGED, 28, 4, fused_qkv=True))
    result.append(Case("ragged, non-causal, 28/4", [1000, 1, 4096, 300, 64], 28, 4, causal=False))
    result.append(
        Case(
            "large scores (q x 30, scale 1), 28/4",
            [500, 2000, 129],
            28,
            4,
            query_scale=30.0,
            softmax_scale=1.0,
        )
    )
    return result


def make_inputs(case: Case, seed: int = 0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    total = sum(case.seq_lens)
    q_width, kv_width = case.num_q_heads * HEAD_DIM, case.num_kv_heads * HEAD_DIM

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16)

    if case.fused_qkv:
        query, key, value = randn(total, q_width + 2 * kv_width).split(
            [q_width, kv_width, kv_width], dim=-1
        )
    else:
        query, key, value = randn(total, q_width), randn(total, kv_width), randn(total, kv_width)

    query = query.view(total, case.num_q_heads, HEAD_DIM)
    key = key.view(total, case.num_kv_heads, HEAD_DIM)
    value = value.view(total, case.num_kv_heads, HEAD_DIM)

    if case.query_scale != 1.0:
        query = query * case.query_scale

    cu_seqlens = torch.tensor(
        [0] + torch.tensor(case.seq_lens).cumsum(0).tolist(), dtype=torch.int32, device="cuda"
    )
    return query, key, value, cu_seqlens


def run_fa2(case: Case, query, key, value, cu_seqlens) -> torch.Tensor:
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    return flash_attn_varlen_func(
        q=query,
        k=key,
        v=value,
        max_seqlen_q=max(case.seq_lens),
        cu_seqlens_q=cu_seqlens,
        max_seqlen_k=max(case.seq_lens),
        cu_seqlens_k=cu_seqlens,
        softmax_scale=case.softmax_scale,
        causal=case.causal,
        fa_version=2,
    )


def run_sdpa(case: Case, query, key, value, cu_seqlens) -> torch.Tensor:
    out = torch.empty_like(query)
    bounds = cu_seqlens.tolist()

    for start, end in itertools.pairwise(bounds):
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out[start:end] = F.scaled_dot_product_attention(
                query[start:end].transpose(0, 1)[None],
                key[start:end].transpose(0, 1)[None],
                value[start:end].transpose(0, 1)[None],
                is_causal=case.causal,
                scale=case.softmax_scale,
                enable_gqa=True,
            )[0].transpose(0, 1)

    return out


def measure(name: str, implementation: str, actual: torch.Tensor, expected: torch.Tensor) -> Error:
    difference = (actual.float() - expected).abs()
    max_abs = float(difference.max())
    passed = bool((difference <= ATOL + RTOL * expected.abs()).all()) and bool(
        torch.isfinite(actual).all()
    )
    return Error(name, implementation, max_abs, max_abs / float(expected.abs().max()), passed)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--variants", nargs="+", choices=PREFILL_VARIANTS, default=PREFILL_VARIANTS)
    parser.add_argument(
        "--output", type=Path, help="write one JSON line per (case, implementation)"
    )
    args = parser.parse_args()

    info = environment()
    errors: list[Error] = []

    for case in cases():
        inputs = make_inputs(case)
        expected = flash_prefill_attention_reference(
            *inputs, case.softmax_scale, causal=case.causal
        )

        for variant in args.variants:
            actual = flash_prefill_attention(
                *inputs[:4],
                max(case.seq_lens),
                case.softmax_scale,
                causal=case.causal,
                variant=variant,
            )
            errors.append(measure(case.name, f"lab-{variant}", actual, expected))

        errors.append(measure(case.name, "vllm-fa2", run_fa2(case, *inputs), expected))
        errors.append(measure(case.name, "sdpa-flash", run_sdpa(case, *inputs), expected))
        del inputs, expected
        torch.cuda.empty_cache()

    shown = [f"lab-{PREFILL_DEFAULT_VARIANT}", "vllm-fa2", "sdpa-flash"]
    print("# " + ", ".join(f"{key}={value}" for key, value in info.items()))
    print(
        f"# errors vs the FP32 reference; pass = |out - ref| <= {ATOL} + {RTOL} * |ref| everywhere"
    )
    print()
    print("| case | " + " | ".join(f"{name} max abs / max rel / pass" for name in shown) + " |")
    print("|---|" + "---|" * len(shown))

    for case in cases():
        cells = []

        for name in shown:
            error = next(e for e in errors if e.case == case.name and e.implementation == name)
            cells.append(
                f"{error.max_abs:.2e} / {error.max_rel:.2e} / {'pass' if error.passed else 'FAIL'}"
            )

        print(f"| {case.name} | " + " | ".join(cells) + " |")

    print()
    print("| implementation | worst max abs | worst max rel | shapes passed |")
    print("|---|---:|---:|---:|")

    for name in [f"lab-{variant}" for variant in args.variants] + ["vllm-fa2", "sdpa-flash"]:
        mine = [e for e in errors if e.implementation == name]
        print(
            f"| {name} | {max(e.max_abs for e in mine):.2e} | {max(e.max_rel for e in mine):.2e} | "
            f"{sum(e.passed for e in mine)}/{len(mine)} |"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            "".join(json.dumps({**asdict(e), "env": info}) + "\n" for e in errors)
        )


if __name__ == "__main__":
    main()
