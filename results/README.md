# Results and profiler evidence

This directory contains the evidence behind the repository's performance claims. Serving results use
Qwen2.5-7B-Instruct in BF16 on an NVIDIA A100-SXM4-40GB. Standalone results use synthetic tensors at the model's
dimensions on that GPU.

## Start here

- [Custom fused RMSNorm case study](a100-qwen2.5-7b-custom-rmsnorm-analysis.md): kernel design, standalone results,
  vLLM integration, matched traces, Amdahl's-law interpretation, and limitations.
- [Serving baseline analysis](a100-qwen2.5-7b-prefix-cache-off-analysis.md): workload protocol, concurrency scaling,
  profiler findings, and optimization selection.
- [Generated benchmark summary](generated-summary.md): deterministic medians, ranges, and serving aggregates used by
  the README charts.

## Evidence for the decode-attention case study

All measured on an A100-SXM4-40GB on 2026-11-07 unless marked; the [report](../docs/decode-attention.md) interprets them.

| Engineering result | Where to verify it |
|---|---|
| 125 correctness tests pass | [A100 log](data/attention/a100-sxm4-40gb-tests-20261107-a100-split-model.txt), [tests](../tests/test_paged_decode_attention.py) |
| 1.03–1.48x faster than vLLM's FlashAttention-2 in 12/12 shapes; FlashInfer comparison | [Kernel grid](data/attention/a100-sxm4-40gb-qwen2.5-7b-20261107-a100-split-model.md), [with the 3080 Ti split constants](data/attention/a100-sxm4-40gb-qwen2.5-7b-20261107.md), [RTX 3080 Ti](data/attention/rtx3080ti-qwen2.5-7b-20261107.md) |
| Split model fitted per GPU: mean regret 5.18% → 0.64% on the A100 | [Fit report](data/attention/a100-sxm4-40gb-split-fit-wide-20261107.md), [sweep](data/attention/a100-sxm4-40gb-split-fit-sweep-20261107.md) |
| Greedy output within the divergence between stock backends | [All pairs](data/attention/a100-sxm4-40gb-vllm-check-pairs-20261107.md), [raw outputs](data/attention/a100-sxm4-40gb-vllm-check-outputs-20261107/) |
| 1.01–1.29x serving throughput | [Summary](data/attention/serving/a100-sxm4-40gb-20261107/summary.md), [every run and server log](data/attention/serving/a100-sxm4-40gb-20261107/) |
| Mixed prefill + decode attention 1,967 → ~790 µs per layer; scheduler hidden | [Attention by launch grid](data/attention/serving-profile/a100-sxm4-40gb-20261107/attention-kernels-by-grid.md), [trace summaries](data/attention/serving-profile/a100-sxm4-40gb-20261107/) |
| Wave count, bandwidth and combine regimes against FA2 and FlashInfer | [Nsight Compute summary](data/attention/ncu/a100-sxm4-40gb-decode-ncu-summary.md), [full details](data/attention/ncu/) |

## Evidence for the RMSNorm case study

| Engineering result | Where to verify it |
|---|---|
| Packed transfers, register caching, FP32 accumulation, warp reduction | [RMSNorm source](../csrc/fused_add_rms_norm_kernel.cu), [correctness tests](../tests/test_fused_add_rms_norm.py) |
| Parity with vLLM's kernel under CUDA-graph timing (0.99–1.03x at 1–2,048 rows) | [A100](data/elementwise/a100-sxm4-40gb-graphs-20261107.md), [RTX 3080 Ti](data/elementwise/rtx3080ti-graphs-20261107.md) |
| Superseded: 1.143x / 1.165x / 1.153x at 1 / 8 / 128 rows from eager-launch timing | [Exact medians and ranges](generated-summary.md#standalone-cuda-kernels), [all kernel runs](kernels/) |
| Five leading BF16 GEMM variants occupy ~89% of decode GPU kernel time | [Baseline analysis](a100-qwen2.5-7b-prefix-cache-off-analysis.md#nsight-systems-findings), [decode trace summary](profiling/decode-c8-stats.txt) |
| Concurrency 16 → 32: +2.9% throughput and 2.7x mean TTFT for 2048/32 requests | [Baseline aggregates](data/prefix-cache-off/summary.md), [raw runs](data/prefix-cache-off/) |
| 6,104 custom launches and no material serving throughput gain | [Provider integration](../src/inference_performance_lab/kernels/vllm_plugin.py), [matched analysis](a100-qwen2.5-7b-custom-rmsnorm-analysis.md), [custom trace](profiling/vllm-rmsnorm-custom-kernels.csv), [baseline trace](profiling/vllm-rmsnorm-baseline-kernels.csv) |

The ~1.6% RMSNorm share comes from the original baseline profiles; the 1.5% figure comes from the later matched-provider
profiles. They describe different captures. Packed SiLU results are an additional standalone experiment and do not
establish an end-to-end serving improvement.

## Artifact map

| Path | Contents |
|---|---|
| [data/attention/](data/attention/) | Decode attention: kernel grids, split sweeps and fits, vLLM checks, serving A/B, Nsight Systems and Compute |
| [data/elementwise/](data/elementwise/) | RMSNorm and SiLU-and-multiply against vLLM under CUDA-graph replay |
| [kernels/](kernels/) | Repeated CUDA-event microbenchmarks for RMSNorm and SiLU-and-multiply (eager launches) |
| [data/prefix-cache-off/](data/prefix-cache-off/) | Baseline: 32 requests, concurrency 1–32, three runs per configuration |
| [data/rmsnorm/inference-lab/](data/rmsnorm/inference-lab/) | Custom provider: 64 requests, concurrency 1 / 8 / 32, three runs per configuration |
| [data/rmsnorm/vllm-c/](data/rmsnorm/vllm-c/) | vLLM provider: the same 64-request protocol |
| [profiling/](profiling/) | Nsight Systems kernel summaries, Nsight Compute reports, and extracted CSV data |

## Rebuild the presentation artifacts

From the repository root:

```bash
uv run --locked --no-sync python scripts/generate_readme_assets.py
```

The generator reads the five repeated packed-kernel reports and the baseline serving JSON. It writes this repository's
generated summary plus `docs/assets/kernel-speedups.svg` and `docs/assets/serving-scaling.svg`. It includes no time,
host, or random metadata, so a clean checkout regenerates byte-identical artifacts.

## Interpretation rules

- Standalone speedups are not presented as end-to-end serving speedups.
- Kernel charts use the median across five runs and show the observed minimum-to-maximum range.
- Medians are computed from the recorded three-decimal speedup ratios. The eight-row RMSNorm outlier is retained in
  the range; these are observed ranges, not confidence intervals. Exact medians are 1.143x, 1.165x, and 1.153x for
  1, 8, and 128 rows; the earlier 1.14–1.16x wording is an approximate summary.
- Those kernel runs, the generated summary and `kernel-speedups.svg` come from CUDA events around a Python loop of
  eager launches, which at small shapes includes launch overhead. CUDA-graph replay
  (`benchmarks/benchmark_elementwise_graphs.py`, `data/elementwise/`) supersedes them for kernel comparisons; they are
  kept as the original record.
- Serving chart points are arithmetic means across three successful runs with 32 requests per run.
- Effective bandwidth in microbenchmark output is based on minimum semantic memory traffic; it is not a claim about
  measured DRAM bytes.
- Raw results are retained even when a measurement is anomalous. Aggregation makes the handling reproducible instead
  of deleting or manually selecting runs.
- The A100 measurements predate the Stable LibTorch ABI migration. The current source has been correctness-tested
  locally, but no new A100 performance measurement is implied by the host API or documentation changes.
