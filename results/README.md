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

## Evidence for the main case study

| Engineering result | Where to verify it |
|---|---|
| Packed transfers, register caching, FP32 accumulation, warp reduction | [RMSNorm source](../csrc/fused_add_rms_norm_kernel.cu), [correctness tests](../tests/test_fused_add_rms_norm.py) |
| Approximately 1.14–1.16x RMSNorm speedup at 1–128 rows | [Exact medians and ranges](generated-summary.md#standalone-cuda-kernels), [all kernel runs](kernels/) |
| Five leading BF16 GEMM variants occupy ~89% of decode GPU kernel time | [Baseline analysis](a100-qwen2.5-7b-prefix-cache-off-analysis.md#nsight-systems-findings), [decode trace summary](profiling/decode-c8-stats.txt) |
| Concurrency 16 → 32: +2.9% throughput and 2.7x mean TTFT for 2048/32 requests | [Baseline aggregates](data/prefix-cache-off/summary.md), [raw runs](data/prefix-cache-off/) |
| 6,104 custom launches and no material serving throughput gain | [Provider integration](../src/inference_performance_lab/kernels/vllm_plugin.py), [matched analysis](a100-qwen2.5-7b-custom-rmsnorm-analysis.md), [custom trace](profiling/vllm-rmsnorm-custom-kernels.csv), [baseline trace](profiling/vllm-rmsnorm-baseline-kernels.csv) |

The ~1.6% RMSNorm share comes from the original baseline profiles; the 1.5% figure comes from the later matched-provider
profiles. They describe different captures. Packed SiLU results are an additional standalone experiment and do not
establish an end-to-end serving improvement.

## Artifact map

| Path | Contents |
|---|---|
| [kernels/](kernels/) | Repeated CUDA-event microbenchmarks for RMSNorm and SiLU-and-multiply |
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
- Serving chart points are arithmetic means across three successful runs with 32 requests per run.
- Effective bandwidth in microbenchmark output is based on minimum semantic memory traffic; it is not a claim about
  measured DRAM bytes.
- Raw results are retained even when a measurement is anomalous. Aggregation makes the handling reproducible instead
  of deleting or manually selecting runs.
- The A100 measurements predate the Stable LibTorch ABI migration. The current source has been correctness-tested
  locally, but no new A100 performance measurement is implied by the host API or documentation changes.
