# Qwen2.5-7B on A100: Prefix-Cache-Off Analysis

## Environment

- GPU: NVIDIA A100-SXM4-40GB
- Model: Qwen/Qwen2.5-7B-Instruct
- Revision: `a09a35458c702b33eeacc393d103063234e8bc28`
- Precision: BF16
- vLLM: 0.27.1
- PyTorch: 2.13.0+cu130
- Maximum model length: 4096
- Prefix caching: explicitly disabled

## Benchmark protocol

- Workloads: 256 input / 128 output and 2048 input / 32 output
- Concurrency: 1, 2, 4, 8, 16, 32
- Requests per run: 32
- Repetitions: 3
- Request rate: infinite/burst
- Temperature: 0
- EOS ignored to enforce exact output lengths
- Seed: 0

Full aggregates: [summary](data/prefix-cache-off/summary.md).

## Scaling results

For 256-input/128-output, output throughput increased from 81.306 tok/s at
concurrency 1 to 1839.046 tok/s at concurrency 32: 22.62x scaling versus an
ideal 32x. Mean TPOT increased from 12.190 ms to 14.592 ms, while mean TTFT
increased from 25.892 ms to 346.949 ms.

For 2048-input/32-output, throughput began flattening near concurrency 16.
Concurrency 16 reached 211.835 output tok/s, versus 218.034 at concurrency 32,
a gain of only 2.93%. Over the same step, mean TTFT increased from 842.920 ms
to 2280.549 ms and mean E2E latency from 2368.085 ms to 4485.301 ms.

Concurrency 8 is a reasonable latency-aware operating point for the
prefill-heavy workload; concurrency 16 is approximately the throughput knee.

## Nsight Systems findings

Two bounded Nsight Systems captures compared decode-heavy and prefill-heavy
execution.

- Decode-heavy: the five largest BF16 GEMM rows represented about 89.1% of
  aggregate GPU kernel duration.
- Prefill-heavy: GEMMs represented approximately 87%.
- FlashAttention grew from roughly 4% in decode-heavy execution to about 6.5%
  in prefill-heavy execution.
- Fused add/RMSNorm represented about 1.6% in both traces.
- Fused SwiGLU represented about 0.7% in decode and 2.5% in prefill.
- CUDA graphs were active, confirmed by `cudaGraphLaunch` calls.
- `cudaEventSynchronize` time represents host waiting and is not treated as
  directly removable overhead.

Profiler summaries:

- [Decode-heavy c8](profiling/decode-c8-stats.txt)
- [Prefill-heavy c8](profiling/prefill-c8-stats.txt)

## Optimization decision

The first custom CUDA kernel will be fused residual-add plus RMSNorm. It is a
common inference primitive, appears consistently in both phases, and provides
a bounded introduction to memory-oriented kernel optimization.

Its measured system-level ceiling is small—approximately 1.6% of aggregate
kernel time—so isolated kernel speedups will not be presented as equivalent
end-to-end gains.

Fused SwiGLU is the next candidate. Paged/decode attention is reserved as a
higher-complexity follow-up. Custom GEMM is not the initial target because the
current workload already uses highly optimized Ampere tensor-core kernels.

## Limitations

- One model and one GPU architecture
- Synthetic fixed-length prompts
- Burst traffic rather than arrival-rate sweeps
- No prefix-cache comparison yet
- Nsight runs used fewer requests and instrumentation, so profiler-run latency
  is not compared directly against the uninstrumented baseline