# Inference Performance Lab

A reproducible lab for LLM inference serving, profiling, benchmarking, and custom GPU-kernel optimization.

## Initial target

- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Temporary validation model: `Qwen/Qwen2.5-7B-Instruct` while Llama access is pending
- Serving engine: vLLM
- Primary cloud baseline: NVIDIA A100-SXM4-40GB
- Secondary GPU comparisons: added later using identical model precision and workload settings
- Package management: uv with a committed lockfile

## Setup

Install [uv](https://docs.astral.sh/uv/), then run:

```bash
./scripts/bootstrap.sh
./scripts/doctor.sh


```

Run project commands without manually activating the environment:

```bash
uv run --locked <command>
```

## Reproducibility

The repository tracks source code, scripts, configuration, and `uv.lock`. Virtual environments, credentials, model weights, caches, and large raw profiler artifacts are intentionally excluded.

Provider storage is treated as a working cache. Git and external artifact storage are the durable sources of truth.

## Roadmap

1. Launch a reproducible vLLM baseline.
2. Define fixed prompt/output workloads.
3. Measure TTFT, inter-token latency, throughput, and memory use.
4. Profile bottlenecks with Nsight Systems and Nsight Compute.
5. Replace profiler-selected operations with custom CUDA or Triton kernels.
6. Publish controlled before/after benchmarks.
7. Extend to additional GPU architectures and multi-GPU parallelism.
