# Qwen2.5-7B vLLM baseline on A100 40 GB

Date: 2026-08-19

Status: Initial single-instance serving baseline. This is not yet a cross-hardware or kernel-optimization comparison.

## Environment

- Provider: Lambda Cloud
- GPU: 1× NVIDIA A100-SXM4-40GB
- Driver: 580.126.20
- CUDA build: 13.0
- Python: 3.12.3
- PyTorch: 2.13.0+cu130
- vLLM: 0.27.1
- Model: `Qwen/Qwen2.5-7B-Instruct`
- Model revision: `a09a35458c702b33eeacc393d103063234e8bc28`
- Precision: BF16
- Maximum model length: 4,096 tokens
- GPU memory utilization setting: 0.90
- Idle initialized GPU memory: 35,967 MiB used, 4,474 MiB free
- API transport: loopback; network latency is excluded

## Workload

Each result is aggregated from four measured runs.

- Backend: OpenAI-compatible completions
- Requests per run: 32
- Input length: exactly 256 tokens
- Output length: exactly 128 tokens
- Temperature: 0
- Ignore EOS: enabled
- Warmup requests per run: 2
- Request rate: unlimited burst
- Maximum concurrency: 1 or 8
- Random seed: 0
- Failed requests: 0

Values below are mean ± sample standard deviation across four runs.

## Results

| Metric | Concurrency 1 | Concurrency 8 |
|---|---:|---:|
| Request throughput (req/s) | 0.638 ± 0.000 | 4.990 ± 0.004 |
| Output throughput (tok/s) | 81.693 ± 0.007 | 638.728 ± 0.539 |
| Mean TTFT (ms) | 20.248 ± 0.133 | 42.385 ± 0.660 |
| P99 TTFT (ms) | 21.259 ± 0.246 | 54.191 ± 0.769 |
| Mean TPOT (ms) | 12.176 ± 0.000 | 12.253 ± 0.002 |
| P99 TPOT (ms) | 12.180 ± 0.001 | 12.274 ± 0.002 |
| P99 ITL (ms) | 12.686 ± 0.031 | 13.124 ± 0.043 |
| Mean end-to-end latency (ms) | 1,566.578 ± 0.127 | 1,598.570 ± 0.698 |
| P99 end-to-end latency (ms) | 1,567.429 ± 0.223 | 1,610.105 ± 0.947 |

## Interpretation

Increasing maximum concurrency from 1 to 8 produced approximately 7.82× output-token throughput. Mean TPOT increased by approximately 0.63%, while mean end-to-end latency increased by approximately 2.04%.

Mean TTFT increased by approximately 2.09× and P99 TTFT by approximately 2.55×. This demonstrates the expected serving tradeoff: continuous batching substantially improves aggregate throughput while increasing request startup latency under burst load.

## Reproduction

Start the server:

```bash
./scripts/bootstrap.sh
./scripts/doctor.sh
./scripts/serve.sh