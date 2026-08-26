# Custom Fused RMSNorm in vLLM on NVIDIA A100

## Summary

A custom CUDA fused residual-add and RMSNorm kernel was implemented for the
Qwen2.5-7B hidden size of 3584, integrated into vLLM through its IR operation
dispatch system, and evaluated against vLLM's CUDA implementation.

The custom packed, register-cached kernel was approximately 1.14–1.16x faster
for small and medium standalone workloads and approximately 1.02x faster at
2,048 tokens. Nsight Systems confirmed 6,104 invocations of the custom kernel
during real vLLM serving.

End-to-end serving throughput did not materially improve. The custom RMSNorm
accounted for only 1.5% of traced GPU kernel time, making the expected
application-level improvement very small under this workload.

## Environment

- GPU: NVIDIA A100-SXM4-40GB
- Model: Qwen/Qwen2.5-7B-Instruct
- Precision: BF16
- Hidden size: 3584
- vLLM: 0.27.1
- PyTorch: 2.13.0+cu130
- CUDA toolkit/build: 13.0
- Maximum model length: 4096
- Prefix caching: disabled

## Kernel design

The specialized CUDA kernel uses:

- One CUDA block per token row
- 256 threads per block
- 16-byte packed loads containing eight BF16 elements
- Register caching of the fused residual values
- FP32 sum-of-squares accumulation
- Warp-shuffle and shared-memory block reduction
- A specialization for hidden size 3584
- Scalar/generic fallback for unsupported layouts
- In-place output and residual updates matching vLLM semantics

Correctness was tested against a PyTorch reference and vLLM across token counts
1, 8, 128, and 2048, including aligned and deliberately misaligned tensors.

## Standalone kernel benchmark

Representative repeated measurements showed:

| Tokens | Approximate custom speedup over vLLM |
|---:|---:|
| 1 | 1.14x |
| 8 | 1.16x |
| 128 | 1.15x |
| 2048 | 1.02x |

One anomalous vLLM measurement at eight tokens was excluded from this summary;
the raw repeated measurements remain available in `results/kernels/`.

## vLLM integration

The implementation is registered as the `inference_lab` provider for vLLM's
`fused_add_rms_norm` IR operation. The serving script explicitly selects either
the custom provider or `vllm_c`, allowing both implementations to be tested
through the same model-serving path.

## End-to-end serving results

Each configuration used 64 requests and three repetitions. Traffic was issued
at infinite request rate with fixed input/output lengths.

Percentage changes below compare `inference_lab` against `vllm_c`. Positive
throughput is better; negative latency is better.

| Input/output | Concurrency | Output throughput | Mean TTFT | Mean TPOT | Mean E2E |
|---|---:|---:|---:|---:|---:|
| 256/128 | 1 | -0.25% | -0.04% | +0.25% | +0.25% |
| 256/128 | 8 | -0.42% | -2.08% | +0.60% | +0.41% |
| 256/128 | 32 | -0.16% | -2.95% | +0.68% | +0.16% |
| 2048/32 | 1 | -0.06% | -0.52% | +0.27% | +0.06% |
| 2048/32 | 8 | +0.01% | -1.23% | +0.74% | -0.02% |
| 2048/32 | 32 | +0.04% | -0.07% | -0.04% | -0.05% |

The differences are small and mixed. They should not be interpreted as a
meaningful end-to-end serving speedup.

A matched `vllm_c` trace contained the same 6,104 RMSNorm launches but used
50.373 ms of total GPU time, averaging 8.252 us with a median of 3.456 us.
The custom implementation therefore consumed 2.72% more aggregate RMSNorm GPU
time in this serving workload, despite outperforming vLLM in the standalone
microbenchmark. This indicates that the isolated token-count benchmark did not
fully represent the dynamic shapes or execution conditions encountered during
serving.

## Production-path profiling evidence

Nsight Systems captured both providers under the same prefill-heavy vLLM
serving workload.

| Provider | Launches | Total GPU time | Average | Median | Kernel-time share |
|---|---:|---:|---:|---:|---:|
| `vllm_c` | 6,104 | 50.373 ms | 8.252 us | 3.456 us | 1.5% |
| `inference_lab` | 6,104 | 51.742 ms | 8.477 us | 3.904 us | 1.5% |

The custom trace contained:

`fused_add_rms_norm_bf16_packed_cached_kernel<3584>`

This confirms that vLLM selected and executed the custom CUDA kernel through
the production serving path rather than only in a microbenchmark.

The custom implementation consumed 2.72% more aggregate RMSNorm GPU time in
the matched serving workload, despite outperforming vLLM in the standalone
microbenchmark. The isolated benchmark therefore did not fully represent the
dynamic shapes or execution conditions encountered during serving.

## Interpretation

The standalone optimization is real, but RMSNorm is too small a fraction of
the tested serving workload to substantially change application throughput.

Using the observed 1.5% kernel-time share, even if the approximately 15%
standalone RMSNorm speedup had transferred perfectly to serving, Amdahl's law
would limit the overall improvement to roughly 0.2%, before considering CPU
scheduling, launch overhead, attention, GEMMs, and other serving costs.

The main project outcome is therefore not an unsupported end-to-end speedup
claim. It is a reproducible demonstration of:

1. Identifying and implementing a GPU kernel optimization
2. Establishing numerical correctness
3. Measuring standalone kernel performance
4. Integrating a custom implementation into vLLM dispatch
5. Proving production-path execution with a system profiler
6. Distinguishing kernel-level gains from application-level impact

## Limitations

- Results use one A100 GPU and one model hidden size.
- Workloads are synthetic and use fixed token lengths.
- Each serving configuration has three repetitions.
- Provider runs were not interleaved, so temporal system effects may remain.
- The custom specialization only targets BF16 hidden size 3584.
- The trace reports kernel-time share for one controlled workload, not every
  possible serving workload.
  