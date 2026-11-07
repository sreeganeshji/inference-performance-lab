# FLASH_ATTN serving profile, cell 4096:256:64

NVIDIA A100-SXM4-40GB, driver 620.70.06, vLLM 0.33.0, NVIDIA Nsight Systems version 2025.3.2.474-253236389321v0

- recorded window: 59670 ms; benchmark output throughput under tracing: 586.2 tok/s
- GPU busy 99.8% of the window (gaps under 10 µs count as busy)
- lab decode kernels: 0.0% of GPU kernel time
- FlashAttention-2 kernels: 33.7% of GPU kernel time
- scheduler CPU time: 1.8% of the window; while the GPU was idle: 0.6 ms (0.00% of the window)

| range | calls | total ms | median µs | p90 µs | p99 µs | max µs |
|---|---:|---:|---:|---:|---:|---:|
| scheduler.schedule | 909 | 639.8 | 892.0 | 1128.0 | 1325.4 | 4624.7 |
| scheduler.update_from_output | 908 | 420.0 | 525.0 | 783.1 | 1086.6 | 3464.9 |
| block_pool.get_new_blocks | 2467 | 14.4 | 2.7 | 16.8 | 39.1 | 96.4 |
| block_pool.free_blocks | 129 | 20.4 | 141.0 | 166.9 | 204.3 | 1963.7 |

| kernel | share | total ms | count |
|---|---:|---:|---:|
| `ampere_bf16_s16816gemm_bf16_256x128_ldg8_f2f_stages_64x3_tn` | 46.4% | 27551.9 | 31864 |
| `flash_fwd_splitkv_kernel` | 33.5% | 19903.6 | 25396 |
| `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn` | 4.6% | 2757.1 | 22501 |
| `ampere_bf16_s16816gemm_bf16_256x128_ldg8_relu_f2f_stages_64x3_tn` | 3.7% | 2207.7 | 7448 |
| `ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_tn` | 3.6% | 2147.9 | 7420 |
| `triton_poi_fused_mul_silu_slice_1` | 2.2% | 1329.9 | 25396 |
| `ampere_bf16_s16816gemm_bf16_256x64_ldg8_f2f_stages_64x3_tn` | 1.5% | 884.2 | 3354 |
| `triton_red_fused_fused_add_rms_norm_2` | 0.7% | 400.2 | 25396 |
| `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_relu_f2f_stages_64x6_tn` | 0.6% | 337.2 | 9996 |
| `triton_red_fused_fused_add_rms_norm_0` | 0.5% | 286.9 | 25396 |
| `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn` | 0.4% | 265.7 | 10136 |
| `ampere_bf16_s16816gemm_bf16_64x64_ldg8_relu_f2f_stages_64x5_tn` | 0.4% | 240.0 | 7952 |
