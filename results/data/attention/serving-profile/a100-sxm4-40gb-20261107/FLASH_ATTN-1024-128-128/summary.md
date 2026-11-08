# FLASH_ATTN serving profile, cell 1024:128:128

NVIDIA A100-SXM4-40GB, driver 620.70.06, vLLM 0.33.0, NVIDIA Nsight Systems version 2025.3.2.474-253236389321v0

- recorded window: 25509 ms; benchmark output throughput under tracing: 1378.5 tok/s
- GPU busy 99.6% of the window (gaps under 10 µs count as busy)
- lab decode kernels: 0.0% of GPU kernel time
- FlashAttention-2 kernels: 20.8% of GPU kernel time
- scheduler CPU time: 3.2% of the window; while the GPU was idle: 0.4 ms (0.00% of the window)

| range | calls | total ms | median µs | p90 µs | p99 µs | max µs |
|---|---:|---:|---:|---:|---:|---:|
| scheduler.schedule | 459 | 527.2 | 1452.5 | 1929.4 | 2054.4 | 5206.8 |
| scheduler.update_from_output | 458 | 279.2 | 705.0 | 1084.5 | 1360.3 | 3480.6 |
| block_pool.get_new_blocks | 2446 | 9.2 | 2.4 | 9.3 | 15.9 | 41.4 |
| block_pool.free_blocks | 257 | 9.4 | 34.0 | 46.2 | 72.4 | 74.3 |

| kernel | share | total ms | count |
|---|---:|---:|---:|
| `ampere_bf16_s16816gemm_bf16_256x128_ldg8_f2f_stages_64x3_tn` | 54.9% | 13899.4 | 13784 |
| `flash_fwd_splitkv_kernel` | 20.6% | 5217.3 | 12796 |
| `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn` | 5.3% | 1329.6 | 11040 |
| `ampere_bf16_s16816gemm_bf16_256x128_ldg8_relu_f2f_stages_64x3_tn` | 4.5% | 1126.6 | 3892 |
| `ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_tn` | 4.3% | 1094.0 | 3724 |
| `triton_poi_fused_mul_silu_slice_1` | 2.7% | 684.8 | 12796 |
| `ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_64x3_tn` | 1.9% | 485.2 | 3892 |
| `triton_red_fused_fused_add_rms_norm_2` | 0.8% | 205.6 | 12796 |
| `ampere_bf16_s16816gemm_bf16_128x64_ldg8_relu_f2f_stages_64x4_tn` | 0.7% | 167.4 | 4340 |
| `triton_red_fused_fused_add_rms_norm_0` | 0.6% | 148.5 | 12796 |
| `ampere_bf16_s16816gemm_bf16_256x64_ldg8_f2f_stages_64x3_tn` | 0.6% | 141.5 | 749 |
| `ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x4_tn` | 0.5% | 129.6 | 4340 |
