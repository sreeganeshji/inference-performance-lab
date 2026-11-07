# CUSTOM serving profile, cell 4096:256:64

NVIDIA A100-SXM4-40GB, driver 620.70.06, vLLM 0.33.0, NVIDIA Nsight Systems version 2025.3.2.474-253236389321v0

- recorded window: 48613 ms; benchmark output throughput under tracing: 729.1 tok/s
- GPU busy 99.8% of the window (gaps under 10 µs count as busy)
- lab decode kernels: 12.6% of GPU kernel time
- FlashAttention-2 kernels: 6.6% of GPU kernel time
- scheduler CPU time: 2.1% of the window; while the GPU was idle: 2.1 ms (0.00% of the window)

| range | calls | total ms | median µs | p90 µs | p99 µs | max µs |
|---|---:|---:|---:|---:|---:|---:|
| scheduler.schedule | 909 | 634.0 | 892.1 | 1117.1 | 1344.0 | 3951.5 |
| scheduler.update_from_output | 908 | 383.9 | 428.5 | 759.4 | 1079.8 | 4065.7 |
| block_pool.get_new_blocks | 2467 | 14.6 | 2.9 | 17.2 | 38.2 | 84.9 |
| block_pool.free_blocks | 129 | 18.5 | 139.2 | 172.5 | 195.2 | 1646.0 |

| kernel | share | total ms | count |
|---|---:|---:|---:|
| `ampere_bf16_s16816gemm_bf16_256x128_ldg8_f2f_stages_64x3_tn` | 56.7% | 27395.5 | 31864 |
| `paged_decode_attention_tc_kernel` | 12.5% | 6048.6 | 25284 |
| `flash_fwd_splitkv_kernel` | 6.6% | 3187.2 | 7476 |
| `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn` | 5.7% | 2753.3 | 22501 |
| `ampere_bf16_s16816gemm_bf16_256x128_ldg8_relu_f2f_stages_64x3_tn` | 4.5% | 2193.8 | 7448 |
| `ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_tn` | 4.4% | 2135.8 | 7420 |
| `triton_poi_fused_mul_silu_slice_1` | 2.8% | 1329.8 | 25396 |
| `ampere_bf16_s16816gemm_bf16_256x64_ldg8_f2f_stages_64x3_tn` | 1.8% | 884.0 | 3354 |
| `triton_red_fused_fused_add_rms_norm_2` | 0.8% | 398.6 | 25396 |
| `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_relu_f2f_stages_64x6_tn` | 0.7% | 337.0 | 9996 |
| `triton_red_fused_fused_add_rms_norm_0` | 0.6% | 288.4 | 25396 |
| `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn` | 0.6% | 266.8 | 10136 |
