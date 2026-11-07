# CUSTOM serving profile, cell 1024:128:128

NVIDIA A100-SXM4-40GB, driver 620.70.06, vLLM 0.33.0, NVIDIA Nsight Systems version 2025.3.2.474-253236389321v0

- recorded window: 22361 ms; benchmark output throughput under tracing: 1585.2 tok/s
- GPU busy 99.5% of the window (gaps under 10 µs count as busy)
- lab decode kernels: 7.5% of GPU kernel time
- FlashAttention-2 kernels: 2.3% of GPU kernel time
- scheduler CPU time: 3.6% of the window; while the GPU was idle: 0.4 ms (0.00% of the window)

| range | calls | total ms | median µs | p90 µs | p99 µs | max µs |
|---|---:|---:|---:|---:|---:|---:|
| scheduler.schedule | 459 | 529.4 | 1473.9 | 1957.7 | 2078.2 | 2269.3 |
| scheduler.update_from_output | 458 | 268.2 | 698.8 | 1085.6 | 1369.7 | 1469.8 |
| block_pool.get_new_blocks | 2446 | 9.3 | 2.3 | 9.7 | 16.2 | 30.2 |
| block_pool.free_blocks | 257 | 8.9 | 31.4 | 45.6 | 81.2 | 85.0 |

| kernel | share | total ms | count |
|---|---:|---:|---:|
| `ampere_bf16_s16816gemm_bf16_256x128_ldg8_f2f_stages_64x3_tn` | 62.7% | 13900.7 | 13784 |
| `paged_decode_attention_tc_kernel` | 7.4% | 1644.8 | 12740 |
| `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn` | 6.0% | 1328.8 | 11040 |
| `ampere_bf16_s16816gemm_bf16_256x128_ldg8_relu_f2f_stages_64x3_tn` | 5.1% | 1126.6 | 3892 |
| `ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_tn` | 4.9% | 1093.9 | 3724 |
| `triton_poi_fused_mul_silu_slice_1` | 3.1% | 684.8 | 12796 |
| `flash_fwd_splitkv_kernel` | 2.3% | 518.6 | 3920 |
| `ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_64x3_tn` | 2.2% | 486.2 | 3892 |
| `triton_red_fused_fused_add_rms_norm_2` | 0.9% | 205.7 | 12796 |
| `ampere_bf16_s16816gemm_bf16_128x64_ldg8_relu_f2f_stages_64x4_tn` | 0.8% | 169.0 | 4340 |
| `triton_red_fused_fused_add_rms_norm_0` | 0.7% | 152.0 | 12796 |
| `ampere_bf16_s16816gemm_bf16_256x64_ldg8_f2f_stages_64x3_tn` | 0.6% | 141.3 | 749 |
