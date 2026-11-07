| report | kernel | grid | duration | DRAM % | SM % | issue % | regs | occ. theory % | occ. achieved % | waves | L2 hit % | cycles/issue |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a100-sxm4-40gb-lab-gqa-split-tc-b1-c16384 | `paged_decode_attention_tc_kernel` | 4,1,45 | 41.5 | 55.78 | 11.32 | 11.32 | 142 | 18.75 | 9.55 | 0.56 | 10.25 | 11.42 |
| a100-sxm4-40gb-lab-gqa-split-tc-b1-c16384 | `combine_splits_kernel` | 28,1,1 | 19.0 | 2.24 | 1.28 | 1.28 | 32 | 100 | 6.16 | 0.02 | 23.95 | 16.71 |
| a100-sxm4-40gb-vllm-fa2-b1-c16384 | `flash_fwd_splitkv_kernel` | 1,43,4 | 40.9 | 56.50 | 21.80 | 13.27 | 252 | 12.50 | 9.44 | 0.80 | 10.50 | 8.87 |
| a100-sxm4-40gb-vllm-fa2-b1-c16384 | `flash_fwd_splitkv_combine_kernel` | 7,1,1 | 30.1 | 1.38 | 0.26 | 0.26 | 58 | 50 | 6.29 | 0.01 | 29.21 | 22.88 |
| a100-sxm4-40gb-flashinfer-tensor-cores-b1-c16384 | `BatchPrefillWithPagedKVCacheKernel` | 54,1,4 | 41.7 | 55.33 | 13.85 | 13.85 | 180 | 12.50 | 11.66 | 1 | 9.13 | 11.95 |
| a100-sxm4-40gb-flashinfer-tensor-cores-b1-c16384 | `PersistentVariableLengthMergeStatesKernel` | 108,1,1 | 8.5 | 3.13 | 2.81 | 2.81 | 47 | 62.50 | 6.15 | 0.10 | 36.47 | 10.87 |
| a100-sxm4-40gb-lab-gqa-split-tc-b16-c16384 | `paged_decode_attention_tc_kernel` | 4,16,3 | 411.2 | 84.37 | 12.11 | 12.11 | 142 | 18.75 | 11.06 | 0.59 | 24.21 | 13.96 |
| a100-sxm4-40gb-lab-gqa-split-tc-b16-c16384 | `combine_splits_kernel` | 28,16,1 | 6.5 | 7.03 | 11.99 | 11.99 | 32 | 100 | 23.93 | 0.26 | 30.73 | 19.83 |
| a100-sxm4-40gb-vllm-fa2-b16-c16384 | `flash_fwd_splitkv_kernel` | 1,3,64 | 410.1 | 84.65 | 34.59 | 16.58 | 252 | 12.50 | 11.24 | 0.89 | 24.19 | 9.77 |
| a100-sxm4-40gb-vllm-fa2-b16-c16384 | `flash_fwd_splitkv_combine_kernel` | 112,1,1 | 10.1 | 4.57 | 6.27 | 6.27 | 56 | 56.25 | 6.43 | 0.12 | 34.22 | 12.28 |
| a100-sxm4-40gb-flashinfer-tensor-cores-b16-c16384 | `BatchPrefillWithPagedKVCacheKernel` | 54,1,4 | 409.2 | 84.78 | 11.22 | 11.22 | 180 | 12.50 | 12.39 | 1 | 23.67 | 14.55 |
| a100-sxm4-40gb-flashinfer-tensor-cores-b16-c16384 | `PersistentVariableLengthMergeStatesKernel` | 540,1,1 | 7.9 | 3.02 | 25.68 | 25.68 | 47 | 62.50 | 24.89 | 0.50 | 58.14 | 10.13 |
| a100-sxm4-40gb-lab-gqa-split-tc-b64-c4096 | `paged_decode_attention_tc_kernel` | 4,64,1 | 400.6 | 86.65 | 12.53 | 12.53 | 142 | 18.75 | 14.72 | 0.79 | 24.20 | 17.72 |
| a100-sxm4-40gb-lab-gqa-split-tc-b64-c4096 | `paged_decode_attention_tc_kernel` | 4,64,1 | 403.1 | 86.13 | 12.45 | 12.45 | 142 | 18.75 | 14.65 | 0.79 | 24.18 | 17.68 |
| a100-sxm4-40gb-vllm-fa2-b64-c4096 | `flash_fwd_splitkv_kernel` | 1,64,4 | 553.3 | 62.79 | 25.64 | 15.14 | 178 | 12.50 | 12.05 | 1.19 | 24.11 | 8.72 |
| a100-sxm4-40gb-vllm-fa2-b64-c4096 | `flash_fwd_splitkv_kernel` | 1,64,4 | 554.0 | 62.71 | 25.61 | 15.13 | 178 | 12.50 | 12.01 | 1.19 | 24.10 | 8.69 |
| a100-sxm4-40gb-flashinfer-tensor-cores-b64-c4096 | `BatchPrefillWithPagedKVCacheKernel` | 91,1,4 | 458.7 | 75.75 | 10.08 | 10.08 | 180 | 12.50 | 12.32 | 1.69 | 23.67 | 14.92 |
| a100-sxm4-40gb-flashinfer-tensor-cores-b64-c4096 | `PersistentVariableLengthMergeStatesKernel` | 1080,1,1 | 7.1 | 4.34 | 16.43 | 16.43 | 47 | 62.50 | 48.17 | 1 | 67.41 | 30.16 |
