| metric | cublas-gemm-8192 | final-skip_masked-1x8192 |
|---|---:|---:|
| kernel time (µs, ncu-locked clock) | 4,780 | 3,456 |
| SM clock (MHz) | 1,094 | 1,093 |
| SMs busy (% of elapsed cycles) | 99.23 | 99.71 |
| registers / thread | 226 | 197 |
| shared memory / block (bytes, incl. 1 KB reserved) | 148,480 | 50,176 |
| block limit: registers | 1.00 | 2.00 |
| block limit: shared memory | 1.00 | 3.00 |
| theoretical occupancy (%) | 12.50 | 12.50 |
| achieved occupancy (%) | 12.50 | 12.47 |
| tensor pipe (HMMA) active (% of peak, SM-active cycles) | 95.82 | 57.84 |
| tensor pipe (HMMA) active (% of peak, elapsed cycles) | 95.07 | 57.67 |
| warp instructions executed | 441,368,576 | 577,385,984 |
| XU pipe (MUFU.EX2) (% of peak) | 0.00 | 15.51 |
| FMA pipe (% of peak) | 0.21 | 19.75 |
| ALU pipe (% of peak) | 4.32 | 17.71 |
| shared load bank conflicts | 0.00 | 164,902 |
| shared store bank conflicts | 0.00 | 10,355 |
| shared load wavefronts | 271,062,102 | 236,427,717 |
| DRAM read (MB) | 1,300 | 94.90 |
| L2 → SM read (MB) | 12,885 | 7,634 |
| L2 hit rate (%) | 83.83 | 97.94 |
| stall: math_pipe_throttle (warps per issue) | 4.32 | 0.25 |
| stall: wait (warps per issue) | 4.05 | 2.60 |
| stall: barrier (warps per issue) | 0.02 | 0.39 |
| stall: short_scoreboard (warps per issue) | 0.02 | 0.56 |
| stall: long_scoreboard (warps per issue) | 0.04 | 0.04 |
| stall: mio_throttle (warps per issue) | 0.02 | 0.17 |
| stall: not_selected (warps per issue) | 0.61 | 0.20 |
| stall: dispatch_stall (warps per issue) | 0.00 | 0.16 |
| stall: lg_throttle (warps per issue) | 0.01 | 0.03 |
| stall: no_instruction (warps per issue) | 0.02 | 0.13 |
| stall: branch_resolving (warps per issue) | 0.03 | 0.10 |

Executed SASS by opcode (warp-level instructions, millions; opcodes above 0.5% of the largest total):

| opcode | cublas-gemm-8192 | final-skip_masked-1x8192 |
|---|---:|---:|
| HMMA | 268.4 | 117.7 |
| LDSM | 67.2 | 58.9 |
| FMUL | 2.1 | 61.9 |
| IADD3 | 28.2 | 39.8 |
| IMAD | 0.2 | 35.8 |
| FMNMX | 0.0 | 35.1 |
| FFMA | 0.0 | 33.3 |
| MUFU | 0.0 | 31.5 |
| FADD | 0.0 | 29.7 |
| LDGSTS | 25.6 | 14.9 |
| LOP3 | 1.2 | 22.0 |
| F2FP | 1.0 | 15.3 |
| BRA | 2.7 | 12.6 |
| ISETP | 5.7 | 10.5 |
| LEA | 1.6 | 9.0 |
| SHF | 0.6 | 7.1 |
| R2P | 6.7 | 0.0 |
| UMOV | 6.4 | 1.8 |
| LDS | 0.5 | 5.6 |
| UISETP | 4.3 | 0.0 |
| UIADD3 | 4.2 | 0.2 |
| WARPSYNC | 0.0 | 3.9 |
| BSYNC | 0.0 | 3.8 |
| BSSY | 0.0 | 3.8 |
| SHFL | 0.0 | 3.8 |
| USHF | 0.0 | 3.7 |
| total | 441.4 | 577.4 |

- cublas-gemm-8192: `ampere_bf16_s16816gemm_bf16_256x128_ldg8_f2f_stages_64x3_nn`
- final-skip_masked-1x8192: `void <unnamed>::flash_prefill_mma_kernel<7>(<unnamed>::PrefillParams)`
