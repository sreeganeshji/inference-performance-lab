| metric | cublas-gemm-8192 | final-skip_masked-1x8192 |
|---|---:|---:|
| kernel time (µs, ncu-locked clock) | 16,487 | 7,471 |
| SM clock (MHz) | 1,660 | 1,661 |
| SMs busy (% of elapsed cycles) | 98.44 | 99.92 |
| registers / thread | 234 | 199 |
| shared memory / block (bytes, incl. 1 KB reserved) | 33,792 | 50,176 |
| block limit: registers | 2.00 | 2.00 |
| block limit: shared memory | 3.00 | 2.00 |
| theoretical occupancy (%) | 16.67 | 16.67 |
| achieved occupancy (%) | 16.54 | 16.63 |
| tensor pipe (HMMA) active (% of peak, SM-active cycles) | 49.82 | 47.46 |
| tensor pipe (HMMA) active (% of peak, elapsed cycles) | 49.04 | 47.42 |
| warp instructions executed | 774,946,816 | 577,400,320 |
| XU pipe (MUFU.EX2) (% of peak) | 0.00 | 6.36 |
| FMA pipe (% of peak) | 0.03 | 4.06 |
| ALU pipe (% of peak) | 1.93 | 7.26 |
| shared load bank conflicts | 113,222 | 135,841 |
| shared store bank conflicts | 7,142,694 | 18,548 |
| shared load wavefronts | 271,443,040 | 236,395,452 |
| DRAM read (MB) | 5,438 | 254 |
| L2 → SM read (MB) | 17,179 | 7,634 |
| L2 hit rate (%) | 75.77 | 96.71 |
| stall: math_pipe_throttle (warps per issue) | 14.54 | 7.40 |
| stall: wait (warps per issue) | 4.60 | 2.22 |
| stall: barrier (warps per issue) | 1.21 | 2.11 |
| stall: short_scoreboard (warps per issue) | 0.04 | 0.21 |
| stall: long_scoreboard (warps per issue) | 0.02 | 0.07 |
| stall: mio_throttle (warps per issue) | 0.03 | 0.24 |
| stall: not_selected (warps per issue) | 0.60 | 0.21 |
| stall: dispatch_stall (warps per issue) | 0.00 | 0.05 |
| stall: lg_throttle (warps per issue) | 0.00 | 0.05 |
| stall: no_instruction (warps per issue) | 0.01 | 0.07 |
| stall: branch_resolving (warps per issue) | 0.02 | 0.09 |

Executed SASS by opcode (warp-level instructions, millions; opcodes above 0.5% of the largest total):

| opcode | cublas-gemm-8192 | final-skip_masked-1x8192 |
|---|---:|---:|
| HMMA | 536.9 | 117.7 |
| LDSM | 67.2 | 58.9 |
| FMUL | 2.1 | 61.9 |
| IADD3 | 55.2 | 39.8 |
| IMAD | 0.3 | 35.9 |
| STS | 35.8 | 0.5 |
| FMNMX | 0.0 | 35.1 |
| LDG | 33.8 | 0.0 |
| FFMA | 0.0 | 33.3 |
| MUFU | 0.0 | 31.5 |
| FADD | 0.0 | 29.7 |
| LOP3 | 9.5 | 21.8 |
| F2FP | 1.0 | 15.3 |
| LDGSTS | 0.0 | 14.9 |
| BRA | 4.7 | 12.6 |
| ISETP | 5.6 | 10.5 |
| LEA | 1.5 | 9.0 |
| R2P | 8.8 | 0.0 |
| SHF | 0.6 | 7.1 |
| LDS | 0.8 | 5.6 |
| BAR | 4.4 | 1.8 |
| ULOP3 | 4.2 | 0.0 |
| WARPSYNC | 0.0 | 3.9 |
| total | 774.9 | 577.4 |

- cublas-gemm-8192: `ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_nn`
- final-skip_masked-1x8192: `void <unnamed>::flash_prefill_mma_kernel<7>(<unnamed>::PrefillParams)`
