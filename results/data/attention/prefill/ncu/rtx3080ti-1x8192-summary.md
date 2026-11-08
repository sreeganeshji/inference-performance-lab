| metric | step-3-cp_async | final-skip_masked | vllm-fa2 |
|---|---:|---:|---:|
| kernel time (µs, ncu-locked clock) | 8,405 | 7,471 | 7,904 |
| SM clock (MHz) | 1,640 | 1,661 | 1,657 |
| SMs busy (% of elapsed cycles) | 96.67 | 99.92 | 95.78 |
| registers / thread | 196 | 199 | 191 |
| shared memory / block (bytes, incl. 1 KB reserved) | 50,176 | 50,176 | 50,176 |
| block limit: registers | 2.00 | 2.00 | 2.00 |
| block limit: shared memory | 2.00 | 2.00 | 2.00 |
| theoretical occupancy (%) | 16.67 | 16.67 | 16.67 |
| achieved occupancy (%) | 16.47 | 16.63 | 16.46 |
| tensor pipe (HMMA) active (% of peak, SM-active cycles) | 44.40 | 47.46 | 47.18 |
| tensor pipe (HMMA) active (% of peak, elapsed cycles) | 42.92 | 47.42 | 45.19 |
| warp instructions executed | 784,903,168 | 577,400,320 | 465,792,656 |
| XU pipe (MUFU.EX2) (% of peak) | 5.92 | 6.36 | 6.28 |
| FMA pipe (% of peak) | 7.00 | 4.06 | 3.40 |
| ALU pipe (% of peak) | 11.20 | 7.26 | 3.91 |
| shared load bank conflicts | 143,965 | 135,841 | 158,005 |
| shared store bank conflicts | 23,840 | 18,548 | 50,107 |
| shared load wavefronts | 237,783,877 | 236,395,452 | 266,950,998 |
| DRAM read (MB) | 86.30 | 254 | 195 |
| L2 → SM read (MB) | 7,634 | 7,634 | 7,634 |
| L2 hit rate (%) | 98.88 | 96.71 | 97.48 |
| stall: math_pipe_throttle (warps per issue) | 7.06 | 7.40 | 11.22 |
| stall: wait (warps per issue) | 1.35 | 2.22 | 2.05 |
| stall: barrier (warps per issue) | 0.45 | 2.11 | 1.61 |
| stall: short_scoreboard (warps per issue) | 0.15 | 0.21 | 0.20 |
| stall: long_scoreboard (warps per issue) | 0.15 | 0.07 | 0.08 |
| stall: mio_throttle (warps per issue) | 0.10 | 0.24 | 0.41 |
| stall: not_selected (warps per issue) | 0.30 | 0.21 | 0.24 |
| stall: dispatch_stall (warps per issue) | 0.13 | 0.05 | 0.06 |
| stall: lg_throttle (warps per issue) | 0.00 | 0.05 | 0.07 |
| stall: no_instruction (warps per issue) | 0.02 | 0.07 | 0.04 |
| stall: branch_resolving (warps per issue) | 0.01 | 0.09 | 0.02 |

Executed SASS by opcode (warp-level instructions, millions; opcodes above 0.5% of the largest total):

| opcode | step-3-cp_async | final-skip_masked | vllm-fa2 |
|---|---:|---:|---:|
| FMUL | 124.8 | 61.9 | 62.9 |
| HMMA | 118.4 | 117.7 | 118.4 |
| IMAD | 110.5 | 35.9 | 12.7 |
| LDSM | 59.3 | 58.9 | 66.6 |
| IADD3 | 28.6 | 39.8 | 11.3 |
| FMNMX | 35.1 | 35.1 | 33.3 |
| LEA | 33.9 | 9.0 | 6.3 |
| FFMA | 33.3 | 33.3 | 31.4 |
| FSETP | 33.3 | 1.9 | 1.9 |
| MUFU | 31.5 | 31.5 | 31.5 |
| FADD | 29.7 | 29.7 | 29.6 |
| ISETP | 28.0 | 10.5 | 2.5 |
| LOP3 | 20.2 | 21.8 | 0.4 |
| SHF | 16.8 | 7.1 | 2.4 |
| F2FP | 15.3 | 15.3 | 15.3 |
| SEL | 15.0 | 0.2 | 0.1 |
| LDGSTS | 14.9 | 14.9 | 14.9 |
| BRA | 4.8 | 12.6 | 2.9 |
| MOV | 10.2 | 0.0 | 2.1 |
| LDS | 5.6 | 5.6 | 5.8 |
| total | 784.9 | 577.4 | 465.8 |

- step-3-cp_async: `void <unnamed>::flash_prefill_mma_kernel<3>(<unnamed>::PrefillParams)`
- final-skip_masked: `void <unnamed>::flash_prefill_mma_kernel<7>(<unnamed>::PrefillParams)`
- vllm-fa2: `void flash_fwd_kernel<Flash_fwd_kernel_traits<128, 64, 64, 4, 0, 0, bfloat16_t, Flash_kernel_traits<128, 64, 64, 4, bfloat16_t>>, 0, 1, 0, 0, 0, 1, 0, 0>(Flash_fwd_params)`
