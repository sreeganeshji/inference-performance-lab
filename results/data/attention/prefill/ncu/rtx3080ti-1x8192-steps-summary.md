| metric | 1-mma | 2-swizzle | 3-cp_async | 4b-hoisted | 6-skip_masked | vllm-fa2 |
|---|---:|---:|---:|---:|---:|---:|
| kernel time (µs, ncu-locked clock) | 16,772 | 8,471 | 8,405 | 7,739 | 7,471 | 7,904 |
| SM clock (MHz) | 1,653 | 1,665 | 1,640 | 1,679 | 1,661 | 1,657 |
| SMs busy (% of elapsed cycles) | 96.57 | 95.90 | 96.67 | 95.22 | 99.92 | 95.78 |
| registers / thread | 200 | 202 | 196 | 198 | 199 | 191 |
| shared memory / block (bytes, incl. 1 KB reserved) | 50,176 | 50,176 | 50,176 | 50,176 | 50,176 | 50,176 |
| block limit: registers | 2.00 | 2.00 | 2.00 | 2.00 | 2.00 | 2.00 |
| block limit: shared memory | 2.00 | 2.00 | 2.00 | 2.00 | 2.00 | 2.00 |
| theoretical occupancy (%) | 16.67 | 16.67 | 16.67 | 16.67 | 16.67 | 16.67 |
| achieved occupancy (%) | 16.49 | 16.48 | 16.47 | 16.46 | 16.63 | 16.46 |
| tensor pipe (HMMA) active (% of peak, SM-active cycles) | 22.10 | 43.74 | 44.40 | 47.84 | 47.46 | 47.18 |
| tensor pipe (HMMA) active (% of peak, elapsed cycles) | 21.34 | 41.95 | 42.92 | 45.55 | 47.42 | 45.19 |
| warp instructions executed | 746,934,272 | 775,319,552 | 784,903,168 | 547,807,232 | 577,400,320 | 465,792,656 |
| XU pipe (MUFU.EX2) (% of peak) | 2.95 | 5.83 | 5.92 | 6.38 | 6.36 | 6.28 |
| FMA pipe (% of peak) | 3.13 | 6.49 | 7.00 | 4.00 | 4.06 | 3.40 |
| ALU pipe (% of peak) | 4.58 | 9.81 | 11.20 | 6.65 | 7.26 | 3.91 |
| shared load bank conflicts | 1,660,332,679 | 164,437 | 143,965 | 160,071 | 135,841 | 158,005 |
| shared store bank conflicts | 12,097,934 | 11,309,077 | 23,840 | 1,758 | 18,548 | 50,107 |
| shared load wavefronts | 1,898,602,536 | 237,933,396 | 237,783,877 | 237,798,693 | 236,395,452 | 266,950,998 |
| DRAM read (MB) | 94.06 | 102 | 86.30 | 89.32 | 254 | 195 |
| L2 → SM read (MB) | 7,629 | 7,627 | 7,634 | 7,634 | 7,634 | 7,634 |
| L2 hit rate (%) | 98.77 | 98.65 | 98.88 | 98.84 | 96.71 | 97.48 |
| stall: math_pipe_throttle (warps per issue) | 3.31 | 6.09 | 7.06 | 9.68 | 7.40 | 11.22 |
| stall: wait (warps per issue) | 1.59 | 1.53 | 1.35 | 1.99 | 2.22 | 2.05 |
| stall: barrier (warps per issue) | 1.00 | 0.50 | 0.45 | 0.48 | 2.11 | 1.61 |
| stall: short_scoreboard (warps per issue) | 4.32 | 0.26 | 0.15 | 0.26 | 0.21 | 0.20 |
| stall: long_scoreboard (warps per issue) | 1.76 | 1.19 | 0.15 | 0.14 | 0.07 | 0.08 |
| stall: mio_throttle (warps per issue) | 9.41 | 0.12 | 0.10 | 0.24 | 0.24 | 0.41 |
| stall: not_selected (warps per issue) | 0.17 | 0.17 | 0.30 | 0.25 | 0.21 | 0.24 |
| stall: dispatch_stall (warps per issue) | 0.09 | 0.13 | 0.13 | 0.07 | 0.05 | 0.06 |
| stall: lg_throttle (warps per issue) | 0.00 | 0.00 | 0.00 | 0.07 | 0.05 | 0.07 |
| stall: no_instruction (warps per issue) | 0.03 | 0.04 | 0.02 | 0.06 | 0.07 | 0.04 |
| stall: branch_resolving (warps per issue) | 0.01 | 0.01 | 0.01 | 0.03 | 0.09 | 0.02 |

Executed SASS by opcode (warp-level instructions, millions; opcodes above 0.5% of the largest total):

| opcode | 1-mma | 2-swizzle | 3-cp_async | 4b-hoisted | 6-skip_masked | vllm-fa2 |
|---|---:|---:|---:|---:|---:|---:|
| FMUL | 124.8 | 124.8 | 124.8 | 61.9 | 61.9 | 62.9 |
| HMMA | 118.4 | 118.4 | 118.4 | 118.4 | 117.7 | 118.4 |
| IMAD | 80.7 | 93.0 | 110.5 | 33.5 | 35.9 | 12.7 |
| LDSM | 59.3 | 59.3 | 59.3 | 59.3 | 58.9 | 66.6 |
| LEA | 46.1 | 48.9 | 33.9 | 7.1 | 9.0 | 6.3 |
| IADD3 | 26.8 | 36.2 | 28.6 | 38.7 | 39.8 | 11.3 |
| FMNMX | 35.1 | 35.1 | 35.1 | 35.1 | 35.1 | 33.3 |
| FFMA | 33.3 | 33.3 | 33.3 | 33.3 | 33.3 | 31.4 |
| FSETP | 33.3 | 33.3 | 33.3 | 1.9 | 1.9 | 1.9 |
| MUFU | 31.5 | 31.5 | 31.5 | 31.5 | 31.5 | 31.5 |
| CS2R | 30.3 | 30.3 | 0.5 | 0.5 | 0.6 | 0.0 |
| FADD | 29.7 | 29.7 | 29.7 | 29.7 | 29.7 | 29.6 |
| ISETP | 14.1 | 15.0 | 28.0 | 7.6 | 10.5 | 2.5 |
| LOP3 | 5.1 | 20.2 | 20.2 | 20.2 | 21.8 | 0.4 |
| SHF | 2.9 | 3.0 | 16.8 | 3.0 | 7.1 | 2.4 |
| STS | 15.4 | 15.4 | 0.5 | 0.5 | 0.5 | 0.6 |
| F2FP | 15.3 | 15.3 | 15.3 | 15.3 | 15.3 | 15.3 |
| SEL | 0.0 | 0.0 | 15.0 | 0.2 | 0.2 | 0.1 |
| LDG | 14.9 | 14.9 | 0.0 | 0.0 | 0.0 | 0.1 |
| LDGSTS | 0.0 | 0.0 | 14.9 | 14.9 | 14.9 | 14.9 |
| MOV | 14.0 | 1.9 | 10.2 | 0.0 | 0.0 | 2.1 |
| BRA | 3.8 | 3.8 | 4.8 | 8.5 | 12.6 | 2.9 |
| LDS | 0.1 | 0.1 | 5.6 | 5.6 | 5.6 | 5.8 |
| total | 746.9 | 775.3 | 784.9 | 547.8 | 577.4 | 465.8 |

- 1-mma: `void <unnamed>::flash_prefill_mma_kernel<1>(<unnamed>::PrefillParams)`
- 2-swizzle: `void <unnamed>::flash_prefill_mma_kernel<2>(<unnamed>::PrefillParams)`
- 3-cp_async: `void <unnamed>::flash_prefill_mma_kernel<3>(<unnamed>::PrefillParams)`
- 4b-hoisted: `void <unnamed>::flash_prefill_mma_kernel<5>(<unnamed>::PrefillParams)`
- 6-skip_masked: `void <unnamed>::flash_prefill_mma_kernel<7>(<unnamed>::PrefillParams)`
- vllm-fa2: `void flash_fwd_kernel<Flash_fwd_kernel_traits<128, 64, 64, 4, 0, 0, bfloat16_t, Flash_kernel_traits<128, 64, 64, 4, bfloat16_t>>, 0, 1, 0, 0, 0, 1, 0, 0>(Flash_fwd_params)`
