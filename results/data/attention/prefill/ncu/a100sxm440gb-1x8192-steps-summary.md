| metric | 1-mma | 2-swizzle | 3-cp_async | 4b-hoisted | 6-skip_masked | vllm-fa2 |
|---|---:|---:|---:|---:|---:|---:|
| kernel time (µs, ncu-locked clock) | 19,298 | 4,750 | 3,955 | 3,471 | 3,456 | 3,067 |
| SM clock (MHz) | 1,095 | 1,094 | 1,094 | 1,094 | 1,093 | 1,094 |
| SMs busy (% of elapsed cycles) | 93.50 | 94.21 | 94.00 | 94.57 | 99.71 | 91.44 |
| registers / thread | 200 | 202 | 196 | 198 | 197 | 255 |
| shared memory / block (bytes, incl. 1 KB reserved) | 50,176 | 50,176 | 50,176 | 50,176 | 50,176 | 66,560 |
| block limit: registers | 2.00 | 2.00 | 2.00 | 2.00 | 2.00 | 2.00 |
| block limit: shared memory | 3.00 | 3.00 | 3.00 | 3.00 | 3.00 | 2.00 |
| theoretical occupancy (%) | 12.50 | 12.50 | 12.50 | 12.50 | 12.50 | 12.50 |
| achieved occupancy (%) | 12.34 | 12.33 | 12.32 | 12.33 | 12.47 | 12.16 |
| tensor pipe (HMMA) active (% of peak, SM-active cycles) | 11.09 | 44.77 | 53.89 | 61.05 | 57.84 | 72.00 |
| tensor pipe (HMMA) active (% of peak, elapsed cycles) | 10.37 | 42.18 | 50.66 | 57.73 | 57.67 | 65.84 |
| warp instructions executed | 746,848,256 | 775,319,552 | 784,903,168 | 547,807,232 | 577,385,984 | 450,989,392 |
| XU pipe (MUFU.EX2) (% of peak) | 2.96 | 11.94 | 14.37 | 16.27 | 15.51 | 19.15 |
| FMA pipe (% of peak) | 6.31 | 26.56 | 33.86 | 20.43 | 19.75 | 19.80 |
| ALU pipe (% of peak) | 4.57 | 20.07 | 27.30 | 16.97 | 17.71 | 14.07 |
| shared load bank conflicts | 1,660,362,856 | 144,207 | 279,288 | 271,169 | 164,902 | 1,978,998 |
| shared store bank conflicts | 11,863,360 | 21,591,666 | 71,485 | 47,204 | 10,355 | 36,920 |
| shared load wavefronts | 1,898,701,498 | 238,178,221 | 237,921,616 | 237,912,903 | 236,427,717 | 151,546,263 |
| DRAM read (MB) | 75.60 | 75.58 | 75.57 | 75.57 | 94.90 | 79.81 |
| L2 → SM read (MB) | 7,625 | 7,593 | 7,634 | 7,634 | 7,634 | 3,898 |
| L2 hit rate (%) | 98.44 | 98.47 | 98.45 | 98.45 | 97.94 | 97.50 |
| stall: math_pipe_throttle (warps per issue) | 0.12 | 0.23 | 0.46 | 0.36 | 0.25 | 1.09 |
| stall: wait (warps per issue) | 1.65 | 1.77 | 1.50 | 2.26 | 2.60 | 2.33 |
| stall: barrier (warps per issue) | 0.79 | 0.26 | 0.21 | 0.34 | 0.39 | 0.15 |
| stall: short_scoreboard (warps per issue) | 10.20 | 0.55 | 0.40 | 0.72 | 0.56 | 0.34 |
| stall: long_scoreboard (warps per issue) | 1.61 | 0.62 | 0.13 | 0.08 | 0.04 | 0.17 |
| stall: mio_throttle (warps per issue) | 6.93 | 0.38 | 0.08 | 0.19 | 0.17 | 0.11 |
| stall: not_selected (warps per issue) | 0.13 | 0.24 | 0.34 | 0.24 | 0.20 | 0.31 |
| stall: dispatch_stall (warps per issue) | 0.08 | 0.25 | 0.22 | 0.22 | 0.16 | 0.15 |
| stall: lg_throttle (warps per issue) | 0.00 | 0.00 | 0.00 | 0.04 | 0.03 | 0.02 |
| stall: no_instruction (warps per issue) | 0.05 | 0.06 | 0.07 | 0.09 | 0.13 | 0.04 |
| stall: branch_resolving (warps per issue) | 0.01 | 0.02 | 0.01 | 0.03 | 0.10 | 0.01 |

Executed SASS by opcode (warp-level instructions, millions; opcodes above 0.5% of the largest total):

| opcode | 1-mma | 2-swizzle | 3-cp_async | 4b-hoisted | 6-skip_masked | vllm-fa2 |
|---|---:|---:|---:|---:|---:|---:|
| FMUL | 124.8 | 124.8 | 124.8 | 61.9 | 61.9 | 63.4 |
| HMMA | 118.4 | 118.4 | 118.4 | 118.4 | 117.7 | 119.3 |
| IMAD | 81.6 | 93.0 | 109.6 | 33.5 | 35.8 | 6.3 |
| LDSM | 59.3 | 59.3 | 59.3 | 59.3 | 58.9 | 37.3 |
| LEA | 46.1 | 48.9 | 34.0 | 7.1 | 9.0 | 3.3 |
| IADD3 | 36.1 | 36.2 | 30.4 | 38.7 | 39.8 | 6.9 |
| FMNMX | 35.1 | 35.1 | 35.1 | 35.1 | 35.1 | 33.5 |
| FFMA | 33.3 | 33.3 | 33.3 | 33.3 | 33.3 | 31.7 |
| FSETP | 33.3 | 33.3 | 33.3 | 1.9 | 1.9 | 1.9 |
| MUFU | 31.5 | 31.5 | 31.5 | 31.5 | 31.5 | 31.7 |
| CS2R | 30.3 | 30.3 | 0.5 | 0.5 | 0.6 | 0.0 |
| FADD | 29.7 | 29.7 | 29.7 | 29.7 | 29.7 | 29.8 |
| ISETP | 15.0 | 15.0 | 28.0 | 7.6 | 10.5 | 2.1 |
| MOV | 3.9 | 1.9 | 7.5 | 0.0 | 0.0 | 26.3 |
| LOP3 | 4.2 | 20.2 | 20.2 | 20.2 | 22.0 | 0.2 |
| SHF | 3.0 | 3.0 | 18.7 | 3.0 | 7.1 | 0.2 |
| F2FP | 15.3 | 15.3 | 15.3 | 15.3 | 15.3 | 15.4 |
| STS | 15.4 | 15.4 | 0.5 | 0.5 | 0.5 | 0.5 |
| SEL | 0.0 | 0.0 | 15.0 | 0.2 | 0.2 | 0.0 |
| LDG | 14.9 | 14.9 | 0.0 | 0.0 | 0.0 | 0.1 |
| LDGSTS | 0.0 | 0.0 | 14.9 | 14.9 | 14.9 | 7.6 |
| BRA | 3.8 | 3.8 | 4.8 | 8.5 | 12.6 | 1.5 |
| LDL | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 6.9 |
| LDS | 0.1 | 0.1 | 5.6 | 5.6 | 5.6 | 3.0 |
| STL | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 5.5 |
| total | 746.8 | 775.3 | 784.9 | 547.8 | 577.4 | 451.0 |

- 1-mma: `void <unnamed>::flash_prefill_mma_kernel<1>(<unnamed>::PrefillParams)`
- 2-swizzle: `void <unnamed>::flash_prefill_mma_kernel<2>(<unnamed>::PrefillParams)`
- 3-cp_async: `void <unnamed>::flash_prefill_mma_kernel<3>(<unnamed>::PrefillParams)`
- 4b-hoisted: `void <unnamed>::flash_prefill_mma_kernel<5>(<unnamed>::PrefillParams)`
- 6-skip_masked: `void <unnamed>::flash_prefill_mma_kernel<7>(<unnamed>::PrefillParams)`
- vllm-fa2: `void flash_fwd_kernel<Flash_fwd_kernel_traits<128, 128, 64, 4, 0, 0, bfloat16_t, Flash_kernel_traits<128, 128, 64, 4, bfloat16_t>>, 0, 1, 0, 0, 0, 1, 0, 0>(Flash_fwd_params)`
