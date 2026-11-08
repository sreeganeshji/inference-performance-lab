| metric | step-3-cp_async | final-skip_masked | vllm-fa2 |
|---|---:|---:|---:|
| kernel time (µs, ncu-locked clock) | 3,955 | 3,456 | 3,067 |
| SM clock (MHz) | 1,094 | 1,093 | 1,094 |
| SMs busy (% of elapsed cycles) | 94.00 | 99.71 | 91.44 |
| registers / thread | 196 | 197 | 255 |
| shared memory / block (bytes, incl. 1 KB reserved) | 50,176 | 50,176 | 66,560 |
| block limit: registers | 2.00 | 2.00 | 2.00 |
| block limit: shared memory | 3.00 | 3.00 | 2.00 |
| theoretical occupancy (%) | 12.50 | 12.50 | 12.50 |
| achieved occupancy (%) | 12.32 | 12.47 | 12.16 |
| tensor pipe (HMMA) active (% of peak, SM-active cycles) | 53.89 | 57.84 | 72.00 |
| tensor pipe (HMMA) active (% of peak, elapsed cycles) | 50.66 | 57.67 | 65.84 |
| warp instructions executed | 784,903,168 | 577,385,984 | 450,989,392 |
| XU pipe (MUFU.EX2) (% of peak) | 14.37 | 15.51 | 19.15 |
| FMA pipe (% of peak) | 33.86 | 19.75 | 19.80 |
| ALU pipe (% of peak) | 27.30 | 17.71 | 14.07 |
| shared load bank conflicts | 279,288 | 164,902 | 1,978,998 |
| shared store bank conflicts | 71,485 | 10,355 | 36,920 |
| shared load wavefronts | 237,921,616 | 236,427,717 | 151,546,263 |
| DRAM read (MB) | 75.57 | 94.90 | 79.81 |
| L2 → SM read (MB) | 7,634 | 7,634 | 3,898 |
| L2 hit rate (%) | 98.45 | 97.94 | 97.50 |
| stall: math_pipe_throttle (warps per issue) | 0.46 | 0.25 | 1.09 |
| stall: wait (warps per issue) | 1.50 | 2.60 | 2.33 |
| stall: barrier (warps per issue) | 0.21 | 0.39 | 0.15 |
| stall: short_scoreboard (warps per issue) | 0.40 | 0.56 | 0.34 |
| stall: long_scoreboard (warps per issue) | 0.13 | 0.04 | 0.17 |
| stall: mio_throttle (warps per issue) | 0.08 | 0.17 | 0.11 |
| stall: not_selected (warps per issue) | 0.34 | 0.20 | 0.31 |
| stall: dispatch_stall (warps per issue) | 0.22 | 0.16 | 0.15 |
| stall: lg_throttle (warps per issue) | 0.00 | 0.03 | 0.02 |
| stall: no_instruction (warps per issue) | 0.07 | 0.13 | 0.04 |
| stall: branch_resolving (warps per issue) | 0.01 | 0.10 | 0.01 |

Executed SASS by opcode (warp-level instructions, millions; opcodes above 0.5% of the largest total):

| opcode | step-3-cp_async | final-skip_masked | vllm-fa2 |
|---|---:|---:|---:|
| FMUL | 124.8 | 61.9 | 63.4 |
| HMMA | 118.4 | 117.7 | 119.3 |
| IMAD | 109.6 | 35.8 | 6.3 |
| LDSM | 59.3 | 58.9 | 37.3 |
| IADD3 | 30.4 | 39.8 | 6.9 |
| FMNMX | 35.1 | 35.1 | 33.5 |
| LEA | 34.0 | 9.0 | 3.3 |
| FFMA | 33.3 | 33.3 | 31.7 |
| FSETP | 33.3 | 1.9 | 1.9 |
| MUFU | 31.5 | 31.5 | 31.7 |
| FADD | 29.7 | 29.7 | 29.8 |
| ISETP | 28.0 | 10.5 | 2.1 |
| MOV | 7.5 | 0.0 | 26.3 |
| LOP3 | 20.2 | 22.0 | 0.2 |
| SHF | 18.7 | 7.1 | 0.2 |
| F2FP | 15.3 | 15.3 | 15.4 |
| SEL | 15.0 | 0.2 | 0.0 |
| LDGSTS | 14.9 | 14.9 | 7.6 |
| BRA | 4.8 | 12.6 | 1.5 |
| LDL | 0.0 | 0.0 | 6.9 |
| LDS | 5.6 | 5.6 | 3.0 |
| STL | 0.0 | 0.0 | 5.5 |
| total | 784.9 | 577.4 | 451.0 |

- step-3-cp_async: `void <unnamed>::flash_prefill_mma_kernel<3>(<unnamed>::PrefillParams)`
- final-skip_masked: `void <unnamed>::flash_prefill_mma_kernel<7>(<unnamed>::PrefillParams)`
- vllm-fa2: `void flash_fwd_kernel<Flash_fwd_kernel_traits<128, 128, 64, 4, 0, 0, bfloat16_t, Flash_kernel_traits<128, 128, 64, 4, bfloat16_t>>, 0, 1, 0, 0, 0, 1, 0, 0>(Flash_fwd_params)`
