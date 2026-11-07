| metric | step-3-cp_async | final-skip_masked | vllm-fa2 |
|---|---:|---:|---:|
| kernel time (µs, ncu-locked clock) | 109 | 79.74 | 116 |
| SM clock (MHz) | 1,090 | 1,089 | 1,091 |
| SMs busy (% of elapsed cycles) | 69.20 | 90.75 | 61.58 |
| registers / thread | 196 | 197 | 255 |
| shared memory / block (bytes, incl. 1 KB reserved) | 50,176 | 50,176 | 66,560 |
| block limit: registers | 2.00 | 2.00 | 2.00 |
| block limit: shared memory | 3.00 | 3.00 | 2.00 |
| theoretical occupancy (%) | 12.50 | 12.50 | 12.50 |
| achieved occupancy (%) | 11.56 | 12.25 | 11.00 |
| tensor pipe (HMMA) active (% of peak, SM-active cycles) | 44.07 | 43.78 | 49.19 |
| tensor pipe (HMMA) active (% of peak, elapsed cycles) | 30.50 | 39.73 | 30.29 |
| warp instructions executed | 14,118,272 | 10,958,528 | 9,415,504 |
| XU pipe (MUFU.EX2) (% of peak) | 12.03 | 12.50 | 13.22 |
| FMA pipe (% of peak) | 29.78 | 18.39 | 16.07 |
| ALU pipe (% of peak) | 25.39 | 18.01 | 14.58 |
| shared load bank conflicts | 27,684 | 19,886 | 63,417 |
| shared store bank conflicts | 7,427 | 0.00 | 157 |
| shared load wavefronts | 4,042,694 | 3,862,337 | 2,701,441 |
| DRAM read (MB) | 9.47 | 9.49 | 9.57 |
| L2 → SM read (MB) | 132 | 132 | 74.99 |
| L2 hit rate (%) | 90.13 | 89.84 | 87.33 |
| stall: math_pipe_throttle (warps per issue) | 0.37 | 0.25 | 0.69 |
| stall: wait (warps per issue) | 1.46 | 2.39 | 2.13 |
| stall: barrier (warps per issue) | 0.14 | 0.44 | 0.13 |
| stall: short_scoreboard (warps per issue) | 0.43 | 0.60 | 0.41 |
| stall: long_scoreboard (warps per issue) | 0.42 | 0.42 | 0.74 |
| stall: mio_throttle (warps per issue) | 0.08 | 0.17 | 0.13 |
| stall: not_selected (warps per issue) | 0.29 | 0.21 | 0.24 |
| stall: dispatch_stall (warps per issue) | 0.20 | 0.13 | 0.12 |
| stall: lg_throttle (warps per issue) | 0.00 | 0.03 | 0.10 |
| stall: no_instruction (warps per issue) | 0.08 | 0.17 | 0.34 |
| stall: branch_resolving (warps per issue) | 0.02 | 0.10 | 0.01 |

Executed SASS by opcode (warp-level instructions, millions; opcodes above 0.5% of the largest total):

| opcode | step-3-cp_async | final-skip_masked | vllm-fa2 |
|---|---:|---:|---:|
| FMUL | 2.2 | 1.1 | 1.1 |
| HMMA | 1.9 | 1.9 | 2.1 |
| IMAD | 2.1 | 1.0 | 0.5 |
| LDSM | 1.0 | 0.9 | 0.6 |
| IADD3 | 0.6 | 0.8 | 0.2 |
| MOV | 0.1 | 0.0 | 0.7 |
| LEA | 0.6 | 0.2 | 0.1 |
| ISETP | 0.6 | 0.3 | 0.2 |
| FMNMX | 0.6 | 0.6 | 0.6 |
| FFMA | 0.6 | 0.6 | 0.5 |
| MUFU | 0.5 | 0.5 | 0.6 |
| FSETP | 0.6 | 0.0 | 0.0 |
| LOP3 | 0.4 | 0.5 | 0.0 |
| FADD | 0.5 | 0.5 | 0.5 |
| SHF | 0.3 | 0.2 | 0.0 |
| F2FP | 0.3 | 0.3 | 0.3 |
| SEL | 0.3 | 0.0 | 0.0 |
| BRA | 0.1 | 0.3 | 0.0 |
| LDGSTS | 0.3 | 0.3 | 0.1 |
| FSEL | 0.1 | 0.1 | 0.2 |
| LDL | 0.0 | 0.0 | 0.2 |
| STL | 0.0 | 0.0 | 0.1 |
| LDS | 0.1 | 0.1 | 0.1 |
| WARPSYNC | 0.0 | 0.1 | 0.0 |
| CS2R | 0.1 | 0.1 | 0.0 |
| BSYNC | 0.0 | 0.1 | 0.0 |
| BSSY | 0.0 | 0.1 | 0.0 |
| SHFL | 0.1 | 0.1 | 0.1 |
| total | 14.1 | 11.0 | 9.4 |

- step-3-cp_async: `void <unnamed>::flash_prefill_mma_kernel<3>(<unnamed>::PrefillParams)`
- final-skip_masked: `void <unnamed>::flash_prefill_mma_kernel<7>(<unnamed>::PrefillParams)`
- vllm-fa2: `void flash_fwd_kernel<Flash_fwd_kernel_traits<128, 128, 64, 4, 0, 0, bfloat16_t, Flash_kernel_traits<128, 128, 64, 4, bfloat16_t>>, 0, 1, 0, 0, 0, 1, 0, 0>(Flash_fwd_params)`
