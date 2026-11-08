| metric | step-3-cp_async | final-skip_masked | vllm-fa2 |
|---|---:|---:|---:|
| kernel time (µs, ncu-locked clock) | 194 | 142 | 170 |
| SM clock (MHz) | 1,636 | 1,653 | 1,677 |
| SMs busy (% of elapsed cycles) | 75.05 | 94.64 | 77.42 |
| registers / thread | 196 | 199 | 191 |
| shared memory / block (bytes, incl. 1 KB reserved) | 50,176 | 50,176 | 50,176 |
| block limit: registers | 2.00 | 2.00 | 2.00 |
| block limit: shared memory | 2.00 | 2.00 | 2.00 |
| theoretical occupancy (%) | 16.67 | 16.67 | 16.67 |
| achieved occupancy (%) | 15.52 | 16.24 | 15.51 |
| tensor pipe (HMMA) active (% of peak, SM-active cycles) | 41.00 | 41.86 | 44.25 |
| tensor pipe (HMMA) active (% of peak, elapsed cycles) | 30.77 | 39.61 | 34.25 |
| warp instructions executed | 14,118,272 | 10,960,320 | 8,839,824 |
| XU pipe (MUFU.EX2) (% of peak) | 5.60 | 5.98 | 5.98 |
| FMA pipe (% of peak) | 6.95 | 4.43 | 3.59 |
| ALU pipe (% of peak) | 11.77 | 8.55 | 5.59 |
| shared load bank conflicts | 11,950 | 5,262 | 19,077 |
| shared store bank conflicts | 1,413 | 2,072 | 3,125 |
| shared load wavefronts | 4,026,315 | 3,847,502 | 4,466,157 |
| DRAM read (MB) | 9.47 | 10.35 | 9.48 |
| L2 → SM read (MB) | 132 | 132 | 132 |
| L2 hit rate (%) | 93.34 | 92.57 | 93.23 |
| stall: math_pipe_throttle (warps per issue) | 6.02 | 6.87 | 9.29 |
| stall: wait (warps per issue) | 1.32 | 2.08 | 1.90 |
| stall: barrier (warps per issue) | 0.34 | 0.99 | 0.78 |
| stall: short_scoreboard (warps per issue) | 0.20 | 0.29 | 0.22 |
| stall: long_scoreboard (warps per issue) | 0.53 | 0.57 | 0.60 |
| stall: mio_throttle (warps per issue) | 0.11 | 0.22 | 0.40 |
| stall: not_selected (warps per issue) | 0.26 | 0.23 | 0.21 |
| stall: dispatch_stall (warps per issue) | 0.13 | 0.05 | 0.06 |
| stall: lg_throttle (warps per issue) | 0.00 | 0.04 | 0.07 |
| stall: no_instruction (warps per issue) | 0.04 | 0.10 | 0.05 |
| stall: branch_resolving (warps per issue) | 0.02 | 0.10 | 0.02 |

Executed SASS by opcode (warp-level instructions, millions; opcodes above 0.5% of the largest total):

| opcode | step-3-cp_async | final-skip_masked | vllm-fa2 |
|---|---:|---:|---:|
| FMUL | 2.2 | 1.1 | 1.0 |
| IMAD | 2.1 | 1.0 | 0.5 |
| HMMA | 1.9 | 1.9 | 1.9 |
| LDSM | 1.0 | 0.9 | 1.1 |
| IADD3 | 0.6 | 0.8 | 0.3 |
| LEA | 0.6 | 0.2 | 0.2 |
| ISETP | 0.6 | 0.3 | 0.2 |
| FMNMX | 0.6 | 0.6 | 0.5 |
| FFMA | 0.6 | 0.6 | 0.5 |
| FSETP | 0.6 | 0.0 | 0.0 |
| MUFU | 0.5 | 0.5 | 0.5 |
| LOP3 | 0.4 | 0.5 | 0.1 |
| FADD | 0.5 | 0.5 | 0.5 |
| F2FP | 0.3 | 0.3 | 0.3 |
| SHF | 0.3 | 0.2 | 0.1 |
| SEL | 0.3 | 0.0 | 0.0 |
| BRA | 0.1 | 0.3 | 0.1 |
| LDGSTS | 0.3 | 0.3 | 0.3 |
| MOV | 0.2 | 0.0 | 0.1 |
| FSEL | 0.1 | 0.1 | 0.1 |
| LDS | 0.1 | 0.1 | 0.1 |
| WARPSYNC | 0.0 | 0.1 | 0.0 |
| CS2R | 0.1 | 0.1 | 0.0 |
| BSSY | 0.0 | 0.1 | 0.0 |
| BSYNC | 0.0 | 0.1 | 0.0 |
| STS | 0.1 | 0.1 | 0.1 |
| total | 14.1 | 11.0 | 8.8 |

- step-3-cp_async: `void <unnamed>::flash_prefill_mma_kernel<3>(<unnamed>::PrefillParams)`
- final-skip_masked: `void <unnamed>::flash_prefill_mma_kernel<7>(<unnamed>::PrefillParams)`
- vllm-fa2: `void flash_fwd_kernel<Flash_fwd_kernel_traits<128, 64, 64, 4, 0, 0, bfloat16_t, Flash_kernel_traits<128, 64, 64, 4, bfloat16_t>>, 0, 1, 0, 0, 0, 1, 0, 0>(Flash_fwd_params)`
