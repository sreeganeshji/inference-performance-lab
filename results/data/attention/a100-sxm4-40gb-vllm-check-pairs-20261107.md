| pair | identical greedy prefix | first divergence per prompt | max abs logprob diff |
|---|---:|---|---:|
| CUSTOM vs FLASHINFER | 286/384 | [48, 0, 48, 48, 42, 48, 4, 48] | 0.122 |
| CUSTOM vs FLASH_ATTN | 334/384 | [48, 48, 48, 48, 42, 48, 4, 48] | 0.104 |
| CUSTOM vs TRITON_ATTN | 342/384 | [48, 48, 48, 48, 48, 48, 6, 48] | 0.0944 |
| FLASHINFER vs FLASH_ATTN | 315/384 | [48, 0, 48, 48, 48, 48, 27, 48] | 0.066 |
| FLASHINFER vs TRITON_ATTN | 286/384 | [48, 0, 48, 48, 42, 48, 4, 48] | 0.104 |
| FLASH_ATTN vs TRITON_ATTN | 334/384 | [48, 48, 48, 48, 42, 48, 4, 48] | 0.104 |
