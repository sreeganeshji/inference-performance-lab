## FLASH_ATTN-i4096-o256-c64

| kernel | gridX | gridY (sequences) | gridZ | calls | total s | mean µs |
|---|---:|---|---:|---:|---:|---:|
| `flash_fwd_splitkv_kernel` | 32 | 1–64 | 28 | 4088 | 8.04 | 1967.4 |
| `flash_fwd_splitkv_kernel` | 1 | 48–64 | 4 | 8624 | 4.13 | 478.4 |
| `flash_fwd_splitkv_kernel` | 22 | 27–63 | 28 | 308 | 0.69 | 2236.5 |
| `flash_fwd_splitkv_kernel` | 24 | 25–63 | 28 | 280 | 0.66 | 2353.0 |
| `flash_fwd_splitkv_kernel` | 27 | 20–64 | 28 | 280 | 0.61 | 2161.7 |
| `flash_fwd_splitkv_kernel` | 29 | 16–64 | 28 | 280 | 0.60 | 2151.6 |

## CUSTOM-i4096-o256-c64

| kernel | gridX | gridY (sequences) | gridZ | calls | total s | mean µs |
|---|---:|---|---:|---:|---:|---:|
| `paged_decode_attention_tc_kernel` | 4 | 22–64 | 1 | 15932 | 5.71 | 358.1 |
| `flash_fwd_splitkv_kernel` | 32 | 1–2 | 28 | 4088 | 1.75 | 428.8 |
| `paged_decode_attention_tc_kernel` | 4 | 2–2 | 12 | 7168 | 0.16 | 22.9 |
| `flash_fwd_splitkv_kernel` | 22 | 2–2 | 28 | 308 | 0.13 | 425.2 |
| `paged_decode_attention_tc_kernel` | 4 | 11–21 | 2 | 1092 | 0.13 | 115.8 |
| `flash_fwd_splitkv_kernel` | 27 | 2–2 | 28 | 280 | 0.12 | 441.7 |

## FLASH_ATTN-i1024-o128-c128

| kernel | gridX | gridY (sequences) | gridZ | calls | total s | mean µs |
|---|---:|---|---:|---:|---:|---:|
| `flash_fwd_splitkv_kernel` | 16 | 1–128 | 28 | 3864 | 4.09 | 1059.7 |
| `flash_fwd_splitkv_kernel` | 1 | 48–128 | 4 | 4760 | 0.98 | 205.0 |
| `flash_fwd_splitkv_kernel` | 1 | 23–23 | 8 | 3556 | 0.05 | 12.7 |
| `flash_fwd_splitkv_kernel` | 11 | 128–128 | 28 | 28 | 0.04 | 1419.7 |
| `flash_fwd_splitkv_combine_kernel` | 14 | 1–1 | 1 | 3556 | 0.04 | 10.2 |
| `flash_fwd_splitkv_kernel` | 2 | 128–128 | 28 | 28 | 0.04 | 1265.6 |

## CUSTOM-i1024-o128-c128

| kernel | gridX | gridY (sequences) | gridZ | calls | total s | mean µs |
|---|---:|---|---:|---:|---:|---:|
| `paged_decode_attention_tc_kernel` | 4 | 23–128 | 1 | 8008 | 1.49 | 186.3 |
| `flash_fwd_splitkv_kernel` | 16 | 1–3 | 28 | 3864 | 0.52 | 133.4 |
| `paged_decode_attention_tc_kernel` | 4 | 11–101 | 2 | 560 | 0.06 | 102.5 |
| `paged_decode_attention_tc_kernel` | 4 | 2–2 | 12 | 3584 | 0.04 | 11.6 |
| `paged_decode_attention_tc_kernel` | 4 | 7–88 | 4 | 168 | 0.02 | 145.3 |
| `combine_splits_kernel` | 28 | 1–101 | 1 | 4732 | 0.02 | 4.5 |

