# qwen2.5-7b: 28 query heads, 4 KV heads, head_dim 128, KV layout vllm
# date=2026-11-07T21:02:12+00:00, gpu=NVIDIA A100-SXM4-40GB, sm_count=108, l2_bytes=41943040, torch=2.14.0+cu130, cuda=13.0, python=3.12.3, vllm=0.33.0, flashinfer=0.7.2.post1, driver=620.70.06

| batch | context | implementation | splits | median µs | min–max µs | GB/s |
|---:|---:|---|---:|---:|---:|---:|
# skipped flashinfer: ead_dim_qk_128_head_dim_vo_128_posenc_0_use_swa_False_use_logits_cap_False/batch_decode.cu:63: Unsupported group_size: 7
| 1 | 1024 | lab-per-query-head | 1 | 106.35 | 106.23–106.42 | 20 |
| 1 | 1024 | lab-gqa | 1 | 154.94 | 154.83–155.00 | 14 |
| 1 | 1024 | lab-gqa-split | 16 | 19.92 | 19.92–19.93 | 106 |
| 1 | 1024 | lab-gqa-split-tc | 13 | 13.56 | 13.55–13.56 | 156 |
| 1 | 1024 | vllm-fa2 |  | 15.84 | 15.83–15.84 | 133 |
| 1 | 1024 | flashinfer-tensor-cores |  | 14.50 | 14.50–14.51 | 146 |
# skipped flashinfer: ead_dim_qk_128_head_dim_vo_128_posenc_0_use_swa_False_use_logits_cap_False/batch_decode.cu:63: Unsupported group_size: 7
| 1 | 4096 | lab-per-query-head | 1 | 419.99 | 419.46–420.86 | 20 |
| 1 | 4096 | lab-gqa | 1 | 604.57 | 604.34–604.84 | 14 |
| 1 | 4096 | lab-gqa-split | 32 | 35.54 | 35.51–35.55 | 236 |
| 1 | 4096 | lab-gqa-split-tc | 26 | 20.09 | 20.08–20.09 | 418 |
| 1 | 4096 | vllm-fa2 |  | 27.24 | 27.23–27.26 | 308 |
| 1 | 4096 | flashinfer-tensor-cores |  | 19.28 | 19.27–19.28 | 436 |
# skipped flashinfer: ead_dim_qk_128_head_dim_vo_128_posenc_0_use_swa_False_use_logits_cap_False/batch_decode.cu:63: Unsupported group_size: 7
| 1 | 16384 | lab-per-query-head | 1 | 1679.02 | 1670.83–1686.87 | 20 |
| 1 | 16384 | lab-gqa | 1 | 2394.03 | 2392.58–2394.97 | 14 |
| 1 | 16384 | lab-gqa-split | 64 | 74.11 | 74.06–74.14 | 453 |
| 1 | 16384 | lab-gqa-split-tc | 45 | 42.83 | 42.80–42.86 | 784 |
| 1 | 16384 | vllm-fa2 |  | 50.35 | 50.29–50.38 | 667 |
| 1 | 16384 | flashinfer-tensor-cores |  | 40.31 | 40.26–40.35 | 833 |
# skipped flashinfer: ead_dim_qk_128_head_dim_vo_128_posenc_0_use_swa_False_use_logits_cap_False/batch_decode.cu:63: Unsupported group_size: 7
| 4 | 1024 | lab-per-query-head | 1 | 107.32 | 107.23–107.38 | 79 |
| 4 | 1024 | lab-gqa | 1 | 157.54 | 157.50–157.66 | 54 |
| 4 | 1024 | lab-gqa-split | 8 | 32.61 | 32.60–32.62 | 259 |
| 4 | 1024 | lab-gqa-split-tc | 6 | 17.68 | 17.67–17.68 | 478 |
| 4 | 1024 | vllm-fa2 |  | 26.13 | 26.10–26.14 | 323 |
| 4 | 1024 | flashinfer-tensor-cores |  | 18.57 | 18.56–18.57 | 455 |
# skipped flashinfer: ead_dim_qk_128_head_dim_vo_128_posenc_0_use_swa_False_use_logits_cap_False/batch_decode.cu:63: Unsupported group_size: 7
| 4 | 4096 | lab-per-query-head | 1 | 424.79 | 423.81–425.98 | 79 |
| 4 | 4096 | lab-gqa | 1 | 613.29 | 613.09–613.43 | 55 |
| 4 | 4096 | lab-gqa-split | 16 | 67.95 | 67.91–67.96 | 495 |
| 4 | 4096 | lab-gqa-split-tc | 11 | 38.29 | 38.28–38.33 | 878 |
| 4 | 4096 | vllm-fa2 |  | 48.07 | 48.02–48.12 | 699 |
| 4 | 4096 | flashinfer-tensor-cores |  | 39.00 | 38.96–39.04 | 862 |
# skipped flashinfer: ead_dim_qk_128_head_dim_vo_128_posenc_0_use_swa_False_use_logits_cap_False/batch_decode.cu:63: Unsupported group_size: 7
| 4 | 16384 | lab-per-query-head | 1 | 1691.22 | 1684.39–1694.72 | 79 |
| 4 | 16384 | lab-gqa | 1 | 2432.23 | 2430.63–2432.46 | 55 |
| 4 | 16384 | lab-gqa-split | 20 | 192.41 | 192.36–192.52 | 698 |
| 4 | 16384 | lab-gqa-split-tc | 12 | 111.62 | 111.53–111.72 | 1203 |
| 4 | 16384 | vllm-fa2 |  | 123.93 | 123.84–123.97 | 1084 |
| 4 | 16384 | flashinfer-tensor-cores |  | 112.88 | 112.71–113.08 | 1190 |
# skipped flashinfer: ead_dim_qk_128_head_dim_vo_128_posenc_0_use_swa_False_use_logits_cap_False/batch_decode.cu:63: Unsupported group_size: 7
| 16 | 1024 | lab-per-query-head | 1 | 127.58 | 127.49–128.00 | 265 |
| 16 | 1024 | lab-gqa | 1 | 159.46 | 159.41–160.49 | 212 |
| 16 | 1024 | lab-gqa-split | 4 | 66.51 | 66.47–66.53 | 508 |
| 16 | 1024 | lab-gqa-split-tc | 3 | 37.11 | 37.10–37.15 | 910 |
| 16 | 1024 | vllm-fa2 |  | 47.71 | 47.68–47.77 | 708 |
| 16 | 1024 | flashinfer-tensor-cores |  | 40.04 | 40.03–40.07 | 844 |
# skipped flashinfer: ead_dim_qk_128_head_dim_vo_128_posenc_0_use_swa_False_use_logits_cap_False/batch_decode.cu:63: Unsupported group_size: 7
| 16 | 4096 | lab-per-query-head | 1 | 498.82 | 498.64–498.95 | 270 |
| 16 | 4096 | lab-gqa | 1 | 620.21 | 619.72–620.44 | 217 |
| 16 | 4096 | lab-gqa-split | 5 | 190.41 | 190.30–190.51 | 706 |
| 16 | 4096 | lab-gqa-split-tc | 3 | 110.30 | 110.23–110.43 | 1219 |
| 16 | 4096 | vllm-fa2 |  | 122.63 | 122.39–122.70 | 1096 |
| 16 | 4096 | flashinfer-tensor-cores |  | 111.89 | 111.73–112.00 | 1202 |
# skipped flashinfer: ead_dim_qk_128_head_dim_vo_128_posenc_0_use_swa_False_use_logits_cap_False/batch_decode.cu:63: Unsupported group_size: 7
| 16 | 16384 | lab-per-query-head | 1 | 1975.91 | 1975.50–1977.34 | 272 |
| 16 | 16384 | lab-gqa | 1 | 2457.71 | 2456.12–2459.76 | 219 |
| 16 | 16384 | lab-gqa-split | 5 | 710.55 | 710.14–711.28 | 756 |
| 16 | 16384 | lab-gqa-split-tc | 3 | 402.24 | 401.80–402.45 | 1335 |
| 16 | 16384 | vllm-fa2 |  | 413.57 | 413.27–414.25 | 1299 |
| 16 | 16384 | flashinfer-tensor-cores |  | 402.30 | 401.77–402.61 | 1335 |
# skipped flashinfer: ead_dim_qk_128_head_dim_vo_128_posenc_0_use_swa_False_use_logits_cap_False/batch_decode.cu:63: Unsupported group_size: 7
| 64 | 1024 | lab-per-query-head | 1 | 322.60 | 322.47–322.65 | 419 |
| 64 | 1024 | lab-gqa | 1 | 225.49 | 225.41–225.57 | 599 |
| 64 | 1024 | lab-gqa-split | 1 | 225.49 | 225.44–225.63 | 599 |
| 64 | 1024 | lab-gqa-split-tc | 1 | 108.47 | 108.37–108.58 | 1246 |
| 64 | 1024 | vllm-fa2 |  | 141.14 | 141.00–141.23 | 957 |
| 64 | 1024 | flashinfer-tensor-cores |  | 119.26 | 119.24–119.41 | 1133 |
# skipped flashinfer: ead_dim_qk_128_head_dim_vo_128_posenc_0_use_swa_False_use_logits_cap_False/batch_decode.cu:63: Unsupported group_size: 7
| 64 | 4096 | lab-per-query-head | 1 | 1274.56 | 1272.45–1275.90 | 422 |
| 64 | 4096 | lab-gqa | 1 | 872.94 | 872.63–873.47 | 616 |
| 64 | 4096 | lab-gqa-split | 1 | 872.94 | 872.27–873.65 | 616 |
| 64 | 4096 | lab-gqa-split-tc | 1 | 398.02 | 397.59–398.44 | 1351 |
| 64 | 4096 | vllm-fa2 |  | 505.77 | 505.52–506.16 | 1063 |
| 64 | 4096 | flashinfer-tensor-cores |  | 433.63 | 432.85–435.32 | 1240 |
# skipped flashinfer: ead_dim_qk_128_head_dim_vo_128_posenc_0_use_swa_False_use_logits_cap_False/batch_decode.cu:63: Unsupported group_size: 7
| 64 | 16384 | lab-per-query-head | 1 | 5178.62 | 5156.61–5187.58 | 415 |
| 64 | 16384 | lab-gqa | 1 | 3460.44 | 3459.07–3464.02 | 621 |
| 64 | 16384 | lab-gqa-split | 5 | 2694.91 | 2693.50–2696.32 | 797 |
| 64 | 16384 | lab-gqa-split-tc | 1 | 1593.11 | 1589.48–1598.46 | 1349 |
| 64 | 16384 | vllm-fa2 |  | 1972.74 | 1971.30–1974.58 | 1089 |
| 64 | 16384 | flashinfer-tensor-cores |  | 1709.40 | 1707.69–1713.92 | 1257 |
