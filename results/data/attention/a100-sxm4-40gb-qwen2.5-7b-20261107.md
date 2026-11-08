# qwen2.5-7b: 28 query heads, 4 KV heads, head_dim 128, KV layout vllm
# date=2026-11-07T19:38:13+00:00, gpu=NVIDIA A100-SXM4-40GB, sm_count=108, l2_bytes=41943040, torch=2.14.0+cu130, cuda=13.0, python=3.12.3, vllm=0.33.0, flashinfer=0.7.2.post1, driver=620.70.06

| batch | context | implementation | splits | median µs | min–max µs | GB/s |
|---:|---:|---|---:|---:|---:|---:|
# skipped flashinfer: ead_dim_qk_128_head_dim_vo_128_posenc_0_use_swa_False_use_logits_cap_False/batch_decode.cu:63: Unsupported group_size: 7
| 1 | 1024 | lab-per-query-head | 1 | 106.41 | 106.25–106.47 | 20 |
| 1 | 1024 | lab-gqa | 1 | 154.98 | 154.91–155.11 | 14 |
| 1 | 1024 | lab-gqa-split | 16 | 19.92 | 19.92–19.94 | 106 |
| 1 | 1024 | lab-gqa-split-tc | 16 | 12.28 | 12.27–12.29 | 172 |
| 1 | 1024 | vllm-fa2 |  | 15.86 | 15.84–15.95 | 133 |
| 1 | 1024 | flashinfer-tensor-cores |  | 14.50 | 14.50–14.55 | 146 |
# skipped flashinfer: ead_dim_qk_128_head_dim_vo_128_posenc_0_use_swa_False_use_logits_cap_False/batch_decode.cu:63: Unsupported group_size: 7
| 1 | 4096 | lab-per-query-head | 1 | 419.75 | 419.00–420.61 | 20 |
| 1 | 4096 | lab-gqa | 1 | 604.77 | 604.32–604.96 | 14 |
| 1 | 4096 | lab-gqa-split | 64 | 32.76 | 32.74–32.76 | 257 |
| 1 | 4096 | lab-gqa-split-tc | 22 | 19.68 | 19.67–19.68 | 427 |
| 1 | 4096 | vllm-fa2 |  | 27.28 | 27.25–27.34 | 308 |
| 1 | 4096 | flashinfer-tensor-cores |  | 19.27 | 19.26–19.28 | 436 |
# skipped flashinfer: ead_dim_qk_128_head_dim_vo_128_posenc_0_use_swa_False_use_logits_cap_False/batch_decode.cu:63: Unsupported group_size: 7
| 1 | 16384 | lab-per-query-head | 1 | 1682.18 | 1675.95–1689.51 | 20 |
| 1 | 16384 | lab-gqa | 1 | 2395.48 | 2394.54–2396.33 | 14 |
| 1 | 16384 | lab-gqa-split | 64 | 74.11 | 74.09–74.17 | 453 |
| 1 | 16384 | lab-gqa-split-tc | 22 | 44.03 | 44.00–44.05 | 762 |
| 1 | 16384 | vllm-fa2 |  | 50.38 | 50.33–50.50 | 666 |
| 1 | 16384 | flashinfer-tensor-cores |  | 40.36 | 40.32–40.43 | 832 |
# skipped flashinfer: ead_dim_qk_128_head_dim_vo_128_posenc_0_use_swa_False_use_logits_cap_False/batch_decode.cu:63: Unsupported group_size: 7
| 4 | 1024 | lab-per-query-head | 1 | 107.28 | 107.20–107.43 | 79 |
| 4 | 1024 | lab-gqa | 1 | 157.63 | 157.54–157.67 | 54 |
| 4 | 1024 | lab-gqa-split | 16 | 26.25 | 26.24–26.25 | 322 |
| 4 | 1024 | lab-gqa-split-tc | 5 | 18.37 | 18.36–18.39 | 460 |
| 4 | 1024 | vllm-fa2 |  | 26.20 | 26.18–26.38 | 322 |
| 4 | 1024 | flashinfer-tensor-cores |  | 18.65 | 18.64–18.72 | 453 |
# skipped flashinfer: ead_dim_qk_128_head_dim_vo_128_posenc_0_use_swa_False_use_logits_cap_False/batch_decode.cu:63: Unsupported group_size: 7
| 4 | 4096 | lab-per-query-head | 1 | 425.02 | 423.17–425.60 | 79 |
| 4 | 4096 | lab-gqa | 1 | 613.58 | 613.15–613.72 | 55 |
| 4 | 4096 | lab-gqa-split | 20 | 59.04 | 58.57–59.06 | 569 |
| 4 | 4096 | lab-gqa-split-tc | 6 | 40.24 | 40.20–40.29 | 835 |
| 4 | 4096 | vllm-fa2 |  | 48.04 | 48.01–48.08 | 700 |
| 4 | 4096 | flashinfer-tensor-cores |  | 39.12 | 39.10–39.16 | 859 |
# skipped flashinfer: ead_dim_qk_128_head_dim_vo_128_posenc_0_use_swa_False_use_logits_cap_False/batch_decode.cu:63: Unsupported group_size: 7
| 4 | 16384 | lab-per-query-head | 1 | 1692.33 | 1686.95–1697.11 | 79 |
| 4 | 16384 | lab-gqa | 1 | 2432.91 | 2430.18–2434.28 | 55 |
| 4 | 16384 | lab-gqa-split | 20 | 192.48 | 192.32–192.61 | 698 |
| 4 | 16384 | lab-gqa-split-tc | 6 | 124.57 | 124.50–124.70 | 1078 |
| 4 | 16384 | vllm-fa2 |  | 123.97 | 123.84–124.05 | 1083 |
| 4 | 16384 | flashinfer-tensor-cores |  | 113.03 | 112.89–113.15 | 1188 |
# skipped flashinfer: ead_dim_qk_128_head_dim_vo_128_posenc_0_use_swa_False_use_logits_cap_False/batch_decode.cu:63: Unsupported group_size: 7
| 16 | 1024 | lab-per-query-head | 1 | 127.56 | 127.51–127.61 | 265 |
| 16 | 1024 | lab-gqa | 1 | 159.41 | 159.34–159.48 | 212 |
| 16 | 1024 | lab-gqa-split | 5 | 57.42 | 57.39–57.45 | 588 |
| 16 | 1024 | lab-gqa-split-tc | 2 | 39.11 | 39.07–39.12 | 864 |
| 16 | 1024 | vllm-fa2 |  | 47.60 | 47.54–47.62 | 710 |
| 16 | 1024 | flashinfer-tensor-cores |  | 40.21 | 40.19–40.24 | 840 |
# skipped flashinfer: ead_dim_qk_128_head_dim_vo_128_posenc_0_use_swa_False_use_logits_cap_False/batch_decode.cu:63: Unsupported group_size: 7
| 16 | 4096 | lab-per-query-head | 1 | 498.79 | 498.58–498.90 | 270 |
| 16 | 4096 | lab-gqa | 1 | 619.78 | 619.42–620.32 | 217 |
| 16 | 4096 | lab-gqa-split | 5 | 190.33 | 190.22–190.41 | 706 |
| 16 | 4096 | lab-gqa-split-tc | 2 | 121.58 | 121.49–121.69 | 1106 |
| 16 | 4096 | vllm-fa2 |  | 121.93 | 121.80–122.04 | 1103 |
| 16 | 4096 | flashinfer-tensor-cores |  | 111.81 | 111.64–112.11 | 1203 |
# skipped flashinfer: ead_dim_qk_128_head_dim_vo_128_posenc_0_use_swa_False_use_logits_cap_False/batch_decode.cu:63: Unsupported group_size: 7
| 16 | 16384 | lab-per-query-head | 1 | 1976.60 | 1975.39–1978.93 | 272 |
| 16 | 16384 | lab-gqa | 1 | 2458.17 | 2457.37–2474.67 | 218 |
| 16 | 16384 | lab-gqa-split | 5 | 710.47 | 710.11–710.95 | 756 |
| 16 | 16384 | lab-gqa-split-tc | 2 | 434.53 | 434.41–434.83 | 1236 |
| 16 | 16384 | vllm-fa2 |  | 413.17 | 412.57–413.42 | 1300 |
| 16 | 16384 | flashinfer-tensor-cores |  | 402.16 | 400.73–402.32 | 1336 |
# skipped flashinfer: ead_dim_qk_128_head_dim_vo_128_posenc_0_use_swa_False_use_logits_cap_False/batch_decode.cu:63: Unsupported group_size: 7
| 64 | 1024 | lab-per-query-head | 1 | 322.85 | 322.52–323.22 | 419 |
| 64 | 1024 | lab-gqa | 1 | 225.65 | 225.48–225.74 | 599 |
| 64 | 1024 | lab-gqa-split | 1 | 225.61 | 225.44–225.68 | 599 |
| 64 | 1024 | lab-gqa-split-tc | 1 | 108.40 | 108.34–108.59 | 1247 |
| 64 | 1024 | vllm-fa2 |  | 140.90 | 140.86–141.04 | 959 |
| 64 | 1024 | flashinfer-tensor-cores |  | 119.55 | 119.26–119.68 | 1130 |
# skipped flashinfer: ead_dim_qk_128_head_dim_vo_128_posenc_0_use_swa_False_use_logits_cap_False/batch_decode.cu:63: Unsupported group_size: 7
| 64 | 4096 | lab-per-query-head | 1 | 1274.05 | 1272.45–1275.14 | 422 |
| 64 | 4096 | lab-gqa | 1 | 873.43 | 872.91–873.66 | 616 |
| 64 | 4096 | lab-gqa-split | 5 | 704.00 | 703.82–704.33 | 764 |
| 64 | 4096 | lab-gqa-split-tc | 1 | 402.20 | 401.69–402.71 | 1337 |
| 64 | 4096 | vllm-fa2 |  | 506.70 | 506.46–507.30 | 1061 |
| 64 | 4096 | flashinfer-tensor-cores |  | 436.35 | 435.37–436.60 | 1232 |
# skipped flashinfer: ead_dim_qk_128_head_dim_vo_128_posenc_0_use_swa_False_use_logits_cap_False/batch_decode.cu:63: Unsupported group_size: 7
| 64 | 16384 | lab-per-query-head | 1 | 5177.60 | 5158.91–5191.42 | 415 |
| 64 | 16384 | lab-gqa | 1 | 3465.22 | 3462.83–3466.92 | 620 |
| 64 | 16384 | lab-gqa-split | 5 | 2695.68 | 2694.40–2697.22 | 797 |
| 64 | 16384 | lab-gqa-split-tc | 1 | 1599.41 | 1596.26–1604.61 | 1343 |
| 64 | 16384 | vllm-fa2 |  | 1975.81 | 1974.17–1977.34 | 1087 |
| 64 | 16384 | flashinfer-tensor-cores |  | 1714.18 | 1710.59–1718.27 | 1253 |
