# qwen2.5-7b: 28 query heads, 4 KV heads, head_dim 128
# date=2026-11-07T16:35:08+00:00, gpu=NVIDIA GeForce RTX 3080 Ti, sm_count=80, l2_bytes=6291456, torch=2.14.0+cu130, cuda=13.0, python=3.12.3, vllm=0.33.0, flashinfer=0.7.2.post1, driver=595.95
| batch | context | implementation | splits | median µs | min–max µs | GB/s |
|---:|---:|---|---:|---:|---:|---:|
# skipped flashinfer: ead_dim_qk_128_head_dim_vo_128_posenc_0_use_swa_False_use_logits_cap_False/batch_decode.cu:63: Unsupported group_size: 7
| 1 | 1024 | lab-per-query-head | 1 | 75.78 | 75.69–82.73 | 28 |
| 1 | 1024 | lab-gqa | 1 | 101.28 | 100.08–116.61 | 21 |
| 1 | 1024 | lab-gqa-split | 16 | 12.66 | 12.54–14.68 | 167 |
| 1 | 1024 | lab-gqa-split-tc | 16 | 9.09 | 9.05–10.49 | 232 |
| 1 | 1024 | vllm-fa2 |  | 14.76 | 14.65–17.17 | 143 |
| 1 | 1024 | flashinfer-tensor-cores |  | 12.04 | 11.99–14.18 | 175 |
# skipped flashinfer: ead_dim_qk_128_head_dim_vo_128_posenc_0_use_swa_False_use_logits_cap_False/batch_decode.cu:63: Unsupported group_size: 7
| 1 | 4096 | lab-per-query-head | 1 | 301.35 | 298.29–308.33 | 28 |
| 1 | 4096 | lab-gqa | 1 | 393.96 | 387.24–395.07 | 21 |
| 1 | 4096 | lab-gqa-split | 52 | 23.16 | 23.03–23.86 | 363 |
| 1 | 4096 | lab-gqa-split-tc | 16 | 17.01 | 16.96–17.41 | 494 |
| 1 | 4096 | vllm-fa2 |  | 29.19 | 29.07–30.17 | 288 |
| 1 | 4096 | flashinfer-tensor-cores |  | 20.36 | 20.30–20.41 | 413 |
# skipped flashinfer: ead_dim_qk_128_head_dim_vo_128_posenc_0_use_swa_False_use_logits_cap_False/batch_decode.cu:63: Unsupported group_size: 7
| 1 | 16384 | lab-per-query-head | 1 | 1196.88 | 1184.63–1199.22 | 28 |
| 1 | 16384 | lab-gqa | 1 | 1573.26 | 1549.55–1633.67 | 21 |
| 1 | 16384 | lab-gqa-split | 57 | 55.02 | 54.73–56.90 | 610 |
| 1 | 16384 | lab-gqa-split-tc | 16 | 46.99 | 46.78–48.37 | 714 |
| 1 | 16384 | vllm-fa2 |  | 68.13 | 67.63–72.31 | 493 |
| 1 | 16384 | flashinfer-tensor-cores |  | 51.10 | 50.28–52.45 | 657 |
# skipped flashinfer: ead_dim_qk_128_head_dim_vo_128_posenc_0_use_swa_False_use_logits_cap_False/batch_decode.cu:63: Unsupported group_size: 7
| 4 | 1024 | lab-per-query-head | 1 | 79.85 | 78.87–81.57 | 106 |
| 4 | 1024 | lab-gqa | 1 | 101.53 | 100.35–102.51 | 83 |
| 4 | 1024 | lab-gqa-split | 13 | 20.90 | 20.71–21.98 | 404 |
| 4 | 1024 | lab-gqa-split-tc | 4 | 16.58 | 16.49–16.67 | 509 |
| 4 | 1024 | vllm-fa2 |  | 29.05 | 28.62–29.24 | 291 |
| 4 | 1024 | flashinfer-tensor-cores |  | 20.11 | 20.03–24.75 | 420 |
# skipped flashinfer: ead_dim_qk_128_head_dim_vo_128_posenc_0_use_swa_False_use_logits_cap_False/batch_decode.cu:63: Unsupported group_size: 7
| 4 | 4096 | lab-per-query-head | 1 | 312.28 | 308.74–314.75 | 108 |
| 4 | 4096 | lab-gqa | 1 | 400.60 | 394.43–414.98 | 84 |
| 4 | 4096 | lab-gqa-split | 15 | 53.01 | 52.86–54.36 | 634 |
| 4 | 4096 | lab-gqa-split-tc | 4 | 46.87 | 46.43–47.23 | 717 |
| 4 | 4096 | vllm-fa2 |  | 73.91 | 73.44–76.64 | 455 |
| 4 | 4096 | flashinfer-tensor-cores |  | 49.81 | 49.59–58.55 | 675 |
# skipped flashinfer: ead_dim_qk_128_head_dim_vo_128_posenc_0_use_swa_False_use_logits_cap_False/batch_decode.cu:63: Unsupported group_size: 7
| 4 | 16384 | lab-per-query-head | 1 | 1272.26 | 1246.66–1350.71 | 106 |
| 4 | 16384 | lab-gqa | 1 | 1623.66 | 1599.09–1665.26 | 83 |
| 4 | 16384 | lab-gqa-split | 15 | 177.33 | 176.41–178.46 | 757 |
| 4 | 16384 | lab-gqa-split-tc | 4 | 164.11 | 163.59–166.19 | 818 |
| 4 | 16384 | vllm-fa2 |  | 196.38 | 191.05–215.13 | 684 |
| 4 | 16384 | flashinfer-tensor-cores |  | 168.24 | 165.86–169.04 | 798 |
# skipped flashinfer: ead_dim_qk_128_head_dim_vo_128_posenc_0_use_swa_False_use_logits_cap_False/batch_decode.cu:63: Unsupported group_size: 7
| 16 | 1024 | lab-per-query-head | 1 | 97.78 | 95.81–101.70 | 346 |
| 16 | 1024 | lab-gqa | 1 | 108.07 | 105.82–112.86 | 313 |
| 16 | 1024 | lab-gqa-split | 3 | 60.60 | 58.64–64.64 | 558 |
| 16 | 1024 | lab-gqa-split-tc | 1 | 44.86 | 44.62–46.09 | 753 |
| 16 | 1024 | vllm-fa2 |  | 67.19 | 65.32–68.96 | 503 |
| 16 | 1024 | flashinfer-tensor-cores |  | 52.37 | 51.01–53.99 | 645 |
# skipped flashinfer: ead_dim_qk_128_head_dim_vo_128_posenc_0_use_swa_False_use_logits_cap_False/batch_decode.cu:63: Unsupported group_size: 7
| 16 | 4096 | lab-per-query-head | 1 | 383.78 | 374.29–398.32 | 350 |
| 16 | 4096 | lab-gqa | 1 | 426.97 | 417.11–434.79 | 315 |
| 16 | 4096 | lab-gqa-split | 7 | 195.77 | 189.02–213.03 | 687 |
| 16 | 4096 | lab-gqa-split-tc | 1 | 163.31 | 161.44–163.89 | 823 |
| 16 | 4096 | vllm-fa2 |  | 196.68 | 193.56–201.01 | 684 |
| 16 | 4096 | flashinfer-tensor-cores |  | 170.18 | 167.90–180.09 | 790 |
# skipped flashinfer: ead_dim_qk_128_head_dim_vo_128_posenc_0_use_swa_False_use_logits_cap_False/batch_decode.cu:63: Unsupported group_size: 7
| 16 | 16384 | lab-per-query-head | 1 | 1570.54 | 1526.95–1650.35 | 342 |
| 16 | 16384 | lab-gqa | 1 | 1731.41 | 1688.34–1795.31 | 310 |
| 16 | 16384 | lab-gqa-split | 15 | 703.65 | 688.05–741.21 | 763 |
| 16 | 16384 | lab-gqa-split-tc | 1 | 637.14 | 630.14–639.65 | 843 |
| 16 | 16384 | vllm-fa2 |  | 680.73 | 664.33–691.29 | 789 |
| 16 | 16384 | flashinfer-tensor-cores |  | 634.68 | 633.41–644.27 | 846 |
# skipped flashinfer: ead_dim_qk_128_head_dim_vo_128_posenc_0_use_swa_False_use_logits_cap_False/batch_decode.cu:63: Unsupported group_size: 7
| 64 | 1024 | lab-per-query-head | 1 | 319.18 | 310.47–346.04 | 423 |
| 64 | 1024 | lab-gqa | 1 | 275.54 | 268.27–279.92 | 490 |
| 64 | 1024 | lab-gqa-split | 4 | 227.01 | 218.55–236.81 | 595 |
| 64 | 1024 | lab-gqa-split-tc | 1 | 182.39 | 178.81–183.43 | 741 |
| 64 | 1024 | vllm-fa2 |  | 202.75 | 196.38–205.64 | 667 |
| 64 | 1024 | flashinfer-tensor-cores |  | 172.30 | 169.54–175.03 | 784 |
# skipped flashinfer: ead_dim_qk_128_head_dim_vo_128_posenc_0_use_swa_False_use_logits_cap_False/batch_decode.cu:63: Unsupported group_size: 7
| 64 | 4096 | lab-per-query-head | 1 | 1280.54 | 1252.71–1319.09 | 420 |
| 64 | 4096 | lab-gqa | 1 | 1073.57 | 1049.34–1101.16 | 501 |
| 64 | 4096 | lab-gqa-split | 8 | 787.61 | 751.03–840.66 | 683 |
| 64 | 4096 | lab-gqa-split-tc | 4 | 657.51 | 647.85–660.28 | 818 |
| 64 | 4096 | vllm-fa2 |  | 779.92 | 762.32–805.79 | 690 |
| 64 | 4096 | flashinfer-tensor-cores |  | 645.78 | 635.65–649.14 | 833 |
# skipped flashinfer: ead_dim_qk_128_head_dim_vo_128_posenc_0_use_swa_False_use_logits_cap_False/batch_decode.cu:63: Unsupported group_size: 7
| 64 | 16384 | lab-per-query-head | 1 | 5221.99 | 5077.20–5300.63 | 411 |
| 64 | 16384 | lab-gqa | 1 | 4274.18 | 4202.29–4482.24 | 503 |
| 64 | 16384 | lab-gqa-split | 13 | 2925.86 | 2790.24–3087.36 | 734 |
| 64 | 16384 | lab-gqa-split-tc | 4 | 2570.50 | 2532.48–2575.23 | 836 |
| 64 | 16384 | vllm-fa2 |  | 3081.07 | 3030.14–3244.15 | 697 |
| 64 | 16384 | flashinfer-tensor-cores |  | 2503.17 | 2498.94–2547.07 | 858 |
