# date=2026-11-07T19:47:22+00:00, gpu=NVIDIA GeForce RTX 3080 Ti, sm_count=80, l2_bytes=6291456, torch=2.14.0+cu130, cuda=13.0, python=3.12.3, vllm=0.33.0, flashinfer=0.7.2.post1, driver=595.95
# errors vs the FP32 reference; pass = |out - ref| <= 0.01 + 0.016 * |ref| everywhere

| case | lab-skip_masked max abs / max rel / pass | vllm-fa2 max abs / max rel / pass | sdpa-flash max abs / max rel / pass |
|---|---|---|---|
| len 1, 28/4 | 0.00e+00 / 0.00e+00 / pass | 0.00e+00 / 0.00e+00 / pass | 0.00e+00 / 0.00e+00 / pass |
| len 15, 28/4 | 8.70e-03 / 2.37e-03 / pass | 8.70e-03 / 2.37e-03 / pass | 8.70e-03 / 2.37e-03 / pass |
| len 16, 28/4 | 8.70e-03 / 2.37e-03 / pass | 8.70e-03 / 2.37e-03 / pass | 8.70e-03 / 2.37e-03 / pass |
| len 17, 28/4 | 8.70e-03 / 2.37e-03 / pass | 8.70e-03 / 2.37e-03 / pass | 8.70e-03 / 2.37e-03 / pass |
| len 63, 28/4 | 8.70e-03 / 2.37e-03 / pass | 8.70e-03 / 2.37e-03 / pass | 8.70e-03 / 2.37e-03 / pass |
| len 64, 28/4 | 8.70e-03 / 2.37e-03 / pass | 8.70e-03 / 2.37e-03 / pass | 8.70e-03 / 2.37e-03 / pass |
| len 65, 28/4 | 8.70e-03 / 2.37e-03 / pass | 8.70e-03 / 2.37e-03 / pass | 8.70e-03 / 2.37e-03 / pass |
| len 127, 28/4 | 8.70e-03 / 2.37e-03 / pass | 8.70e-03 / 2.37e-03 / pass | 8.70e-03 / 2.37e-03 / pass |
| len 128, 28/4 | 8.70e-03 / 2.37e-03 / pass | 8.70e-03 / 2.37e-03 / pass | 8.70e-03 / 2.37e-03 / pass |
| len 129, 28/4 | 8.70e-03 / 2.37e-03 / pass | 8.70e-03 / 2.37e-03 / pass | 8.70e-03 / 2.37e-03 / pass |
| len 1000, 28/4 | 7.91e-03 / 2.65e-03 / pass | 7.91e-03 / 2.65e-03 / pass | 7.91e-03 / 2.65e-03 / pass |
| len 4096, 28/4 | 7.50e-03 / 2.46e-03 / pass | 7.50e-03 / 2.46e-03 / pass | 7.50e-03 / 2.46e-03 / pass |
| len 8192, 28/4 | 8.62e-03 / 2.32e-03 / pass | 8.62e-03 / 2.32e-03 / pass | 8.62e-03 / 2.32e-03 / pass |
| ragged x12, 28/4 | 1.04e-02 / 2.60e-03 / pass | 1.04e-02 / 2.60e-03 / pass | 1.04e-02 / 2.60e-03 / pass |
| ragged x12, 12/2 | 1.04e-02 / 2.56e-03 / pass | 1.04e-02 / 2.56e-03 / pass | 1.04e-02 / 2.56e-03 / pass |
| ragged x12, 16/2 | 9.93e-03 / 2.51e-03 / pass | 9.93e-03 / 2.51e-03 / pass | 9.93e-03 / 2.51e-03 / pass |
| ragged x12, 32/8 | 9.92e-03 / 2.54e-03 / pass | 9.92e-03 / 2.54e-03 / pass | 9.92e-03 / 2.54e-03 / pass |
| ragged x12, 8/8 | 8.75e-03 / 2.22e-03 / pass | 8.75e-03 / 2.22e-03 / pass | 8.75e-03 / 2.22e-03 / pass |
| ragged, fused-QKV strides, 28/4 | 9.43e-03 / 2.56e-03 / pass | 9.43e-03 / 2.56e-03 / pass | 9.43e-03 / 2.56e-03 / pass |
| ragged, non-causal, 28/4 | 3.87e-03 / 1.26e-03 / pass | 3.87e-03 / 1.26e-03 / pass | 3.87e-03 / 1.26e-03 / pass |
| large scores (q x 30, scale 1), 28/4 | 1.24e-02 / 2.49e-03 / pass | 1.24e-02 / 2.49e-03 / pass | 1.24e-02 / 2.49e-03 / pass |

| implementation | worst max abs | worst max rel | shapes passed |
|---|---:|---:|---:|
| lab-simt | 1.24e-02 | 2.59e-03 | 21/21 |
| lab-mma | 1.24e-02 | 2.65e-03 | 21/21 |
| lab-swizzle | 1.24e-02 | 2.65e-03 | 21/21 |
| lab-cp_async | 1.24e-02 | 2.65e-03 | 21/21 |
| lab-fast_exp2 | 1.24e-02 | 2.65e-03 | 21/21 |
| lab-hoisted | 1.24e-02 | 2.65e-03 | 21/21 |
| lab-heavy_first | 1.24e-02 | 2.65e-03 | 21/21 |
| lab-skip_masked | 1.24e-02 | 2.65e-03 | 21/21 |
| vllm-fa2 | 1.24e-02 | 2.65e-03 | 21/21 |
| sdpa-flash | 1.24e-02 | 2.65e-03 | 21/21 |
