# Prefill attention measurements

Evidence for `csrc/flash_prefill_attention_kernel.cu`. [docs/flash-prefill.md](../../../../docs/flash-prefill.md)
explains the kernel and reads these numbers. Every file here comes from
`scripts/flash_prefill_gpu_run.sh` on one RTX 3080 Ti (sm_86, 80 SMs, 12 GB). The steps are listed
below; GPU steps ran under the shared `flock` GPU lock, so no other job used the GPU at the same
time.

| file | step | contents |
|---|---|---|
| `rtx3080ti-machine.txt` | `machine` | GPU, driver, clocks, power limit, toolchain versions |
| `rtx3080ti-bf16-gemm-peak.{md,json}` | `peak` | cuBLAS BF16 GEMM with FP32 accumulation: the measured tensor peak behind every "% of peak" |
| `rtx3080ti-steps.{md,jsonl}` | `steps` | every development step (variants `simt` .. `skip_masked`) at 1×512, 1×2048, 1×8192 and 4×8192 |
| `rtx3080ti-qwen2.5-7b-grid.{md,jsonl}` | `grid` | Qwen2.5-7B (28 query heads, 4 KV heads), batch 1–8 × 512–16K tokens, against FA2 and SDPA |
| `rtx3080ti-accuracy.{md,jsonl}` | `accuracy` | max abs / max rel error per shape against the FP32 reference, for every variant, FA2 and SDPA |
| `ncu/rtx3080ti-<shape>-summary.md` | `ncu` | Nsight Compute, side by side: step 3, the final kernel and FA2 (occupancy, pipes, bank conflicts, stalls, SASS mix) |
| `ncu/rtx3080ti-<shape>-<impl>-details.txt` | `ncu` | the full `--set full` details page for the final kernel and FA2 |

Timing method: each implementation's calls are captured in one CUDA graph and replayed. Inputs
rotate until a replay touches three times the L2 capacity. Trials replay the graphs in shuffled
order, and the median per-call time is reported. The SM clock is sampled through NVML during the
trials and reported per row: GeForce boost clocks follow power and temperature. Nsight Compute
locks the clock for its own runs, so its kernel times are slower than the benchmark's and are
comparable only with each other.

The `.ncu-rep` files themselves (40–80 MB each) are left out of the repository. The script writes
them to `~/.cache/ipl-flash-ncu`.
