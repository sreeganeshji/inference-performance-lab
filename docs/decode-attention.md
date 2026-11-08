# Paged GQA decode attention for Qwen2.5-7B

A CUDA decode-attention kernel for vLLM's paged KV cache, served inside vLLM 0.33 as a custom
attention backend. It targets Qwen2.5-7B's attention (28 query heads, 4 KV heads, head dimension
128, BF16) and is measured on an A100-SXM4-40GB and an RTX 3080 Ti. Every number below links to the
result file it came from.

| Result | Measurement |
|---|---|
| Kernel vs vLLM's FlashAttention-2, A100, 12 batch × context shapes | **1.03–1.48× faster in 12/12** |
| Kernel vs FlashInfer's tensor-core decode, A100 | faster in 9/12 (up to 1.10×), tied at 16 × 16K; 0.94–0.96× at batch 1 with 4K–16K context |
| Peak achieved bandwidth, A100 | **1,351 GB/s**, 87% of the 1,555 GB/s HBM2 peak |
| vLLM serving, real Qwen2.5-7B-Instruct, A100, CUSTOM vs FLASH_ATTN | **1.01–1.29×** output throughput across 6 workloads |
| Where the largest serving gain comes from | mixed prefill + decode steps: attention 2.5× faster per layer |
| Greedy output against FLASH_ATTN, real weights | 334/384 tokens identical, the same as vLLM's Triton backend |
| Scheduler CPU time exposed as GPU idle | about 0 ms per 22–60 s trace (async scheduling hides it) |

## The problem

At decode time each sequence contributes one query token per step, and attention reads that
sequence's whole KV cache: 4 KV heads × 128 dimensions × (K + V) × 2 bytes = 2 KiB per token per
layer, 56 KiB per token across 28 layers. One KV row (512 bytes, K and V for one head) serves the
7 query heads of its GQA group, which do 7 × 128 × 2 × 2 = 3,584 FLOP on it: **7 FLOP per byte**.
The A100's FP32 ridge point is about 12.5 FLOP/byte (19.5 TFLOP/s over 1.555 TB/s) and its tensor
ridge about 200, so decode attention is memory-bound on paper. It still needs care: at 7 FLOP/byte
a CUDA-core kernel spends several instructions per FLOP on loads, shuffles and conversions, and
can become issue-bound before it saturates DRAM.

vLLM stores the cache in blocks of 16 tokens per layer as `[num_blocks, num_kv_heads, block_size,
2 * head_size]` logically, laid out `[blocks, tokens, heads, K|V]` physically, so K and V of one
(token, head) are interleaved and a block table maps each sequence's logical blocks to physical
ones.

## Kernel design

`csrc/paged_decode_attention_kernel.cu` holds four variants, each a step in a ladder. All of them
use online softmax in the exp2 domain (scores scaled by `scale × log2 e`), merge per-warp state
`(m, ℓ, acc)` through shared memory, and treat a padded batch slot (`seq_len == 0`) as all-zero
output.

1. **One block per query head** (`per_query_head`). 16 lanes share each 256-byte K or V row with
   one 16-byte load per lane; row dot products reduce with `__shfl_xor_sync` over 8, 4, 2, 1. Each
   KV row is read 7 times, once per query head of its group.
2. **One block per KV head** (`gqa`). The block holds all 7 query heads of the group, so each KV row
   is read once. At small batch this is slower: batch 1 × 4 KV heads is only 4 blocks for 108 SMs.
3. **Split-KV** (`gqa_split`). A third grid dimension splits each sequence's KV range; each split
   writes an FP32 partial output and its log-sum-exp, and a combine kernel weights the splits by
   `exp2(lse − max)`. This is flash-decoding; it turns 4 blocks into enough to fill the GPU.
4. **Tensor cores** (`gqa_split_tc`, the default). S = QKᵀ and P·V run on `mma.sync.m16n8k16`
   (BF16 in, FP32 accumulate) over 16-token chunks, one chunk per warp in turn.
   - K and V go straight from global memory into fragment registers with 16-byte loads, with no
     shared-memory staging. This works because the order in which S sums over head dimensions only
     has to match between Q and K, and the output's dimension order is a relabeling: the kernel
     picks both so every lane's fragment elements are contiguous in memory.
   - S's accumulator fragment is reused, after the softmax, as P's operand fragment.
   - P·V is computed transposed, Oᵀ = Vᵀ Pᵀ, so the 7 query heads sit on the MMA's 8-wide N side
     instead of being zero-padded to 16 rows on M. The first version (heads on M) kept half its O
     accumulators as padding and needed 178 registers (2 blocks/SM); the transposed one uses 142
     (3 blocks/SM).
   - Slots past a sequence's end load zeros into the V fragments, so stale cache contents, possibly
     NaN, never reach the MMA (0 × NaN is NaN); the score mask handles K.

### Choosing the split count

`choose_num_splits` (`src/inference_performance_lab/kernels/extension.py`) minimizes a cost model
in bytes of equivalent DRAM traffic. DRAM saturates once `saturating` blocks run at once; a wave of
n blocks each reading W bytes costs `W × max(n, saturating)`; splits add their FP32 partials
(weighted, because the combine pass is far from streaming speed) and a fixed combine cost. The
resident-block capacity comes from `cudaOccupancyMaxActiveBlocksPerMultiprocessor`, so wave
quantization is modeled: at batch 4 × 4K on the 3080 Ti, 15 splits fill exactly one wave of 240
blocks and 16 splits spill into a second.

The constants are fitted per GPU by `benchmarks/fit_split_model.py`, a grid search that scores each
candidate by its mean regret against the fastest measured split count in a sweep:

| GPU | saturating blocks per SM | mean regret | worst regret | source |
|---|---:|---:|---:|---|
| RTX 3080 Ti (default model, fitted by hand to its sweeps) | 0.8 | — | — | — |
| A100, with the 3080 Ti model | 0.8 | 5.18% | 12.76% | [fit report](../results/data/attention/a100-sxm4-40gb-split-fit-wide-20261107.md) |
| A100, fitted | 1.7 | **0.64%** | **2.30%** | same |

The A100 needs about twice as many blocks in flight per SM before DRAM saturates. That is Little's
law: bytes in flight = bandwidth × latency, and the A100 moves more bytes per SM (14.4 GB/s/SM
against the 3080 Ti's 11.4).

## vLLM integration

`src/inference_performance_lab/kernels/vllm_attention.py` registers `AttentionBackendEnum.CUSTOM`
through the lab's vLLM plugin; serve with `--attention-backend CUSTOM`. It subclasses vLLM's
FlashAttention backend with two changes:

- The metadata builder sets `reorder_batch_threshold = 1`, so vLLM puts decode requests first in
  every batch, and allocates a split workspace once, so full CUDA graphs can capture the kernel.
- `forward` sends the decode rows to the lab kernel and any remaining rows (prefill, chunked
  prefill) to FlashAttention-2, using sliced FlashAttention metadata. Layers the kernel does not
  cover (other head sizes, FP8 KV cache, sliding window, ALiBi, soft-capping, sinks, GQA groups
  above 8, decode context parallelism) fall back to FlashAttention-2 for every row.

Under full CUDA graphs a captured graph replays for any context length, so the split count may
depend only on the number of decode rows; the kernel reads each sequence's length from device
memory and divides its own range into equal splits.

## Correctness

- `tests/test_paged_decode_attention.py`: 125 tests pass on the A100 and on the 3080 Ti
  ([A100 log](../results/data/attention/a100-sxm4-40gb-tests-20261107-a100-split-model.txt)). Every
  variant is compared with an FP32 reference over 5 head layouts (28/4, 12/2, 16/2, 32/8, 8/8),
  block edges, ragged batches, padded slots, forced split counts up to 64, block sizes 16 and 32,
  strided queries, 32K context and large scores, and against vLLM's `flash_attn_varlen_func`.
- A stale-cache test fills every cache slot no valid token owns with NaN. Loading V without the
  zero-fill makes the tensor-core cases fail; with it, all pass.
- `benchmarks/vllm_attention_check.py` runs greedy decoding twice in fresh vLLM processes and
  compares tokens and logprobs. With real Qwen2.5-7B-Instruct weights, every pair of backends
  diverges a little, because BF16 reductions in different orders flip near-ties
  ([pairs](../results/data/attention/a100-sxm4-40gb-vllm-check-pairs-20261107.md)):

| pair | identical greedy tokens | max \|Δ logprob\| |
|---|---:|---:|
| FLASH_ATTN vs CUSTOM (lab) | 334/384 | 0.104 |
| FLASH_ATTN vs TRITON_ATTN | 334/384 | 0.104 |
| FLASH_ATTN vs FLASHINFER | 315/384 | 0.066 |
| CUSTOM vs TRITON_ATTN | 342/384 | 0.094 |

  The lab backend sits inside the range two stock backends show. With the 4-layer dummy-weight
  model, CUSTOM and FLASH_ATTN agree on 384/384 tokens.

## Kernel results

CUDA-graph replay, KV pools rotated until one replay reads at least 3× the L2, interleaved shuffled
trials with one discarded warm-up round, medians. vLLM's KV layout. Times in µs.

**A100-SXM4-40GB** ([grid](../results/data/attention/a100-sxm4-40gb-qwen2.5-7b-20261107-a100-split-model.md)),
with the A100 split model:

| batch × context | lab (splits) | vLLM FA2 | FlashInfer TC | vs FA2 | vs FlashInfer | lab GB/s |
|---|---:|---:|---:|---:|---:|---:|
| 1 × 1K | 13.6 (13) | 15.8 | 14.5 | 1.17× | 1.07× | 156 |
| 1 × 4K | 20.1 (26) | 27.2 | 19.3 | 1.36× | 0.96× | 418 |
| 1 × 16K | 42.8 (45) | 50.4 | 40.3 | 1.18× | 0.94× | 784 |
| 4 × 1K | 17.7 (6) | 26.1 | 18.6 | 1.48× | 1.05× | 478 |
| 4 × 4K | 38.3 (11) | 48.1 | 39.0 | 1.26× | 1.02× | 878 |
| 4 × 16K | 111.6 (12) | 123.9 | 112.9 | 1.11× | 1.01× | 1,203 |
| 16 × 1K | 37.1 (3) | 47.7 | 40.0 | 1.29× | 1.08× | 910 |
| 16 × 4K | 110.3 (3) | 122.6 | 111.9 | 1.11× | 1.01× | 1,219 |
| 16 × 16K | 402.2 (3) | 413.6 | 402.3 | 1.03× | 1.00× | 1,335 |
| 64 × 1K | 108.5 (1) | 141.1 | 119.3 | 1.30× | 1.10× | 1,246 |
| 64 × 4K | 398.0 (1) | 505.8 | 433.6 | 1.27× | 1.09× | 1,351 |
| 64 × 16K | 1,593.1 (1) | 1,972.7 | 1,709.4 | 1.24× | 1.07× | 1,349 |

With the 3080 Ti's split constants
([grid](../results/data/attention/a100-sxm4-40gb-qwen2.5-7b-20261107.md)), the same kernel tied
FA2 at 4 × 16K and 16 × 4K and lost to it by 5% at 16 × 16K; the refit fixed those three. It made
1 × 1K slower (12.3 → 13.6 µs): the model picks 13 splits there, a count the fit's sweep never
measured.

**RTX 3080 Ti** ([grid](../results/data/attention/rtx3080ti-qwen2.5-7b-20261107.md), measured
with separate K and V caches before the benchmark gained vLLM's interleaved layout): faster than
FA2 in 12/12 shapes (1.07–1.75×), and faster than FlashInfer TC except at batch 64 and 16 × 16K,
where FlashInfer is 0.4–6% faster. Peak 843 GB/s.

## Serving results

`benchmarks/serving_attention_ab.py` alternates `vllm serve` instances (FLASH_ATTN, then CUSTOM,
then the reverse) for 3 rounds and runs `vllm bench serve` per cell: random-token prompts of fixed
length, EOS ignored, prefix caching off, burst arrival at the given concurrency, 4 prompts per
concurrency slot. vLLM's defaults otherwise apply (full and piecewise CUDA graphs, async
scheduling, chunked prefill). The baseline runs without the lab's plugin. Medians of 3 rounds
([summary](../results/data/attention/serving/a100-sxm4-40gb-20261107/summary.md)):

| input : output : concurrency | FLASH_ATTN tok/s | CUSTOM tok/s | ratio | TPOT p50 ms |
|---|---:|---:|---:|---|
| 4096 : 256 : 1 | 73.4 | 75.5 | 1.03× | 12.57 → 12.21 |
| 4096 : 256 : 8 | 372.5 | 376.7 | 1.01× | 18.51 → 18.32 |
| 4096 : 256 : 32 | 597.9 | 650.7 | 1.09× | 50.08 → 45.50 |
| 4096 : 256 : 64 | 574.1 | 741.8 | **1.29×** | 110.1 → 81.7 |
| 16384 : 256 : 8 | 134.7 | 142.1 | 1.05× | 49.01 → 46.99 |
| 16384 : 256 : 16 | 139.1 | 160.1 | 1.15× | 104.9 → 89.9 |

One FLASH_ATTN run (round 1 of 4096 : 256 : 32) is excluded, because a mistakenly started split
sweep shared the GPU for 35 s of it; including it changes that cell's ratio from 1.088× to 1.089×.
These runs used the 3080 Ti split constants (the A100 fit came later), so they understate the
kernel's gain at small decode batches.

## Profiling: where the serving gain comes from

`benchmarks/profile_serving_steps.py` traces one serving cell per backend with Nsight Systems
(`--trace=cuda,nvtx --cuda-graph-trace=node`, collection paused through a warm-up run) and reduces
the trace; `benchmarks/attention_kernels_by_grid.py` splits attention time by launch grid
([profiles](../results/data/attention/serving-profile/a100-sxm4-40gb-20261107/)).

At 4096 : 256 : 64, attention is 33.7% of GPU kernel time with FLASH_ATTN and 19.2% with CUSTOM
(lab decode 12.6%, FlashAttention-2 on prefill rows 6.6%). Attention time drops by 10.7 s and the
recorded window by 11.1 s, so attention accounts for about 96% of the end-to-end difference. The
launch grids show why
([by grid](../results/data/attention/serving-profile/a100-sxm4-40gb-20261107/attention-kernels-by-grid.md)):

| step type | FLASH_ATTN | CUSTOM |
|---|---|---|
| decode only | FA2 packs each GQA group into one block (gridZ = 4 KV heads): **478 µs** per layer | lab kernel: **358 µs** |
| prefill chunk + decodes | one FA2 varlen launch sized for the chunk (gridX = 32 query tiles, gridZ = 28 query heads), so decode rows lose GQA packing: **1,967 µs** per layer | FA2 on the prefill rows (429 µs) + lab kernel on the decode rows (358 µs): **≈ 790 µs** |

So most of the 1.29× comes from the backend's design, keeping decode rows out of the prefill
launch, and the rest from the faster kernel. The same split would help any decode kernel; it is
why the gain grows with concurrency (more decode rows ride along with each prefill chunk).

**Scheduler.** In all four traces the GPU is busy 99.5–99.8% of the window. `Scheduler.schedule`
takes a median 0.9 ms per step at 64 running requests and 1.45 ms at 128, and block allocation and
freeing (`BlockPool.get_new_blocks`, `free_blocks`) total 18–35 ms per trace; but the scheduler time
that falls while the GPU is idle totals 0.4–2.1 ms over 22–60 s windows. vLLM's async scheduling
overlaps it with the GPU, so a faster scheduler or block pool would not change these workloads'
throughput.

## Nsight Compute on the A100

`ncu --set full` of one call per implementation at three shapes: one where the lab kernel wins most,
one tie and the one where FlashInfer wins
([summary](../results/data/attention/ncu/a100-sxm4-40gb-decode-ncu-summary.md), made by
`benchmarks/summarize_ncu_details.py` from the `*-details.txt` reports beside it). ncu locks clocks
and flushes caches between kernels, so its durations are longer than the benchmark's, most of all
for the small combine kernels; compare implementations with each other.

| shape | implementation | main kernel µs | DRAM % | regs | waves | combine / merge µs |
|---|---|---:|---:|---:|---:|---:|
| 64 × 4K | lab (1 split) | 400.6 | **86.7** | 142 | **0.79** | — |
| | FA2 | 553.3 | 62.8 | 178 | 1.19 | — |
| | FlashInfer TC | 458.7 | 75.8 | 180 | 1.69 | 7.1 |
| 16 × 16K | lab (3 splits) | 411.2 | 84.4 | 142 | 0.59 | 6.5 |
| | FA2 | 410.1 | 84.7 | 252 | 0.89 | 10.1 |
| | FlashInfer TC | 409.2 | 84.8 | 180 | 1.00 | 7.9 |
| 1 × 16K | lab (45 splits) | 41.5 | 55.8 | 142 | 0.56 | 19.0 |
| | FA2 | 40.9 | 56.5 | 252 | 0.80 | 30.1 |
| | FlashInfer TC | 41.7 | 55.3 | 180 | 1.00 | **8.5** |

Three regimes:

- **Wave quantization decides batch 64.** All three launch 256–364 blocks. At 142 registers the lab
  kernel fits 3 blocks per SM, so its 256 blocks finish in one partial wave (0.79) and keep DRAM 87%
  busy. FA2's split kernel uses 178 registers, fits 2 blocks per SM and needs 1.19 waves; the second,
  mostly empty wave runs at a fraction of the bandwidth, and DRAM averages 63%.
- **Bandwidth decides batch 16 × 16K.** Each kernel streams at about 85% of peak and their times are
  within 0.5%. Nothing but moving fewer bytes would help.
- **The combine decides batch 1.** The main kernels are equal (41–42 µs), and each issues for only
  11–14% of cycles at about 10% achieved occupancy: 180 blocks cannot hide DRAM latency, and only 56%
  of bandwidth is used. The difference is the second kernel. The lab's combine runs one block per
  query head (28 blocks) over 45 splits, while FlashInfer's persistent merge spreads the same work
  over all 108 SMs. A combine with more parallelism, or one fused into the split kernel (the last
  block of a sequence merges), is the next step at small batch.

## Limitations

- Serving results cover one GPU (A100-SXM4-40GB), one model, BF16, synthetic fixed-length prompts
  and 3 rounds per cell.
- FlashInfer's tensor-core decode is 4–6% faster at batch 1 with 4K–16K context on the A100,
  because of the combine kernel (see Nsight Compute above).
- The split model chooses among all counts but was fitted on 16 measured ones; at 1 × 1K it picks an
  unmeasured 13 and loses 10% to the 3080 Ti constants. Fitting on every count would fix it.
- Under CUDA graphs the split count assumes a 32K context, since it may depend only on the decode
  count.
- The kernel covers head dimension 128, GQA groups up to 8 and a BF16 KV cache.

## Reproduce

```bash
scripts/attention_gpu_run.sh setup      # fresh GPU machine: driver, CUDA 13, venv, vLLM 0.33
scripts/attention_gpu_run.sh test
scripts/attention_gpu_run.sh kernels    # kernel grid and split sweep
scripts/attention_gpu_run.sh check      # greedy check on Qwen2.5-7B-Instruct (15 GB download)
scripts/attention_gpu_run.sh controls   # the same check between stock backends
scripts/attention_gpu_run.sh split-fit  # wide sweep and fit of the split model
scripts/attention_gpu_run.sh serving    # serving A/B, about an hour on an A100
scripts/attention_gpu_run.sh profile    # Nsight Systems traces
```
