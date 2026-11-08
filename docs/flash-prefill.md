# FlashAttention-2 causal prefill on an RTX 3080 Ti

`csrc/flash_prefill_attention_kernel.cu` is a FlashAttention-2-style forward pass for causal
prefill over packed variable-length sequences: the tiled, IO-aware attention of
[FlashAttention](https://arxiv.org/abs/2205.14135) with the work partitioning of
[FlashAttention-2](https://arxiv.org/abs/2307.08691). It is written from the papers, the PTX ISA
(`mma.sync.m16n8k16`, `ldmatrix`, `cp.async`) and the CUDA C++ Programming Guide, and it reuses
the fragment helpers of this repository's decode kernel. This document is the decision journal:
every step was measured before the next one was written. It then walks the kernel from the
perspective of one block, one warp and one thread.

**Result (RTX 3080 Ti, Qwen2.5-7B shapes: 28 query heads, 4 KV heads, head dim 128).** The
final kernel is faster than vLLM's FlashAttention-2 in **all 24 measured shapes** (batch 1–8 ×
512–16,384 tokens): **1.015× to 1.225×, geometric mean 1.066×**. It is 1.03–1.25× faster than
SDPA's flash backend, and reaches 69.4 TFLOP/s at 1×8192, 92% of the measured 75.5 TFLOP/s cuBLAS
BF16 peak. Every tested shape matches an FP32 reference within the same error as FA2.

Inside an SM the two kernels are equally efficient: the tensor pipe is busy 47.5% vs 47.2% of
cycles, against a ceiling of about 50% on this GPU. The difference is scheduling. Launching each
sequence's longest causal tiles first keeps the SMs busy 94.6% of the time at 1×1024 (FA2: 77.4%)
and 99.9% at 1×8192 (FA2: 95.8%). So the gain is largest on short prompts and shrinks to 1.5–2%
at 8×16K, where the tail is a small part of the run.

## Interface and scope

```python
from inference_performance_lab.kernels.extension import flash_prefill_attention

out = flash_prefill_attention(query, key, value, cu_seqlens, max_seqlen, scale)  # causal=True
```

| argument | shape / type | notes |
|---|---|---|
| `query` | `[total_tokens, num_q_heads, 128]` BF16 | any token and head strides that are multiples of 8 elements (e.g. views of a fused QKV output) |
| `key`, `value` | `[total_tokens, num_kv_heads, 128]` BF16 | the same tokens as `query`: a fresh prompt, no prefix cache |
| `cu_seqlens` | int32 `[num_seqs + 1]` | sequence *s* owns tokens `cu_seqlens[s]:cu_seqlens[s+1]` of all three; empty sequences are allowed |
| `max_seqlen` | int | at least the longest sequence; it sizes the grid, so the launch needs no device-to-host copy |
| `scale`, `causal` | float > 0, bool | non-causal is supported (all tiles visited, only the ragged end masked) |
| `out` | like `query`, BF16 | optional; may be strided |

GQA needs no data movement: query head *h* reads KV head *h* / (`num_q_heads` / `num_kv_heads`).
The kernel does not support other head sizes, FP8, sliding windows, ALiBi, soft-capping, a returned
LSE, or a K/V history in the paged cache (chunked prefill or prefix-cache hits). The tensor-core
variants need sm_80 or later.

Reproduce everything on a CUDA machine with the lab venv (`scripts/attention_gpu_run.sh setup`):

```bash
scripts/flash_prefill_gpu_run.sh all      # machine, test, peak, steps, grid, accuracy, ncu
```

Each subcommand also runs on its own. Results land in `results/data/attention/prefill/`
([index](../results/data/attention/prefill/README.md)).

## The algorithm in one page

For one query row *i* of a sequence, causal attention is
`o_i = Σ_{j ≤ i} softmax_j(s_ij) v_j` with `s_ij = scale · q_i · k_j`. FlashAttention never forms
the `L × L` score matrix. A block owns `Br` query rows and streams the keys and values through
shared memory `Bc` rows at a time. Each row keeps three pieces of state in registers: a running
max `m`, a running sum `l` and an unnormalized output `o`. After each K/V tile:

```
m_new = max(m, max_j s_ij)                  over the tile's columns
p_ij  = exp(s_ij - m_new)
l     = l · exp(m - m_new) + Σ_j p_ij
o     = o · exp(m - m_new) + Σ_j p_ij v_j
m     = m_new
```

and `o / l` at the end is exact softmax attention. Subtracting the running max keeps every
exponent ≤ 0, so nothing overflows however large the scores are. FlashAttention-2 changes three
things, and this kernel follows all three:

1. **Parallelize over query tiles.** One block per (query tile, head, sequence) instead of one per
   (head, sequence), so even a single long prompt fills the GPU.
2. **Split warps over rows, not over K.** Each warp owns 16 of the tile's 64 rows across the whole
   head dimension. A row's max, sum and output therefore live in one warp, and warps never
   exchange partial results. FlashAttention-1's "split-K" layout had to combine warps through
   shared memory on every tile.
3. **Keep non-matmul work small.** Normalize by `l` once, at the end, and skip work the causal mask
   removes. Tiles wholly above the diagonal are never loaded; only the diagonal tile is masked
   element by element.

The kernel computes in the exp2 domain: `exp(x) = exp2(x · log2 e)`, with `log2 e` folded into
the scale. Scale and max subtraction then become one FFMA ahead of one `MUFU.EX2`.

## Decision journal

All times are medians of CUDA-graph replays on the RTX 3080 Ti, from the final measurement run
(`rtx3080ti-steps.md`). "% of peak" is relative to the **measured** cuBLAS BF16 GEMM throughput,
75.5 TFLOP/s (next section). Each variant adds one change to the one before it. All of them
stay callable (`variant=` in the Python wrapper), so every row below can be reproduced.

| step | variant | 1×512 | 1×2048 | 1×8192 | 4×8192 |
|---|---|---:|---:|---:|---:|
| 0 | `lab-simt` | 195.4 µs (13%) | 2,193.3 µs (18%) | 31,885.2 µs (20%) | 126,834.7 µs (20%) |
| 1 | `lab-mma` | 109.4 µs (23%) | 1,101.7 µs (36%) | 15,065.9 µs (42%) | 58,975.2 µs (43%) |
| 2 | `lab-swizzle` | 58.7 µs (42%) | 565.3 µs (70%) | 7,741.0 µs (82%) | 30,611.5 µs (83%) |
| 3 | `lab-cp_async` | 54.4 µs (46%) | 551.4 µs (72%) | 7,619.9 µs (84%) | 30,044.2 µs (85%) |
| 4a | `lab-fast_exp2` | 54.6 µs (46%) | 541.1 µs (74%) | 7,540.3 µs (85%) | 29,516.3 µs (86%) |
| 4b | `lab-hoisted` | 53.2 µs (47%) | 514.8 µs (77%) | 7,095.9 µs (90%) | 27,837.4 µs (92%) |
| 5 | `lab-heavy_first` | 45.7 µs (55%) | 451.2 µs (88%) | 6,922.5 µs (92%) | 28,084.7 µs (91%) |
| 6 | `lab-skip_masked` | 44.2 µs (56%) | 451.2 µs (88%) | 6,839.9 µs (93%) | 27,569.2 µs (92%) |
|  | `vllm-fa2` | 51.6 µs (48%) | 518.5 µs (77%) | 7,195.3 µs (89%) | 28,750.3 µs (89%) |
|  | `sdpa-flash` | 52.2 µs (48%) | 521.2 µs (76%) | 7,245.5 µs (88%) | 28,908.5 µs (88%) |

Speedup of `lab-skip_masked` over FA2: 1×512 1.168×, 1×2048 1.149×, 1×8192 1.052×, 4×8192 1.043×.

Cells are median µs (and % of the 75.5 TFLOP/s peak). The SM clock was 1,920 MHz in the 1×512
trials and 1,800 MHz in the 4×8192 trials (`rtx3080ti-steps.md` lists it per shape).

### The tensor-core peak of this GPU

| M | N | K | median TFLOP/s | min–max TFLOP/s | SM clock MHz (median, min–max) |
|---:|---:|---:|---:|---:|---|
| 4096 | 4096 | 4096 | 70.3 | 69.3–70.9 | 1920 (210–1935) |
| 8192 | 8192 | 8192 | 71.1 | 70.9–71.7 | 1770 (1755–1905) |
| 16384 | 8192 | 8192 | 75.5 | 74.9–75.8 | 1890 (1770–1905) |
| 8192 | 8192 | 16384 | 67.1 | 63.8–68.2 | 1695 (1635–1875) |

The best size, 16384 × 8192 × 8192 at a median 1,890 MHz, reaches 499 FLOP/clk/SM. That is 97%
of 512 FLOP/clk/SM, GA102's rate for HMMA with FP32 accumulation, half its FP16-accumulation rate.
GA102 is the GeForce Ampere die, so the measurement confirms the half rate on this card. One
`mma.m16n8k16` (2,048 multiply-adds) therefore occupies a sub-partition's tensor core for 32
cycles. Attention must accumulate in FP32, so this is the ceiling that matters. Nsight Compute
reports the HMMA pipe against the FP16-accumulation rate, so
cuBLAS's own GEMM peaks at 49.8% "tensor pipe active". On this GPU, 50% on that metric is a full
pipe.

### Step 0: the smallest correct tiled kernel (CUDA cores)

The block and tile structure of the final kernel (64-row query tile, 64-row K/V tiles, four warps
of 16 rows, causal tile skipping), with FP32 math on CUDA cores:

- Lane *i* of a warp computes scores for key columns *i* and *i* + 32 of the warp's 16 rows.
- Shared memory holds Q (read as warp-wide broadcasts) and K, whose rows are padded by one 32-bit
  word so that lane *c*'s column reads land in bank (*c* + *w*) mod 32.
- V is read as consecutive words.
- The probabilities go to shared memory, so the P·V product can read them back as `float4`
  broadcasts.

It passes every test. At 1×8192 it runs at 15.1 TFLOP/s, about 40% of the CUDA-core FP32 rate at
the measured 1,815 MHz (80 SMs × 128 FMA/clk × 2 FLOP × 1.815 GHz = 37.2 TFLOP/s). It is the
yardstick for what tensor cores buy.

### The tile decision (asked before step 1)

The options, worked out for this GPU (100 KB of shared memory per SM, 1 KB of it reserved per
block, 64K registers, 48 warps):

| | Br × Bc | warps | shared memory | blocks/SM | warps/SM |
|---|---|---|---|---|---|
| **A (chosen)** | 64 × 64 | 4 | Q, K, V tiles: 48 KB | 2 (≤ 255 regs) | 8 (16.7%) |
| B | 128 × 64 | 8 | 64 KB | 1 | 8 (16.7%), half the L2 bytes per FLOP, half the blocks |
| C | 64 × 64 | 4 | Q staged through V: 32 KB | 3 only if ≤ 168 regs | 12 (25%) |
| D | 64 × 64 | 4 | K and V double-buffered: 64 KB | 1 | 4 (8.3%) |

A won for three reasons:

- **Least diagonal waste.** With `Br = Bc`, exactly one K/V tile per query tile is partly masked.
- **Twice B's block count.** That matters for short prompts and for the causal tail.
- **Overlap without double buffering.** cp.async can alternate K and V through single buffers.

It is also FlashAttention-2's own choice here. Profiling vLLM's FA2 and PyTorch SDPA's flash
backend on this GPU shows both launching `flash_fwd_kernel<Flash_fwd_kernel_traits<128, 64, 64, 4,
false, false, bf16>>` for causal head-dim-128 prefill. The two kernels therefore compete with the
same tile, the same occupancy and the same shared-memory budget. Whatever separates them is code
quality and scheduling. Option C was not pursued: the tensor-core kernel needs about 200
registers, so it would never fit three blocks.

### Step 1: tensor cores

Every warp now computes, per K/V tile:

```
S[16 × 64]   = Q[16 × 128] · Kᵀ   8 k-steps × 8 n-tiles  = 64 mma.m16n8k16
O[16 × 128] += P[16 × 64] · V     4 k-steps × 16 n-tiles = 64 mma.m16n8k16
```

with fragments from `ldmatrix`:

- **Q** is loaded once, then lives in 32 registers.
- **K** uses plain `ldmatrix`. K's rows, contiguous along the head dimension, are already the
  "column-major" B operand of Q·Kᵀ.
- **V** uses `ldmatrix.trans`. V's rows must be transposed to serve as the B operand of P·V.
- **P** never leaves registers. The S accumulator layout (lane `(g, q)` holds rows g and g + 8,
  columns 8n + 2q and the next) packed to BF16 pairs is exactly the m16n8k16 A-fragment layout.
  n-tiles 2t and 2t + 1 become P's k-step t.

The result was 2.1× the CUDA-core kernel and 42% of peak at 1×8192. Nsight Compute showed why it
was not more: **1.66 billion shared-load bank conflicts**, with `mio_throttle` and
`short_scoreboard` as the top stalls (summary tables in `ncu/`). Each `ldmatrix` phase reads one
16-byte chunk from each of eight consecutive rows. With 256-byte rows those eight chunks sit at the
same offset modulo 128
bytes, which is the same four banks. That is an 8-way conflict on every phase.

### Step 2: XOR-swizzled shared memory

16-byte chunk *c* of row *r* is stored at chunk position `c ^ (r % 8)`. Eight consecutive rows at
the same logical chunk now occupy eight different positions, covering all 32 banks. The global →
shared stores stay conflict-free too: each 8-thread phase writes a permutation of one row's eight
chunks. Shared-load bank conflicts dropped from 1.66 G to 0.16 M. The tensor pipe went from 22% to
44% busy, and throughput doubled again.

### Step 3: cp.async, FlashAttention-2's single-buffer pipeline

Loads go straight from global to shared memory with `cp.async.cg` (16 bytes, bypassing L1). Rows
past the sequence end are zero-filled with a zero source size, so stale or NaN data never reaches
an MMA. The loads overlap the math: V(j) streams in while the warps compute S = Q·K(j)ᵀ, and
K(j+1) streams in during the softmax and O += P·V(j). Two block barriers per tile order the
buffers:

```
wait K(j); __syncthreads   → K(j) visible; every warp done with V(j-1) → issue V(j)
S = Q K(j)ᵀ
wait V(j); __syncthreads   → V(j) visible; every warp done with K(j)   → issue K(j+1)
softmax; O += P V(j)
```

This bought 1.6–2.5% at 2K–8K tokens, and 8% at 1×512. The likely reason short prompts gain
more: each block there visits only a few tiles, so its first, exposed load weighs more. The gain
is small because the load latency was already hidden: two blocks share each SM, so one block's
loads overlap the other's MMAs. The counters agree. `long_scoreboard` stalls (waiting
on global loads) fell from 1.19 to 0.15 warps per issued instruction, but the kernel was already
waiting on the tensor pipe instead (`math_pipe_throttle`).

### Step 4: as lean an instruction stream as FA2's

At this point FA2 was still about 6% faster at 1×8192 (7,195 vs 7,620 µs) with the same tile,
the same occupancy and the same HMMA count. The difference was in the instruction count. Nsight
Compute's executed-SASS mix at 1×8192:

| opcode | step 3 | FA2 | what it is |
|---|---:|---:|---|
| HMMA, LDGSTS, MUFU, F2FP | identical | identical | the algorithm |
| IMAD + LEA + IADD3 + SHF + LOP3 + SEL + ISETP | 253 M | 36 M | per-copy 64-bit `row × stride` products, row predicates and pointer selects, and per-`ldmatrix` swizzle arithmetic, recomputed every tile |
| FMUL + FSETP | 158 M | 65 M | IEEE `exp2f`: a range check, select and two multiplies around every `MUFU.EX2` |
| total | 785 M | 466 M | |

- **4a, `ex2.approx.ftz`.** One `MUFU.EX2` per exponential, which is what FA2 gets from
  `--use_fast_math`. Flushing results below 2⁻¹²⁶ to zero cannot change a BF16 output. About 1%.
- **4b, addresses computed once.** The swizzle depends only on `row % 8`, and every 16-row block
  starts at a multiple of 8. Each lane therefore needs one `ldmatrix` offset per k-step (K) and
  per n-tile pair (V), 16 registers computed once, and the 16-row block index becomes an
  immediate in the address. Full K/V tiles use an unpredicated copy: one source pointer advanced
  eight rows per pass, and destinations as one base plus immediates. Only a sequence's ragged last
  tile takes the predicated path. This cut instructions from 785 M to 548 M and raised the tensor
  pipe from 44.4% to 47.8% of peak during active cycles. That is about 96% of the 50% ceiling, and
  level with FA2's 47.2%; FA2 still executes fewer instructions (466 M). At this step the kernel
  first matched FA2 at long prompts (7,096 vs 7,195 µs at 1×8192).

### Step 5: heavy-first scheduling

Causal query tiles are unequal. Tile *t* visits *t* + 1 K/V tiles, so the last tile of a
1,024-token prompt does 16× the work of the first. With FA2's grid order (query tile fastest), the
first wave holds every tile of the first heads, and the most expensive tiles of the last heads start
late and run into a tail where most SMs are idle. Step 5 makes the query tile the slowest grid
dimension, counted back from each sequence's last tile. Every head's and every sequence's most
expensive tile launches in the first wave, and the one-tile blocks fill the end. This is
longest-processing-time-first scheduling. It costs nothing in-kernel, and it is where the kernel
pulls clearly ahead of FA2. At 1×1024, Nsight Compute shows the SMs busy 94.6% of elapsed cycles
against FA2's 77.4% (step 3, in FA2's order: 75.1%), and at 1×8192 99.9% against 95.8%. At 4×8192
the run is about 90 waves (14,336 blocks for 160 slots) and the tail is a small fraction, so the
order hardly matters there. The order costs some L2 locality: at 1×8192, DRAM reads rise from
89 MB (step 4b) to 254 MB (final) and the L2 hit rate falls from 98.8% to 96.7%. That traffic is
trivial next to the 7.6 GB that L2
serves to the SMs.

### Step 6: skip the fully masked part of the diagonal tile

In the diagonal tile, warp *w*'s rows (16*w* .. 16*w* + 15 of the tile) need only key columns ≤
16*w* + 15. Only the first *w* + 1 16-column blocks hold unmasked scores, so the warp skips the
Q·Kᵀ MMAs of the rest and the matching P·V k-steps. This removes 37.5% of the diagonal tile's MMAs
on average. Only one tile per query tile has fewer than four active blocks, so it takes a separate
code path and the full tiles pay nothing for it.

Over the full grid, step 6 is faster than step 5 in 20 of 24 shapes, by up to 3.6% (at 1×512). It
is slower in four short shapes, by 0.3–2.0%, worst at 1×1024. The final profile shows more
`barrier` stalls than step 4b's: 2.11 vs 0.48 warps per issued instruction at 1×8192. I have not
separated how much of that comes from step 5 and how much from step 6. In step 6, warps that skip
work in the diagonal tile reach the next barrier early and wait for warp 3. Step 6 stays the
default because it wins on most shapes.

### Considered and not done

- **Br = 128 (option B).** It halves the L2 → SM bytes per FLOP, but L2 is not the limit: the L2
  hit rate is 97–99% and the kernel waits on the tensor pipe. It would also halve the block count.
- **Re-reading Q fragments from shared memory every tile, as FA2 does.** Keeping Q in registers costs
  32 registers that fit under the 2-blocks/SM budget and saves 8 `ldmatrix` per tile per warp. FA2
  executes 66.6 M LDSM at 1×8192 against this kernel's 58.9 M.
- **Packing a GQA group's 7 query heads into one block** to share K/V tiles in shared memory. The 7
  blocks of a group already read the same K/V tiles through L2 at about the same time, and DRAM
  reads are 254 MB per 1×8192 call against 7.6 GB of L2 → SM traffic.
- **Split-KV for prefill.** Even one 512-token prompt makes 224 blocks for 160 slots.

## One block, one warp, one thread

Final kernel: `flash_prefill_mma_kernel<PrefillVariant::kSkipMasked>`, 128 threads,
48 KB of dynamic shared memory, 199 registers, two blocks per SM.

### One block

**Which work.** The grid is (query head, sequence, query tile counted back from the sequence's last
tile). `block_tile()` reads `cu_seqlens[seq]` and `cu_seqlens[seq + 1]` and computes the sequence
length. It turns `blockIdx.z` into the tile's first row, `q_begin`, and exits at once if the tile
does not exist (the grid is sized for `max_seqlen`). It then derives the KV head and the number of
K/V tiles to visit: `ceil(min(seq_len, q_begin + 64) / 64)` when causal, so every tile right of
the diagonal is skipped without being loaded.

**Shared memory.** Three 64-row × 256-byte tiles, `sQ`, `sK` and `sV` (16 KB each), all
XOR-swizzled. `sQ` is read once, into registers. It is reused at the end to stage the output.

**Global reads.** The Q tile (16 KB) once, then K and V for each visited tile (32 KB per tile),
through `cp.async.cg`. Sixteen consecutive threads copy one 256-byte row, so every warp-wide copy
covers two whole rows: coalesced, full 32-byte sectors. A block on query tile *t* reads 16 KB +
(*t* + 1) × 32 KB. **Global writes:** its 64 output rows (16 KB), as 16-byte stores of whole rows.
Rows past the sequence are never written.

**Timeline.**

```
prologue  cp.async Q → sQ, K(0) → sK; commit
tile j    wait_group 0; __syncthreads()      K(j) landed for every thread; V(j-1) no longer read
          (j = 0: ldmatrix Q → registers)
          cp.async V(j) → sV; commit         streams in during the next line
          S = Q K(j)ᵀ                        64 HMMA per warp
          wait_group 0; __syncthreads()      V(j) landed; K(j) no longer read
          cp.async K(j+1) → sK; commit       streams in during the next two lines
          mask (diagonal / ragged tile only); online softmax
          O += P V(j)                        64 HMMA per warp
epilogue  normalize; each warp stages its 16 rows in its own rows of sQ; 16-byte global stores
```

The block's warps meet only at those two barriers per tile. They share K and V tiles, never
results.

### One warp

Warp *w* owns tile rows 16*w* .. 16*w* + 15 and the whole head dimension. Per K/V tile it executes:

- 32 `ldmatrix.x4` on K. Each yields the B fragments (b0, b1) of two 8-token n-tiles for one
  16-dimension k-step. Lanes 0–7 and 8–15 address tokens 0–7 at the step's two 8-dimension halves;
  lanes 16–31 do the same for tokens 8–15.
- 64 HMMA for S, in step-outer order. Eight independent accumulators interleave, so no HMMA waits
  on the previous one.
- The online softmax, all in registers. Each row's max and sum are reduced across the row's quad of
  four lanes with two `shfl.xor` (offsets 1 and 2). There is no shared memory and no cross-warp
  reduction.
- 32 `ldmatrix.x4.trans` on V and 64 HMMA for O += P·V.

In the diagonal tile, warp *w* runs only *w* + 1 of the four 16-column blocks (step 6).

### One thread

Lane *ℓ* has `group = ℓ / 4` and `quad = ℓ % 4`. It holds rows `group` and `group + 8` of its
warp's 16:

- **Q:** `q[8][4]`, the A fragments of all 8 k-steps (32 registers, loaded once).
- **S:** `s[8][4]`. n-tile *n* holds columns 8*n* + 2·`quad` and the next, for both rows (32 values).
- **O:** `o[16][4]`, output dimensions 8*n* + 2·`quad` and the next for both rows (64 FP32 values).
- **Softmax state:** `m[2]` (the row max, the same in all four lanes of the quad, in raw score
  units) and `l[2]` (this lane's share of the row sum, finished with two shuffles at the end).

Per tile it issues 16 `cp.async` of 16 bytes: 8 for K and 8 for V, one chunk of rows *r*, *r* + 8,
…, *r* + 56 where *r* = `threadIdx.x / 16`. It also executes 34 `MUFU.EX2`: 32 probabilities and two
rescale factors. Its probabilities become P's A fragments by packing: `p[t] = {s[2t][0..1],
s[2t][2..3], s[2t+1][0..1], s[2t+1][2..3]}` as BF16 pairs. At the end it writes two BF16 values per
n-tile and row into `sQ`, then copies whole 16-byte chunks to global memory.

### Synchronization and reductions, summarized

| what | how | why |
|---|---|---|
| K/V buffer reuse | `cp.async.wait_group 0` + `__syncthreads()`, twice per tile | a buffer is overwritten only after every warp has read it, and read only after every thread's copy landed |
| row max and row sum | 2 × `__shfl_xor_sync` within a quad | a row's 64 scores live in the four lanes of one quad |
| final row sum | 2 more shuffles in the epilogue | `l` is kept as per-lane partials until then |
| output staging | `__syncwarp()` | each warp writes and reads only its own 16 rows of `sQ` |
| across warps | nothing | warps split rows (FlashAttention-2), so no warp needs another's state |

### Numerics

- FP32 accumulation everywhere.
- P is rounded to BF16 before P·V, as FA2 does.
- The running max is taken over raw scores and the scale is applied inside the exponent:
  `exp2(s · scale·log2e − m · scale·log2e)` as one FFMA. This is valid because `scale > 0`, which
  the wrapper checks.
- A running max that is still −∞ is replaced by 0 before subtraction, so a fully masked row yields
  zeros rather than `exp2(−∞ + ∞) = NaN`.
- Errors against the FP32 reference, for every tested shape:

| case | lab-skip_masked max abs / max rel / pass | vllm-fa2 max abs / max rel / pass | sdpa-flash max abs / max rel / pass |
|---|---|---|---|
| len 1, 28/4 | 0.00e+00 / 0.00e+00 / pass | 0.00e+00 / 0.00e+00 / pass | 0.00e+00 / 0.00e+00 / pass |
| len 129, 28/4 | 8.70e-03 / 2.37e-03 / pass | 8.70e-03 / 2.37e-03 / pass | 8.70e-03 / 2.37e-03 / pass |
| len 8192, 28/4 | 8.62e-03 / 2.32e-03 / pass | 8.62e-03 / 2.32e-03 / pass | 8.62e-03 / 2.32e-03 / pass |
| ragged x12, 28/4 | 1.04e-02 / 2.60e-03 / pass | 1.04e-02 / 2.60e-03 / pass | 1.04e-02 / 2.60e-03 / pass |
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

The worst errors of the lab kernel, FA2 and SDPA agree to three significant digits in every shape,
because they are set by the final rounding of the output to BF16. The worst relative error, about
2.4e-3, is about half a BF16 ulp. All 21 shapes pass for every variant; the full per-shape table
is `rtx3080ti-accuracy.md`. Repeated runs are bitwise identical: the kernel has no atomics and a
fixed reduction order.

## Where the time goes (bottleneck)

Nsight Compute at 1×8192, every tensor-core step against FA2 (from
`ncu/rtx3080ti-1x8192-steps-summary.md`; `--set full` under ncu's locked clock, so its kernel
times are slower than the benchmark's):

| metric | 1-mma | 2-swizzle | 3-cp_async | 4b-hoisted | 6-skip_masked | vllm-fa2 |
|---|---:|---:|---:|---:|---:|---:|
| kernel time (µs, ncu-locked clock) | 16,772 | 8,471 | 8,405 | 7,739 | 7,471 | 7,904 |
| SMs busy (% of elapsed cycles) | 96.57 | 95.90 | 96.67 | 95.22 | 99.92 | 95.78 |
| registers / thread | 200 | 202 | 196 | 198 | 199 | 191 |
| achieved occupancy (%) | 16.49 | 16.48 | 16.47 | 16.46 | 16.63 | 16.46 |
| tensor pipe (HMMA) active (% of peak, SM-active cycles) | 22.10 | 43.74 | 44.40 | 47.84 | 47.46 | 47.18 |
| tensor pipe (HMMA) active (% of peak, elapsed cycles) | 21.34 | 41.95 | 42.92 | 45.55 | 47.42 | 45.19 |
| warp instructions executed | 746,934,272 | 775,319,552 | 784,903,168 | 547,807,232 | 577,400,320 | 465,792,656 |
| shared load bank conflicts | 1,660,332,679 | 164,437 | 143,965 | 160,071 | 135,841 | 158,005 |
| DRAM read (MB) | 94.06 | 102 | 86.30 | 89.32 | 254 | 195 |
| L2 → SM read (MB) | 7,629 | 7,627 | 7,634 | 7,634 | 7,634 | 7,634 |
| L2 hit rate (%) | 98.77 | 98.65 | 98.88 | 98.84 | 96.71 | 97.48 |
| stall: math_pipe_throttle (warps per issue) | 3.31 | 6.09 | 7.06 | 9.68 | 7.40 | 11.22 |
| stall: barrier (warps per issue) | 1.00 | 0.50 | 0.45 | 0.48 | 2.11 | 1.61 |
| stall: short_scoreboard (warps per issue) | 4.32 | 0.26 | 0.15 | 0.26 | 0.21 | 0.20 |
| stall: long_scoreboard (warps per issue) | 1.76 | 1.19 | 0.15 | 0.14 | 0.07 | 0.08 |
| stall: mio_throttle (warps per issue) | 9.41 | 0.12 | 0.10 | 0.24 | 0.24 | 0.41 |

The same comparison at 1×1024, where the causal tail matters most
(`ncu/rtx3080ti-1x1024-summary.md`):

| metric | step-3-cp_async | final-skip_masked | vllm-fa2 |
|---|---:|---:|---:|
| kernel time (µs, ncu-locked clock) | 194 | 142 | 170 |
| SMs busy (% of elapsed cycles) | 75.05 | 94.64 | 77.42 |
| tensor pipe (HMMA) active (% of peak, SM-active cycles) | 41.00 | 41.86 | 44.25 |
| tensor pipe (HMMA) active (% of peak, elapsed cycles) | 30.77 | 39.61 | 34.25 |
| warp instructions executed | 14,118,272 | 10,960,320 | 8,839,824 |

Calibration (`ncu/rtx3080ti-cublas-calibration-summary.md`): cuBLAS's own BF16 GEMM against the
final kernel. This is what a full tensor pipe reads on this GPU:

| metric | cublas-gemm-8192 | final-skip_masked-1x8192 |
|---|---:|---:|
| SMs busy (% of elapsed cycles) | 98.44 | 99.92 |
| tensor pipe (HMMA) active (% of peak, SM-active cycles) | 49.82 | 47.46 |
| tensor pipe (HMMA) active (% of peak, elapsed cycles) | 49.04 | 47.42 |

**The kernel is tensor-pipe bound.** The top stall is `math_pipe_throttle` (warps ready but the
HMMA pipe busy), and the HMMA pipe is active 47.5% of cycles against a ceiling of about 50%
(cuBLAS: 49.8%). Measured over elapsed cycles, including the causal tail, it is 47.4% against
FA2's 45.2%. That 5% is the end-to-end gap at 1×8192. Occupancy is 16.5% by design. Two blocks of
4 warps leave each SM sub-partition two warps, which is enough here: each warp has eight independent
HMMA chains per GEMM, and the
other block covers barriers and loads. Memory is not the limit:

- The L2 hit rate is 96.7% because the 7 query heads of a GQA group, and the query tiles of one
  head, stream the same K/V tiles.
- DRAM reads are 254 MB per 1×8192 call, about 35 GB/s over the call.
- Shared-memory bank conflicts are effectively zero.

The kernel still executes 24% more instructions than FA2 (577 M vs 466 M at 1×8192), mostly
integer addressing (IMAD, IADD3, LOP3) and the step-6 branches. The tensor pipe saturates anyway,
because the issue slots are far from full, so this costs little time today. It is still the first
thing to remove before any change that adds non-matmul work. What is left between the kernel and
the 75.5 TFLOP/s GEMM peak is mostly structural:

- the softmax and rescale instructions issued between the two GEMMs of a tile;
- the diagonal tile's masked work;
- the two barriers per tile;
- the causal tail at short sequence lengths.

## Benchmark: Qwen2.5-7B shapes

28 query heads, 4 KV heads, head dim 128, causal, uniform-length batches packed as varlen.
Implementations:

- `lab-skip_masked`: this kernel, all steps.
- `vllm-fa2`: `vllm.vllm_flash_attn.flash_attn_varlen_func(fa_version=2)`, with the arguments
  vLLM's FlashAttention backend would pass for a fresh prompt.
- `sdpa-flash`: PyTorch SDPA forced to its flash backend, `enable_gqa=True`, on
  `[batch, heads, seq, dim]` views of the same memory. Its time excludes a copy into a caller's
  buffer, since SDPA has no `out=` argument, which slightly favours it.

FLOPs count the causal work exactly: `4 · 128 · 28 · L(L+1)/2` per sequence. "% of peak" is
relative to the measured 75.5 TFLOP/s.

| batch | seq | lab µs | lab TFLOP/s | lab % of peak | FA2 µs | FA2 TFLOP/s | FA2 % of peak | SDPA µs | speedup vs FA2 | speedup vs SDPA | SM MHz |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 512 | 44.5 | 42.3 | 56.0 | 52.0 | 36.2 | 48.0 | 52.8 | **1.167×** | 1.186× | 1905 |
| 1 | 1024 | 127.8 | 58.9 | 78.0 | 156.5 | 48.1 | 63.7 | 159.8 | **1.225×** | 1.250× | 1845 |
| 1 | 2048 | 456.7 | 65.9 | 87.3 | 526.4 | 57.1 | 75.7 | 532.0 | **1.153×** | 1.165× | 1815 |
| 1 | 4096 | 1,750.2 | 68.7 | 91.0 | 1,900.9 | 63.3 | 83.8 | 1,936.4 | **1.086×** | 1.106× | 1800 |
| 1 | 8192 | 6,934.3 | 69.4 | 91.9 | 7,303.5 | 65.9 | 87.3 | 7,366.0 | **1.053×** | 1.062× | 1800 |
| 1 | 16384 | 27,841.5 | 69.1 | 91.6 | 28,789.7 | 66.8 | 88.5 | 29,248.0 | **1.034×** | 1.051× | 1785 |
| 2 | 512 | 74.3 | 50.7 | 67.2 | 87.6 | 43.0 | 57.0 | 89.3 | **1.179×** | 1.202× | 1815 |
| 2 | 1024 | 249.6 | 60.3 | 79.9 | 278.9 | 54.0 | 71.5 | 282.6 | **1.117×** | 1.132× | 1800 |
| 2 | 2048 | 921.3 | 65.3 | 86.5 | 988.7 | 60.8 | 80.6 | 1,004.4 | **1.073×** | 1.090× | 1785 |
| 2 | 4096 | 3,550.7 | 67.8 | 89.8 | 3,731.4 | 64.5 | 85.4 | 3,804.8 | **1.051×** | 1.072× | 1762 |
| 2 | 8192 | 14,381.3 | 66.9 | 88.6 | 14,803.2 | 65.0 | 86.1 | 15,016.6 | **1.029×** | 1.044× | 1725 |
| 2 | 16384 | 57,539.6 | 66.9 | 88.6 | 59,437.9 | 64.7 | 85.8 | 60,260.3 | **1.033×** | 1.047× | 1695 |
| 4 | 512 | 147.3 | 51.1 | 67.7 | 161.2 | 46.7 | 61.9 | 162.5 | **1.095×** | 1.103× | 1695 |
| 4 | 1024 | 501.0 | 60.1 | 79.6 | 532.7 | 56.5 | 74.8 | 541.4 | **1.063×** | 1.081× | 1725 |
| 4 | 2048 | 1,862.4 | 64.6 | 85.6 | 1,934.9 | 62.2 | 82.4 | 1,966.4 | **1.039×** | 1.056× | 1725 |
| 4 | 4096 | 7,340.3 | 65.5 | 86.8 | 7,535.6 | 63.9 | 84.6 | 7,655.1 | **1.027×** | 1.043× | 1740 |
| 4 | 8192 | 29,274.0 | 65.7 | 87.1 | 29,895.7 | 64.4 | 85.3 | 30,382.1 | **1.021×** | 1.038× | 1680 |
| 4 | 16384 | 118,162.4 | 65.1 | 86.3 | 120,344.6 | 64.0 | 84.7 | 121,737.2 | **1.018×** | 1.030× | 1665 |
| 8 | 512 | 286.7 | 52.5 | 69.6 | 301.1 | 50.0 | 66.3 | 303.6 | **1.050×** | 1.059× | 1665 |
| 8 | 1024 | 1,000.2 | 60.2 | 79.7 | 1,035.9 | 58.1 | 77.0 | 1,050.9 | **1.036×** | 1.051× | 1695 |
| 8 | 2048 | 3,800.4 | 63.3 | 83.9 | 3,885.4 | 61.9 | 82.0 | 3,949.0 | **1.022×** | 1.039× | 1695 |
| 8 | 4096 | 14,807.6 | 65.0 | 86.1 | 15,095.8 | 63.7 | 84.4 | 15,335.9 | **1.019×** | 1.036× | 1695 |
| 8 | 8192 | 58,615.8 | 65.7 | 87.0 | 59,943.9 | 64.2 | 85.0 | 60,679.2 | **1.023×** | 1.035× | 1688 |
| 8 | 16384 | 237,901.8 | 64.7 | 85.7 | 241,532.9 | 63.7 | 84.4 | 246,902.8 | **1.015×** | 1.038× | 1650 |

`lab-skip_masked` is faster than FA2 in 24 of 24 shapes: 1.015× to 1.225×, geometric mean 1.066×.

The SM clock column is the median NVML reading during each shape's trials. The card's clock fell
from 1,905 MHz to 1,650 MHz over the 40-minute grid as it heated. The peak was measured at
1,890 MHz, so "% of peak" understates the later rows by up to 13%. The speedups are unaffected:
within a shape, all four implementations ran interleaved in the same trials.

## Integration: routing fresh prefills from the CUSTOM vLLM backend

Designed and tested, not wired in. `vllm_attention.py` is unchanged. The callable is
`inference_performance_lab.kernels.vllm_prefill.fresh_prefill_attention`, with the host-side check
`is_fresh_prefill`. `tests/test_flash_prefill_attention.py::test_fresh_prefill_routing` exercises
both on a vLLM-ordered batch (three decodes, then three fresh prompts) built from fused-QKV views.

What vLLM 0.33 hands the backend:

- **`forward(query, key, value, ...)` receives the step's own K and V.** They are
  `[num_tokens, num_kv_heads, 128]` views of the QKV projection, BF16 even when the cache is not.
- **The KV-cache write happens separately,** in `do_kv_cache_update` (`reshape_and_cache_flash`).
- **Batches are ordered decode-first.** The CUSTOM builder already sets
  `reorder_batch_threshold = 1`.

For a request whose whole prompt is in this step (`query_len == seq_len`), attention needs nothing
from the cache. `forward`'s `key` and `value` slices are exactly its K and V, so FA2's paged gather
is unnecessary.

1. **Builder (host side, no sync).** After `split_decodes_and_prefills`, set
   `lab_fresh_prefill = is_fresh_prefill(common.query_start_loc_cpu, common.seq_lens_cpu_upper_bound,
   num_decodes)`. `seq_lens_cpu_upper_bound` is exact for prefill rows. If it is `None`, or there are
   no prefills, use False. Build `lab_prefill` (the prefill rows' metadata, `query_start_loc`
   rebased to the first prefill token) also when `num_decodes == 0`: today the builder returns early
   there.
2. **Forward.** After the decode kernel, or directly in a prefill-only batch: if
   `lab_fresh_prefill`, call

   ```python
   t0, t1 = num_decode_tokens, attn_metadata.num_actual_tokens
   fresh_prefill_attention(
       output[t0:t1],
       query[t0:t1],
       key[t0:t1],
       value[t0:t1],
       attn_metadata.lab_prefill.query_start_loc,
       attn_metadata.lab_prefill.max_query_len,
       self.scale,
   )
   ```

   Otherwise keep today's FA2 call. Chunked-prefill continuations and prefix-cache hits must read
   earlier K/V from the paged cache, which this kernel does not do.
3. **Eligibility.** The same as the decode path: `lab_enabled` (head size 128, `kv_cache_dtype ==
   "auto"`, no ALiBi, sliding window, soft-cap or sinks, no DCP), decoder self-attention, no cascade
   attention (`common_prefix_len == 0`), and BF16 queries. Unlike the decode kernel, any GQA group
   size works.
4. **Graphs.** Prefill and mixed batches run attention eagerly under vLLM's piecewise CUDA graphs,
   and the kernel takes `max_query_len` as a host integer, so nothing needs capturing. A
   full-graph path would need a fixed `max_seqlen` bound.

The kernel-level gains in the grid above are an upper bound on what this changes in serving:
attention is one part of a prefill step, and mixed batches route only their fresh-prefill rows.
This has not been measured end to end.

## Next steps

- Wire the routing above into `vllm_attention.py` and A/B it with `benchmarks/serving_attention_ab.py`
  on prefill-heavy workloads, then check greedy-output equality as the decode check does.
- Return the log-sum-exp, then accept a K/V prefix from the paged cache. That covers chunked
  prefill and prefix-cache hits, and lets split-KV partials merge.
- Reduce the softmax's instruction count between the GEMMs (e.g. rescale O only when the max
  changes), and interleave the next tile's first `ldmatrix` with the softmax.
- Re-tune for sm_80 (A100: 164 KB shared memory per SM, full-rate FP32-accumulate HMMA). The 2-blocks
  argument and the 50% pipe ceiling are GA102-specific.

## Check questions

Try each before opening the answer.

1. **Bank conflicts.** In step 1 the 16-byte stores that fill a K tile were conflict-free, but the
   `ldmatrix` reads of the same tile were 8-way conflicted. Which 16-byte chunks does one
   `ldmatrix` phase touch? Why does `chunk ^ (row % 8)` fix the reads without breaking the stores?

   <details><summary>Answer</summary>

   A phase reads one chunk column from eight consecutive rows. With 256-byte rows those chunks are
   256 bytes apart, the same offset modulo 128 bytes, so the same 4 banks. A store phase is 8
   threads writing 8 consecutive chunks of one row (128 contiguous bytes, all 32 banks). XOR with
   `row % 8` gives the eight rows eight different chunk positions. Within one row it only permutes
   the chunks, so a store phase still covers all 32 banks.
   </details>

2. **Fragments.** Lane 13 holds `s[5][2]` after Q·Kᵀ. Which of the warp's 16 rows, and which of
   the tile's 64 key columns, is that? Which register of which P A-fragment does it become?

   <details><summary>Answer</summary>

   group = 13 / 4 = 3 and quad = 13 % 4 = 1. Element 2 is row group + 8 = 11, column
   8·5 + 2·1 = 42. n-tile 5 is the second half of k-step t = 2 (tokens 32–47), so it lands in
   `p[2][3]`, the low half of `pack(s[5][2], s[5][3])`.
   </details>

3. **Pipelining.** cp.async bought only 1.6–2.5% at long prompts. Use the stall counters to explain
   why. What would
   you expect if each block needed 64 KB of shared memory, so only one block fits per SM?

   <details><summary>Answer</summary>

   With two blocks per SM, one block's loads overlap the other block's MMAs, so the synchronous
   kernel already hid most of the load latency. `long_scoreboard` stalls fell from 1.19 to 0.15
   warps per issued instruction, but `math_pipe_throttle` already dominated. With one block per SM
   nothing else covers the loads, and cp.async overlap (or double buffering) would matter much more.
   </details>

4. **Masking.** With Br = Bc = 64, why does each causal query tile mask exactly one K/V tile? How
   many would be partly masked with Br = 128, Bc = 64, and how much work could each warp skip?

   <details><summary>Answer</summary>

   Tile *t*'s rows 64t .. 64t + 63 see columns up to 64t + 63. K/V tiles 0 .. t − 1 are fully
   visible, tile t straddles the diagonal, and later tiles are skipped. With Br = 128, two K/V tiles
   straddle the diagonal. With 8 warps of 16 rows, warp w needs only the 16-column blocks up to its
   own rows, a staircase across both tiles.
   </details>

5. **Scheduling.** Why does heavy-first ordering help causal prefill but not non-causal? Why is the
   gain largest around 1×1024 and smallest at 8×16384?

   <details><summary>Answer</summary>

   Causal query tile *t* costs *t* + 1 K/V tiles, so blocks differ up to L/64-fold in cost.
   Non-causal blocks all cost the same. At 1×1024 there are 448 blocks for 160 slots, so a badly
   ordered final wave leaves most SMs idle (FA2: SMs busy 77.4%). At 8×16384 there are 57,344
   blocks, about 360 waves, and the tail is a fraction of a percent of the run.
   </details>

6. **Numerics.** The kernel takes each row's max over raw scores and applies `scale · log2(e)`
   inside one FFMA before `ex2`. What property of `scale` makes that valid, and what breaks
   without it? Why does a row whose scores are all masked produce zeros rather than NaN?

   <details><summary>Answer</summary>

   It needs scale > 0, because max(scale · s) = scale · max(s) only for a positive scale. With a
   negative scale the "max" would be the minimum, and the exponents could become positive and
   overflow. A running max still at −∞ is replaced by 0 before subtraction (`finite_max`), so
   p = exp2(−∞) = 0 and α = 0, never exp2(−∞ − (−∞)) = NaN. The epilogue writes 0 when l = 0.
   </details>

7. **Roofline.** Nsight Compute reports the HMMA pipe 47.5% "of peak". Why is that about 95% of what
   this GPU can do, and which measurement shows it?

   <details><summary>Answer</summary>

   On GA102, HMMA with FP32 accumulation runs at half the FP16-accumulation rate, and Nsight's
   "peak" is the FP16-accumulation rate. cuBLAS's own BF16 GEMM, the best this GPU does (75.5
   TFLOP/s), reads 49.8% on the same metric. 47.5 / 49.8 ≈ 95%.
   </details>
