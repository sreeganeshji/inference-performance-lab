# Decode attention: ten hard questions

Questions an interviewer who knows GPUs and vLLM could ask about this project, each answered from
the measurements in [decode-attention.md](decode-attention.md). Try each one before reading the
answer.

---

### 1. Decode attention does 7 FLOP per byte, under the A100's FP32 ridge. Why bother with tensor cores?

Being under the ridge only says DRAM can be the limit; whether it is depends on whether the SM can
issue enough instructions to keep DRAM busy. A CUDA-core kernel spends several instructions per
FLOP on loads, BF16 unpacking, shuffles for the row reductions and FP32 math, so at 7 FLOP/byte it
needs a high issue rate per byte moved. The tensor-core kernel does the same arithmetic in a few
`mma.sync` instructions per 16-token chunk, so it reaches 86–87% of DRAM bandwidth on the A100
while issuing on only 12–13% of cycles (`Issue Slots Busy` in the ncu summary). The split CUDA-core
kernel was 1.4–2.1× slower than the tensor-core one at every A100 shape in the grid. Tensor cores
here buy instruction throughput, not FLOPs.

### 2. Derive the merge of two partial softmax states, and say why the kernel works in exp2.

A partial state over a set of keys is (m, ℓ, o): the running max score, ℓ = Σ exp(sᵢ − m) and
o = Σ exp(sᵢ − m)·vᵢ. To merge (m₁, ℓ₁, o₁) with (m₂, ℓ₂, o₂), take m = max(m₁, m₂) and rescale
each side by exp(mₖ − m): ℓ = ℓ₁e^(m₁−m) + ℓ₂e^(m₂−m), and likewise for o. The final output is o/ℓ.
The same rule merges lanes, warps, splits (the combine kernel) and new tokens into a running state.
Scaling scores by scale × log₂e up front turns every exp into `exp2`, a single MUFU instruction,
and saves a multiply per score. A state with m = −∞ (no valid tokens) must get weight 0 explicitly,
or −∞ − (−∞) produces NaN.

### 3. Grouped-query attention made the kernel read each KV row once instead of seven times. Why was it slower at batch 1?

Qwen2.5-7B has 28 query heads sharing 4 KV heads, so one block per KV head reads each 512-byte row
once for all 7 query heads. But at batch 1 that is 4 blocks for a 108-SM GPU (the per-query-head
kernel launched 28). Four SMs cannot pull more than a small fraction of HBM bandwidth however
efficient each is: at 1 × 4K the GQA kernel ran at 14 GB/s. Parallelism has to come from somewhere
else, which is what split-KV provides. The lesson is that removing redundant traffic only helps
once enough of the machine is working.

### 4. How do you choose the number of KV splits, and what is wave quantization?

More splits add parallelism but cost FP32 partial outputs and a combine pass. The heuristic models
DRAM as saturating once a certain number of blocks run at once, and counts how many "slots" each
wave of blocks occupies: resident capacity comes from the occupancy API (3 tensor-core blocks per SM
here). If the grid is slightly more than one wave, the second wave runs nearly empty and at low
bandwidth. On the 3080 Ti at batch 4 × 4K, 15 splits make 240 blocks, exactly one wave of 240, and
16 splits spill into a second wave. The constants are fitted per GPU to measured sweeps; on the A100
the fit cut mean regret against the best measured split count from 5.2% to 0.64%.

### 5. The tensor-core kernel loads K and V straight from global memory into MMA fragment registers. Why is that legal, given `mma.sync`'s fixed fragment layouts?

Two orderings are free. S = QKᵀ sums over the 128 head dimensions, and a sum does not care about
order as long as Q's and K's fragments enumerate dimensions the same way; so each lane can be given
the dimensions that are contiguous in memory (32i + 8·quad … +8) and load them with one 16-byte
load. For P·V, which physical dimension each row of the output stands for is a relabeling, fixed up
when the result is stored. With both chosen this way, every lane's fragment elements are contiguous
and no shared-memory staging or `ldmatrix` is needed. The accumulator fragment of S then has exactly
the layout P needs as the A operand of the next MMA, so the softmax happens in registers.

### 6. Why compute Oᵀ = VᵀPᵀ instead of O = PV?

`m16n8k16` has a 16-row M side and an 8-wide N side. A GQA group has 7 query heads. With heads on M
they are padded to 16 rows and half of every O accumulator register holds padding: that version
needed 178 registers per thread and fit 2 blocks per SM. Transposing puts the 7 heads on the 8-wide
N side and the 128 output dimensions on M, so almost nothing is padding: 142 registers and 3 blocks
per SM. The ncu batch 64 × 4K profile shows why residency matters here: the lab kernel's 256 blocks
fit in 0.79 of a wave, while FA2 at 178 registers needs 1.19 waves and averages 63% of DRAM against
the lab kernel's 87%.

### 7. The split model saturates DRAM with 0.8 blocks per SM on the 3080 Ti and 1.7 on the A100. Why would they differ?

Little's law: bytes in flight = bandwidth × latency. To keep DRAM busy each SM needs its share of
bandwidth times the memory latency in outstanding loads. The A100 delivers about 14.4 GB/s per SM
(1,555 GB/s over 108 SMs) against the 3080 Ti's 11.4 (912 over 80), and HBM latency is not shorter,
so each SM needs more requests outstanding, which here means more resident blocks. The 3080 Ti
constants left the kernel 8–13% off the best split count at mid batch on the A100. Tuning by
measurement on each target is the practical answer.

### 8. Under vLLM's full CUDA graphs the split count may depend only on the number of decode rows. Why, and what does the kernel do about it?

A CUDA graph captures the launch configuration, grid included, and is replayed for every later step
with the same batch size. Context lengths change every step, but the graph does not. So the split
count is chosen from the decode count alone (assuming a long context), the partial-output workspace
is allocated once at its maximum, and the kernel reads each sequence's length from device memory,
dividing its own range into equal splits at run time. Choosing splits from the current context
length would require recapturing graphs or falling back to eager launches.

### 9. Serving throughput improved 1.29× at concurrency 64, about the same as the kernel's 1.27× at 64 × 4K. Is the serving gain the kernel's?

Mostly not. In the FLASH_ATTN trace, attention in decode-only steps took 4.1 s of the 59.7 s
window, so even zero-cost attention there could give at most 1.07×. The Nsight Systems trace splits
attention by launch grid:
in steps that mix a prefill chunk with decodes, vLLM's FlashAttention backend runs one varlen launch
sized for the chunk, with one block per query head (gridZ 28) instead of per KV head, so decode rows
lose GQA packing; those launches average 1,967 µs per layer. The lab backend runs FA2 on the prefill
rows only (429 µs) and its own kernel on the decodes (358 µs). The backend design, routing decode
rows separately, is most of the win; the faster kernel adds the rest.

### 10. Would rewriting vLLM's scheduler or KV-block free list in C++ speed up serving?

Not on these workloads. With NVTX ranges around `Scheduler.schedule`, `update_from_output` and the
block pool, the traces show the scheduler at 1.8–3.6% of wall time, schedule() at 0.9–1.5 ms per
step, and block allocation and freeing at 18–35 ms per 22–60 s trace. But the GPU is busy 99.5–99.8%
of the time, and scheduler time that falls while the GPU is idle totals 0.4–2.1 ms per trace:
vLLM's async scheduling builds step N + 1 while the GPU runs step N. Speeding up hidden work does not
change throughput; it would matter only where the CPU becomes the critical path (much smaller
models, many tiny requests or speculative decoding).

---

**Follow-ups worth having an answer for**

- *Why does FlashInfer win at batch 1 × 16K?* The main kernels are equal; FlashInfer's persistent
  merge kernel uses all 108 SMs while the lab's combine uses one block per query head.
- *How do you know the greedy-output divergence on real weights is not a bug?* The same check
  between two stock vLLM backends diverges as much (FLASH_ATTN vs TRITON_ATTN: the same 334/384).
- *Why zero-fill V past the end of a sequence but only mask K?* Masked scores become −∞, so P is
  exactly 0 there, but 0 × NaN is still NaN in P·V, and freed cache blocks are never cleared.
