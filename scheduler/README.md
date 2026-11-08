# LLM request scheduler

A small C++ version of the request scheduler inside an LLM serving engine: a KV block
pool with a prefix cache, a continuous-batching FCFS scheduler with recompute preemption,
and an async two-batch dispatch loop with a fake GPU worker thread.

Behaviour is modeled on vLLM 0.33's V1 scheduler (Apache-2.0); the code is written from
scratch. It covers one model and one KV group. There is no LoRA, multimodal input,
connectors, speculative decoding, or priority policy.

It is not meant to beat vLLM. The comparison below checks scheduling decisions only and
makes no speed claim.

## Run it

```bash
./scripts/test_scheduler.sh      # unit tests under ASan+UBSan and TSan
./scripts/compare_scheduler.sh   # replay every trace through vLLM and the C++ scheduler
```

`compare_scheduler.sh` needs vLLM 0.33.0 in `~/.venvs/ipl-vllm` (or set `VLLM_PYTHON`).
It runs vLLM's real `Scheduler` with a fake model and no GPU.

## Layout

| Path | What it holds |
| --- | --- |
| `src/block_pool` | Null block, free queue, ref counts, hashed block reuse |
| `src/kv_cache` | Chained block hashes, block-aligned prefix hits, per-request block tables |
| `src/scheduler` | Waiting and running queues, token budget, admission, preemption |
| `src/fake_gpu`, `src/engine` | The worker thread and the batch-queue engine loop |
| `src/replay.cpp`, `compare/` | C++ CLI, trace generator, and the vLLM replay |

## Result

**Matched 10378 of 10378 scheduling steps across 55 traces** on vLLM 0.33.0.
The output, seeds, git commit, and machine are in
[`results/scheduler/comparison.json`](../results/scheduler/comparison.json).

A step is one `schedule()` plus its `update_from_output()`. It matches only if all eight
recorded fields are equal: new request ids, their prefix-hit tokens, the running queue,
tokens per request, preempted ids, finished ids, and free blocks after each half.

| Family | Traces | Steps matched |
| --- | ---: | ---: |
| light load | 5 | 317 of 317 |
| token budget pressure | 5 | 682 of 682 |
| KV pressure with preemption | 5 | 1134 of 1134 |
| shared prefixes with eviction | 5 | 933 of 933 |
| async | 15 | 2953 of 2953 |
| pressure (budget and KV together) | 10 | 1971 of 1971 |
| tight (chunked admission, resume) | 10 | 2388 of 2388 |

The first five families were fixed before the first run and matched 6019 of 6019. The
last two were added afterwards, to reach paths that first set missed, and also matched.

## Known gaps

- Prefix hashes are a 64-bit chain, not SHA-256, so a collision would reuse the wrong
  block. That is fine for this comparison but not for a shared server.
- No abort path: requests only leave by finishing.
- Stale-output handling is covered by unit tests, not traces. With a two-deep queue the
  in-flight step always returns before the next schedule, so the gate cannot fire in a
  replay.
- Caching a block when its last output token returns changes a result only if that
  request is skipped on the very next step. No trace does this.
