"""Replay a trace through vLLM 0.33's real Scheduler, with a fake model and no GPU.

The step record is fixed here and in replay.cpp. A step is one schedule() plus its
update_from_output(), and the two sides must agree on every field to count as a match:

  new                  request ids admitted this step, new then resumed
  hits                 prefix-hit tokens for each of those requests
  running              the running queue, in order, right after schedule()
  scheduled            (request id, tokens) pairs in scheduling order
  preempted            ids preempted by this schedule(), sorted
  finished             ids finished by this update_from_output(), in order
  free_after_schedule  free KV blocks right after schedule()
  free_after_update    free KV blocks right after update_from_output()
"""

import collections
from pathlib import Path

import torch
from vllm.config import CacheConfig, ModelConfig, ParallelConfig, SchedulerConfig, VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

FAKE_MODEL = str(Path(__file__).parent / "fake-qwen2")
EOS_TOKEN = 50256

init_none_hash(sha256)


def read_trace(path):
    config = {}
    requests = []
    for line in Path(path).read_text().splitlines():
        key, *values = line.split()
        if key == "request":
            numbers = [int(value) for value in values]
            request_id, arrival, max_tokens, stop_after, salt, _ = numbers[:6]
            requests.append(
                {
                    "id": request_id,
                    "arrival": arrival,
                    "max_tokens": max_tokens,
                    "stop_after": stop_after,
                    "salt": salt,
                    "prompt": numbers[6:],
                }
            )
        elif key == "watermark":
            config[key] = float(values[0])
        else:
            config[key] = int(values[0])
    return config, requests


def trace_token(request, output_index):
    if output_index + 1 == request["stop_after"]:
        return EOS_TOKEN
    return 30001 + (request["salt"] * 31 + output_index * 7) % 9973


def make_scheduler(config):
    model_config = ModelConfig(
        model=FAKE_MODEL,
        dtype="float16",
        seed=42,
        skip_tokenizer_init=True,
        max_model_len=config["max_model_len"],
    )
    vllm_config = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_seqs=config["max_running_requests"],
            max_num_batched_tokens=config["max_tokens_per_step"],
            max_model_len=config["max_model_len"],
            long_prefill_token_threshold=config["long_prefill_threshold"],
            enable_chunked_prefill=bool(config["chunked_prefill"]),
            async_scheduling=bool(config["async_scheduling"]),
            is_encoder_decoder=False,
            watermark=config["watermark"],
            policy="fcfs",
        ),
        model_config=model_config,
        cache_config=CacheConfig(
            block_size=config["block_size"],
            gpu_memory_utilization=0.9,
            cache_dtype="auto",
            enable_prefix_caching=bool(config["prefix_caching"]),
        ),
        parallel_config=ParallelConfig(pipeline_parallel_size=1, data_parallel_size=1),
    )
    vllm_config.cache_config.num_gpu_blocks = config["num_blocks"]
    register_all_kvcache_specs(vllm_config)

    spec = FullAttentionSpec(
        block_size=config["block_size"], num_kv_heads=1, head_size=1, dtype=torch.float32
    )
    scheduler_class = AsyncScheduler if config["async_scheduling"] else Scheduler
    scheduler = scheduler_class(
        vllm_config=vllm_config,
        kv_cache_config=KVCacheConfig(
            num_blocks=config["num_blocks"],
            kv_cache_tensors=[],
            kv_cache_groups=[KVCacheGroupSpec(["layer"], spec)],
        ),
        block_size=config["block_size"],
        log_stats=False,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )
    # Match the V2 model runner, vLLM 0.33's default, with resumed requests in scheduled_new_reqs.
    scheduler.use_v2_model_runner = True
    return scheduler


def free_blocks(scheduler):
    return scheduler.kv_cache_manager.block_pool.get_num_free_blocks()


class Engine:
    """The same loop as EngineCore.step and step_with_batch_queue: a batch queue of
    depth 1 is a synchronous step, and depth 2 runs one step ahead of the GPU."""

    def __init__(self, config, requests):
        self.scheduler = make_scheduler(config)
        self.requests = {request["id"]: request for request in requests}
        self.depth = 2 if config["async_scheduling"] else 1
        self.block_hasher = get_request_block_hasher(config["block_size"], sha256)
        self.num_sampled = collections.Counter()
        self.in_flight = collections.deque()

    def add_request(self, request):
        sampling_params = SamplingParams(max_tokens=request["max_tokens"])
        sampling_params.update_from_generation_config({}, EOS_TOKEN)
        self.scheduler.add_request(
            Request(
                request_id=str(request["id"]),
                prompt_token_ids=request["prompt"],
                sampling_params=sampling_params,
                pooling_params=None,
                block_hasher=self.block_hasher,
            )
        )

    def busy(self):
        return self.scheduler.has_requests() or bool(self.in_flight)

    def sample(self, req_id):
        request_id = int(req_id)
        output_index = self.num_sampled[request_id]
        self.num_sampled[request_id] += 1
        return trace_token(self.requests[request_id], output_index)

    def start_step(self):
        output = self.scheduler.schedule()
        record = {
            "new": [int(new.req_id) for new in output.scheduled_new_reqs],
            "hits": [new.num_computed_tokens for new in output.scheduled_new_reqs],
            "running": [int(r.request_id) for r in self.scheduler.running],
            "scheduled": [[int(k), v] for k, v in output.num_scheduled_tokens.items()],
            "preempted": sorted(int(req_id) for req_id in output.preempted_req_ids),
            "free_after_schedule": free_blocks(self.scheduler),
        }
        sampled_token_ids = []
        for req_id in output.num_scheduled_tokens:
            sampled = not self.scheduler.requests[req_id].is_prefill_chunk
            sampled_token_ids.append([self.sample(req_id)] if sampled else [])
        return output, record, sampled_token_ids

    def finish_step(self, step):
        output, record, sampled_token_ids = step
        req_ids = list(output.num_scheduled_tokens)
        engine_outputs = self.scheduler.update_from_output(
            output,
            ModelRunnerOutput(
                req_ids=req_ids,
                req_id_to_index={req_id: index for index, req_id in enumerate(req_ids)},
                sampled_token_ids=sampled_token_ids,
            ),
        )
        record["finished"] = [
            int(o.request_id)
            for client in engine_outputs.values()
            for o in client.outputs
            if o.finish_reason is not None
        ]
        record["free_after_update"] = free_blocks(self.scheduler)
        return record

    def step(self):
        if self.scheduler.has_requests():
            step = self.start_step()
            self.in_flight.appendleft(step)
            model_executed = step[0].total_num_scheduled_tokens > 0
            if len(self.in_flight) < self.depth and (
                model_executed or self.scheduler.has_requests()
            ):
                return None
        elif not self.in_flight:
            return None
        return self.finish_step(self.in_flight.pop())


def replay(path):
    config, requests = read_trace(path)
    engine = Engine(config, requests)
    pending = collections.deque(requests)
    records = []

    for iteration in range(config["max_iterations"]):
        while pending and pending[0]["arrival"] <= iteration:
            engine.add_request(pending.popleft())
        if not engine.busy() and not pending:
            break
        record = engine.step()
        if record is not None:
            records.append(record)

    return records
