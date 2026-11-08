"""Generate the comparison traces from fixed seeds.

A trace is a config plus requests that arrive at fixed engine iterations. Both replays
read the same trace files, and the fake model's output is a pure function of the trace.
"""

import random
from pathlib import Path

LIGHT_SEEDS = range(101, 106)
BUDGET_SEEDS = range(201, 206)
KV_SEEDS = range(301, 306)
PREFIX_SEEDS = range(401, 406)
ASYNC_SEEDS = range(501, 516)
PRESSURE_SEEDS = range(601, 611)
TIGHT_SEEDS = range(701, 711)

BLOCK_SIZE = 16
MAX_ITERATIONS = 50_000


def base_config(**overrides):
    config = {
        "block_size": BLOCK_SIZE,
        "num_blocks": 4096,
        "max_tokens_per_step": 8192,
        "max_running_requests": 32,
        "max_model_len": 8192,
        "long_prefill_threshold": 0,
        "watermark": 0.0,
        "prefix_caching": 0,
        "chunked_prefill": 1,
        "async_scheduling": 0,
        "max_iterations": MAX_ITERATIONS,
    }
    config.update(overrides)
    return config


def random_tokens(rng, length):
    return [rng.randint(1, 30000) for _ in range(length)]


def make_request(rng, request_id, arrival, prompt, max_tokens, salt=None):
    stop_after = rng.randint(1, max_tokens) if rng.random() < 0.3 else -1
    return {
        "id": request_id,
        "arrival": arrival,
        "max_tokens": max_tokens,
        "stop_after": stop_after,
        "salt": request_id if salt is None else salt,
        "prompt": prompt,
    }


def ordered_by_arrival(requests):
    requests.sort(key=lambda request: request["arrival"])
    for new_id, request in enumerate(requests):
        if request["salt"] == request["id"]:
            request["salt"] = new_id
        request["id"] = new_id
    return requests


def clamp_to_pool(prompt_len, max_tokens, config):
    capacity = (config["num_blocks"] - 2) * BLOCK_SIZE
    capacity -= int(config["watermark"] * config["num_blocks"]) * BLOCK_SIZE
    prompt_len = min(prompt_len, capacity - 1)
    return prompt_len, min(max_tokens, capacity - prompt_len)


def light(seed):
    rng = random.Random(seed)
    config = base_config(prefix_caching=seed % 2)
    requests = [
        make_request(
            rng, i, rng.randint(0, 40), random_tokens(rng, rng.randint(8, 300)), rng.randint(1, 40)
        )
        for i in range(24)
    ]
    return config, ordered_by_arrival(requests)


def budget(seed):
    rng = random.Random(seed)
    index = seed % 5
    config = base_config(
        max_tokens_per_step=[32, 64, 128, 256, 512][index],
        long_prefill_threshold=[0, 16, 48, 0, 96][index],
        prefix_caching=seed % 2,
        max_running_requests=12,
    )
    max_prompt = 1200
    if index == 4:
        # Chunked prefill can only be off when one step can hold a whole sequence.
        config.update(chunked_prefill=0, max_tokens_per_step=1024, max_model_len=1024)
        max_prompt = 800

    requests = [
        make_request(
            rng,
            i,
            rng.randint(0, 20),
            random_tokens(rng, rng.randint(20, max_prompt)),
            rng.randint(1, 24),
        )
        for i in range(16)
    ]
    return config, ordered_by_arrival(requests)


def kv_pressure(seed):
    rng = random.Random(seed)
    index = seed % 5
    config = base_config(
        num_blocks=[32, 40, 48, 64, 96][index],
        watermark=[0.0, 0.0, 0.05, 0.1, 0.0][index],
        prefix_caching=seed % 2,
        max_running_requests=16,
    )
    if index == 3:
        config["max_model_len"] = 256

    requests = []
    for i in range(20):
        prompt_len, max_tokens = clamp_to_pool(rng.randint(16, 200), rng.randint(8, 100), config)
        if index == 3:
            prompt_len = min(prompt_len, 250)
        requests.append(
            make_request(rng, i, rng.randint(0, 10), random_tokens(rng, prompt_len), max_tokens)
        )
    return config, ordered_by_arrival(requests)


def shared_prefix(seed):
    rng = random.Random(seed)
    config = base_config(
        num_blocks=[64, 96, 128, 192, 256][seed % 5], prefix_caching=1, max_running_requests=16
    )
    prefixes = [
        random_tokens(rng, rng.choice([32, 47, 48, 49, 80, 96, 160, 200])) for _ in range(3)
    ]

    requests = []
    for i in range(40):
        arrival = rng.randint(0, 200)
        if rng.random() < 0.75:
            group = rng.randrange(len(prefixes))
            suffix = rng.randint(0, 40)
            prompt = prefixes[group] + random_tokens(rng, suffix)
            salt = group if suffix == 0 else None
        else:
            prompt = random_tokens(rng, rng.randint(8, 80))
            salt = None

        prompt_len, max_tokens = clamp_to_pool(len(prompt), rng.randint(2, 24), config)
        requests.append(make_request(rng, i, arrival, prompt[:prompt_len], max_tokens, salt))
    return config, ordered_by_arrival(requests)


def pressure(seed):
    rng = random.Random(seed)
    index = seed % 5
    config = base_config(
        num_blocks=[24, 32, 40, 48, 64][index],
        max_tokens_per_step=[16, 32, 48, 64, 96][index],
        long_prefill_threshold=[0, 8, 0, 24, 0][index],
        prefix_caching=1,
        max_running_requests=[2, 4, 8, 6, 8][index],
        async_scheduling=seed % 2,
    )
    prompts = [random_tokens(rng, rng.choice([32, 48, 64, 96, 128])) for _ in range(4)]

    requests = []
    for i in range(30):
        group = rng.randrange(len(prompts))
        prompt_len, max_tokens = clamp_to_pool(len(prompts[group]), rng.randint(8, 48), config)
        prompt = prompts[group][:prompt_len]
        requests.append(make_request(rng, i, rng.randint(0, 60), prompt, max_tokens, salt=group))
    return config, ordered_by_arrival(requests)


def tight(seed):
    rng = random.Random(seed)
    if seed % 2 == 0:
        # A small step budget chunks long prompts: the first chunk fits but the whole prompt does not.
        config = base_config(
            num_blocks=rng.choice([32, 40, 48]),
            max_tokens_per_step=rng.choice([32, 64]),
            prefix_caching=1,
            max_running_requests=8,
        )
        prompt_lengths = (200, 320)
        output_lengths = (8, 16)
    else:
        # Long outputs under async, so preemption lands on blocks filled by output tokens.
        config = base_config(
            num_blocks=rng.choice([24, 32, 40]),
            prefix_caching=1,
            max_running_requests=8,
            async_scheduling=1,
        )
        prompt_lengths = (16, 48)
        output_lengths = (48, 96)

    requests = []
    for i in range(20):
        prompt_len, max_tokens = clamp_to_pool(
            rng.randint(*prompt_lengths), rng.randint(*output_lengths), config
        )
        prompt = random_tokens(rng, prompt_len)
        requests.append(make_request(rng, i, rng.randint(0, 30), prompt, max_tokens))
    return config, ordered_by_arrival(requests)


def async_trace(seed):
    variant = [kv_pressure, shared_prefix, budget][seed % 3]
    config, requests = variant(seed)
    config["async_scheduling"] = 1
    return config, requests


FAMILIES = {
    "light": (light, LIGHT_SEEDS),
    "budget": (budget, BUDGET_SEEDS),
    "kv": (kv_pressure, KV_SEEDS),
    "prefix": (shared_prefix, PREFIX_SEEDS),
    "async": (async_trace, ASYNC_SEEDS),
    "pressure": (pressure, PRESSURE_SEEDS),
    "tight": (tight, TIGHT_SEEDS),
}


def write_trace(path, config, requests):
    lines = [f"{key} {value}" for key, value in config.items()]
    for r in requests:
        header = f"{r['id']} {r['arrival']} {r['max_tokens']} {r['stop_after']} {r['salt']}"
        tokens = " ".join(str(token) for token in r["prompt"])
        lines.append(f"request {header} {len(r['prompt'])} {tokens}")
    Path(path).write_text("\n".join(lines) + "\n")


def write_all_traces(directory):
    """Write every trace and return (family, seed, path) in a stable order."""
    written = []
    for family, (generate, seeds) in FAMILIES.items():
        for seed in seeds:
            path = Path(directory) / f"{family}-{seed}.trace"
            write_trace(path, *generate(seed))
            written.append((family, seed, path))
    return written
