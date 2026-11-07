"""End-to-end check of the CUSTOM attention backend inside vLLM, without downloading weights.

Runs the same greedy generation twice in fresh processes, once with vLLM's FLASH_ATTN backend and
once with this lab's CUSTOM backend, and compares the generated tokens and their logprobs. By
default the model is a 4-layer copy of Qwen2.5-7B's architecture with vLLM's dummy weights, so it
exercises the real shapes (28 query heads, 4 KV heads, head_dim 128) on a 12 GB GPU. Pass --model
to use a real checkpoint instead.

A small token budget forces chunked prefill, so batches mix decode rows (lab kernel) and prefill
rows (FlashAttention-2).

    python benchmarks/vllm_attention_check.py               # CUDA graphs (vLLM's default)
    python benchmarks/vllm_attention_check.py --enforce-eager
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

QWEN25_7B_CONFIG = {
    "architectures": ["Qwen2ForCausalLM"],
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "eos_token_id": 151645,
    "hidden_act": "silu",
    "hidden_size": 3584,
    "initializer_range": 0.02,
    "intermediate_size": 18944,
    "max_position_embeddings": 32768,
    "max_window_layers": 28,
    "model_type": "qwen2",
    "num_attention_heads": 28,
    "num_hidden_layers": 4,
    "num_key_value_heads": 4,
    "rms_norm_eps": 1e-06,
    "rope_theta": 1000000.0,
    "sliding_window": 131072,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "use_cache": True,
    "use_sliding_window": False,
    "vocab_size": 152064,
}

PROMPT_LENGTHS = [17, 300, 1000, 2500, 40, 4000, 5, 1500]


def dummy_model_dir() -> Path:
    directory = Path.home() / ".cache" / "ipl-models" / "qwen2.5-7b-arch-4-layers"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(QWEN25_7B_CONFIG, indent=2))
    return directory


def run_worker(args: argparse.Namespace) -> None:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(
        model=args.model,
        revision=args.revision,
        load_format="dummy" if args.dummy else "auto",
        skip_tokenizer_init=args.dummy,
        dtype="bfloat16",
        seed=0,
        max_model_len=8192,
        max_num_batched_tokens=1024,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        attention_backend=args.backend,
    )

    prompts = [
        TokensPrompt(prompt_token_ids=[(7919 * i + 13 * j) % 150000 + 100 for j in range(length)])
        for i, length in enumerate(PROMPT_LENGTHS)
    ]
    params = SamplingParams(
        temperature=0.0, max_tokens=args.max_tokens, logprobs=1, detokenize=False
    )
    outputs = llm.generate(prompts, params, use_tqdm=False)

    result = []

    for output in outputs:
        completion = output.outputs[0]
        logprobs = [
            entry[token].logprob for entry, token in zip(completion.logprobs, completion.token_ids)
        ]
        result.append({"tokens": list(completion.token_ids), "logprobs": logprobs})

    Path(args.result).write_text(json.dumps(result))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", help="a real checkpoint; default: dummy Qwen2.5-7B architecture")
    parser.add_argument("--revision", help="the checkpoint's Hugging Face revision")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--backend", help=argparse.SUPPRESS)
    parser.add_argument("--result", help=argparse.SUPPRESS)
    parser.add_argument("--dummy", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        run_worker(args)
        return

    dummy = args.model is None
    model = str(dummy_model_dir()) if dummy else args.model
    results = {}

    for backend in ("FLASH_ATTN", "CUSTOM"):
        result_path = Path.home() / ".cache" / "ipl-models" / f"check-{backend}.json"
        command = [
            sys.executable,
            __file__,
            "--worker",
            "--backend",
            backend,
            "--model",
            model,
            "--result",
            str(result_path),
            "--max-tokens",
            str(args.max_tokens),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
        ]
        command += ["--dummy"] if dummy else []
        command += ["--revision", args.revision] if args.revision else []
        command += ["--enforce-eager"] if args.enforce_eager else []

        completed = subprocess.run(
            command,
            env=os.environ.copy(),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        print(f"[{backend}] exit {completed.returncode}")

        for line in completed.stdout.splitlines():
            if "backend" in line.lower() or "Lab decode" in line:
                print(f"  {line.split(']', 1)[-1].strip()[:150]}")

        if completed.returncode != 0:
            print(completed.stdout[-4000:])
            sys.exit(1)

        results[backend] = json.loads(result_path.read_text())

    reference, custom = results["FLASH_ATTN"], results["CUSTOM"]
    matching_tokens = total_tokens = 0
    first_divergence = []
    max_logprob_delta = 0.0

    for ref, lab in zip(reference, custom):
        prefix = 0

        while (
            prefix < min(len(ref["tokens"]), len(lab["tokens"]))
            and ref["tokens"][prefix] == lab["tokens"][prefix]
        ):
            prefix += 1

        first_divergence.append(prefix)
        matching_tokens += prefix
        total_tokens += len(ref["tokens"])

        for a, b in zip(ref["logprobs"][:prefix], lab["logprobs"][:prefix]):
            if math.isfinite(a) and math.isfinite(b):
                max_logprob_delta = max(max_logprob_delta, abs(a - b))

    finite = all(math.isfinite(x) for item in custom for x in item["logprobs"])
    print(
        f"identical greedy prefix: {matching_tokens}/{total_tokens} tokens; per prompt {first_divergence}"
    )
    print(
        f"max |logprob difference| over the identical prefix: {max_logprob_delta:.3g}; all finite: {finite}"
    )


if __name__ == "__main__":
    main()
