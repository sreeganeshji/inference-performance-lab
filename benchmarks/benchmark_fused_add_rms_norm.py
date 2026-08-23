from collections.abc import Callable

import torch
from vllm import _custom_ops as vllm_ops

from inference_performance_lab.kernels.extension import (
    fused_add_rms_norm as custom_fused_add_rms_norm,
)


Implementation = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, float],
    None,
]

HIDDEN_SIZE = 3584
EPSILON = 1e-6
TOKEN_COUNTS = [1, 8, 128, 2048]
ITERATIONS = {
    1: 2000,
    8: 1000,
    128: 300,
    2048: 50,
}
WARMUP_ITERATIONS = 25

# Minimum semantic traffic per element:
# read x, residual, and weight; write x and residual.
MINIMUM_BYTES_PER_ELEMENT = 10


def benchmark(
    implementation: Implementation,
    num_tokens: int,
) -> tuple[float, float]:
    torch.manual_seed(0)

    x = torch.randn(
        num_tokens,
        HIDDEN_SIZE,
        device="cuda",
        dtype=torch.bfloat16,
    )
    residual = torch.randn_like(x)
    weight = (
        torch.randn(
            HIDDEN_SIZE,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.1
        + 1.0
    )

    for _ in range(WARMUP_ITERATIONS):
        implementation(x, residual, weight, EPSILON)

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    iterations = ITERATIONS[num_tokens]

    start.record()

    for _ in range(iterations):
        implementation(x, residual, weight, EPSILON)

    end.record()
    end.synchronize()

    milliseconds = start.elapsed_time(end) / iterations
    microseconds = milliseconds * 1000.0

    minimum_bytes = (
        num_tokens
        * HIDDEN_SIZE
        * MINIMUM_BYTES_PER_ELEMENT
    )
    effective_gbps = (
        minimum_bytes
        / (microseconds * 1e-6)
        / 1e9
    )

    return microseconds, effective_gbps


def main() -> None:
    implementations = {
        "vLLM": vllm_ops.fused_add_rms_norm,
        "custom": custom_fused_add_rms_norm,
    }
    results: dict[str, dict[int, tuple[float, float]]] = {}

    for name, implementation in implementations.items():
        results[name] = {}

        for num_tokens in TOKEN_COUNTS:
            results[name][num_tokens] = benchmark(
                implementation,
                num_tokens,
            )

    print(
        "| Tokens | vLLM (us) | Custom (us) | "
        "Speedup | vLLM GB/s | Custom GB/s |"
    )
    print("|---:|---:|---:|---:|---:|---:|")

    for num_tokens in TOKEN_COUNTS:
        vllm_us, vllm_gbps = results["vLLM"][num_tokens]
        custom_us, custom_gbps = results["custom"][num_tokens]

        print(
            f"| {num_tokens} "
            f"| {vllm_us:.3f} "
            f"| {custom_us:.3f} "
            f"| {vllm_us / custom_us:.3f}x "
            f"| {vllm_gbps:.1f} "
            f"| {custom_gbps:.1f} |"
        )


if __name__ == "__main__":
    main()