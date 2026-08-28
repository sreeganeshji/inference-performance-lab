from collections.abc import Callable

import torch
import vllm._custom_ops

from inference_performance_lab.kernels.extension import (
    load_extension,
)


Implementation = Callable[
    [torch.Tensor, torch.Tensor],
    None,
]

INTERMEDIATE_SIZE = 18944
TOKEN_COUNTS = [1, 8, 128, 2048]
ITERATIONS = {
    1: 2000,
    8: 1000,
    128: 300,
    2048: 50,
}
WARMUP_ITERATIONS = 25

# Read gate and up BF16 values, then write one BF16 output.
MINIMUM_BYTES_PER_OUTPUT = 6


def vllm_silu_and_mul(
        output: torch.Tensor,
        x: torch.Tensor,
    ) -> None:
    torch.ops._C.silu_and_mul(output, x)


def custom_scalar_silu_and_mul(
        output: torch.Tensor,
        x: torch.Tensor,
    ) -> None:
    torch.ops.inference_performance_lab.silu_and_mul(
        output,
        x,
    )

def custom_packed_silu_and_mul(
        output: torch.Tensor,
        x: torch.Tensor,
    ) -> None:
    torch.ops.inference_performance_lab.silu_and_mul_packed(output, x)


@torch.inference_mode()
def benchmark(
    implementation: Implementation,
    num_tokens: int,
) -> tuple[float, float]:
    torch.manual_seed(0)

    x = torch.randn(
        (num_tokens, 2 * INTERMEDIATE_SIZE),
        device="cuda",
        dtype=torch.bfloat16,
    )
    output = torch.empty(
        (num_tokens, INTERMEDIATE_SIZE),
        device="cuda",
        dtype=torch.bfloat16,
    )

    for _ in range(WARMUP_ITERATIONS):
        implementation(output, x)

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    iterations = ITERATIONS[num_tokens]

    start.record()

    for _ in range(iterations):
        implementation(output, x)

    end.record()
    end.synchronize()

    milliseconds = start.elapsed_time(end) / iterations
    microseconds = milliseconds * 1000.0

    minimum_bytes = (
        num_tokens
        * INTERMEDIATE_SIZE
        * MINIMUM_BYTES_PER_OUTPUT
    )
    effective_gbps = (
        minimum_bytes
        / (microseconds * 1e-6)
        / 1e9
    )

    return microseconds, effective_gbps


def main() -> None:
    load_extension()

    implementations = {
        "vLLM": vllm_silu_and_mul,
        "Custom scalar": custom_scalar_silu_and_mul,
        "Custom packed": custom_packed_silu_and_mul,
    }
    results: dict[
        str,
        dict[int, tuple[float, float]],
    ] = {}

    for name, implementation in implementations.items():
        results[name] = {}

        for num_tokens in TOKEN_COUNTS:
            results[name][num_tokens] = benchmark(
                implementation,
                num_tokens,
            )

    print(
        "| Tokens | vLLM (us) | Scalar (us) | Packed (us) "
        "| Scalar speedup | Packed speedup "
        "| vLLM GB/s | Scalar GB/s | Packed GB/s |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for num_tokens in TOKEN_COUNTS:
        vllm_us, vllm_gbps = results["vLLM"][num_tokens]
        scalar_us, scalar_gbps = results["Custom scalar"][num_tokens]
        packed_us, packed_gbps = results["Custom packed"][num_tokens]

        print(
            f"| {num_tokens} "
            f"| {vllm_us:.3f} "
            f"| {scalar_us:.3f} "
            f"| {packed_us:.3f} "
            f"| {vllm_us / scalar_us:.3f}x "
            f"| {vllm_us / packed_us:.3f}x "
            f"| {vllm_gbps:.1f} "
            f"| {scalar_gbps:.1f} "
            f"| {packed_gbps:.1f} |"
        )


if __name__ == "__main__":
    main()