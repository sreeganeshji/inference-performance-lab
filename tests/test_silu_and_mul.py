import pytest
import torch
import vllm._custom_ops

from inference_performance_lab.kernels.extension import (
    silu_and_mul as custom_silu_and_mul,
    silu_and_mul_packed as custom_packed_silu_and_mul,
)
from inference_performance_lab.kernels.reference import (
    silu_and_mul_reference,
)


INTERMEDIATE_SIZE = 18944


def vllm_silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    output_shape = x.shape[:-1] + (x.shape[-1] // 2,)
    output = torch.empty(
        output_shape,
        dtype=x.dtype,
        device=x.device,
    )
    torch.ops._C.silu_and_mul(output, x)
    return output


@pytest.mark.parametrize(
    "implementation",
    [
        pytest.param(
            vllm_silu_and_mul,
            id="vllm",
        ),
        pytest.param(
            custom_silu_and_mul,
            id="custom",
        ),
        pytest.param(
            custom_packed_silu_and_mul,
            id="custom-packed",
        ),
    ],
)
@pytest.mark.parametrize("num_tokens", [1, 8, 128, 2048])
@torch.inference_mode()
def test_matches_reference(
    implementation,
    num_tokens: int,
) -> None:
    torch.manual_seed(0)

    x = torch.randn(
        (num_tokens, 2 * INTERMEDIATE_SIZE),
        device="cuda",
        dtype=torch.bfloat16,
    )
    original = x.clone()

    expected = silu_and_mul_reference(x)
    actual = implementation(x)

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-2,
        atol=1e-2,
    )
    torch.testing.assert_close(
        x,
        original,
        rtol=0,
        atol=0,
    )


def test_reference_rejects_odd_final_dimension() -> None:
    x = torch.empty((2, 7))

    with pytest.raises(
        ValueError,
        match="final tensor dimension must be even",
    ):
        silu_and_mul_reference(x)