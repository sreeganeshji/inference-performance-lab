from collections.abc import Callable

import pytest
import torch

from inference_performance_lab.kernels.extension import silu_and_mul as custom_silu_and_mul
from inference_performance_lab.kernels.extension import (
    silu_and_mul_packed as custom_packed_silu_and_mul,
)
from inference_performance_lab.kernels.reference import silu_and_mul_reference

INTERMEDIATE_SIZE = 18944
TOKEN_COUNTS = [1, 8, 128, 2048]

Implementation = Callable[[torch.Tensor], torch.Tensor]
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")


def assert_matches_reference(implementation: Implementation, num_tokens: int) -> None:
    torch.manual_seed(0)

    x = torch.randn((num_tokens, 2 * INTERMEDIATE_SIZE), device="cuda", dtype=torch.bfloat16)
    original = x.clone()

    expected = silu_and_mul_reference(x)
    actual = implementation(x)

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(x, original, rtol=0, atol=0)


@pytest.mark.gpu
@requires_cuda
@pytest.mark.parametrize(
    "implementation",
    [
        pytest.param(custom_silu_and_mul, id="custom-scalar"),
        pytest.param(custom_packed_silu_and_mul, id="custom-packed"),
    ],
)
@pytest.mark.parametrize("num_tokens", TOKEN_COUNTS)
@torch.inference_mode()
def test_custom_matches_reference(implementation: Implementation, num_tokens: int) -> None:
    assert_matches_reference(implementation, num_tokens)


@pytest.mark.gpu
@pytest.mark.vllm
@requires_cuda
@pytest.mark.parametrize("num_tokens", TOKEN_COUNTS)
@torch.inference_mode()
def test_vllm_matches_reference(num_tokens: int) -> None:
    pytest.importorskip("vllm._custom_ops")

    def vllm_silu_and_mul(x: torch.Tensor) -> torch.Tensor:
        output_shape = x.shape[:-1] + (x.shape[-1] // 2,)
        output = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        torch.ops._C.silu_and_mul(output, x)
        return output

    assert_matches_reference(vllm_silu_and_mul, num_tokens)


def test_reference_rejects_odd_final_dimension() -> None:
    x = torch.empty((2, 7))

    with pytest.raises(ValueError, match="final tensor dimension must be even"):
        silu_and_mul_reference(x)
