import pytest
import torch
from vllm import _custom_ops as vllm_ops

from inference_performance_lab.kernels.extension import (
    fused_add_rms_norm as custom_fused_add_rms_norm,
)
from inference_performance_lab.kernels.reference import (
    fused_add_rms_norm_reference,
)


HIDDEN_SIZE = 3584
EPSILON = 1e-6


@pytest.mark.parametrize(
    "implementation",
    [
        pytest.param(
            vllm_ops.fused_add_rms_norm,
            id="vllm",
        ),
        pytest.param(
            custom_fused_add_rms_norm,
            id="custom",
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

    expected_output, expected_residual = (
        fused_add_rms_norm_reference(
            x,
            residual,
            weight,
            EPSILON,
        )
    )

    actual_output = x.clone()
    actual_residual = residual.clone()

    implementation(
        actual_output,
        actual_residual,
        weight,
        EPSILON,
    )

    torch.testing.assert_close(
        actual_residual,
        expected_residual,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        actual_output,
        expected_output,
        rtol=1e-2,
        atol=1e-2,
    )

def misaligned_contiguous_copy(
    tensor: torch.Tensor,
) -> torch.Tensor:
    storage = torch.empty(
        tensor.numel() + 1,
        device=tensor.device,
        dtype=tensor.dtype,
    )

    result = storage[1:].view_as(tensor)
    result.copy_(tensor)

    assert result.is_contiguous()
    assert result.data_ptr() % 16 != 0

    return result


@pytest.mark.parametrize(
    "misaligned_argument",
    ["x", "residual", "weight"],
)
@torch.inference_mode()
def test_custom_handles_misaligned_contiguous_tensors(
    misaligned_argument: str,
) -> None:
    torch.manual_seed(0)

    x = torch.randn(
        8,
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

    expected_output, expected_residual = (
        fused_add_rms_norm_reference(
            x,
            residual,
            weight,
            EPSILON,
        )
    )

    actual_output = x.clone()
    actual_residual = residual.clone()
    actual_weight = weight.clone()

    if misaligned_argument == "x":
        actual_output = misaligned_contiguous_copy(x)
    elif misaligned_argument == "residual":
        actual_residual = misaligned_contiguous_copy(
            residual
        )
    else:
        actual_weight = misaligned_contiguous_copy(
            weight
        )

    custom_fused_add_rms_norm(
        actual_output,
        actual_residual,
        actual_weight,
        EPSILON,
    )

    torch.testing.assert_close(
        actual_residual,
        expected_residual,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        actual_output,
        expected_output,
        rtol=1e-2,
        atol=1e-2,
    )