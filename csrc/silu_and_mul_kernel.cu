#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/BFloat16.h>

#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int kThreadsPerBlock = 256;

constexpr int kVectorWidth = 8;

struct alignas(16) BFloat16x8 {
    c10::BFloat16 values[kVectorWidth];
};

static_assert(sizeof(BFloat16x8) == 16);

__global__ void silu_and_mul_bf16_scalar_kernel(
    c10::BFloat16* output,
    const c10::BFloat16* input,
    int64_t intermediate_size) {
    const int64_t row = blockIdx.y;
    const int64_t column = blockIdx.x * blockDim.x + threadIdx.x;

    if (column >= intermediate_size) {
        return;
    }

    const int64_t input_row_offset = row * 2 * intermediate_size;
    const int64_t output_row_offset = row * intermediate_size;

    const float gate = static_cast<float>(input[input_row_offset + column]);
    const float up = static_cast<float>(input[input_row_offset+ intermediate_size+ column]);

    const float silu = gate / (1.0f + expf(-gate));

    output[output_row_offset + column] = c10::BFloat16(silu * up);
}

__global__ void silu_and_mul_bf16_packed_kernel(
    c10::BFloat16* output,
    const c10::BFloat16* input,
    int64_t intermediate_size) {
    const int64_t row = blockIdx.y;
    const int64_t pack_column =
        blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t packs_per_row =
        intermediate_size / kVectorWidth;

    if (pack_column >= packs_per_row) {
        return;
    }

    const int64_t input_row_offset =
        row * 2 * intermediate_size;
    const int64_t output_row_offset =
        row * intermediate_size;

    const auto* gate_packs =
        reinterpret_cast<const BFloat16x8*>(
            input + input_row_offset);
    const auto* up_packs =
        reinterpret_cast<const BFloat16x8*>(
            input
            + input_row_offset
            + intermediate_size);
    auto* output_packs =
        reinterpret_cast<BFloat16x8*>(
            output + output_row_offset);

    const BFloat16x8 gate_pack =
        gate_packs[pack_column];
    const BFloat16x8 up_pack =
        up_packs[pack_column];

    BFloat16x8 output_pack;

#pragma unroll
    for (int element = 0;
         element < kVectorWidth;
         ++element) {
        const float gate = static_cast<float>(
            gate_pack.values[element]);
        const float up = static_cast<float>(
            up_pack.values[element]);

        const float silu =
            gate / (1.0f + expf(-gate));

        output_pack.values[element] =
            c10::BFloat16(silu * up);
    }

    output_packs[pack_column] = output_pack;
}

}  // namespace

void silu_and_mul_cuda(at::Tensor output, const at::Tensor& input) {
    TORCH_CHECK(input.is_cuda() && output.is_cuda(), "input and output must be CUDA tensors");
    TORCH_CHECK(input.scalar_type() == at::kBFloat16 && output.scalar_type() == at::kBFloat16, "input and output must use bfloat16");
    TORCH_CHECK(input.is_contiguous() && output.is_contiguous(), "input and output must be contiguous");
    TORCH_CHECK(input.device() == output.device(), "input and output must be on the same device");
    TORCH_CHECK(input.dim() >= 1 && input.size(-1) % 2 == 0, "the final input dimension must be even");
    TORCH_CHECK(output.dim() == input.dim(), "input and output must have the same rank");

    const int64_t intermediate_size = input.size(-1) / 2;

    TORCH_CHECK(output.size(-1) == intermediate_size, "the output width must be half the input width");

    for (int64_t dimension = 0; dimension < input.dim() - 1; ++dimension) {
        TORCH_CHECK(output.size(dimension) == input.size(dimension), "input and output leading dimensions must match");
    }

    if (output.numel() == 0) {
        return;
    }

    const int64_t rows = output.numel() / intermediate_size;

    c10::cuda::CUDAGuard device_guard(input.device());

    const cudaDeviceProp* properties = at::cuda::getCurrentDeviceProperties();

    TORCH_CHECK(rows <= properties->maxGridSize[1], "the number of rows exceeds CUDA grid.y capacity");

    const dim3 block(kThreadsPerBlock);
    const dim3 grid((intermediate_size + kThreadsPerBlock - 1)/ kThreadsPerBlock, rows);

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    silu_and_mul_bf16_scalar_kernel<<<grid,block,0,stream>>>(
            output.data_ptr<c10::BFloat16>(),
            input.data_ptr<c10::BFloat16>(),
            intermediate_size);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void silu_and_mul_packed_cuda(
    at::Tensor output,
    const at::Tensor& input) {
    TORCH_CHECK(
        input.is_cuda() && output.is_cuda(),
        "input and output must be CUDA tensors");
    TORCH_CHECK(
        input.scalar_type() == at::kBFloat16
            && output.scalar_type() == at::kBFloat16,
        "input and output must use bfloat16");
    TORCH_CHECK(
        input.is_contiguous() && output.is_contiguous(),
        "input and output must be contiguous");
    TORCH_CHECK(
        input.device() == output.device(),
        "input and output must be on the same device");
    TORCH_CHECK(
        input.dim() >= 1 && input.size(-1) % 2 == 0,
        "the final input dimension must be even");
    TORCH_CHECK(
        output.dim() == input.dim(),
        "input and output must have the same rank");

    const int64_t intermediate_size =
        input.size(-1) / 2;

    TORCH_CHECK(
        output.size(-1) == intermediate_size,
        "the output width must be half the input width");

    for (int64_t dimension = 0;
         dimension < input.dim() - 1;
         ++dimension) {
        TORCH_CHECK(
            output.size(dimension)
                == input.size(dimension),
            "input and output leading dimensions must match");
    }

    if (output.numel() == 0) {
        return;
    }

    TORCH_CHECK(
        intermediate_size % kVectorWidth == 0,
        "the output width must be divisible by 8");

    const auto input_address =
        reinterpret_cast<std::uintptr_t>(
            input.data_ptr<c10::BFloat16>());
    const auto output_address =
        reinterpret_cast<std::uintptr_t>(
            output.data_ptr<c10::BFloat16>());

    TORCH_CHECK(
        input_address % alignof(BFloat16x8) == 0
            && output_address % alignof(BFloat16x8) == 0,
        "input and output must be 16-byte aligned");

    const int64_t rows =
        output.numel() / intermediate_size;
    const int64_t packs_per_row =
        intermediate_size / kVectorWidth;

    c10::cuda::CUDAGuard device_guard(
        input.device());

    const cudaDeviceProp* properties =
        at::cuda::getCurrentDeviceProperties();

    TORCH_CHECK(
        rows <= properties->maxGridSize[1],
        "the number of rows exceeds CUDA grid.y capacity");

    const dim3 block(kThreadsPerBlock);
    const dim3 grid(
        (packs_per_row + kThreadsPerBlock - 1)
            / kThreadsPerBlock,
        rows);

    const cudaStream_t stream =
        at::cuda::getCurrentCUDAStream();

    silu_and_mul_bf16_packed_kernel<<<
        grid,
        block,
        0,
        stream>>>(
            output.data_ptr<c10::BFloat16>(),
            input.data_ptr<c10::BFloat16>(),
            intermediate_size);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}