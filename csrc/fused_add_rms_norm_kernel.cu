#include <ATen/ATen.h>
// #include <ATen/BFloat16.h>
#include <c10/util/BFloat16.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>

#include <limits>


namespace {

constexpr int kBlockSize = 256;
constexpr int kWarpSize = 32;
constexpr int kVectorWidth = 8;

struct alignas(16) BFloat16x8 {
    at::BFloat16 values[kVectorWidth];
};

static_assert(sizeof(BFloat16x8) == 16);
static_assert(alignof(BFloat16x8) == 16);


__device__ __forceinline__ float warp_reduce_sum(float value) {
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }

    return value;
}


__global__ void fused_add_rms_norm_bf16_generic_kernel(
    at::BFloat16* x,
    at::BFloat16* residual,
    const at::BFloat16* weight,
    int hidden_size,
    float epsilon) {
    const int row = blockIdx.x;
    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;
    const int num_warps = blockDim.x / kWarpSize;

    x += static_cast<int64_t>(row) * hidden_size;
    residual += static_cast<int64_t>(row) * hidden_size;

    float sum_of_squares = 0.0f;

    for (int column = threadIdx.x;
         column < hidden_size;
         column += blockDim.x) {
        const float value =
            static_cast<float>(x[column])
            + static_cast<float>(residual[column]);

        sum_of_squares = fmaf(
            value,
            value,
            sum_of_squares);
    }

    sum_of_squares = warp_reduce_sum(sum_of_squares);

    __shared__ float warp_sums[kBlockSize / kWarpSize];
    __shared__ float inverse_rms;

    if (lane == 0) {
        warp_sums[warp] = sum_of_squares;
    }

    __syncthreads();

    if (warp == 0) {
        float block_sum =
            lane < num_warps
                ? warp_sums[lane]
                : 0.0f;

        block_sum = warp_reduce_sum(block_sum);

        if (lane == 0) {
            inverse_rms = rsqrtf(
                block_sum / static_cast<float>(hidden_size)
                + epsilon);
        }
    }

    __syncthreads();

    for (int column = threadIdx.x;
         column < hidden_size;
         column += blockDim.x) {
        const float residual_value =
            static_cast<float>(x[column])
            + static_cast<float>(residual[column]);

        const float output_value =
            residual_value
            * inverse_rms
            * static_cast<float>(weight[column]);

        residual[column] =
            static_cast<at::BFloat16>(residual_value);

        x[column] =
            static_cast<at::BFloat16>(output_value);
    }
}

template <int HiddenSize>
__global__ __launch_bounds__(kBlockSize)
void fused_add_rms_norm_bf16_cached_kernel(
    at::BFloat16* x,
    at::BFloat16* residual,
    const at::BFloat16* weight,
    float epsilon) {
    static_assert(
        HiddenSize % kBlockSize == 0,
        "hidden size must divide evenly across the block");

    constexpr int kElementsPerThread =
        HiddenSize / kBlockSize;

    const int row = blockIdx.x;
    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;
    const int num_warps = blockDim.x / kWarpSize;

    x += static_cast<int64_t>(row) * HiddenSize;
    residual += static_cast<int64_t>(row) * HiddenSize;

    float values[kElementsPerThread];
    float sum_of_squares = 0.0f;

#pragma unroll
    for (int item = 0;
         item < kElementsPerThread;
         ++item) {
        const int column =
            threadIdx.x + item * kBlockSize;

        const float value =
            static_cast<float>(x[column])
            + static_cast<float>(residual[column]);

        values[item] = value;
        sum_of_squares = fmaf(
            value,
            value,
            sum_of_squares);
    }

    sum_of_squares = warp_reduce_sum(sum_of_squares);

    __shared__ float warp_sums[kBlockSize / kWarpSize];
    __shared__ float inverse_rms;

    if (lane == 0) {
        warp_sums[warp] = sum_of_squares;
    }

    __syncthreads();

    if (warp == 0) {
        float block_sum =
            lane < num_warps
                ? warp_sums[lane]
                : 0.0f;

        block_sum = warp_reduce_sum(block_sum);

        if (lane == 0) {
            inverse_rms = rsqrtf(
                block_sum
                    / static_cast<float>(HiddenSize)
                + epsilon);
        }
    }

    __syncthreads();

#pragma unroll
    for (int item = 0;
         item < kElementsPerThread;
         ++item) {
        const int column =
            threadIdx.x + item * kBlockSize;
        const float residual_value = values[item];

        residual[column] =
            static_cast<at::BFloat16>(residual_value);

        x[column] = static_cast<at::BFloat16>(
            residual_value
            * inverse_rms
            * static_cast<float>(weight[column]));
    }
}

template <int HiddenSize>
__global__ __launch_bounds__(kBlockSize)
void fused_add_rms_norm_bf16_packed_cached_kernel(
    at::BFloat16* x,
    at::BFloat16* residual,
    const at::BFloat16* weight,
    float epsilon) {
    static_assert(HiddenSize % kVectorWidth == 0);

    constexpr int kVectorsPerRow = HiddenSize / kVectorWidth;
    constexpr int kSecondVectorCount = kVectorsPerRow - kBlockSize;

    static_assert(kVectorsPerRow > kBlockSize);
    static_assert(kVectorsPerRow <= 2 * kBlockSize);

    const int row = blockIdx.x;
    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;
    const int num_warps = blockDim.x / kWarpSize;

    auto* x_vectors = reinterpret_cast<BFloat16x8*>(
        x + static_cast<int64_t>(row) * HiddenSize);
    auto* residual_vectors = reinterpret_cast<BFloat16x8*>(
        residual + static_cast<int64_t>(row) * HiddenSize);
    const auto* weight_vectors =
        reinterpret_cast<const BFloat16x8*>(weight);

    const int first_vector = threadIdx.x;
    const int second_vector =
        threadIdx.x + kBlockSize;

    float values_0[kVectorWidth];
    float values_1[kVectorWidth];
    float sum_of_squares = 0.0f;

    // Every thread owns one packed vector.
    const BFloat16x8 x_0 = x_vectors[first_vector];
    const BFloat16x8 residual_0 =
        residual_vectors[first_vector];

    BFloat16x8 residual_output_0;

#pragma unroll
    for (int element = 0;
         element < kVectorWidth;
         ++element) {
        const float value =
            static_cast<float>(x_0.values[element])
            + static_cast<float>(
                residual_0.values[element]);

        values_0[element] = value;
        sum_of_squares = fmaf(
            value,
            value,
            sum_of_squares);

        residual_output_0.values[element] =
            static_cast<at::BFloat16>(value);
    }

    residual_vectors[first_vector] =
        residual_output_0;

    // Only 192 threads own a second vector:
    // 448 total vectors - 256 first vectors = 192.
    if (threadIdx.x < kSecondVectorCount) {
        const BFloat16x8 x_1 =
            x_vectors[second_vector];
        const BFloat16x8 residual_1 =
            residual_vectors[second_vector];

        BFloat16x8 residual_output_1;

#pragma unroll
        for (int element = 0;
             element < kVectorWidth;
             ++element) {
            const float value =
                static_cast<float>(
                    x_1.values[element])
                + static_cast<float>(
                    residual_1.values[element]);

            values_1[element] = value;
            sum_of_squares = fmaf(
                value,
                value,
                sum_of_squares);

            residual_output_1.values[element] =
                static_cast<at::BFloat16>(value);
        }

        residual_vectors[second_vector] =
            residual_output_1;
    }

    // First reduction level: one partial sum per warp.
    sum_of_squares =
        warp_reduce_sum(sum_of_squares);

    __shared__ float warp_sums[
        kBlockSize / kWarpSize];
    __shared__ float inverse_rms;

    if (lane == 0) {
        warp_sums[warp] = sum_of_squares;
    }

    __syncthreads();

    // Second reduction level: warp 0 reduces
    // the eight warp-level partial sums.
    if (warp == 0) {
        float block_sum =
            lane < num_warps
                ? warp_sums[lane]
                : 0.0f;

        block_sum = warp_reduce_sum(block_sum);

        if (lane == 0) {
            inverse_rms = rsqrtf(
                block_sum
                    / static_cast<float>(HiddenSize)
                + epsilon);
        }
    }

    __syncthreads();

    // Normalize the first packed vector using
    // the FP32 values retained in registers.
    const BFloat16x8 weight_0 =
        weight_vectors[first_vector];

    BFloat16x8 output_0;

#pragma unroll
    for (int element = 0;
         element < kVectorWidth;
         ++element) {
        output_0.values[element] =
            static_cast<at::BFloat16>(
                values_0[element]
                * inverse_rms
                * static_cast<float>(
                    weight_0.values[element]));
    }

    x_vectors[first_vector] = output_0;

    if (threadIdx.x < kSecondVectorCount) {
        const BFloat16x8 weight_1 =
            weight_vectors[second_vector];

        BFloat16x8 output_1;

#pragma unroll
        for (int element = 0;
             element < kVectorWidth;
             ++element) {
            output_1.values[element] =
                static_cast<at::BFloat16>(
                    values_1[element]
                    * inverse_rms
                    * static_cast<float>(
                        weight_1.values[element]));
        }

        x_vectors[second_vector] = output_1;
    }
}

}  // namespace


void fused_add_rms_norm_cuda(
    at::Tensor x,
    at::Tensor residual,
    const at::Tensor& weight,
    double epsilon) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(
        residual.is_cuda(),
        "residual must be a CUDA tensor");
    TORCH_CHECK(
        weight.is_cuda(),
        "weight must be a CUDA tensor");

    TORCH_CHECK(
        x.scalar_type() == at::kBFloat16,
        "x must use bfloat16");
    TORCH_CHECK(
        residual.scalar_type() == at::kBFloat16,
        "residual must use bfloat16");
    TORCH_CHECK(
        weight.scalar_type() == at::kBFloat16,
        "weight must use bfloat16");

    TORCH_CHECK(
        x.is_contiguous(),
        "x must be contiguous");
    TORCH_CHECK(
        residual.is_contiguous(),
        "residual must be contiguous");
    TORCH_CHECK(
        weight.is_contiguous(),
        "weight must be contiguous");

    TORCH_CHECK(
        x.sizes() == residual.sizes(),
        "x and residual must have identical shapes");
    TORCH_CHECK(
        x.dim() >= 1,
        "x must have at least one dimension");
    TORCH_CHECK(
        weight.dim() == 1,
        "weight must be one-dimensional");
    TORCH_CHECK(
        weight.size(0) == x.size(-1),
        "weight must match the final tensor dimension");
    TORCH_CHECK(
        epsilon > 0.0,
        "epsilon must be positive");

    const int64_t hidden_size = x.size(-1);
    const int64_t rows = x.numel() / hidden_size;

    TORCH_CHECK(
        hidden_size <= std::numeric_limits<int>::max(),
        "hidden dimension is too large");
    TORCH_CHECK(
        rows <= std::numeric_limits<int>::max(),
        "number of rows is too large");

    const c10::cuda::CUDAGuard device_guard(x.device());
    const cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(x.get_device());

auto* x_pointer = x.data_ptr<at::BFloat16>();
auto* residual_pointer = residual.data_ptr<at::BFloat16>();
const auto* weight_pointer = weight.data_ptr<at::BFloat16>();

const bool pointers_are_16_byte_aligned = reinterpret_cast<std::uintptr_t>(x_pointer) % 16 == 0 &&
    reinterpret_cast<std::uintptr_t>(residual_pointer) % 16 == 0 &&
    reinterpret_cast<std::uintptr_t>(weight_pointer) % 16 == 0;

if (hidden_size == 3584 && pointers_are_16_byte_aligned) {
    fused_add_rms_norm_bf16_packed_cached_kernel<3584><<<static_cast<int>(rows),kBlockSize,0,stream>>>(
        x_pointer,
        residual_pointer,
        weight_pointer,
        static_cast<float>(epsilon));
} else if (hidden_size == 3584) {
    fused_add_rms_norm_bf16_cached_kernel<3584><<<static_cast<int>(rows),kBlockSize,0,stream>>>(
        x_pointer,
        residual_pointer,
        weight_pointer,
        static_cast<float>(epsilon));
} else {
    fused_add_rms_norm_bf16_generic_kernel<<<static_cast<int>(rows),kBlockSize,0,stream>>>(
        x_pointer,
        residual_pointer,
        weight_pointer,
        static_cast<int>(hidden_size),
        static_cast<float>(epsilon));
}

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}