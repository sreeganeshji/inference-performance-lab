#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>

namespace ts = torch::stable;

void fused_add_rms_norm_cuda(ts::Tensor& x, ts::Tensor& residual, const ts::Tensor& weight, double epsilon);

void silu_and_mul_cuda(ts::Tensor& output, const ts::Tensor& input);

void silu_and_mul_packed_cuda(ts::Tensor& output, const ts::Tensor& input);

STABLE_TORCH_LIBRARY(inference_performance_lab, library) {
    library.def("fused_add_rms_norm(Tensor(a!) x, Tensor(b!) residual, Tensor weight, float epsilon) -> ()");
    library.def("silu_and_mul(Tensor(a!) output, Tensor input) -> ()");
    library.def("silu_and_mul_packed(Tensor(a!) output, Tensor input) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(inference_performance_lab, CUDA, library) {
    library.impl("fused_add_rms_norm", TORCH_BOX(&fused_add_rms_norm_cuda));
    library.impl("silu_and_mul", TORCH_BOX(&silu_and_mul_cuda));
    library.impl("silu_and_mul_packed", TORCH_BOX(&silu_and_mul_packed_cuda));
}
