#include <ATen/ATen.h>
#include <torch/library.h>


void fused_add_rms_norm_cuda(
    at::Tensor x,
    at::Tensor residual,
    const at::Tensor& weight,
    double epsilon);

void silu_and_mul_cuda(
    at::Tensor output,
    const at::Tensor& input);

void silu_and_mul_packed_cuda(
    at::Tensor output,
    const at::Tensor& input);

TORCH_LIBRARY(inference_performance_lab, library) {
    library.def("fused_add_rms_norm(""Tensor(a!) x, ""Tensor(b!) residual, ""Tensor weight, ""float epsilon"") -> ()");
    library.def("silu_and_mul(""Tensor(a!) output, ""Tensor input"") -> ()");
    library.def("silu_and_mul_packed(Tensor(a!) output, Tensor input) -> ()");
}


TORCH_LIBRARY_IMPL(inference_performance_lab, CUDA, library) {
    library.impl("fused_add_rms_norm",&fused_add_rms_norm_cuda);
    library.impl("silu_and_mul",&silu_and_mul_cuda);
    library.impl("silu_and_mul_packed", &silu_and_mul_packed_cuda);
}