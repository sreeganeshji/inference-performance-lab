#include <ATen/ATen.h>
#include <torch/library.h>


void fused_add_rms_norm_cuda(
    at::Tensor x,
    at::Tensor residual,
    const at::Tensor& weight,
    double epsilon);


TORCH_LIBRARY(inference_performance_lab, library) {
    library.def(
        "fused_add_rms_norm("
        "Tensor(a!) x, "
        "Tensor(b!) residual, "
        "Tensor weight, "
        "float epsilon"
        ") -> ()");
}


TORCH_LIBRARY_IMPL(inference_performance_lab, CUDA, library) {
    library.impl(
        "fused_add_rms_norm",
        &fused_add_rms_norm_cuda);
}