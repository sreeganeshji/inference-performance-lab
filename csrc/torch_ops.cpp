#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>

namespace ts = torch::stable;

void fused_add_rms_norm_cuda(ts::Tensor& x, ts::Tensor& residual, const ts::Tensor& weight, double epsilon);

void silu_and_mul_cuda(ts::Tensor& output, const ts::Tensor& input);

void silu_and_mul_packed_cuda(ts::Tensor& output, const ts::Tensor& input);

void paged_decode_attention_cuda(ts::Tensor& out, ts::Tensor& partial_out, ts::Tensor& partial_lse,
                                 const ts::Tensor& query, const ts::Tensor& key_cache, const ts::Tensor& value_cache,
                                 const ts::Tensor& block_table, const ts::Tensor& seq_lens, double scale,
                                 int64_t queries_per_cta, int64_t num_splits, bool tensor_cores);

int64_t paged_decode_attention_blocks_per_sm(const ts::Tensor& like, int64_t queries_per_cta, bool tensor_cores);

STABLE_TORCH_LIBRARY(inference_performance_lab, library) {
    library.def("fused_add_rms_norm(Tensor(a!) x, Tensor(b!) residual, Tensor weight, float epsilon) -> ()");
    library.def("silu_and_mul(Tensor(a!) output, Tensor input) -> ()");
    library.def("silu_and_mul_packed(Tensor(a!) output, Tensor input) -> ()");
    library.def("paged_decode_attention(Tensor(a!) out, Tensor(b!) partial_out, Tensor(c!) partial_lse, "
                "Tensor query, Tensor key_cache, Tensor value_cache, Tensor block_table, Tensor seq_lens, "
                "float scale, int queries_per_cta, int num_splits, bool tensor_cores) -> ()");
    library.def("paged_decode_attention_blocks_per_sm(Tensor like, int queries_per_cta, bool tensor_cores) -> int");
}

STABLE_TORCH_LIBRARY_IMPL(inference_performance_lab, CUDA, library) {
    library.impl("fused_add_rms_norm", TORCH_BOX(&fused_add_rms_norm_cuda));
    library.impl("silu_and_mul", TORCH_BOX(&silu_and_mul_cuda));
    library.impl("silu_and_mul_packed", TORCH_BOX(&silu_and_mul_packed_cuda));
    library.impl("paged_decode_attention", TORCH_BOX(&paged_decode_attention_cuda));
    library.impl("paged_decode_attention_blocks_per_sm", TORCH_BOX(&paged_decode_attention_blocks_per_sm));
}

// Prefill attention over packed variable-length sequences (csrc/flash_prefill_attention_kernel.cu).
void flash_prefill_attention_cuda(ts::Tensor& out, const ts::Tensor& query, const ts::Tensor& key,
                                  const ts::Tensor& value, const ts::Tensor& cu_seqlens, int64_t max_seqlen,
                                  double scale, bool causal, int64_t variant);

STABLE_TORCH_LIBRARY_FRAGMENT(inference_performance_lab, library) {
    library.def("flash_prefill_attention(Tensor(a!) out, Tensor query, Tensor key, Tensor value, "
                "Tensor cu_seqlens, int max_seqlen, float scale, bool causal, int variant) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(inference_performance_lab, CUDA, library) {
    library.impl("flash_prefill_attention", TORCH_BOX(&flash_prefill_attention_cuda));
}
