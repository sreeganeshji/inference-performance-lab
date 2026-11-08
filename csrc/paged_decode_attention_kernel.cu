#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/BFloat16.h>
#include <torch/headeronly/util/Exception.h>

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace ts = torch::stable;

namespace {

using BFloat16 = c10::BFloat16;
using ScalarType = torch::headeronly::ScalarType;

// Decode attention: one new query token per sequence attends to that sequence's paged KV cache,
// stored in vLLM's FLASH_ATTN layout [num_blocks, block_size, num_kv_heads, head_dim].
//
// Thread mapping: sixteen lanes (an aligned half-warp) share one K or V row, and each lane owns
// eight consecutive head dimensions loaded with a single 16-byte access. A 128-thread block
// therefore works on eight tokens per step.
constexpr int kHeadDim = 128;
constexpr int kVectorWidth = 8;
constexpr int kLanesPerRow = kHeadDim / kVectorWidth;
constexpr int kWarpSize = 32;
constexpr int kThreadsPerBlock = 128;
constexpr int kWarpsPerBlock = kThreadsPerBlock / kWarpSize;
constexpr int kRowsPerStep = kThreadsPerBlock / kLanesPerRow;
constexpr int kMaxQueriesPerCta = 8;
constexpr unsigned kFullMask = 0xffffffffu;
constexpr float kLog2e = 1.4426950408889634f;

static_assert(kLanesPerRow == 16, "row reductions assume an aligned half-warp per row");
static_assert(kThreadsPerBlock == kHeadDim, "the epilogue maps one thread to one output dimension");

struct DecodeParams {
    BFloat16* out;         // [num_seqs, num_q_heads, head_dim], token stride out_stride
    float* partial_out;    // [num_seqs, num_q_heads, num_splits, head_dim], used when num_splits > 1
    float* partial_lse;    // [num_seqs, num_q_heads, num_splits], log2 domain
    const BFloat16* query; // [num_seqs, num_q_heads, head_dim], token stride query_stride
    const BFloat16* key_cache;
    const BFloat16* value_cache;
    const int32_t* block_table; // [num_seqs, max_blocks_per_seq]
    const int32_t* seq_lens;    // [num_seqs]; zero marks a padded slot
    int64_t query_stride;
    int64_t out_stride;
    int64_t kv_block_stride;
    int64_t kv_token_stride;
    int64_t kv_head_stride;
    int64_t block_table_stride;
    int num_q_heads;
    int num_kv_heads;
    int block_size;
    int num_splits;
    float scale_log2; // softmax scale × log2(e), so the softmax can use exp2
};

void check_cuda(cudaError_t error, const char* operation) {
    STD_TORCH_CHECK(error == cudaSuccess, operation, ": ", cudaGetErrorString(error));
}

__device__ __forceinline__ void unpack_bf16x8(uint4 packed, float (&values)[kVectorWidth]) {
    const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};

#pragma unroll
    for (int word = 0; word < 4; ++word) {
        values[2 * word] = __uint_as_float(words[word] << 16);
        values[2 * word + 1] = __uint_as_float(words[word] & 0xffff0000u);
    }
}

// Sum across the 16 lanes of one row. XOR offsets below 16 never leave the aligned half-warp.
__device__ __forceinline__ float row_reduce_sum(float value) {
#pragma unroll
    for (int offset = kLanesPerRow / 2; offset > 0; offset /= 2) {
        value += __shfl_xor_sync(kFullMask, value, offset);
    }

    return value;
}

// Weight that moves a softmax state with running max m onto the larger max m_new. An empty
// state (m = -inf) gets weight zero, which also keeps -inf - -inf out of exp2.
__device__ __forceinline__ float rescale(float m, float m_new) { return m == -INFINITY ? 0.0f : exp2f(m - m_new); }

// Epilogue shared by both kernels. Each warp has left its softmax state (m, l) and unnormalized
// output rows in shared memory; thread d merges the warps for output dimension d and writes either
// the final BF16 output or, when the sequence is split, this split's normalized FP32 output and LSE.
template <int kQueriesPerCta, int kAccStride>
__device__ __forceinline__ void
store_block_result(const float (&shared_acc)[kWarpsPerBlock][kQueriesPerCta][kAccStride],
                   const float (&shared_m)[kWarpsPerBlock][kQueriesPerCta],
                   const float (&shared_l)[kWarpsPerBlock][kQueriesPerCta], const DecodeParams& params, int seq,
                   int split, int first_q_head) {
    const int dim = threadIdx.x;

#pragma unroll
    for (int g = 0; g < kQueriesPerCta; ++g) {
        float m_max = -INFINITY;

#pragma unroll
        for (int w = 0; w < kWarpsPerBlock; ++w) {
            m_max = fmaxf(m_max, shared_m[w][g]);
        }

        float sum = 0.0f;
        float output = 0.0f;

#pragma unroll
        for (int w = 0; w < kWarpsPerBlock; ++w) {
            const float weight = rescale(shared_m[w][g], m_max);
            sum += shared_l[w][g] * weight;
            output += shared_acc[w][g][dim] * weight;
        }

        const int q_head = first_q_head + g;

        if (params.num_splits == 1) {
            params.out[seq * params.out_stride + q_head * kHeadDim + dim] = BFloat16(sum > 0.0f ? output / sum : 0.0f);
        } else {
            const int64_t slot = (static_cast<int64_t>(seq) * params.num_q_heads + q_head) * params.num_splits + split;

            params.partial_out[slot * kHeadDim + dim] = sum > 0.0f ? output / sum : 0.0f;

            if (dim == 0) {
                params.partial_lse[slot] = sum > 0.0f ? m_max + log2f(sum) : -INFINITY;
            }
        }
    }
}

// One thread block per (query-head group, sequence, split). kQueriesPerCta = 1 gives every query
// head its own block, so a GQA model reads each K/V row once per query head. kQueriesPerCta =
// group size keeps all query heads that share a KV head in one block, so each row is read once.
template <int kQueriesPerCta>
__global__ void __launch_bounds__(kThreadsPerBlock) paged_decode_attention_kernel(const DecodeParams params) {
    const int seq = blockIdx.y;
    const int split = blockIdx.z;
    const int first_q_head = blockIdx.x * kQueriesPerCta;
    const int kv_head = first_q_head / (params.num_q_heads / params.num_kv_heads);

    const int row = threadIdx.x / kLanesPerRow;
    const int lane = threadIdx.x % kLanesPerRow;
    const int dim0 = lane * kVectorWidth;

    // The queries stay in registers for the whole kernel, pre-scaled for exp2.
    float q[kQueriesPerCta][kVectorWidth];
    const BFloat16* query = params.query + seq * params.query_stride + first_q_head * kHeadDim + dim0;

#pragma unroll
    for (int g = 0; g < kQueriesPerCta; ++g) {
        unpack_bf16x8(*reinterpret_cast<const uint4*>(query + g * kHeadDim), q[g]);

#pragma unroll
        for (int i = 0; i < kVectorWidth; ++i) {
            q[g][i] *= params.scale_log2;
        }
    }

    // This split's share of the sequence, in whole KV blocks.
    const int seq_len = params.seq_lens[seq];
    const int seq_blocks = (seq_len + params.block_size - 1) / params.block_size;
    const int blocks_per_split = (seq_blocks + params.num_splits - 1) / params.num_splits;
    const int token_begin = split * blocks_per_split * params.block_size;
    const int token_end = min(token_begin + blocks_per_split * params.block_size, seq_len);

    // Online-softmax state per query head: running max m (log2 domain), running sum l, and the
    // unnormalized output for this lane's eight dimensions.
    float m[kQueriesPerCta];
    float l[kQueriesPerCta];
    float acc[kQueriesPerCta][kVectorWidth];

#pragma unroll
    for (int g = 0; g < kQueriesPerCta; ++g) {
        m[g] = -INFINITY;
        l[g] = 0.0f;

#pragma unroll
        for (int i = 0; i < kVectorWidth; ++i) {
            acc[g][i] = 0.0f;
        }
    }

    const int32_t* block_table = params.block_table + seq * params.block_table_stride;
    const int64_t head_offset = kv_head * params.kv_head_stride + dim0;

    // The loop bound is uniform across the block, so every lane reaches the row shuffles; rows
    // past the end of the split load nothing and skip the update.
    for (int base = token_begin; base < token_end; base += kRowsPerStep) {
        const int token = base + row;
        const bool valid = token < token_end;

        uint4 k_packed = make_uint4(0, 0, 0, 0);
        uint4 v_packed = make_uint4(0, 0, 0, 0);

        if (valid) {
            const int64_t block = block_table[token / params.block_size];
            const int64_t offset =
                block * params.kv_block_stride + (token % params.block_size) * params.kv_token_stride + head_offset;

            k_packed = *reinterpret_cast<const uint4*>(params.key_cache + offset);
            v_packed = *reinterpret_cast<const uint4*>(params.value_cache + offset);
        }

        float k[kVectorWidth];
        unpack_bf16x8(k_packed, k);

        float score[kQueriesPerCta];

#pragma unroll
        for (int g = 0; g < kQueriesPerCta; ++g) {
            float partial = 0.0f;

#pragma unroll
            for (int i = 0; i < kVectorWidth; ++i) {
                partial = fmaf(q[g][i], k[i], partial);
            }

            score[g] = row_reduce_sum(partial);
        }

        if (valid) {
            float v[kVectorWidth];
            unpack_bf16x8(v_packed, v);

#pragma unroll
            for (int g = 0; g < kQueriesPerCta; ++g) {
                const float m_new = fmaxf(m[g], score[g]);
                const float alpha = exp2f(m[g] - m_new);
                const float p = exp2f(score[g] - m_new);

                l[g] = fmaf(l[g], alpha, p);

#pragma unroll
                for (int i = 0; i < kVectorWidth; ++i) {
                    acc[g][i] = fmaf(acc[g][i], alpha, p * v[i]);
                }

                m[g] = m_new;
            }
        }
    }

    // Merge the warp's two rows (lanes i and i + 16 own the same dimensions).
#pragma unroll
    for (int g = 0; g < kQueriesPerCta; ++g) {
        const float m_other = __shfl_xor_sync(kFullMask, m[g], kLanesPerRow);
        const float l_other = __shfl_xor_sync(kFullMask, l[g], kLanesPerRow);
        const float m_new = fmaxf(m[g], m_other);
        const float self_weight = rescale(m[g], m_new);
        const float other_weight = rescale(m_other, m_new);

        l[g] = l[g] * self_weight + l_other * other_weight;

#pragma unroll
        for (int i = 0; i < kVectorWidth; ++i) {
            const float acc_other = __shfl_xor_sync(kFullMask, acc[g][i], kLanesPerRow);
            acc[g][i] = acc[g][i] * self_weight + acc_other * other_weight;
        }

        m[g] = m_new;
    }

    // Merge the block's warps through shared memory.
    __shared__ float shared_acc[kWarpsPerBlock][kQueriesPerCta][kHeadDim];
    __shared__ float shared_m[kWarpsPerBlock][kQueriesPerCta];
    __shared__ float shared_l[kWarpsPerBlock][kQueriesPerCta];

    const int warp = threadIdx.x / kWarpSize;

    if (threadIdx.x % kWarpSize < kLanesPerRow) {
#pragma unroll
        for (int g = 0; g < kQueriesPerCta; ++g) {
#pragma unroll
            for (int i = 0; i < kVectorWidth; ++i) {
                shared_acc[warp][g][dim0 + i] = acc[g][i];
            }

            if (lane == 0) {
                shared_m[warp][g] = m[g];
                shared_l[warp][g] = l[g];
            }
        }
    }

    __syncthreads();

    store_block_result(shared_acc, shared_m, shared_l, params, seq, split, first_q_head);
}

// Flash-decoding's second pass: weight each split's normalized output by exp2(lse - max lse).
__global__ void __launch_bounds__(kHeadDim) combine_splits_kernel(const DecodeParams params) {
    const int q_head = blockIdx.x;
    const int seq = blockIdx.y;
    const int dim = threadIdx.x;
    const int64_t first_slot = (static_cast<int64_t>(seq) * params.num_q_heads + q_head) * params.num_splits;

    float lse_max = -INFINITY;

    for (int split = 0; split < params.num_splits; ++split) {
        lse_max = fmaxf(lse_max, params.partial_lse[first_slot + split]);
    }

    float sum = 0.0f;
    float output = 0.0f;

    for (int split = 0; split < params.num_splits; ++split) {
        const float weight = rescale(params.partial_lse[first_slot + split], lse_max);
        sum += weight;
        output += weight * params.partial_out[(first_slot + split) * kHeadDim + dim];
    }

    params.out[seq * params.out_stride + q_head * kHeadDim + dim] = BFloat16(sum > 0.0f ? output / sum : 0.0f);
}

// Tensor-core variant (sm_80 and later): mma.sync m16n8k16 with BF16 inputs and FP32 accumulation.
//
// Each warp owns whole 16-token chunks of the split and keeps its own softmax state; the block's
// four warps meet only in the shared epilogue. Per chunk:
//
//   S[16 x 16]    = Q[16 x 128] K^T   query heads on M (the group, zero-padded to 16), tokens on N
//   O^T[128 x 8] += V^T P^T           head dimensions on M, query heads on N (padded to 8)
//
// Computing O transposed keeps every accumulator register useful: with heads on N, the 8-wide
// dimension holds the GQA group instead of 16 rows that are more than half padding. S's
// accumulator registers already are P^T's B fragment, so the two MMAs chain without shuffles.
//
// K and V go straight from global memory into fragment registers with 16-byte loads, with no
// shared-memory staging. Two orderings are free to choose: the order in which S sums over head
// dimensions (it must only match between Q and K), and which physical dimension each row of O^T
// stands for. The kernel picks both so that every lane's fragment elements are contiguous.
constexpr int kChunkTokens = 16;
// Row stride of the epilogue's shared accumulator. The padding suits the earlier layout with heads
// on M; with O^T each lane stores dimensions 8*group + j, which Nsight Compute shows 8-way
// bank-conflicted at stride 128 and 132 alike (129 would give 2-way). The epilogue runs once per
// block, so the conflicts cost nothing measurable.
constexpr int kTcAccStride = kHeadDim + 4;

__device__ __forceinline__ void mma_m16n8k16_bf16(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                  uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
                 "{%8, %9}, {%0, %1, %2, %3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    __trap();
#endif
}

// Two floats to a BF16 pair; cvt puts its first source in the upper half.
__device__ __forceinline__ uint32_t pack_bf16x2(float low, float high) {
    uint32_t packed;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(packed) : "f"(high), "f"(low));
    return packed;
}

__device__ __forceinline__ uint32_t word(const uint4& packed, int index) {
    return index == 0 ? packed.x : index == 1 ? packed.y : index == 2 ? packed.z : packed.w;
}

template <int kQueriesPerCta>
__global__ void __launch_bounds__(kThreadsPerBlock) paged_decode_attention_tc_kernel(const DecodeParams params) {
    const int seq = blockIdx.y;
    const int split = blockIdx.z;
    const int first_q_head = blockIdx.x * kQueriesPerCta;
    const int kv_head = first_q_head / (params.num_q_heads / params.num_kv_heads);

    const int warp = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;
    const int group = lane / 4; // fragment row, or B-fragment column
    const int quad = lane % 4;  // lane within its group of four

    // Q's A fragments for S. Row `group` is query head first_q_head + group; rows past the GQA group
    // (and rows 8-15, which never exist) are zero. q_frag[i] holds dimensions 32i + 8*quad .. +8:
    // words 0 and 1 feed k-step 2i, words 2 and 3 feed k-step 2i + 1.
    uint4 q_frag[4];
    const BFloat16* query = params.query + seq * params.query_stride + (first_q_head + group) * kHeadDim + 8 * quad;

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        q_frag[i] = group < kQueriesPerCta ? *reinterpret_cast<const uint4*>(query + 32 * i) : make_uint4(0, 0, 0, 0);
    }

    const int seq_len = params.seq_lens[seq];
    const int seq_blocks = (seq_len + params.block_size - 1) / params.block_size;
    const int blocks_per_split = (seq_blocks + params.num_splits - 1) / params.num_splits;
    const int token_begin = split * blocks_per_split * params.block_size;
    const int token_end = min(token_begin + blocks_per_split * params.block_size, seq_len);

    // Softmax state for query head `group`: m is uniform across the quad; l is this lane's share
    // until the final quad reduction.
    float m = -INFINITY;
    float l = 0.0f;

    // O^T accumulators. M-tile j has rows (dimensions) 8*group + j and 64 + 8*group + j, and columns
    // (query heads) 2*quad and 2*quad + 1:
    //   o[j][0] = O[2*quad][8*group + j]        o[j][1] = O[2*quad + 1][8*group + j]
    //   o[j][2] = O[2*quad][64 + 8*group + j]   o[j][3] = O[2*quad + 1][64 + 8*group + j]
    float o[8][4] = {};

    const int32_t* block_table = params.block_table + seq * params.block_table_stride;
    const BFloat16* key_head = params.key_cache + kv_head * params.kv_head_stride;
    const BFloat16* value_head = params.value_cache + kv_head * params.kv_head_stride;
    const int end_chunk = (token_end + kChunkTokens - 1) / kChunkTokens;

    for (int chunk = token_begin / kChunkTokens + warp; chunk < end_chunk; chunk += kWarpsPerBlock) {
        const int chunk_token = chunk * kChunkTokens;
        const int64_t block = block_table[chunk_token / params.block_size];
        const int64_t row_offset =
            block * params.kv_block_stride + (chunk_token % params.block_size) * params.kv_token_stride;

        // K: this lane supplies tokens `group` and `group + 8`, the B-fragment columns of S's two
        // n-tiles, at the same dimensions its Q fragment holds. Tokens past the split load zeros.
        uint4 k_frag[2][4];

#pragma unroll
        for (int n = 0; n < 2; ++n) {
            const int token = group + 8 * n;
            const BFloat16* row = key_head + row_offset + token * params.kv_token_stride + 8 * quad;
            const bool valid = chunk_token + token < token_end;

#pragma unroll
            for (int i = 0; i < 4; ++i) {
                k_frag[n][i] = valid ? *reinterpret_cast<const uint4*>(row + 32 * i) : make_uint4(0, 0, 0, 0);
            }
        }

        // V: this lane supplies tokens 2*quad, 2*quad + 1, 2*quad + 8 and 2*quad + 9 (V^T's k
        // dimension) at dimensions 64h + 8*group .. +8 (its rows of O^T). Zeros past the split keep
        // stale cache entries, possibly NaN, out of the MMA.
        uint4 v_frag[4][2];

#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int token = 2 * quad + (r % 2) + 8 * (r / 2);
            const BFloat16* row = value_head + row_offset + token * params.kv_token_stride + 8 * group;
            const bool valid = chunk_token + token < token_end;

#pragma unroll
            for (int h = 0; h < 2; ++h) {
                v_frag[r][h] = valid ? *reinterpret_cast<const uint4*>(row + 64 * h) : make_uint4(0, 0, 0, 0);
            }
        }

        // S = Q K^T: two n-tiles of eight tokens, eight k-steps of sixteen dimensions each.
        float s[2][4] = {};

#pragma unroll
        for (int n = 0; n < 2; ++n) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
#pragma unroll
                for (int step = 0; step < 2; ++step) {
                    mma_m16n8k16_bf16(s[n], word(q_frag[i], 2 * step), 0, word(q_frag[i], 2 * step + 1), 0,
                                      word(k_frag[n][i], 2 * step), word(k_frag[n][i], 2 * step + 1));
                }
            }
        }

        // Head `group`'s four scores in this lane: tokens 2*quad and 2*quad + 1 of each n-tile.
        float x[4] = {s[0][0], s[0][1], s[1][0], s[1][1]};
        float chunk_max = -INFINITY;

#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int token = chunk_token + 8 * (e / 2) + 2 * quad + (e % 2);
            x[e] = token < token_end ? x[e] * params.scale_log2 : -INFINITY;
            chunk_max = fmaxf(chunk_max, x[e]);
        }

        chunk_max = fmaxf(chunk_max, __shfl_xor_sync(kFullMask, chunk_max, 1));
        chunk_max = fmaxf(chunk_max, __shfl_xor_sync(kFullMask, chunk_max, 2));

        // Every processed chunk holds at least one valid token, so m_new is finite.
        const float m_new = fmaxf(m, chunk_max);
        const float alpha = exp2f(m - m_new);
        float p[4];

#pragma unroll
        for (int e = 0; e < 4; ++e) {
            p[e] = exp2f(x[e] - m_new);
        }

        l = fmaf(l, alpha, (p[0] + p[1]) + (p[2] + p[3]));
        m = m_new;

        // This lane's O^T columns are heads 2*quad and 2*quad + 1, whose rescale factors live in the
        // lanes of groups 2*quad and 2*quad + 1.
        const float alpha_even = __shfl_sync(kFullMask, alpha, 8 * quad);
        const float alpha_odd = __shfl_sync(kFullMask, alpha, 8 * quad + 4);

#pragma unroll
        for (int j = 0; j < 8; ++j) {
            o[j][0] *= alpha_even;
            o[j][1] *= alpha_odd;
            o[j][2] *= alpha_even;
            o[j][3] *= alpha_odd;
        }

        // P^T's B fragment: column `group` (head), rows tokens 2*quad, 2*quad + 1 and +8, +9. These
        // are exactly the scores this lane computed.
        const uint32_t p_low = pack_bf16x2(p[0], p[1]);
        const uint32_t p_high = pack_bf16x2(p[2], p[3]);

        // O^T += V^T P^T. M-tile j's A fragment is element j of the lane's V vectors: half 0 for rows
        // 0-7 (dimension 8*group + j), half 1 for rows 8-15 (dimension 64 + 8*group + j).
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const uint32_t selector = j % 2 == 0 ? 0x5410 : 0x7632; // low or high BF16 halves
            const int w = j / 2;
            const uint32_t a0 = __byte_perm(word(v_frag[0][0], w), word(v_frag[1][0], w), selector);
            const uint32_t a1 = __byte_perm(word(v_frag[0][1], w), word(v_frag[1][1], w), selector);
            const uint32_t a2 = __byte_perm(word(v_frag[2][0], w), word(v_frag[3][0], w), selector);
            const uint32_t a3 = __byte_perm(word(v_frag[2][1], w), word(v_frag[3][1], w), selector);

            mma_m16n8k16_bf16(o[j], a0, a1, a2, a3, p_low, p_high);
        }
    }

    l += __shfl_xor_sync(kFullMask, l, 1);
    l += __shfl_xor_sync(kFullMask, l, 2);

    __shared__ float shared_acc[kWarpsPerBlock][kQueriesPerCta][kTcAccStride];
    __shared__ float shared_m[kWarpsPerBlock][kQueriesPerCta];
    __shared__ float shared_l[kWarpsPerBlock][kQueriesPerCta];

#pragma unroll
    for (int column = 0; column < 2; ++column) {
        const int head = 2 * quad + column;

        if (head < kQueriesPerCta) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                shared_acc[warp][head][8 * group + j] = o[j][column];
                shared_acc[warp][head][64 + 8 * group + j] = o[j][2 + column];
            }
        }
    }

    if (group < kQueriesPerCta && quad == 0) {
        shared_m[warp][group] = m;
        shared_l[warp][group] = l;
    }

    __syncthreads();

    store_block_result(shared_acc, shared_m, shared_l, params, seq, split, first_q_head);
}

template <int kQueriesPerCta>
void launch_decode(const DecodeParams& params, dim3 grid, bool tensor_cores, cudaStream_t stream) {
    if (tensor_cores) {
        paged_decode_attention_tc_kernel<kQueriesPerCta><<<grid, kThreadsPerBlock, 0, stream>>>(params);
    } else {
        paged_decode_attention_kernel<kQueriesPerCta><<<grid, kThreadsPerBlock, 0, stream>>>(params);
    }
}

template <int kQueriesPerCta> int64_t resident_blocks_per_sm(bool tensor_cores) {
    int blocks = 0;
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,
                                                             tensor_cores
                                                                 ? paged_decode_attention_tc_kernel<kQueriesPerCta>
                                                                 : paged_decode_attention_kernel<kQueriesPerCta>,
                                                             kThreadsPerBlock, 0),
               "occupancy query failed");
    return blocks;
}

bool is_16_byte_aligned(const void* pointer) { return reinterpret_cast<std::uintptr_t>(pointer) % 16 == 0; }

} // namespace

// How many decode thread blocks one SM of `like`'s device holds at once; the split heuristic sizes
// grids in whole waves.
int64_t paged_decode_attention_blocks_per_sm(const ts::Tensor& like, int64_t queries_per_cta, bool tensor_cores) {
    const ts::accelerator::DeviceGuard device_guard(like.get_device_index());

    switch (queries_per_cta) {
    case 1:
        return resident_blocks_per_sm<1>(tensor_cores);
    case 2:
        return resident_blocks_per_sm<2>(tensor_cores);
    case 3:
        return resident_blocks_per_sm<3>(tensor_cores);
    case 4:
        return resident_blocks_per_sm<4>(tensor_cores);
    case 5:
        return resident_blocks_per_sm<5>(tensor_cores);
    case 6:
        return resident_blocks_per_sm<6>(tensor_cores);
    case 7:
        return resident_blocks_per_sm<7>(tensor_cores);
    case 8:
        return resident_blocks_per_sm<8>(tensor_cores);
    default:
        STD_TORCH_CHECK(false, "unsupported queries_per_cta");
    }

    return 0;
}

void paged_decode_attention_cuda(ts::Tensor& out, ts::Tensor& partial_out, ts::Tensor& partial_lse,
                                 const ts::Tensor& query, const ts::Tensor& key_cache, const ts::Tensor& value_cache,
                                 const ts::Tensor& block_table, const ts::Tensor& seq_lens, double scale,
                                 int64_t queries_per_cta, int64_t num_splits, bool tensor_cores) {
    STD_TORCH_CHECK(out.is_cuda() && query.is_cuda() && key_cache.is_cuda() && value_cache.is_cuda() &&
                        block_table.is_cuda() && seq_lens.is_cuda(),
                    "all tensors must be CUDA tensors");
    STD_TORCH_CHECK(query.device() == out.device() && key_cache.device() == query.device() &&
                        value_cache.device() == query.device() && block_table.device() == query.device() &&
                        seq_lens.device() == query.device(),
                    "all tensors must be on the same device");
    STD_TORCH_CHECK(query.scalar_type() == ScalarType::BFloat16 && out.scalar_type() == ScalarType::BFloat16 &&
                        key_cache.scalar_type() == ScalarType::BFloat16 &&
                        value_cache.scalar_type() == ScalarType::BFloat16,
                    "query, out and the KV cache must use bfloat16");
    STD_TORCH_CHECK(block_table.scalar_type() == ScalarType::Int && seq_lens.scalar_type() == ScalarType::Int,
                    "block_table and seq_lens must be int32");

    STD_TORCH_CHECK(query.dim() == 3 && out.dim() == 3, "query and out must be [num_seqs, num_q_heads, head_dim]");
    STD_TORCH_CHECK(key_cache.dim() == 4 && value_cache.dim() == 4,
                    "the KV cache must be [num_blocks, block_size, num_kv_heads, head_dim]");

    const int64_t num_seqs = query.size(0);
    const int64_t num_q_heads = query.size(1);
    const int64_t block_size = key_cache.size(1);
    const int64_t num_kv_heads = key_cache.size(2);

    STD_TORCH_CHECK(query.size(2) == kHeadDim && key_cache.size(3) == kHeadDim, "head_dim must be 128");
    STD_TORCH_CHECK(out.size(0) == num_seqs && out.size(1) == num_q_heads && out.size(2) == kHeadDim,
                    "out must match query's shape");

    for (int64_t dimension = 0; dimension < 4; ++dimension) {
        STD_TORCH_CHECK(value_cache.size(dimension) == key_cache.size(dimension) &&
                            value_cache.stride(dimension) == key_cache.stride(dimension),
                        "the key and value caches must have the same shape and strides");
    }

    STD_TORCH_CHECK(query.stride(2) == 1 && query.stride(1) == kHeadDim && out.stride(2) == 1 &&
                        out.stride(1) == kHeadDim,
                    "query and out must be contiguous within a token");
    STD_TORCH_CHECK(key_cache.stride(3) == 1, "the KV cache must be contiguous along head_dim");
    STD_TORCH_CHECK(query.stride(0) % kVectorWidth == 0 && key_cache.stride(0) % kVectorWidth == 0 &&
                        key_cache.stride(1) % kVectorWidth == 0 && key_cache.stride(2) % kVectorWidth == 0,
                    "query and KV strides must be multiples of 8 elements");
    STD_TORCH_CHECK(is_16_byte_aligned(query.const_data_ptr<BFloat16>()) &&
                        is_16_byte_aligned(key_cache.const_data_ptr<BFloat16>()) &&
                        is_16_byte_aligned(value_cache.const_data_ptr<BFloat16>()),
                    "query and the KV cache must be 16-byte aligned");

    STD_TORCH_CHECK(num_kv_heads > 0 && num_q_heads % num_kv_heads == 0,
                    "num_q_heads must be a multiple of num_kv_heads");

    const int64_t group_size = num_q_heads / num_kv_heads;

    STD_TORCH_CHECK(queries_per_cta == 1 || queries_per_cta == group_size,
                    "queries_per_cta must be 1 or the GQA group size");
    STD_TORCH_CHECK(queries_per_cta <= kMaxQueriesPerCta, "GQA groups larger than 8 are not supported");

    if (tensor_cores) {
        int major = 0;
        check_cuda(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, query.get_device_index()),
                   "failed to read the compute capability");
        STD_TORCH_CHECK(major >= 8, "the tensor-core kernel needs sm_80 or later");
        STD_TORCH_CHECK(block_size % kChunkTokens == 0, "the tensor-core kernel needs block_size divisible by 16");
        STD_TORCH_CHECK(queries_per_cta == group_size, "the tensor-core kernel keeps a whole GQA group per block");
    }

    STD_TORCH_CHECK(block_table.dim() == 2 && block_table.size(0) == num_seqs && block_table.stride(1) == 1,
                    "block_table must be [num_seqs, max_blocks_per_seq] with contiguous rows");
    STD_TORCH_CHECK(seq_lens.dim() == 1 && seq_lens.size(0) == num_seqs && seq_lens.is_contiguous(),
                    "seq_lens must be a contiguous [num_seqs] tensor");

    STD_TORCH_CHECK(num_splits >= 1 && num_splits <= 65535, "num_splits must be in [1, 65535]");
    STD_TORCH_CHECK(num_seqs <= 65535, "at most 65535 sequences per launch");

    if (num_splits > 1) {
        STD_TORCH_CHECK(partial_out.is_cuda() && partial_lse.is_cuda() && partial_out.device() == query.device() &&
                            partial_lse.device() == query.device(),
                        "the split workspace must be on the query's device");
        STD_TORCH_CHECK(partial_out.scalar_type() == ScalarType::Float &&
                            partial_lse.scalar_type() == ScalarType::Float,
                        "the split workspace must be float32");
        STD_TORCH_CHECK(partial_out.is_contiguous() && partial_lse.is_contiguous(),
                        "the split workspace must be contiguous");
        STD_TORCH_CHECK(partial_out.dim() == 4 && partial_out.size(0) == num_seqs &&
                            partial_out.size(1) == num_q_heads && partial_out.size(2) == num_splits &&
                            partial_out.size(3) == kHeadDim,
                        "partial_out must be [num_seqs, num_q_heads, num_splits, head_dim]");
        STD_TORCH_CHECK(partial_lse.dim() == 3 && partial_lse.size(0) == num_seqs &&
                            partial_lse.size(1) == num_q_heads && partial_lse.size(2) == num_splits,
                        "partial_lse must be [num_seqs, num_q_heads, num_splits]");
    }

    if (num_seqs == 0) {
        return;
    }

    DecodeParams params{};
    params.out = out.mutable_data_ptr<BFloat16>();
    params.partial_out = num_splits > 1 ? partial_out.mutable_data_ptr<float>() : nullptr;
    params.partial_lse = num_splits > 1 ? partial_lse.mutable_data_ptr<float>() : nullptr;
    params.query = query.const_data_ptr<BFloat16>();
    params.key_cache = key_cache.const_data_ptr<BFloat16>();
    params.value_cache = value_cache.const_data_ptr<BFloat16>();
    params.block_table = block_table.const_data_ptr<int32_t>();
    params.seq_lens = seq_lens.const_data_ptr<int32_t>();
    params.query_stride = query.stride(0);
    params.out_stride = out.stride(0);
    params.kv_block_stride = key_cache.stride(0);
    params.kv_token_stride = key_cache.stride(1);
    params.kv_head_stride = key_cache.stride(2);
    params.block_table_stride = block_table.stride(0);
    params.num_q_heads = static_cast<int>(num_q_heads);
    params.num_kv_heads = static_cast<int>(num_kv_heads);
    params.block_size = static_cast<int>(block_size);
    params.num_splits = static_cast<int>(num_splits);
    params.scale_log2 = static_cast<float>(scale) * kLog2e;

    const ts::accelerator::DeviceGuard device_guard(query.get_device_index());
    const cudaStream_t stream =
        static_cast<cudaStream_t>(ts::accelerator::getCurrentStream(query.get_device_index()).nativeHandle());

    const dim3 grid(static_cast<unsigned>(num_q_heads / queries_per_cta), static_cast<unsigned>(num_seqs),
                    static_cast<unsigned>(num_splits));

    switch (queries_per_cta) {
    case 1:
        launch_decode<1>(params, grid, tensor_cores, stream);
        break;
    case 2:
        launch_decode<2>(params, grid, tensor_cores, stream);
        break;
    case 3:
        launch_decode<3>(params, grid, tensor_cores, stream);
        break;
    case 4:
        launch_decode<4>(params, grid, tensor_cores, stream);
        break;
    case 5:
        launch_decode<5>(params, grid, tensor_cores, stream);
        break;
    case 6:
        launch_decode<6>(params, grid, tensor_cores, stream);
        break;
    case 7:
        launch_decode<7>(params, grid, tensor_cores, stream);
        break;
    case 8:
        launch_decode<8>(params, grid, tensor_cores, stream);
        break;
    default:
        STD_TORCH_CHECK(false, "unsupported queries_per_cta");
    }

    check_cuda(cudaGetLastError(), "paged decode attention kernel launch failed");

    if (num_splits > 1) {
        const dim3 combine_grid(static_cast<unsigned>(num_q_heads), static_cast<unsigned>(num_seqs));
        combine_splits_kernel<<<combine_grid, kHeadDim, 0, stream>>>(params);

        check_cuda(cudaGetLastError(), "split combine kernel launch failed");
    }
}
