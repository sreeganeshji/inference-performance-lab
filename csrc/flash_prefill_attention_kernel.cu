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

// Prefill attention over packed variable-length sequences (the FlashAttention-2 forward pass).
//
// Q is [total_tokens, num_q_heads, 128] and K, V are [total_tokens, num_kv_heads, 128]; sequence s
// owns tokens cu_seqlens[s] .. cu_seqlens[s + 1] of all three, so every query row attends to keys
// of its own sequence only (a fresh prompt with no prefix cache). One thread block computes one
// kBlockM-row query tile of one query head of one sequence. It streams the sequence's K and V rows
// through shared memory kBlockN rows at a time and keeps the online-softmax state in registers.
// Its four warps split the tile's rows, sixteen each, and never exchange data.
//
// Under the causal mask a query tile visits only the K/V tiles at or left of the diagonal: tiles
// wholly right of it are skipped, and only the diagonal tile (and the sequence's ragged last tile)
// is masked element by element. GQA needs no data movement: query head h reads KV head h / group.
constexpr int kHeadDim = 128;
constexpr int kBlockM = 64;
constexpr int kBlockN = 64;
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 4;
constexpr int kThreadsPerBlock = kWarpsPerBlock * kWarpSize;
constexpr int kRowsPerWarp = kBlockM / kWarpsPerBlock;
constexpr int kChunksPerRow = kHeadDim / 8;                    // 16-byte chunks in one 256-byte row
constexpr int kRowsPerPass = kThreadsPerBlock / kChunksPerRow; // rows one block-wide load covers
constexpr int kWordsPerRow = kHeadDim / 2;                     // 32-bit words (BF16 pairs) per row
constexpr unsigned kFullMask = 0xffffffffu;
constexpr float kLog2e = 1.4426950408889634f;

static_assert(kBlockM == kBlockN, "causal tile skipping assumes one diagonal tile per query tile");
static_assert(kRowsPerWarp == 16, "each warp owns one 16-row MMA tile");

struct PrefillParams {
    BFloat16* out;             // [total_tokens, num_q_heads, 128]
    const BFloat16* query;     // [total_tokens, num_q_heads, 128]
    const BFloat16* key;       // [total_tokens, num_kv_heads, 128]
    const BFloat16* value;     // [total_tokens, num_kv_heads, 128]
    const int32_t* cu_seqlens; // [num_seqs + 1]
    int64_t q_token_stride;
    int64_t q_head_stride;
    int64_t k_token_stride;
    int64_t k_head_stride;
    int64_t v_token_stride;
    int64_t v_head_stride;
    int64_t out_token_stride;
    int64_t out_head_stride;
    int num_q_heads;
    int num_kv_heads;
    bool causal;
    float scale_log2; // softmax scale × log2(e), so the softmax can use exp2
};

// The kernel's development steps; each variant adds one change to the one before it, so variant n
// is steps 0..n together (docs/flash-prefill.md measures every step).
enum class PrefillVariant : int64_t {
    kSimt,       // 0: tiled, CUDA cores, FP32
    kMma,        // 1: tensor cores (mma.sync, ldmatrix), plain 256-byte shared-memory rows
    kSwizzle,    // 2: XOR-swizzled shared memory, conflict-free ldmatrix
    kCpAsync,    // 3: cp.async, K and V loads overlapped with the two GEMMs
    kFastExp2,   // 4a: exp2 as a bare MUFU.EX2
    kHoisted,    // 4b: per-thread copy and ldmatrix addresses computed once
    kHeavyFirst, // 5: longest query tiles launch first
    kSkipMasked, // 6: warps skip 16-column blocks of the diagonal tile that are fully masked
    kCount,
};

void check_cuda(cudaError_t error, const char* operation) {
    STD_TORCH_CHECK(error == cudaSuccess, operation, ": ", cudaGetErrorString(error));
}

// The block's share of the problem. Grid: x = query tile, y = query head, z = sequence; or, for
// heavy-first scheduling (step 5), x = query head, y = sequence, z = query tile counted back from
// the sequence's last. Blocks launch in x-fastest order, so heavy-first puts every head's and every
// sequence's most expensive causal tile (it visits the most K/V tiles) in the first wave and the
// one-tile blocks last, where they fill the tail of the final wave.
struct BlockTile {
    const BFloat16* query; // row 0 of the sequence, at this block's query head
    const BFloat16* key;   // row 0 of the sequence, at the matching KV head
    const BFloat16* value;
    BFloat16* out;
    int seq_len;
    int q_begin;      // first query row of the tile, relative to the sequence
    int num_kv_tiles; // K/V tiles this query tile visits; the rest are fully masked
};

template <bool kHeavyFirst> __device__ __forceinline__ bool block_tile(const PrefillParams& p, BlockTile& tile) {
    const int seq = kHeavyFirst ? blockIdx.y : blockIdx.z;
    const int q_head = kHeavyFirst ? blockIdx.x : blockIdx.y;
    const int64_t seq_begin = p.cu_seqlens[seq];

    tile.seq_len = p.cu_seqlens[seq + 1] - static_cast<int>(seq_begin);

    // The grid is sized for the longest sequence; shorter ones leave their surplus blocks idle.
    if constexpr (kHeavyFirst) {
        const int q_tile = (tile.seq_len + kBlockM - 1) / kBlockM - 1 - static_cast<int>(blockIdx.z);

        if (q_tile < 0) {
            return false;
        }

        tile.q_begin = q_tile * kBlockM;
    } else {
        tile.q_begin = blockIdx.x * kBlockM;

        if (tile.q_begin >= tile.seq_len) {
            return false;
        }
    }

    const int kv_head = q_head / (p.num_q_heads / p.num_kv_heads);
    const int kv_end = p.causal ? min(tile.seq_len, tile.q_begin + kBlockM) : tile.seq_len;

    tile.query = p.query + seq_begin * p.q_token_stride + q_head * p.q_head_stride;
    tile.key = p.key + seq_begin * p.k_token_stride + kv_head * p.k_head_stride;
    tile.value = p.value + seq_begin * p.v_token_stride + kv_head * p.v_head_stride;
    tile.out = p.out + seq_begin * p.out_token_stride + q_head * p.out_head_stride;
    tile.num_kv_tiles = (kv_end + kBlockN - 1) / kBlockN;

    return true;
}

// Whether a K/V tile needs the element mask: it crosses the diagonal or the end of the sequence.
__device__ __forceinline__ bool tile_needs_mask(const PrefillParams& p, const BlockTile& tile, int k_begin) {
    return (p.causal && k_begin + kBlockN - 1 > tile.q_begin) || k_begin + kBlockN > tile.seq_len;
}

__device__ __forceinline__ bool masked_out(const PrefillParams& p, int seq_len, int row, int column) {
    return column >= seq_len || (p.causal && column > row);
}

// Copies a 64 x 128 tile of rows first_row.. into shared memory. Sixteen consecutive threads cover
// one 256-byte row, so each warp's loads coalesce into two full rows. Rows at or past valid_rows
// store zeros: they belong to the next sequence or lie past the end of the tensor.
template <typename Store>
__device__ __forceinline__ void load_tile(const BFloat16* rows, int64_t token_stride, int first_row, int valid_rows,
                                          Store store) {
    const int chunk = threadIdx.x % kChunksPerRow;

#pragma unroll
    for (int pass = 0; pass < kBlockN / kRowsPerPass; ++pass) {
        const int row = pass * kRowsPerPass + threadIdx.x / kChunksPerRow;
        uint4 data = make_uint4(0, 0, 0, 0);

        if (first_row + row < valid_rows) {
            data = *reinterpret_cast<const uint4*>(rows + (first_row + row) * token_stride + chunk * 8);
        }

        store(row, chunk, data);
    }
}

__device__ __forceinline__ float2 unpack_bf16x2(uint32_t packed) {
    return make_float2(__uint_as_float(packed << 16), __uint_as_float(packed & 0xffff0000u));
}

// Two floats to a BF16 pair; cvt puts its first source in the upper half.
__device__ __forceinline__ uint32_t pack_bf16x2(float low, float high) {
    uint32_t packed;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(packed) : "f"(high), "f"(low));
    return packed;
}

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        value = fmaxf(value, __shfl_xor_sync(kFullMask, value, offset));
    }

    return value;
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        value += __shfl_xor_sync(kFullMask, value, offset);
    }

    return value;
}

// A running max that is still -inf (no unmasked score seen yet) is replaced by 0 before it is
// subtracted, so exp2(-inf - -inf) never produces NaN: the row's probabilities and its rescale
// factor then both come out as exp2(-inf) = 0.
__device__ __forceinline__ float finite_max(float m) { return m == -INFINITY ? 0.0f : m; }

// ---------------------------------------------------------------------------------------------
// Step 0: the smallest correct tiled kernel, on CUDA cores in FP32.
//
// Shared memory, in 32-bit words (BF16 pairs):
//   sQ [64][64]       the query tile; every lane of a warp reads the same word (a broadcast)
//   sK [64][65]       one K/V tile of keys, rows padded by a word so lane c's reads of column w
//                     land in bank (c + w) % 32
//   sV [64][64]       values; lane i reads words i and 32 + i of a row, so a warp covers 32 banks
//   sP [64][64]       FP32 probabilities, each warp's sixteen rows written and read by that warp
//
// S = Q K^T: lane i owns key columns i and i + 32 of its warp's sixteen rows (32 scores).
// O += P V: lane i owns output dimensions 2i, 2i + 1, 64 + 2i and 65 + 2i of the same rows.
// ---------------------------------------------------------------------------------------------
constexpr int kSimtKStride = kWordsPerRow + 1;
constexpr size_t kSimtSmemBytes =
    sizeof(uint32_t) * (kBlockM * kWordsPerRow + kBlockN * kSimtKStride + kBlockN * kWordsPerRow) +
    sizeof(float) * kBlockM * kBlockN;

__global__ void __launch_bounds__(kThreadsPerBlock) flash_prefill_simt_kernel(const PrefillParams p) {
    BlockTile tile;

    if (!block_tile<false>(p, tile)) {
        return;
    }

    extern __shared__ uint4 shared_storage[];
    uint32_t* sQ = reinterpret_cast<uint32_t*>(shared_storage);
    uint32_t* sK = sQ + kBlockM * kWordsPerRow;
    uint32_t* sV = sK + kBlockN * kSimtKStride;
    float* sP = reinterpret_cast<float*>(sV + kBlockN * kWordsPerRow);

    const int warp = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;
    const int warp_row = warp * kRowsPerWarp;      // first tile row this warp owns
    const int first_row = tile.q_begin + warp_row; // the same row, relative to the sequence

    load_tile(tile.query, p.q_token_stride, tile.q_begin, tile.seq_len, [&](int row, int chunk, uint4 data) {
        *reinterpret_cast<uint4*>(sQ + row * kWordsPerRow + chunk * 4) = data;
    });

    float m[kRowsPerWarp]; // running max per row, log2 domain; the same in every lane
    float l[kRowsPerWarp]; // this lane's share of the running sum, reduced across the warp at the end
    float o[kRowsPerWarp][4] = {};

#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
        m[r] = -INFINITY;
        l[r] = 0.0f;
    }

    for (int kv_tile = 0; kv_tile < tile.num_kv_tiles; ++kv_tile) {
        const int k_begin = kv_tile * kBlockN;

        __syncthreads(); // every warp is done reading the previous tile (and sQ is complete)

        load_tile(tile.key, p.k_token_stride, k_begin, tile.seq_len, [&](int row, int chunk, uint4 data) {
            uint32_t* dst = sK + row * kSimtKStride + chunk * 4; // padded rows are only 4-byte aligned
            dst[0] = data.x;
            dst[1] = data.y;
            dst[2] = data.z;
            dst[3] = data.w;
        });
        load_tile(tile.value, p.v_token_stride, k_begin, tile.seq_len, [&](int row, int chunk, uint4 data) {
            *reinterpret_cast<uint4*>(sV + row * kWordsPerRow + chunk * 4) = data;
        });

        __syncthreads();

        float s[kRowsPerWarp][2] = {};

#pragma unroll 4
        for (int w = 0; w < kWordsPerRow; ++w) {
            const float2 k0 = unpack_bf16x2(sK[lane * kSimtKStride + w]);
            const float2 k1 = unpack_bf16x2(sK[(lane + 32) * kSimtKStride + w]);

#pragma unroll
            for (int r = 0; r < kRowsPerWarp; ++r) {
                const float2 q = unpack_bf16x2(sQ[(warp_row + r) * kWordsPerRow + w]);
                s[r][0] = fmaf(q.x, k0.x, fmaf(q.y, k0.y, s[r][0]));
                s[r][1] = fmaf(q.x, k1.x, fmaf(q.y, k1.y, s[r][1]));
            }
        }

        const bool needs_mask = tile_needs_mask(p, tile, k_begin);

#pragma unroll
        for (int r = 0; r < kRowsPerWarp; ++r) {
            float x0 = s[r][0] * p.scale_log2;
            float x1 = s[r][1] * p.scale_log2;

            if (needs_mask) {
                x0 = masked_out(p, tile.seq_len, first_row + r, k_begin + lane) ? -INFINITY : x0;
                x1 = masked_out(p, tile.seq_len, first_row + r, k_begin + lane + 32) ? -INFINITY : x1;
            }

            const float m_new = fmaxf(m[r], warp_max(fmaxf(x0, x1)));
            const float m_safe = finite_max(m_new);
            const float alpha = exp2f(m[r] - m_safe);
            const float p0 = exp2f(x0 - m_safe);
            const float p1 = exp2f(x1 - m_safe);

            l[r] = fmaf(l[r], alpha, p0 + p1);
            m[r] = m_new;

#pragma unroll
            for (int i = 0; i < 4; ++i) {
                o[r][i] *= alpha;
            }

            sP[(warp_row + r) * kBlockN + lane] = p0;
            sP[(warp_row + r) * kBlockN + lane + 32] = p1;
        }

        __syncwarp();

#pragma unroll 2
        for (int c = 0; c < kBlockN; c += 4) {
            float2 v_low[4];
            float2 v_high[4];

#pragma unroll
            for (int i = 0; i < 4; ++i) {
                v_low[i] = unpack_bf16x2(sV[(c + i) * kWordsPerRow + lane]);
                v_high[i] = unpack_bf16x2(sV[(c + i) * kWordsPerRow + 32 + lane]);
            }

#pragma unroll
            for (int r = 0; r < kRowsPerWarp; ++r) {
                const float4 prob = *reinterpret_cast<const float4*>(sP + (warp_row + r) * kBlockN + c);
                const float pr[4] = {prob.x, prob.y, prob.z, prob.w};

#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    o[r][0] = fmaf(pr[i], v_low[i].x, o[r][0]);
                    o[r][1] = fmaf(pr[i], v_low[i].y, o[r][1]);
                    o[r][2] = fmaf(pr[i], v_high[i].x, o[r][2]);
                    o[r][3] = fmaf(pr[i], v_high[i].y, o[r][3]);
                }
            }
        }
    }

#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
        const float sum = warp_sum(l[r]);
        const float inverse = sum > 0.0f ? 1.0f / sum : 0.0f;
        const int row = first_row + r;

        if (row < tile.seq_len) {
            uint32_t* out_row = reinterpret_cast<uint32_t*>(tile.out + row * p.out_token_stride);
            out_row[lane] = pack_bf16x2(o[r][0] * inverse, o[r][1] * inverse);
            out_row[32 + lane] = pack_bf16x2(o[r][2] * inverse, o[r][3] * inverse);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Steps 1-3: tensor cores (sm_80 and later), mma.sync m16n8k16 with BF16 inputs and FP32 accumulation.
//
// Same tiles and the same warp-to-row split as step 0. Each warp computes, per K/V tile,
//
//   S[16 x 64]   = Q[16 x 128] K^T     8 k-steps x 8 n-tiles = 64 MMAs
//   O[16 x 128] += P[16 x 64] V        4 k-steps x 16 n-tiles = 64 MMAs
//
// Fragments come from shared memory through ldmatrix: Q's A fragments once (then they stay in 32
// registers), K's B fragments with ldmatrix (K rows are already "column-major" B for Q K^T), and
// V's with ldmatrix.trans (V rows must be transposed to serve as B for P V). S's accumulator
// registers are exactly P's A-fragment layout once packed to BF16, so P never leaves registers.
//
// Shared memory: three 64-row x 256-byte tiles, sQ, sK, sV (48 KB), so two blocks fit in an SM's
// 100 KB. sQ is reused at the end to stage each warp's output rows for coalesced 16-byte stores.
//
// kSwizzle (step 2) stores 16-byte chunk c of row r at chunk position c ^ (r % 8). Every ldmatrix
// phase reads one chunk column from eight consecutive rows; with 256-byte rows those eight chunks
// share the same four banks (an 8-way conflict) unless the XOR spreads them over all 32.
//
// kPipelined (step 3) replaces register-staged loads with cp.async and overlaps them with math,
// as FlashAttention-2 does with one buffer per operand: V(j) streams in during S = Q K(j)^T, and
// K(j + 1) during the softmax and O += P V(j). Two block barriers per tile order the buffers.
// ---------------------------------------------------------------------------------------------
constexpr int kTileBytes = kBlockN * kHeadDim * sizeof(BFloat16);
constexpr size_t kMmaSmemBytes = 3 * kTileBytes;

__device__ __forceinline__ void mma_m16n8k16_bf16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
                 "{%8, %9}, {%0, %1, %2, %3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    __trap();
#endif
}

// Four 8x8 BF16 matrices; lanes 8i .. 8i + 7 supply the row addresses of matrix i, and each lane
// receives two elements of every matrix (row lane / 4, columns 2 * (lane % 4) and the next).
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], uint32_t address) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(address));
}

// The same, transposed: each lane receives rows 2 * (lane % 4) and the next, of column lane / 4.
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4], uint32_t address) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(address));
}

// 16-byte global-to-shared copy that bypasses L1. A zero source size fills the destination with
// zeros, which is how rows past the end of a sequence are padded.
__device__ __forceinline__ void cp_async_16(uint32_t address, const void* source, bool valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(address), "l"(source), "r"(valid ? 16 : 0));
}

__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }

__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_group 0;\n" ::: "memory"); }

// Byte offset of 16-byte chunk `chunk` (dimensions 8 * chunk .. + 8) of tile row `row`.
template <bool kSwizzle> __device__ __forceinline__ uint32_t tile_offset(int row, int chunk) {
    return row * (kHeadDim * sizeof(BFloat16)) + (kSwizzle ? chunk ^ (row % 8) : chunk) * 16;
}

template <bool kSwizzle>
__device__ __forceinline__ void copy_tile_sync(uint32_t tile, const BFloat16* rows, int64_t token_stride, int first_row,
                                               int valid_rows) {
    load_tile(rows, token_stride, first_row, valid_rows, [&](int row, int chunk, uint4 data) {
        asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};\n" ::"r"(tile + tile_offset<kSwizzle>(row, chunk)),
                     "r"(data.x), "r"(data.y), "r"(data.z), "r"(data.w));
    });
}

template <bool kSwizzle>
__device__ __forceinline__ void copy_tile_async(uint32_t tile, const BFloat16* rows, int64_t token_stride,
                                                int first_row, int valid_rows) {
    const int chunk = threadIdx.x % kChunksPerRow;

#pragma unroll
    for (int pass = 0; pass < kBlockN / kRowsPerPass; ++pass) {
        const int row = pass * kRowsPerPass + threadIdx.x / kChunksPerRow;
        const bool valid = first_row + row < valid_rows;
        const BFloat16* source = valid ? rows + (first_row + row) * token_stride + chunk * 8 : rows;

        cp_async_16(tile + tile_offset<kSwizzle>(row, chunk), source, valid);
    }
}

// Step 4b: the same copy for a tile whose 64 rows all exist, with this thread's addresses computed
// once per kernel. Thread t copies chunk t % 16 of rows r, r + 8, ..., r + 56 (r = t / 16 < 8). The
// swizzle depends only on row % 8 = r, so the eight destinations are one base plus immediates and
// the sources one pointer advanced by eight rows per pass; no row is predicated.
struct TileCopy {
    uint32_t smem_offset; // destination of the first pass, within a tile
    int64_t row_offset;   // first pass's source, in elements from the tile's first row
    int64_t pass_stride;  // eight rows, in elements
};

template <bool kSwizzle> __device__ __forceinline__ TileCopy make_tile_copy(int64_t token_stride) {
    const int row = threadIdx.x / kChunksPerRow;
    const int chunk = threadIdx.x % kChunksPerRow;
    return {tile_offset<kSwizzle>(row, chunk), row * token_stride + chunk * 8, kRowsPerPass * token_stride};
}

__device__ __forceinline__ void cp_async_16_unpredicated(uint32_t address, const void* source) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(address), "l"(source));
}

__device__ __forceinline__ void copy_full_tile_async(uint32_t tile, const BFloat16* tile_rows, const TileCopy& copy) {
    const BFloat16* source = tile_rows + copy.row_offset;

#pragma unroll
    for (int pass = 0; pass < kBlockN / kRowsPerPass; ++pass) {
        cp_async_16_unpredicated(tile + copy.smem_offset + pass * kRowsPerPass * kHeadDim * sizeof(BFloat16), source);
        source += copy.pass_stride;
    }
}

// Copies K/V tile `k_begin` with the step-4b addressing when every row exists, else row by row.
template <bool kSwizzle, bool kHoisted>
__device__ __forceinline__ void copy_kv_tile_async(uint32_t tile, const BFloat16* rows, int64_t token_stride,
                                                   const TileCopy& copy, int k_begin, int seq_len) {
    if (kHoisted && k_begin + kBlockN <= seq_len) {
        copy_full_tile_async(tile, rows + k_begin * token_stride, copy);
    } else {
        copy_tile_async<kSwizzle>(tile, rows, token_stride, k_begin, seq_len);
    }
}

// Step 4b: each lane's ldmatrix offsets within a tile, computed once per kernel. A 16-row block
// starts at a multiple of 8 rows, so the swizzle of row 16b + x equals that of row x: K needs one
// offset per k-step (k[step]) and V one per pair of n-tiles (v[pair]), and the 16-row block index
// becomes an immediate (b * 4096 bytes) in the ldmatrix address.
struct FragmentOffsets {
    uint32_t k[8];
    uint32_t v[8];
};

template <bool kSwizzle> __device__ __forceinline__ FragmentOffsets make_fragment_offsets(int lane) {
    FragmentOffsets offsets;

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        offsets.k[i] = tile_offset<kSwizzle>(lane % 8 + 8 * (lane / 16), 2 * i + (lane / 8) % 2);
        offsets.v[i] = tile_offset<kSwizzle>(lane % 8 + 8 * ((lane / 8) % 2), 2 * i + lane / 16);
    }

    return offsets;
}

constexpr uint32_t kBlock16Bytes = 16 * kHeadDim * sizeof(BFloat16);

// Q's A fragments for the warp's sixteen rows: q[step] covers dimensions 16 * step .. + 16. Lanes
// 0-15 address rows 0-15 at the step's first eight dimensions, lanes 16-31 the same rows at the
// next eight, so matrices 0-3 land in a0-a3 of the m16n8k16 A layout.
template <bool kSwizzle>
__device__ __forceinline__ void load_q_fragments(uint32_t sQ, int warp_row, int lane, uint32_t (&q)[8][4]) {
#pragma unroll
    for (int step = 0; step < 8; ++step) {
        ldmatrix_x4(q[step], sQ + tile_offset<kSwizzle>(warp_row + lane % 16, 2 * step + lane / 16));
    }
}

// S = Q K^T for one K tile. One ldmatrix.x4 yields the B fragments (b0, b1) of two eight-token
// n-tiles for one k-step: lanes 0-7 and 8-15 address tokens 0-7 at the step's two eight-dimension
// halves, lanes 16-31 the same for tokens 8-15.
template <bool kSwizzle>
__device__ __forceinline__ void qk_tile(uint32_t sK, int lane, const uint32_t (&q)[8][4], float (&s)[8][4]) {
    const int token = lane % 8 + 8 * (lane / 16);
    const int half = (lane / 8) % 2;

#pragma unroll
    for (int n = 0; n < 8; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            s[n][i] = 0.0f;
        }
    }

#pragma unroll
    for (int step = 0; step < 8; ++step) {
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            uint32_t b[4];
            ldmatrix_x4(b, sK + tile_offset<kSwizzle>(16 * pair + token, 2 * step + half));
            mma_m16n8k16_bf16(s[2 * pair], q[step], b[0], b[1]);
            mma_m16n8k16_bf16(s[2 * pair + 1], q[step], b[2], b[3]);
        }
    }
}

// O += P V for one V tile. ldmatrix.trans turns each 8-token x 8-dimension block of V into a B
// fragment (tokens on k, dimensions on n): lanes 0-7 and 8-15 address tokens 0-7 and 8-15 of the
// k-step at one eight-dimension n-tile, lanes 16-31 the same at the next n-tile.
template <bool kSwizzle>
__device__ __forceinline__ void pv_tile(uint32_t sV, int lane, const uint32_t (&p)[4][4], float (&o)[16][4]) {
    const int token = lane % 8 + 8 * ((lane / 8) % 2);
    const int half = lane / 16;

#pragma unroll
    for (int step = 0; step < 4; ++step) {
#pragma unroll
        for (int pair = 0; pair < 8; ++pair) {
            uint32_t b[4];
            ldmatrix_x4_trans(b, sV + tile_offset<kSwizzle>(16 * step + token, 2 * pair + half));
            mma_m16n8k16_bf16(o[2 * pair], p[step], b[0], b[1]);
            mma_m16n8k16_bf16(o[2 * pair + 1], p[step], b[2], b[3]);
        }
    }
}

// qk_tile and pv_tile with the step-4b precomputed offsets. With kPartial (step 6) only the first
// `blocks` 16-token blocks of the tile take part: S's later n-tiles stay zero (the mask then sets
// them to -inf) and P V skips the matching k-steps, whose probabilities are zero.
template <bool kPartial = false>
__device__ __forceinline__ void qk_tile_hoisted(uint32_t sK, const FragmentOffsets& offsets, const uint32_t (&q)[8][4],
                                                float (&s)[8][4], int blocks = 4) {
#pragma unroll
    for (int n = 0; n < 8; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            s[n][i] = 0.0f;
        }
    }

#pragma unroll
    for (int step = 0; step < 8; ++step) {
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            if (kPartial && pair >= blocks) {
                continue;
            }

            uint32_t b[4];
            ldmatrix_x4(b, sK + offsets.k[step] + pair * kBlock16Bytes);
            mma_m16n8k16_bf16(s[2 * pair], q[step], b[0], b[1]);
            mma_m16n8k16_bf16(s[2 * pair + 1], q[step], b[2], b[3]);
        }
    }
}

template <bool kPartial = false>
__device__ __forceinline__ void pv_tile_hoisted(uint32_t sV, const FragmentOffsets& offsets, const uint32_t (&p)[4][4],
                                                float (&o)[16][4], int blocks = 4) {
#pragma unroll
    for (int step = 0; step < 4; ++step) {
        if (kPartial && step >= blocks) {
            continue;
        }

#pragma unroll
        for (int pair = 0; pair < 8; ++pair) {
            uint32_t b[4];
            ldmatrix_x4_trans(b, sV + offsets.v[pair] + step * kBlock16Bytes);
            mma_m16n8k16_bf16(o[2 * pair], p[step], b[0], b[1]);
            mma_m16n8k16_bf16(o[2 * pair + 1], p[step], b[2], b[3]);
        }
    }
}

// Step 4a: exp2 as one MUFU.EX2. exp2f() keeps IEEE behaviour for results below 2^-126, which costs a
// compare, a select and two multiplies around every MUFU.EX2; flushing those results to zero is
// harmless here, since a probability below 2^-126 of the row's max contributes nothing to a BF16
// output. FlashAttention-2 gets the same instruction from --use_fast_math.
template <bool kFastExp> __device__ __forceinline__ float softmax_exp2(float x) {
    if constexpr (kFastExp) {
        float y;
        asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
        return y;
    } else {
        return exp2f(x);
    }
}

// Online softmax over one tile of S, in the accumulator layout: lane (group, quad) holds rows group
// and group + 8 of the warp's sixteen (index i = 0, 1), at columns 8n + 2 * quad and the next. Each
// row's four lanes (one quad) agree on its running max after two shuffles; the running sum l stays
// a per-lane partial until the epilogue. The max is taken over raw scores (the scale is positive),
// so scaling and the max subtraction fuse into one FFMA ahead of exp2, as in FlashAttention-2.
template <bool kMask, bool kFastExp>
__device__ __forceinline__ void softmax_tile(float (&s)[8][4], float (&m)[2], float (&l)[2], float (&o)[16][4],
                                             uint32_t (&p)[4][4], const PrefillParams& params, int seq_len,
                                             int first_row, int k_begin, int lane) {
    const int group = lane / 4;
    const int quad = lane % 4;

    if constexpr (kMask) {
#pragma unroll
        for (int n = 0; n < 8; ++n) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int row = first_row + group + 8 * (e / 2);
                const int column = k_begin + 8 * n + 2 * quad + e % 2;
                s[n][e] = masked_out(params, seq_len, row, column) ? -INFINITY : s[n][e];
            }
        }
    }

    float m_scaled[2];
    float alpha[2];

#pragma unroll
    for (int i = 0; i < 2; ++i) {
        float tile_max = -INFINITY;

#pragma unroll
        for (int n = 0; n < 8; ++n) {
            tile_max = fmaxf(tile_max, fmaxf(s[n][2 * i], s[n][2 * i + 1]));
        }

        tile_max = fmaxf(tile_max, __shfl_xor_sync(kFullMask, tile_max, 1));
        tile_max = fmaxf(tile_max, __shfl_xor_sync(kFullMask, tile_max, 2));

        const float m_new = fmaxf(m[i], tile_max);
        m_scaled[i] = finite_max(m_new) * params.scale_log2;
        alpha[i] = softmax_exp2<kFastExp>(m[i] * params.scale_log2 - m_scaled[i]);
        m[i] = m_new;
    }

    float sum[2] = {0.0f, 0.0f};

#pragma unroll
    for (int n = 0; n < 8; ++n) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            s[n][e] = softmax_exp2<kFastExp>(fmaf(s[n][e], params.scale_log2, -m_scaled[e / 2]));
            sum[e / 2] += s[n][e];
        }
    }

#pragma unroll
    for (int i = 0; i < 2; ++i) {
        l[i] = fmaf(l[i], alpha[i], sum[i]);
    }

#pragma unroll
    for (int n = 0; n < 16; ++n) {
        o[n][0] *= alpha[0];
        o[n][1] *= alpha[0];
        o[n][2] *= alpha[1];
        o[n][3] *= alpha[1];
    }

    // P's A fragment for k-step t (tokens 16t .. + 16) is S's n-tiles 2t and 2t + 1, packed.
#pragma unroll
    for (int t = 0; t < 4; ++t) {
        p[t][0] = pack_bf16x2(s[2 * t][0], s[2 * t][1]);
        p[t][1] = pack_bf16x2(s[2 * t][2], s[2 * t][3]);
        p[t][2] = pack_bf16x2(s[2 * t + 1][0], s[2 * t + 1][1]);
        p[t][3] = pack_bf16x2(s[2 * t + 1][2], s[2 * t + 1][3]);
    }
}

template <PrefillVariant kVariant>
__global__ void __launch_bounds__(kThreadsPerBlock, 2) flash_prefill_mma_kernel(const PrefillParams params) {
    constexpr bool kSwizzle = kVariant >= PrefillVariant::kSwizzle;
    constexpr bool kPipelined = kVariant >= PrefillVariant::kCpAsync;
    constexpr bool kFastExp = kVariant >= PrefillVariant::kFastExp2;
    constexpr bool kHoisted = kVariant >= PrefillVariant::kHoisted;
    constexpr bool kHeavyFirst = kVariant >= PrefillVariant::kHeavyFirst;
    constexpr bool kSkipMasked = kVariant >= PrefillVariant::kSkipMasked;

    BlockTile tile;

    if (!block_tile<kHeavyFirst>(params, tile)) {
        return;
    }

    extern __shared__ uint4 shared_storage[];
    const uint32_t sQ = static_cast<uint32_t>(__cvta_generic_to_shared(shared_storage));
    const uint32_t sK = sQ + kTileBytes;
    const uint32_t sV = sK + kTileBytes;

    const int warp = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;
    const int warp_row = warp * kRowsPerWarp;
    const int first_row = tile.q_begin + warp_row;

    const TileCopy k_copy = make_tile_copy<kSwizzle>(params.k_token_stride);
    const TileCopy v_copy = make_tile_copy<kSwizzle>(params.v_token_stride);
    const FragmentOffsets offsets = make_fragment_offsets<kSwizzle>(lane);

    uint32_t q[8][4];
    float m[2] = {-INFINITY, -INFINITY};
    float l[2] = {0.0f, 0.0f};
    float o[16][4] = {};

    if constexpr (kPipelined) {
        copy_tile_async<kSwizzle>(sQ, tile.query, params.q_token_stride, tile.q_begin, tile.seq_len);
        copy_kv_tile_async<kSwizzle, kHoisted>(sK, tile.key, params.k_token_stride, k_copy, 0, tile.seq_len);
        cp_async_commit();
    } else {
        copy_tile_sync<kSwizzle>(sQ, tile.query, params.q_token_stride, tile.q_begin, tile.seq_len);
    }

    for (int kv_tile = 0; kv_tile < tile.num_kv_tiles; ++kv_tile) {
        const int k_begin = kv_tile * kBlockN;

        if constexpr (kPipelined) {
            // K(j) has landed, and every warp is done with V(j - 1), so V(j) may overwrite it.
            cp_async_wait_all();
            __syncthreads();
            copy_kv_tile_async<kSwizzle, kHoisted>(sV, tile.value, params.v_token_stride, v_copy, k_begin,
                                                   tile.seq_len);
            cp_async_commit();
        } else {
            __syncthreads(); // every warp is done with the previous K and V tiles
            copy_tile_sync<kSwizzle>(sK, tile.key, params.k_token_stride, k_begin, tile.seq_len);
            copy_tile_sync<kSwizzle>(sV, tile.value, params.v_token_stride, k_begin, tile.seq_len);
            __syncthreads();
        }

        if (kv_tile == 0) {
            load_q_fragments<kSwizzle>(sQ, warp_row, lane, q);
        }

        // Step 6: the 16-column blocks of this tile that hold an unmasked score for one of the warp's
        // rows. Only the diagonal tile and a sequence's ragged last tile have fewer than four, and a
        // warp whose rows all lie past the sequence (their output is never stored) has none.
        int blocks = 4;

        if constexpr (kSkipMasked) {
            const int column_end = params.causal ? min(tile.seq_len, first_row + kRowsPerWarp) : tile.seq_len;
            blocks = first_row >= tile.seq_len ? 0 : min(4, (column_end - k_begin + 15) / 16);
        }

        float s[8][4];

        if constexpr (kSkipMasked) {
            if (blocks == 4) {
                qk_tile_hoisted(sK, offsets, q, s);
            } else {
                qk_tile_hoisted<true>(sK, offsets, q, s, blocks);
            }
        } else if constexpr (kHoisted) {
            qk_tile_hoisted(sK, offsets, q, s);
        } else {
            qk_tile<kSwizzle>(sK, lane, q, s);
        }

        if constexpr (kPipelined) {
            // V(j) has landed, and every warp is done with K(j), so K(j + 1) may overwrite it.
            cp_async_wait_all();
            __syncthreads();

            if (kv_tile + 1 < tile.num_kv_tiles) {
                copy_kv_tile_async<kSwizzle, kHoisted>(sK, tile.key, params.k_token_stride, k_copy, k_begin + kBlockN,
                                                       tile.seq_len);
                cp_async_commit();
            }
        }

        uint32_t p[4][4];

        if (tile_needs_mask(params, tile, k_begin)) {
            softmax_tile<true, kFastExp>(s, m, l, o, p, params, tile.seq_len, first_row, k_begin, lane);
        } else {
            softmax_tile<false, kFastExp>(s, m, l, o, p, params, tile.seq_len, first_row, k_begin, lane);
        }

        if constexpr (kSkipMasked) {
            if (blocks == 4) {
                pv_tile_hoisted(sV, offsets, p, o);
            } else {
                pv_tile_hoisted<true>(sV, offsets, p, o, blocks);
            }
        } else if constexpr (kHoisted) {
            pv_tile_hoisted(sV, offsets, p, o);
        } else {
            pv_tile<kSwizzle>(sV, lane, p, o);
        }
    }

    // Finish each row's sum across its quad, then stage the normalized rows in this warp's own rows
    // of sQ (no other warp reads them, and Q now lives in registers) for 16-byte global stores.
    const int group = lane / 4;
    const int quad = lane % 4;
    float inverse[2];

#pragma unroll
    for (int i = 0; i < 2; ++i) {
        l[i] += __shfl_xor_sync(kFullMask, l[i], 1);
        l[i] += __shfl_xor_sync(kFullMask, l[i], 2);
        inverse[i] = l[i] > 0.0f ? 1.0f / l[i] : 0.0f;
    }

#pragma unroll
    for (int n = 0; n < 16; ++n) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const uint32_t address = sQ + tile_offset<kSwizzle>(warp_row + group + 8 * i, n) + 4 * quad;
            const uint32_t packed = pack_bf16x2(o[n][2 * i] * inverse[i], o[n][2 * i + 1] * inverse[i]);
            asm volatile("st.shared.b32 [%0], %1;\n" ::"r"(address), "r"(packed));
        }
    }

    __syncwarp();

#pragma unroll
    for (int pass = 0; pass < kRowsPerWarp / 2; ++pass) {
        const int row = warp_row + 2 * pass + lane / 16;
        const int chunk = lane % 16;

        if (tile.q_begin + row < tile.seq_len) {
            uint4 data;
            asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];\n"
                         : "=r"(data.x), "=r"(data.y), "=r"(data.z), "=r"(data.w)
                         : "r"(sQ + tile_offset<kSwizzle>(row, chunk)));
            *reinterpret_cast<uint4*>(tile.out + (tile.q_begin + row) * params.out_token_stride + chunk * 8) = data;
        }
    }
}

template <typename Kernel>
void launch(Kernel kernel, size_t smem_bytes, dim3 grid, cudaStream_t stream, const PrefillParams& params) {
    check_cuda(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes)),
               "failed to set the kernel's shared-memory size");
    // Ask for the largest shared-memory carveout so two 48 KB blocks fit on each SM.
    check_cuda(
        cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared),
        "failed to set the kernel's shared-memory carveout");
    kernel<<<grid, kThreadsPerBlock, smem_bytes, stream>>>(params);
}

bool is_16_byte_aligned(const void* pointer) { return reinterpret_cast<std::uintptr_t>(pointer) % 16 == 0; }

void check_token_major(const ts::Tensor& tensor, const char* name) {
    STD_TORCH_CHECK(tensor.dim() == 3 && tensor.size(2) == kHeadDim, name, " must be [total_tokens, heads, 128]");
    STD_TORCH_CHECK(tensor.stride(2) == 1, name, " must be contiguous along head_dim");
    STD_TORCH_CHECK(tensor.stride(0) % 8 == 0 && tensor.stride(1) % 8 == 0, name,
                    " token and head strides must be multiples of 8 elements");
    STD_TORCH_CHECK(is_16_byte_aligned(tensor.const_data_ptr<BFloat16>()), name, " must be 16-byte aligned");
}

} // namespace

void flash_prefill_attention_cuda(ts::Tensor& out, const ts::Tensor& query, const ts::Tensor& key,
                                  const ts::Tensor& value, const ts::Tensor& cu_seqlens, int64_t max_seqlen,
                                  double scale, bool causal, int64_t variant) {
    STD_TORCH_CHECK(out.is_cuda() && query.is_cuda() && key.is_cuda() && value.is_cuda() && cu_seqlens.is_cuda(),
                    "all tensors must be CUDA tensors");
    STD_TORCH_CHECK(out.device() == query.device() && key.device() == query.device() &&
                        value.device() == query.device() && cu_seqlens.device() == query.device(),
                    "all tensors must be on the same device");
    STD_TORCH_CHECK(query.scalar_type() == ScalarType::BFloat16 && key.scalar_type() == ScalarType::BFloat16 &&
                        value.scalar_type() == ScalarType::BFloat16 && out.scalar_type() == ScalarType::BFloat16,
                    "query, key, value and out must use bfloat16");
    STD_TORCH_CHECK(cu_seqlens.scalar_type() == ScalarType::Int, "cu_seqlens must be int32");
    STD_TORCH_CHECK(query.dim() == 3 && key.dim() == 3 && value.dim() == 3 && out.dim() == 3,
                    "query, key, value and out must be [total_tokens, heads, head_dim]");
    STD_TORCH_CHECK(query.size(2) == kHeadDim && key.size(2) == kHeadDim && value.size(2) == kHeadDim,
                    "head_dim must be 128");

    check_token_major(query, "query");
    check_token_major(key, "key");
    check_token_major(value, "value");
    check_token_major(out, "out");

    const int64_t total_tokens = query.size(0);
    const int64_t num_q_heads = query.size(1);
    const int64_t num_kv_heads = key.size(1);

    STD_TORCH_CHECK(key.size(0) == total_tokens && value.size(0) == total_tokens,
                    "query, key and value must hold the same tokens (prefill without a prefix cache)");
    STD_TORCH_CHECK(value.size(1) == num_kv_heads, "key and value must have the same number of heads");
    STD_TORCH_CHECK(out.size(0) == total_tokens && out.size(1) == num_q_heads, "out must match query's shape");
    STD_TORCH_CHECK(num_kv_heads > 0 && num_q_heads % num_kv_heads == 0,
                    "num_q_heads must be a multiple of num_kv_heads");
    STD_TORCH_CHECK(cu_seqlens.dim() == 1 && cu_seqlens.size(0) >= 1 && cu_seqlens.is_contiguous(),
                    "cu_seqlens must be a contiguous [num_seqs + 1] tensor");
    STD_TORCH_CHECK(max_seqlen >= 0 && total_tokens < (int64_t{1} << 31),
                    "max_seqlen must be non-negative and total_tokens below 2^31");

    const int64_t num_seqs = cu_seqlens.size(0) - 1;

    STD_TORCH_CHECK(num_seqs <= 65535 && num_q_heads <= 65535, "at most 65535 sequences and query heads");
    // The tensor-core kernels take each row's max over raw scores, which requires scale > 0.
    STD_TORCH_CHECK(scale > 0.0, "the softmax scale must be positive");
    STD_TORCH_CHECK(variant >= 0 && variant < static_cast<int64_t>(PrefillVariant::kCount),
                    "unknown flash prefill variant");

    if (static_cast<PrefillVariant>(variant) != PrefillVariant::kSimt) {
        int major = 0;
        check_cuda(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, query.get_device_index()),
                   "failed to read the compute capability");
        STD_TORCH_CHECK(major >= 8, "the tensor-core kernels need sm_80 or later");
    }

    if (num_seqs == 0 || max_seqlen == 0 || total_tokens == 0) {
        return;
    }

    PrefillParams params{};
    params.out = out.mutable_data_ptr<BFloat16>();
    params.query = query.const_data_ptr<BFloat16>();
    params.key = key.const_data_ptr<BFloat16>();
    params.value = value.const_data_ptr<BFloat16>();
    params.cu_seqlens = cu_seqlens.const_data_ptr<int32_t>();
    params.q_token_stride = query.stride(0);
    params.q_head_stride = query.stride(1);
    params.k_token_stride = key.stride(0);
    params.k_head_stride = key.stride(1);
    params.v_token_stride = value.stride(0);
    params.v_head_stride = value.stride(1);
    params.out_token_stride = out.stride(0);
    params.out_head_stride = out.stride(1);
    params.num_q_heads = static_cast<int>(num_q_heads);
    params.num_kv_heads = static_cast<int>(num_kv_heads);
    params.causal = causal;
    params.scale_log2 = static_cast<float>(scale) * kLog2e;

    const ts::accelerator::DeviceGuard device_guard(query.get_device_index());
    const cudaStream_t stream =
        static_cast<cudaStream_t>(ts::accelerator::getCurrentStream(query.get_device_index()).nativeHandle());

    // max_seqlen sizes the grid (like FlashAttention's max_seqlen_q) and must be at least the
    // longest sequence; passing it in keeps the launch free of a device-to-host copy.
    const int64_t num_q_tiles = (max_seqlen + kBlockM - 1) / kBlockM;
    const bool heavy_first = variant >= static_cast<int64_t>(PrefillVariant::kHeavyFirst);

    STD_TORCH_CHECK(!heavy_first || num_q_tiles <= 65535, "max_seqlen must be at most 65535 * 64");

    const dim3 grid = heavy_first ? dim3(static_cast<unsigned>(num_q_heads), static_cast<unsigned>(num_seqs),
                                         static_cast<unsigned>(num_q_tiles))
                                  : dim3(static_cast<unsigned>(num_q_tiles), static_cast<unsigned>(num_q_heads),
                                         static_cast<unsigned>(num_seqs));

    switch (static_cast<PrefillVariant>(variant)) {
    case PrefillVariant::kSimt:
        launch(flash_prefill_simt_kernel, kSimtSmemBytes, grid, stream, params);
        break;
    case PrefillVariant::kMma:
        launch(flash_prefill_mma_kernel<PrefillVariant::kMma>, kMmaSmemBytes, grid, stream, params);
        break;
    case PrefillVariant::kSwizzle:
        launch(flash_prefill_mma_kernel<PrefillVariant::kSwizzle>, kMmaSmemBytes, grid, stream, params);
        break;
    case PrefillVariant::kCpAsync:
        launch(flash_prefill_mma_kernel<PrefillVariant::kCpAsync>, kMmaSmemBytes, grid, stream, params);
        break;
    case PrefillVariant::kFastExp2:
        launch(flash_prefill_mma_kernel<PrefillVariant::kFastExp2>, kMmaSmemBytes, grid, stream, params);
        break;
    case PrefillVariant::kHoisted:
        launch(flash_prefill_mma_kernel<PrefillVariant::kHoisted>, kMmaSmemBytes, grid, stream, params);
        break;
    case PrefillVariant::kHeavyFirst:
        launch(flash_prefill_mma_kernel<PrefillVariant::kHeavyFirst>, kMmaSmemBytes, grid, stream, params);
        break;
    case PrefillVariant::kSkipMasked:
        launch(flash_prefill_mma_kernel<PrefillVariant::kSkipMasked>, kMmaSmemBytes, grid, stream, params);
        break;
    default:
        STD_TORCH_CHECK(false, "unknown flash prefill variant");
    }

    check_cuda(cudaGetLastError(), "flash prefill attention kernel launch failed");
}
