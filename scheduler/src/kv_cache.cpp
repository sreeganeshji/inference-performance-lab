#include "kv_cache.h"

#include <algorithm>
#include <cstdint>

namespace {

constexpr std::uint64_t kHashMultiplier = 0x9E3779B97F4A7C15ull;

int ceil_div(int value, int divisor) { return (value + divisor - 1) / divisor; }

BlockHash mix(BlockHash hash, std::uint64_t value) {
    hash = (hash ^ value) * kHashMultiplier;
    return hash ^ (hash >> 32);
}

} // namespace

KvCache::KvCache(int num_blocks, int block_size, int max_model_len, bool enable_caching, int watermark_blocks)
    : pool_(num_blocks, enable_caching), block_size_(block_size), max_model_len_(max_model_len),
      enable_caching_(enable_caching), watermark_blocks_(watermark_blocks) {}

void KvCache::update_block_hashes(Request& request) const {
    const int num_full_blocks = request.num_tokens() / block_size_;

    for (int block = static_cast<int>(request.block_hashes.size()); block < num_full_blocks; ++block) {
        BlockHash hash = block == 0 ? 0 : request.block_hashes.back();
        for (int token = block * block_size_; token < (block + 1) * block_size_; ++token) {
            hash = mix(hash, static_cast<std::uint64_t>(request.tokens[token]));
        }
        request.block_hashes.push_back(hash);
    }
}

PrefixHit KvCache::find_prefix_hit(const Request& request) const {
    PrefixHit hit;
    if (!enable_caching_) {
        return hit;
    }

    // The last token must always be recomputed to get logits, and hits are whole blocks.
    const int max_blocks = (request.num_tokens() - 1) / block_size_;

    for (int block = 0; block < max_blocks; ++block) {
        const auto cached = pool_.lookup(request.block_hashes[block]);
        if (!cached) {
            break;
        }
        hit.blocks.push_back(*cached);
    }

    hit.num_tokens = static_cast<int>(hit.blocks.size()) * block_size_;
    return hit;
}

int KvCache::blocks_to_allocate(const Request& request, int num_tokens, const PrefixHit& hit) const {
    const int held = static_cast<int>(request.blocks.size() + hit.blocks.size());
    const int new_blocks = std::max(ceil_div(num_tokens, block_size_) - held, 0);

    // A hit block sitting in the free queue leaves the queue when it is adopted.
    const auto evictable =
        std::count_if(hit.blocks.begin(), hit.blocks.end(), [&](int block) { return pool_.ref_count(block) == 0; });

    return new_blocks + static_cast<int>(evictable);
}

void KvCache::grow(Request& request, int num_tokens) {
    const int needed = ceil_div(num_tokens, block_size_) - static_cast<int>(request.blocks.size());
    if (needed > 0) {
        const std::vector<int> blocks = pool_.allocate(needed);
        request.blocks.insert(request.blocks.end(), blocks.begin(), blocks.end());
    }
}

bool KvCache::allocate(Request& request, int num_new_tokens) {
    const int num_tokens = request.num_computed_tokens + num_new_tokens;
    const int slots = std::min(num_tokens, max_model_len_);

    if (blocks_to_allocate(request, slots, {}) > num_free_blocks()) {
        return false;
    }

    grow(request, slots);
    cache_blocks(request, std::min(num_tokens, request.num_tokens()));
    return true;
}

bool KvCache::admit(Request& request, int num_new_tokens, const PrefixHit& hit, bool has_running) {
    const int reserved = has_running ? watermark_blocks_ : 0;
    const int full_sequence = std::min(request.num_tokens(), max_model_len_);

    if (blocks_to_allocate(request, full_sequence, hit) + reserved > num_free_blocks()) {
        return false;
    }

    pool_.touch(hit.blocks);
    request.blocks = hit.blocks;
    request.num_cached_blocks = static_cast<int>(hit.blocks.size());

    const int num_tokens = hit.num_tokens + num_new_tokens;
    grow(request, std::min(num_tokens, max_model_len_));
    cache_blocks(request, std::min(num_tokens, request.num_tokens()));
    return true;
}

void KvCache::cache_blocks(Request& request, int num_tokens) {
    if (!enable_caching_) {
        return;
    }

    const int num_hashed_tokens = static_cast<int>(request.block_hashes.size()) * block_size_;
    const int num_full_blocks = std::min(num_tokens, num_hashed_tokens) / block_size_;

    for (int block = request.num_cached_blocks; block < num_full_blocks; ++block) {
        pool_.cache(request.blocks[block], request.block_hashes[block]);
    }

    request.num_cached_blocks = std::max(request.num_cached_blocks, num_full_blocks);
}

void KvCache::free(Request& request) {
    // Free the tail first so a chain is evicted from its end, keeping the shared prefix longest.
    std::vector<int> tail_first(request.blocks.rbegin(), request.blocks.rend());
    pool_.free(tail_first);

    request.blocks.clear();
    request.num_cached_blocks = 0;
}
