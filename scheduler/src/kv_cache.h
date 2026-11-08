#pragma once

#include "block_pool.h"
#include "request.h"

#include <vector>

struct PrefixHit {
    std::vector<int> blocks;
    int num_tokens = 0;
};

// Per-request block tables on top of the block pool, for one full-attention KV group.
class KvCache {
  public:
    KvCache(int num_blocks, int block_size, int max_model_len, bool enable_caching, int watermark_blocks);

    int num_free_blocks() const { return pool_.num_free_blocks(); }
    const BlockPool& pool() const { return pool_; }

    void update_block_hashes(Request& request) const;
    PrefixHit find_prefix_hit(const Request& request) const;

    // Grow a running request's block table. Returns false if there are too few free blocks.
    bool allocate(Request& request, int num_new_tokens);

    // Adopt a waiting request's prefix hit and allocate its first chunk. Admission needs
    // room for the whole sequence, not just the first chunk, so it cannot over-commit.
    bool admit(Request& request, int num_new_tokens, const PrefixHit& hit, bool has_running);

    void cache_blocks(Request& request, int num_tokens);
    void free(Request& request);

  private:
    int blocks_to_allocate(const Request& request, int num_tokens, const PrefixHit& hit) const;
    void grow(Request& request, int num_tokens);

    BlockPool pool_;
    int block_size_;
    int max_model_len_;
    bool enable_caching_;
    int watermark_blocks_;
};
