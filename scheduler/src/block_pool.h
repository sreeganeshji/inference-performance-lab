#pragma once

#include "request.h"

#include <list>
#include <optional>
#include <unordered_map>
#include <vector>

// A fixed pool of KV cache blocks. Block 0 is the null block and is never handed out.
// Free blocks wait in a queue, and a free block keeps its cached prefix until it is reused.
class BlockPool {
  public:
    BlockPool(int num_blocks, bool enable_caching);

    int num_free_blocks() const { return static_cast<int>(free_queue_.size()); }
    int ref_count(int block_id) const { return blocks_[block_id].ref_count; }

    std::vector<int> allocate(int count);
    void touch(const std::vector<int>& block_ids);

    // block_ids are ordered so the block to evict first comes first.
    void free(const std::vector<int>& block_ids);

    void cache(int block_id, BlockHash hash);
    std::optional<int> lookup(BlockHash hash) const;
    std::optional<BlockHash> hash_of(int block_id) const { return blocks_[block_id].hash; }

  private:
    struct Block {
        int ref_count = 0;
        std::optional<BlockHash> hash;
        std::list<int>::iterator free_position;
    };

    void evict(int block_id);

    std::vector<Block> blocks_;
    std::list<int> free_queue_;
    // Requests that race on the same prefix each keep their own block, so one hash can map to several.
    std::unordered_map<BlockHash, std::vector<int>> cached_;
    bool enable_caching_;
};
