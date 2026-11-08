#include "block_pool.h"

#include <algorithm>
#include <cassert>

BlockPool::BlockPool(int num_blocks, bool enable_caching) : blocks_(num_blocks), enable_caching_(enable_caching) {
    for (int id = 1; id < num_blocks; ++id) {
        blocks_[id].free_position = free_queue_.insert(free_queue_.end(), id);
    }
}

std::vector<int> BlockPool::allocate(int count) {
    assert(count <= num_free_blocks());

    std::vector<int> allocated;
    allocated.reserve(count);

    for (int i = 0; i < count; ++i) {
        const int id = free_queue_.front();
        free_queue_.pop_front();
        evict(id);
        blocks_[id].ref_count = 1;
        allocated.push_back(id);
    }

    return allocated;
}

void BlockPool::touch(const std::vector<int>& block_ids) {
    for (int id : block_ids) {
        Block& block = blocks_[id];
        if (block.ref_count == 0) {
            free_queue_.erase(block.free_position);
        }
        ++block.ref_count;
    }
}

void BlockPool::free(const std::vector<int>& block_ids) {
    std::vector<int> hashed;
    const auto front = free_queue_.begin();

    for (int id : block_ids) {
        Block& block = blocks_[id];
        if (--block.ref_count > 0) {
            continue;
        }

        // Unhashed blocks are reused immediately, which also keeps the GPU pages warm.
        // Hashed blocks go to the back so the prefix cache evicts in LRU order.
        if (block.hash && enable_caching_) {
            hashed.push_back(id);
        } else {
            block.free_position = free_queue_.insert(front, id);
        }
    }

    for (int id : hashed) {
        blocks_[id].free_position = free_queue_.insert(free_queue_.end(), id);
    }
}

void BlockPool::cache(int block_id, BlockHash hash) {
    blocks_[block_id].hash = hash;
    cached_[hash].push_back(block_id);
}

std::optional<int> BlockPool::lookup(BlockHash hash) const {
    const auto entry = cached_.find(hash);
    if (entry == cached_.end()) {
        return std::nullopt;
    }
    return entry->second.front();
}

void BlockPool::evict(int block_id) {
    Block& block = blocks_[block_id];
    if (!block.hash) {
        return;
    }

    const auto entry = cached_.find(*block.hash);
    std::erase(entry->second, block_id);
    if (entry->second.empty()) {
        cached_.erase(entry);
    }
    block.hash.reset();
}
