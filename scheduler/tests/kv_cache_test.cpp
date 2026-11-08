#include "check.h"
#include "kv_cache.h"

#include <vector>

using std::vector;

constexpr int kBlockSize = 16;

static Request make_request(int id, int num_tokens, int token = 7) {
    return Request(id, vector<int>(num_tokens, token), 16, -1);
}

static void admit_all(KvCache& cache, Request& request) {
    cache.update_block_hashes(request);
    const PrefixHit hit = cache.find_prefix_hit(request);
    CHECK(cache.admit(request, request.num_tokens() - hit.num_tokens, hit, false));
}

static void test_hashes_chain_over_the_whole_prefix() {
    KvCache cache(100, kBlockSize, 4096, true, 0);

    Request same_first(0, vector<int>(32, 1), 16, -1);
    vector<int> tokens(32, 1);
    tokens[20] = 9;
    Request different_second(1, tokens, 16, -1);
    tokens[3] = 8;
    Request different_first(2, tokens, 16, -1);

    cache.update_block_hashes(same_first);
    cache.update_block_hashes(different_second);
    cache.update_block_hashes(different_first);

    CHECK_EQ(same_first.block_hashes.size(), 2u);
    CHECK_EQ(same_first.block_hashes[0], different_second.block_hashes[0]);
    CHECK(same_first.block_hashes[1] != different_second.block_hashes[1]);
    CHECK(different_first.block_hashes[1] != different_second.block_hashes[1]);
}

static void test_partial_blocks_are_not_hashed() {
    KvCache cache(100, kBlockSize, 4096, true, 0);
    Request request = make_request(0, 47);

    cache.update_block_hashes(request);
    CHECK_EQ(request.block_hashes.size(), 2u);
}

static void test_prefix_hits_are_block_aligned_and_capped() {
    const vector<vector<int>> cases = {{31, 16}, {32, 16}, {33, 32}, {48, 32}, {49, 48}};

    for (const vector<int>& entry : cases) {
        KvCache cache(1000, kBlockSize, 4096, true, 0);
        Request first = make_request(0, entry[0]);
        Request second = make_request(1, entry[0]);

        admit_all(cache, first);
        cache.update_block_hashes(second);

        CHECK_EQ(cache.find_prefix_hit(second).num_tokens, entry[1]);
    }
}

static void test_no_hits_with_caching_disabled() {
    KvCache cache(1000, kBlockSize, 4096, false, 0);
    Request first = make_request(0, 64);
    Request second = make_request(1, 64);

    admit_all(cache, first);
    cache.update_block_hashes(second);
    CHECK_EQ(cache.find_prefix_hit(second).num_tokens, 0);
}

static void test_admission_takes_whole_blocks_only() {
    KvCache cache(11, kBlockSize, 4096, false, 0);
    Request request = make_request(0, 80);

    admit_all(cache, request);
    CHECK_EQ(request.blocks.size(), 5u);
    CHECK_EQ(cache.num_free_blocks(), 5);
}

static void test_admission_needs_room_for_the_whole_sequence() {
    KvCache cache(5, kBlockSize, 4096, false, 0); // four usable blocks
    Request request = make_request(0, 80);

    cache.update_block_hashes(request);
    CHECK(!cache.admit(request, 16, {}, false));
    CHECK_EQ(cache.num_free_blocks(), 4);
}

static void test_watermark_only_applies_with_running_requests() {
    KvCache cache(5, kBlockSize, 4096, false, 2); // four usable blocks
    Request request = make_request(0, 48);
    cache.update_block_hashes(request);

    CHECK(!cache.admit(request, 48, {}, true));
    CHECK(cache.admit(request, 48, {}, false));
    CHECK_EQ(cache.num_free_blocks(), 1);
}

static void test_shared_blocks_are_ref_counted() {
    KvCache cache(100, kBlockSize, 4096, true, 0);
    Request first = make_request(0, 33);
    Request second = make_request(1, 33);

    admit_all(cache, first);
    admit_all(cache, second);

    CHECK_EQ(first.blocks[0], second.blocks[0]);
    CHECK_EQ(first.blocks[1], second.blocks[1]);
    CHECK_EQ(cache.pool().ref_count(first.blocks[0]), 2);

    const int shared = first.blocks[0];
    cache.free(first);
    CHECK_EQ(cache.pool().ref_count(shared), 1);
}

static void test_adopting_a_free_cached_block_counts_against_free_blocks() {
    KvCache cache(4, kBlockSize, 4096, true, 0); // three usable blocks
    Request first = make_request(0, 32);
    admit_all(cache, first);
    cache.free(first);
    CHECK_EQ(cache.num_free_blocks(), 3);

    // The hit adopts one cached block, and the second block is a fresh block.
    Request second = make_request(1, 32);
    cache.update_block_hashes(second);
    const PrefixHit hit = cache.find_prefix_hit(second);
    CHECK_EQ(hit.blocks.size(), 1u);
    CHECK(cache.admit(second, 16, hit, false));
    CHECK_EQ(cache.num_free_blocks(), 1);
}

static void test_running_request_gets_a_new_block_on_a_block_boundary() {
    KvCache cache(10, kBlockSize, 4096, false, 0);
    Request request = make_request(0, 32);
    admit_all(cache, request);
    request.num_computed_tokens = 32;
    CHECK_EQ(request.blocks.size(), 2u);

    CHECK(cache.allocate(request, 1));
    CHECK_EQ(request.blocks.size(), 3u);
}

int main() {
    test_hashes_chain_over_the_whole_prefix();
    test_partial_blocks_are_not_hashed();
    test_prefix_hits_are_block_aligned_and_capped();
    test_no_hits_with_caching_disabled();
    test_admission_takes_whole_blocks_only();
    test_admission_needs_room_for_the_whole_sequence();
    test_watermark_only_applies_with_running_requests();
    test_shared_blocks_are_ref_counted();
    test_adopting_a_free_cached_block_counts_against_free_blocks();
    test_running_request_gets_a_new_block_on_a_block_boundary();
    return 0;
}
