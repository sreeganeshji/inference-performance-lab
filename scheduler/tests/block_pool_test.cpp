#include "block_pool.h"
#include "check.h"

#include <vector>

using std::vector;

static void test_null_block_is_never_allocated() {
    BlockPool pool(6, true);
    CHECK_EQ(pool.num_free_blocks(), 5);

    const vector<int> blocks = pool.allocate(5);
    CHECK(blocks == (vector<int>{1, 2, 3, 4, 5}));
    CHECK_EQ(pool.num_free_blocks(), 0);
}

static void test_unhashed_blocks_are_reused_first() {
    BlockPool pool(5, true);
    const vector<int> blocks = pool.allocate(4);

    pool.free({4, 3});
    CHECK(pool.allocate(1) == vector<int>{4});
    CHECK(pool.allocate(1) == vector<int>{3});
}

static void test_hashed_blocks_go_to_the_back() {
    BlockPool pool(5, true);
    pool.allocate(3);

    pool.cache(2, 0xAA);
    pool.free({3, 2, 1});

    // Block 4 was never used and the hashed block waits behind every unhashed one.
    CHECK(pool.allocate(4) == (vector<int>{3, 1, 4, 2}));
}

static void test_mixed_free_with_empty_queue() {
    BlockPool pool(4, true);
    pool.allocate(3);
    pool.cache(2, 0xAA);

    pool.free({3, 2, 1});
    CHECK(pool.allocate(3) == (vector<int>{3, 1, 2}));
}

static void test_hashed_blocks_keep_their_prefix_until_reused() {
    BlockPool pool(3, true);
    pool.allocate(2);
    pool.cache(1, 0xAA);

    pool.free({2, 1});
    CHECK_EQ(pool.num_free_blocks(), 2);
    CHECK_EQ(*pool.lookup(0xAA), 1);

    pool.allocate(1); // unhashed block 2
    CHECK_EQ(*pool.lookup(0xAA), 1);

    pool.allocate(1); // reusing block 1 evicts its prefix
    CHECK(!pool.lookup(0xAA).has_value());
    CHECK(!pool.hash_of(1).has_value());
}

static void test_ref_counts_and_touch() {
    BlockPool pool(4, true);
    pool.allocate(1);
    pool.cache(1, 0xAA);

    pool.touch({1});
    CHECK_EQ(pool.ref_count(1), 2);

    pool.free({1});
    CHECK_EQ(pool.ref_count(1), 1);
    CHECK_EQ(pool.num_free_blocks(), 2);

    pool.free({1});
    CHECK_EQ(pool.num_free_blocks(), 3);

    // Touching a free cached block takes it back out of the queue.
    pool.touch({1});
    CHECK_EQ(pool.num_free_blocks(), 2);
    CHECK_EQ(pool.ref_count(1), 1);
}

static void test_duplicate_hashes_return_first_inserted() {
    BlockPool pool(4, true);
    pool.allocate(2);
    pool.cache(1, 0xAA);
    pool.cache(2, 0xAA);
    CHECK_EQ(*pool.lookup(0xAA), 1);

    pool.free({1});
    pool.allocate(2); // takes 3, then block 1, which clears only its own entry
    CHECK_EQ(*pool.lookup(0xAA), 2);
}

int main() {
    test_null_block_is_never_allocated();
    test_unhashed_blocks_are_reused_first();
    test_hashed_blocks_go_to_the_back();
    test_mixed_free_with_empty_queue();
    test_hashed_blocks_keep_their_prefix_until_reused();
    test_ref_counts_and_touch();
    test_duplicate_hashes_return_first_inserted();
    return 0;
}
