#include "check.h"
#include "scheduler.h"

#include <optional>
#include <vector>

using std::vector;

constexpr int kEos = 50256;

static vector<int> prompt(int num_tokens, int token) { return vector<int>(num_tokens, token); }

static vector<int> scheduled_tokens(const SchedulerOutput& output) {
    vector<int> tokens;
    for (const ScheduledRequest& request : output.scheduled) {
        tokens.push_back(request.num_tokens);
    }
    return tokens;
}

// Samples token 100 for every request that reached the end of its sequence.
static vector<int> finish_step(Scheduler& scheduler, const SchedulerOutput& output, int token = 100) {
    vector<std::optional<int>> sampled;
    for (const ScheduledRequest& request : output.scheduled) {
        sampled.push_back(request.samples_token ? std::optional<int>(token) : std::nullopt);
    }
    return scheduler.update_from_output(output.scheduled, sampled);
}

static void test_admits_in_arrival_order_up_to_max_running() {
    SchedulerConfig config;
    config.max_running_requests = 2;
    Scheduler scheduler(config);

    for (int id = 0; id < 3; ++id) {
        scheduler.add_request(id, prompt(10, id + 1), 8, kEos);
    }

    const SchedulerOutput output = scheduler.schedule();
    CHECK(output.new_ids == (vector<int>{0, 1}));
    CHECK(scheduler.running_ids() == (vector<int>{0, 1}));
    CHECK(scheduled_tokens(output) == (vector<int>{10, 10}));
}

static void test_token_budget_chunks_the_prefill() {
    SchedulerConfig config;
    config.max_tokens_per_step = 25;
    Scheduler scheduler(config);
    scheduler.add_request(0, prompt(40, 1), 8, kEos);
    scheduler.add_request(1, prompt(10, 2), 8, kEos);

    const SchedulerOutput first = scheduler.schedule();
    CHECK(first.new_ids == vector<int>{0});
    CHECK(scheduled_tokens(first) == vector<int>{25});
    CHECK(!first.scheduled[0].samples_token);
    finish_step(scheduler, first);

    const SchedulerOutput second = scheduler.schedule();
    CHECK(scheduled_tokens(second) == (vector<int>{15, 10}));
    CHECK(second.scheduled[0].samples_token);
    CHECK(second.scheduled[1].samples_token);
}

static void test_no_chunking_stops_at_the_first_request_that_does_not_fit() {
    SchedulerConfig config;
    config.max_tokens_per_step = 20;
    config.chunked_prefill = false;
    Scheduler scheduler(config);
    scheduler.add_request(0, prompt(15, 1), 8, kEos);
    scheduler.add_request(1, prompt(10, 2), 8, kEos);
    scheduler.add_request(2, prompt(2, 3), 8, kEos);

    const SchedulerOutput output = scheduler.schedule();
    CHECK(output.new_ids == vector<int>{0});
}

static void test_long_prefill_threshold_only_applies_with_competition() {
    SchedulerConfig config;
    config.long_prefill_threshold = 20;

    Scheduler alone(config);
    alone.add_request(0, prompt(50, 1), 8, kEos);
    CHECK(scheduled_tokens(alone.schedule()) == vector<int>{50});

    Scheduler shared(config);
    shared.add_request(0, prompt(50, 1), 8, kEos);
    shared.add_request(1, prompt(50, 2), 8, kEos);
    CHECK(scheduled_tokens(shared.schedule()) == (vector<int>{20, 20}));
}

static void test_request_finishes_at_max_tokens() {
    Scheduler scheduler{SchedulerConfig{}};
    scheduler.add_request(0, prompt(10, 1), 3, kEos);

    for (int step = 0; step < 2; ++step) {
        CHECK(finish_step(scheduler, scheduler.schedule()).empty());
    }

    CHECK(finish_step(scheduler, scheduler.schedule()) == vector<int>{0});
    CHECK(scheduler.find(0) == nullptr);
    CHECK_EQ(scheduler.num_free_blocks(), 999);
}

static void test_eos_stops_early() {
    Scheduler scheduler{SchedulerConfig{}};
    scheduler.add_request(0, prompt(10, 1), 100, kEos);

    CHECK(finish_step(scheduler, scheduler.schedule(), kEos) == vector<int>{0});
}

static void test_model_length_caps_output() {
    SchedulerConfig config;
    config.max_model_len = 12;
    Scheduler scheduler(config);
    scheduler.add_request(0, prompt(10, 1), 100, kEos);

    CHECK(finish_step(scheduler, scheduler.schedule()).empty());
    CHECK(finish_step(scheduler, scheduler.schedule()) == vector<int>{0});
}

static void test_a_finish_keeps_the_scheduler_alive_for_one_more_step() {
    Scheduler scheduler{SchedulerConfig{}};
    scheduler.add_request(0, prompt(10, 1), 1, kEos);

    finish_step(scheduler, scheduler.schedule());
    CHECK(scheduler.has_requests());

    finish_step(scheduler, scheduler.schedule());
    CHECK(!scheduler.has_requests());
}

// Three requests of 32 tokens fill six blocks, so the third cannot grow past its second block.
static SchedulerOutput fill_kv_cache(Scheduler& scheduler) {
    scheduler.add_request(0, prompt(32, 1), 4, kEos);
    scheduler.add_request(1, prompt(32, 2), 4, kEos);

    SchedulerOutput output = scheduler.schedule();
    CHECK(output.new_ids == (vector<int>{0, 1}));
    finish_step(scheduler, output);
    return output;
}

static void test_preempts_from_the_back_of_running() {
    SchedulerConfig config;
    config.num_blocks = 6; // five usable, and the two requests already hold four
    Scheduler scheduler(config);
    fill_kv_cache(scheduler);

    // Both requests need a new block for their 33rd token and only one is free.
    const SchedulerOutput output = scheduler.schedule();
    CHECK(output.preempted_ids == vector<int>{1});
    CHECK(scheduler.running_ids() == vector<int>{0});
    CHECK(scheduled_tokens(output) == vector<int>{1});
    CHECK(output.new_ids.empty());

    const Request* victim = scheduler.find(1);
    CHECK(victim->status == RequestStatus::Preempted);
    CHECK_EQ(victim->num_computed_tokens, 0);
    CHECK(victim->blocks.empty());
}

static void test_preempted_request_resumes_with_its_output_tokens() {
    SchedulerConfig config;
    config.num_blocks = 6;
    Scheduler scheduler(config);
    fill_kv_cache(scheduler);
    finish_step(scheduler, scheduler.schedule());

    // Request 0 runs to max_tokens; then request 1 can come back.
    SchedulerOutput output = scheduler.schedule();
    while (output.new_ids.empty()) {
        finish_step(scheduler, output);
        output = scheduler.schedule();
    }

    CHECK(output.new_ids == vector<int>{1});
    CHECK_EQ(output.prefix_hits[0], 0);
    CHECK_EQ(output.scheduled.back().num_tokens, 33); // 32 prompt tokens + 1 output token
}

static void test_a_request_can_preempt_itself() {
    SchedulerConfig config;
    config.num_blocks = 4; // three usable
    Scheduler scheduler(config);
    scheduler.add_request(0, prompt(48, 1), 4, kEos);

    finish_step(scheduler, scheduler.schedule());
    CHECK_EQ(scheduler.num_free_blocks(), 0);

    const SchedulerOutput output = scheduler.schedule();
    CHECK(output.preempted_ids == vector<int>{0});
    CHECK(output.scheduled.empty());
    CHECK_EQ(scheduler.num_free_blocks(), 3);
}

static void test_budget_is_shared_across_running_requests() {
    SchedulerConfig config;
    config.max_tokens_per_step = 6;
    config.long_prefill_threshold = 5;
    Scheduler scheduler(config);
    scheduler.add_request(0, prompt(20, 1), 8, kEos);
    scheduler.add_request(1, prompt(2, 2), 8, kEos);

    const SchedulerOutput first = scheduler.schedule();
    CHECK(scheduled_tokens(first) == (vector<int>{5, 1}));
    finish_step(scheduler, first);

    CHECK(scheduled_tokens(scheduler.schedule()) == (vector<int>{5, 1}));
}

static void test_prefix_cache_hit_on_a_later_request() {
    SchedulerConfig config;
    config.prefix_caching = true;
    Scheduler scheduler(config);

    scheduler.add_request(0, prompt(40, 1), 1, kEos);
    finish_step(scheduler, scheduler.schedule());
    scheduler.schedule();

    scheduler.add_request(1, prompt(40, 1), 1, kEos);
    const SchedulerOutput output = scheduler.schedule();
    CHECK(output.new_ids == vector<int>{1});
    CHECK_EQ(output.prefix_hits[0], 32);
    CHECK(scheduled_tokens(output) == vector<int>{8});
}

int main() {
    test_admits_in_arrival_order_up_to_max_running();
    test_token_budget_chunks_the_prefill();
    test_no_chunking_stops_at_the_first_request_that_does_not_fit();
    test_long_prefill_threshold_only_applies_with_competition();
    test_request_finishes_at_max_tokens();
    test_eos_stops_early();
    test_model_length_caps_output();
    test_a_finish_keeps_the_scheduler_alive_for_one_more_step();
    test_preempts_from_the_back_of_running();
    test_preempted_request_resumes_with_its_output_tokens();
    test_a_request_can_preempt_itself();
    test_budget_is_shared_across_running_requests();
    test_prefix_cache_hit_on_a_later_request();
    return 0;
}
