#include "check.h"
#include "engine.h"

#include <chrono>
#include <random>
#include <thread>
#include <vector>

using std::vector;

constexpr int kEos = 50256;

static SchedulerConfig async_config() {
    SchedulerConfig config;
    config.async_scheduling = true;
    return config;
}

static vector<int> prompt(int num_tokens, int token) { return vector<int>(num_tokens, token); }

static SampledTokens sampled_tokens(const SchedulerOutput& output, int token = 100) {
    SampledTokens tokens;
    for (const ScheduledRequest& request : output.scheduled) {
        tokens.push_back(request.samples_token ? std::optional<int>(token) : std::nullopt);
    }
    return tokens;
}

static void test_schedules_ahead_of_the_gpu() {
    Scheduler scheduler(async_config());
    scheduler.add_request(0, prompt(40, 1), 16, kEos);

    scheduler.schedule();
    scheduler.schedule();

    const Request* request = scheduler.find(0);
    CHECK_EQ(request->num_computed_tokens, 41);
    CHECK_EQ(request->num_in_flight_tokens, 41);
    CHECK_EQ(request->num_output_placeholders, 2);
}

// Request 1 holds three blocks and request 0 holds two, with nothing free. Two more
// schedules grow them by one token each, and on the third request 0 needs a new
// block, which preempts request 1 with 41 tokens still in flight.
struct PreemptedInFlight {
    PreemptedInFlight() : scheduler(make_config()) {
        scheduler.add_request(0, prompt(31, 1), 16, kEos);
        scheduler.add_request(1, prompt(40, 2), 16, kEos);
        first = scheduler.schedule();
        second = scheduler.schedule();
        third = scheduler.schedule();
    }

    static SchedulerConfig make_config() {
        SchedulerConfig config = async_config();
        config.num_blocks = 6;
        return config;
    }

    Scheduler scheduler;
    SchedulerOutput first;
    SchedulerOutput second;
    SchedulerOutput third;
};

static void test_preemption_marks_in_flight_tokens_stale() {
    PreemptedInFlight run;
    CHECK(run.third.preempted_ids == vector<int>{1});

    const Request* request = run.scheduler.find(1);
    CHECK(request->status == RequestStatus::Preempted);
    CHECK_EQ(request->num_computed_tokens, 0);
    CHECK_EQ(request->num_in_flight_tokens, 41);
    CHECK_EQ(request->num_stale_output_tokens, 41);
    CHECK_EQ(request->num_output_placeholders, 0);
}

static void test_stale_output_drains_without_touching_the_reset_counters() {
    PreemptedInFlight run;

    run.scheduler.update_from_output(run.first.scheduled, sampled_tokens(run.first));

    const Request* request = run.scheduler.find(1);
    CHECK_EQ(request->num_in_flight_tokens, 1);
    CHECK_EQ(request->num_stale_output_tokens, 1);
    CHECK_EQ(request->num_output_tokens(), 1);
    CHECK_EQ(request->num_computed_tokens, 0);
    CHECK_EQ(request->num_output_placeholders, 0);
}

static void test_preempted_request_waits_for_its_stale_output() {
    PreemptedInFlight run;
    run.scheduler.add_request(2, prompt(1, 3), 16, kEos);

    // Finishing request 0 frees blocks, and only stale output holds request 1 back.
    run.scheduler.update_from_output(run.first.scheduled, SampledTokens{kEos, 100});

    SchedulerOutput output = run.scheduler.schedule();
    CHECK(output.new_ids == vector<int>{2});

    run.scheduler.update_from_output(run.second.scheduled, sampled_tokens(run.second));
    output = run.scheduler.schedule();
    CHECK(output.new_ids == vector<int>{1});
}

static void test_stale_output_can_finish_a_preempted_request() {
    PreemptedInFlight run;

    run.scheduler.update_from_output(run.first.scheduled, sampled_tokens(run.first, kEos));

    CHECK(run.scheduler.find(0) == nullptr);
    CHECK(run.scheduler.find(1) == nullptr);
}

static void test_no_extra_step_once_max_tokens_is_in_flight() {
    Scheduler scheduler(async_config());
    scheduler.add_request(0, prompt(10, 1), 1, kEos);

    CHECK_EQ(scheduler.schedule().scheduled.size(), 1u);
    CHECK_EQ(scheduler.schedule().scheduled.size(), 0u);
}

static void test_engine_runs_a_synchronous_step_in_one_pass() {
    Engine engine(SchedulerConfig{}, [](int, int) { return 100; });
    engine.scheduler().add_request(0, prompt(10, 1), 2, kEos);

    const auto first = engine.step();
    CHECK(first.has_value());
    CHECK(first->output.new_ids == vector<int>{0});
}

static void test_engine_keeps_two_batches_in_flight() {
    Engine engine(async_config(), [](int, int) { return 100; });
    engine.scheduler().add_request(0, prompt(10, 1), 16, kEos);

    CHECK(!engine.step().has_value());

    const auto first = engine.step();
    CHECK(first.has_value());
    CHECK(first->output.new_ids == vector<int>{0});
}

static void test_engine_under_load_with_a_jittery_gpu() {
    SchedulerConfig config = async_config();
    config.num_blocks = 80;
    config.prefix_caching = true;

    Engine engine(config, [](int, int output_index) {
        thread_local std::minstd_rand generator(std::random_device{}());
        std::this_thread::sleep_for(std::chrono::microseconds(generator() % 50));
        return 100 + output_index;
    });

    std::minstd_rand generator(7);
    const int num_requests = 64;
    for (int id = 0; id < num_requests; ++id) {
        const int num_tokens = 8 + static_cast<int>(generator() % 120);
        engine.scheduler().add_request(id, prompt(num_tokens, 1 + id % 5), 8 + id % 16, kEos);
    }

    int num_finished = 0;
    for (int pass = 0; pass < 200000 && engine.has_work(); ++pass) {
        if (const auto step = engine.step()) {
            num_finished += static_cast<int>(step->finished_ids.size());
        }
    }

    CHECK_EQ(num_finished, num_requests);
    CHECK_EQ(engine.scheduler().num_free_blocks(), 79);
}

int main() {
    test_schedules_ahead_of_the_gpu();
    test_preemption_marks_in_flight_tokens_stale();
    test_stale_output_drains_without_touching_the_reset_counters();
    test_preempted_request_waits_for_its_stale_output();
    test_stale_output_can_finish_a_preempted_request();
    test_no_extra_step_once_max_tokens_is_in_flight();
    test_engine_runs_a_synchronous_step_in_one_pass();
    test_engine_keeps_two_batches_in_flight();
    test_engine_under_load_with_a_jittery_gpu();
    return 0;
}
