#pragma once

#include "kv_cache.h"
#include "request.h"

#include <deque>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

struct SchedulerConfig {
    int block_size = 16;
    int num_blocks = 1000;
    int max_tokens_per_step = 8192;
    int max_running_requests = 16;
    int max_model_len = 8192;
    int long_prefill_threshold = 0;
    double watermark = 0.0;
    bool prefix_caching = false;
    bool chunked_prefill = true;
    bool async_scheduling = false;
};

struct ScheduledRequest {
    int id;
    int num_tokens;
    bool samples_token = false;
};

struct SchedulerOutput {
    std::vector<int> new_ids; // first-time admissions, then resumed requests
    std::vector<int> prefix_hits;
    std::vector<ScheduledRequest> scheduled;
    std::vector<int> preempted_ids;

    int total_tokens() const;
};

// A continuous-batching FCFS scheduler. schedule() builds one step and advances each
// request optimistically, and update_from_output() applies what the GPU returned.
class Scheduler {
  public:
    explicit Scheduler(const SchedulerConfig& config);

    void add_request(int id, std::vector<int> prompt, int max_tokens, int eos_token);
    bool has_requests() const;

    SchedulerOutput schedule();

    // sampled[i] is the token generated for output.scheduled[i], if it generated one.
    // Returns the requests that finished, in scheduling order.
    std::vector<int> update_from_output(const std::vector<ScheduledRequest>& scheduled,
                                        const std::vector<std::optional<int>>& sampled);

    // The request must already be out of running_.
    void preempt(Request& request);

    int num_free_blocks() const { return kv_cache_.num_free_blocks(); }
    std::vector<int> running_ids() const;
    const Request* find(int id) const;
    const KvCache& kv_cache() const { return kv_cache_; }

  private:
    void schedule_running(SchedulerOutput& output, int& budget, int long_prefill_threshold);
    void schedule_waiting(SchedulerOutput& output, int& budget, int long_prefill_threshold);
    void finish_scheduling(SchedulerOutput& output);
    bool has_reached_max_tokens(const Request& request) const;
    bool check_stop(Request& request) const;
    void release(Request& request);

    SchedulerConfig config_;
    KvCache kv_cache_;
    std::unordered_map<int, std::unique_ptr<Request>> requests_;
    std::deque<Request*> waiting_;
    std::vector<Request*> running_;
    bool finished_since_last_schedule_ = false;
};
