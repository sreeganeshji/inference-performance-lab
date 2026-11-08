#pragma once

#include "scheduler.h"

#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>
#include <vector>

using SampledTokens = std::vector<std::optional<int>>;

// A worker thread standing in for the GPU. Batches complete in submission order.
class FakeGpu {
  public:
    using Sampler = std::function<int(int request_id, int output_index)>;

    explicit FakeGpu(Sampler sampler);
    ~FakeGpu();

    void submit(std::vector<ScheduledRequest> batch);
    SampledTokens wait_for_next();

  private:
    void run();
    SampledTokens execute(const std::vector<ScheduledRequest>& batch);

    Sampler sampler_;
    std::unordered_map<int, int> num_sampled_; // only the worker touches this

    std::mutex mutex_;
    std::condition_variable work_available_;
    std::condition_variable batch_finished_;
    std::deque<std::vector<ScheduledRequest>> submitted_;
    std::deque<SampledTokens> finished_;
    bool stopping_ = false;

    std::thread worker_;
};
