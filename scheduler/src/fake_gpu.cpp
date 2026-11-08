#include "fake_gpu.h"

#include <utility>

FakeGpu::FakeGpu(Sampler sampler) : sampler_(std::move(sampler)), worker_([this] { run(); }) {}

FakeGpu::~FakeGpu() {
    {
        std::lock_guard lock(mutex_);
        stopping_ = true;
    }
    work_available_.notify_one();
    worker_.join();
}

void FakeGpu::submit(std::vector<ScheduledRequest> batch) {
    {
        std::lock_guard lock(mutex_);
        submitted_.push_back(std::move(batch));
    }
    work_available_.notify_one();
}

SampledTokens FakeGpu::wait_for_next() {
    std::unique_lock lock(mutex_);
    batch_finished_.wait(lock, [this] { return !finished_.empty(); });

    SampledTokens tokens = std::move(finished_.front());
    finished_.pop_front();
    return tokens;
}

void FakeGpu::run() {
    while (true) {
        std::vector<ScheduledRequest> batch;
        {
            std::unique_lock lock(mutex_);
            work_available_.wait(lock, [this] { return stopping_ || !submitted_.empty(); });
            if (submitted_.empty()) {
                return;
            }
            batch = std::move(submitted_.front());
            submitted_.pop_front();
        }

        SampledTokens tokens = execute(batch);

        {
            std::lock_guard lock(mutex_);
            finished_.push_back(std::move(tokens));
        }
        batch_finished_.notify_one();
    }
}

SampledTokens FakeGpu::execute(const std::vector<ScheduledRequest>& batch) {
    SampledTokens tokens;
    tokens.reserve(batch.size());

    for (const ScheduledRequest& request : batch) {
        if (request.samples_token) {
            tokens.push_back(sampler_(request.id, num_sampled_[request.id]++));
        } else {
            tokens.push_back(std::nullopt);
        }
    }

    return tokens;
}
