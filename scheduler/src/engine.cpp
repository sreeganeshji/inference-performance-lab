#include "engine.h"

#include <utility>

Engine::Engine(const SchedulerConfig& config, FakeGpu::Sampler sampler)
    : scheduler_(config), gpu_(std::move(sampler)), queue_depth_(config.async_scheduling ? 2 : 1) {}

std::optional<StepRecord> Engine::step() {
    if (scheduler_.has_requests()) {
        start_step();

        const bool model_executed = in_flight_.back().output.total_tokens() > 0;
        if (in_flight_.size() < queue_depth_ && (model_executed || scheduler_.has_requests())) {
            return std::nullopt;
        }
    } else if (in_flight_.empty()) {
        return std::nullopt;
    }

    return finish_oldest_step();
}

void Engine::start_step() {
    StepRecord record;
    record.output = scheduler_.schedule();
    record.running_ids = scheduler_.running_ids();
    record.free_after_schedule = scheduler_.num_free_blocks();

    gpu_.submit(record.output.scheduled);
    in_flight_.push_back(std::move(record));
}

StepRecord Engine::finish_oldest_step() {
    const SampledTokens tokens = gpu_.wait_for_next();

    StepRecord record = std::move(in_flight_.front());
    in_flight_.pop_front();

    record.finished_ids = scheduler_.update_from_output(record.output.scheduled, tokens);
    record.free_after_update = scheduler_.num_free_blocks();
    return record;
}
