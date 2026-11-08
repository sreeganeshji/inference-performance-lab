#pragma once

#include "fake_gpu.h"
#include "scheduler.h"

#include <cstddef>
#include <deque>
#include <optional>
#include <vector>

struct StepRecord {
    SchedulerOutput output;
    std::vector<int> running_ids;
    std::vector<int> finished_ids;
    int free_after_schedule = 0;
    int free_after_update = 0;
};

// The engine loop. A batch queue one deep is a synchronous step. Two deep means
// the next batch is scheduled while the GPU runs the last one, so the GPU never
// waits on the scheduler.
class Engine {
  public:
    Engine(const SchedulerConfig& config, FakeGpu::Sampler sampler);

    Scheduler& scheduler() { return scheduler_; }
    bool has_work() const { return scheduler_.has_requests() || !in_flight_.empty(); }

    // Returns the step that finished in this pass, or nothing if none did.
    std::optional<StepRecord> step();

  private:
    void start_step();
    StepRecord finish_oldest_step();

    Scheduler scheduler_;
    FakeGpu gpu_;
    std::size_t queue_depth_;
    std::deque<StepRecord> in_flight_;
};
