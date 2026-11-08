#include "scheduler.h"

#include <algorithm>
#include <cassert>
#include <utility>

int SchedulerOutput::total_tokens() const {
    int total = 0;
    for (const ScheduledRequest& request : scheduled) {
        total += request.num_tokens;
    }
    return total;
}

Scheduler::Scheduler(const SchedulerConfig& config)
    : config_(config), kv_cache_(config.num_blocks, config.block_size, config.max_model_len, config.prefix_caching,
                                 static_cast<int>(config.watermark * config.num_blocks)) {}

void Scheduler::add_request(int id, std::vector<int> prompt, int max_tokens, int eos_token) {
    auto request = std::make_unique<Request>(id, std::move(prompt), max_tokens, eos_token);
    kv_cache_.update_block_hashes(*request);
    waiting_.push_back(request.get());
    requests_.emplace(id, std::move(request));
}

bool Scheduler::has_requests() const { return !running_.empty() || !waiting_.empty() || finished_since_last_schedule_; }

std::vector<int> Scheduler::running_ids() const {
    std::vector<int> ids;
    ids.reserve(running_.size());
    for (const Request* request : running_) {
        ids.push_back(request->id);
    }
    return ids;
}

const Request* Scheduler::find(int id) const {
    const auto entry = requests_.find(id);
    return entry == requests_.end() ? nullptr : entry->second.get();
}

SchedulerOutput Scheduler::schedule() {
    SchedulerOutput output;
    int budget = config_.max_tokens_per_step;

    // The cap protects other requests, so a request running alone may use the full budget.
    const bool has_other_requests = running_.size() + waiting_.size() > 1;
    const int long_prefill_threshold = has_other_requests ? config_.long_prefill_threshold : 0;

    schedule_running(output, budget, long_prefill_threshold);

    // Admitting after a preemption would just preempt something else on the next step.
    if (output.preempted_ids.empty()) {
        schedule_waiting(output, budget, long_prefill_threshold);
    }

    finish_scheduling(output);
    return output;
}

void Scheduler::schedule_running(SchedulerOutput& output, int& budget, int long_prefill_threshold) {
    size_t index = 0;
    while (index < running_.size() && budget > 0) {
        Request& request = *running_[index];

        if (has_reached_max_tokens(request)) {
            ++index;
            continue;
        }

        int num_new_tokens = request.num_tokens() + request.num_output_placeholders - request.num_computed_tokens;
        if (long_prefill_threshold > 0 && num_new_tokens > long_prefill_threshold) {
            num_new_tokens = long_prefill_threshold;
        }
        num_new_tokens = std::min({num_new_tokens, budget, config_.max_model_len - request.num_computed_tokens - 1});

        if (num_new_tokens <= 0) {
            ++index;
            continue;
        }

        // Recompute preemption: evict from the back of the running queue until this
        // request fits. If it becomes the victim itself, nothing behind it can run either.
        bool allocated = kv_cache_.allocate(request, num_new_tokens);
        while (!allocated) {
            Request& victim = *running_.back();
            running_.pop_back();
            preempt(victim);
            output.preempted_ids.push_back(victim.id);

            if (&victim == &request) {
                return;
            }
            allocated = kv_cache_.allocate(request, num_new_tokens);
        }

        output.scheduled.push_back({request.id, num_new_tokens});
        budget -= num_new_tokens;
        ++index;
    }
}

void Scheduler::schedule_waiting(SchedulerOutput& output, int& budget, int long_prefill_threshold) {
    std::vector<int> new_ids;
    std::vector<int> new_hits;
    std::vector<int> resumed_ids;
    std::vector<int> resumed_hits;
    std::vector<Request*> deferred;

    while (budget > 0 && !waiting_.empty() && static_cast<int>(running_.size()) < config_.max_running_requests) {
        Request& request = *waiting_.front();

        // Stale output from before this request's preemption is still in flight.
        // Resuming now could resample a position that output is about to deliver.
        if (request.num_stale_output_tokens > 0) {
            deferred.push_back(&request);
            waiting_.pop_front();
            continue;
        }

        const PrefixHit hit = kv_cache_.find_prefix_hit(request);

        int num_new_tokens = request.num_tokens() - hit.num_tokens;
        if (long_prefill_threshold > 0 && num_new_tokens > long_prefill_threshold) {
            num_new_tokens = long_prefill_threshold;
        }
        if (!config_.chunked_prefill && num_new_tokens > budget) {
            break;
        }
        num_new_tokens = std::min(num_new_tokens, budget);

        if (!kv_cache_.admit(request, num_new_tokens, hit, !running_.empty())) {
            break;
        }

        waiting_.pop_front();
        running_.push_back(&request);

        if (request.status == RequestStatus::Waiting) {
            new_ids.push_back(request.id);
            new_hits.push_back(hit.num_tokens);
        } else {
            resumed_ids.push_back(request.id);
            resumed_hits.push_back(hit.num_tokens);
        }

        output.scheduled.push_back({request.id, num_new_tokens});
        budget -= num_new_tokens;
        request.status = RequestStatus::Running;
        request.num_computed_tokens = hit.num_tokens;
    }

    waiting_.insert(waiting_.begin(), deferred.begin(), deferred.end());

    new_ids.insert(new_ids.end(), resumed_ids.begin(), resumed_ids.end());
    new_hits.insert(new_hits.end(), resumed_hits.begin(), resumed_hits.end());
    output.new_ids = std::move(new_ids);
    output.prefix_hits = std::move(new_hits);
}

// Advance num_computed_tokens now so the next step can be built before this one finishes.
void Scheduler::finish_scheduling(SchedulerOutput& output) {
    for (ScheduledRequest& scheduled : output.scheduled) {
        Request& request = *requests_.at(scheduled.id);

        request.num_computed_tokens += scheduled.num_tokens;
        request.num_in_flight_tokens += scheduled.num_tokens;
        request.is_prefill_chunk = request.num_computed_tokens < request.num_tokens() + request.num_output_placeholders;
        scheduled.samples_token = !request.is_prefill_chunk;

        if (config_.async_scheduling && scheduled.samples_token) {
            ++request.num_output_placeholders;
        }
    }

    finished_since_last_schedule_ = false;
}

// Under async scheduling the last step may already be in flight. The +2 assumes every
// other placeholder is a draft token that could be rejected, so this never overshoots.
bool Scheduler::has_reached_max_tokens(const Request& request) const {
    return request.num_output_placeholders > 0 && request.num_computed_tokens + 2 - request.num_output_placeholders >=
                                                      request.num_prompt_tokens + request.max_tokens;
}

void Scheduler::preempt(Request& request) {
    assert(request.status == RequestStatus::Running);

    kv_cache_.free(request);
    request.status = RequestStatus::Preempted;
    request.num_computed_tokens = 0;
    request.num_stale_output_tokens = request.num_in_flight_tokens;
    request.num_output_placeholders = 0;
    waiting_.push_front(&request);
}

std::vector<int> Scheduler::update_from_output(const std::vector<ScheduledRequest>& scheduled,
                                               const std::vector<std::optional<int>>& sampled) {
    std::vector<int> finished;

    for (size_t i = 0; i < scheduled.size(); ++i) {
        const auto entry = requests_.find(scheduled[i].id);
        if (entry == requests_.end()) {
            continue; // finished while this step was in flight
        }
        Request& request = *entry->second;

        request.num_in_flight_tokens -= scheduled[i].num_tokens;
        const bool stale = request.num_stale_output_tokens > 0;
        if (stale) {
            request.num_stale_output_tokens -= scheduled[i].num_tokens;
            assert(request.num_stale_output_tokens >= 0);
        }

        if (!sampled[i]) {
            continue;
        }

        const bool had_placeholders = request.num_output_placeholders > 0;
        const RequestStatus status_before_stop = request.status;

        request.tokens.push_back(*sampled[i]);
        kv_cache_.update_block_hashes(request);
        const bool stopped = check_stop(request);

        if (!stale && had_placeholders) {
            --request.num_output_placeholders;
        }
        if (!stale && had_placeholders && status_before_stop == RequestStatus::Running) {
            kv_cache_.cache_blocks(request, request.num_computed_tokens - request.num_in_flight_tokens);
        }

        if (stopped) {
            finished.push_back(request.id);
            release(request);
        }
    }

    return finished;
}

bool Scheduler::check_stop(Request& request) const {
    if (request.tokens.back() == request.eos_token) {
        request.status = RequestStatus::FinishedStopped;
        return true;
    }
    if (request.num_tokens() >= config_.max_model_len || request.num_output_tokens() >= request.max_tokens) {
        request.status = RequestStatus::FinishedLengthCapped;
        return true;
    }
    return false;
}

void Scheduler::release(Request& request) {
    std::erase(running_, &request);
    std::erase(waiting_, &request);
    kv_cache_.free(request);
    requests_.erase(request.id);
    finished_since_last_schedule_ = true;
}
