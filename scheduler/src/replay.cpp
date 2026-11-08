#include "scheduler.h"
#include "trace.h"

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <optional>
#include <vector>

namespace {

void print_ids(std::ostream& out, const char* key, const std::vector<int>& ids) {
    out << '"' << key << "\":[";
    for (std::size_t i = 0; i < ids.size(); ++i) {
        out << (i == 0 ? "" : ",") << ids[i];
    }
    out << "],";
}

void print_step(std::ostream& out, const SchedulerOutput& output, const std::vector<int>& running_ids,
                const std::vector<int>& finished_ids, int free_after_schedule, int free_after_update) {
    std::vector<int> preempted = output.preempted_ids;
    std::sort(preempted.begin(), preempted.end());

    out << '{';
    print_ids(out, "new", output.new_ids);
    print_ids(out, "hits", output.prefix_hits);
    print_ids(out, "running", running_ids);

    out << "\"scheduled\":[";
    for (std::size_t i = 0; i < output.scheduled.size(); ++i) {
        out << (i == 0 ? "" : ",") << '[' << output.scheduled[i].id << ',' << output.scheduled[i].num_tokens << ']';
    }
    out << "],";

    print_ids(out, "preempted", preempted);
    print_ids(out, "finished", finished_ids);
    out << "\"free_after_schedule\":" << free_after_schedule << ",\"free_after_update\":" << free_after_update << "}\n";
}

} // namespace

// Replay a trace through the scheduler and print one JSON line per step.
int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: replay <trace-file>\n";
        return 1;
    }

    const Trace trace = read_trace(argv[1]);
    if (trace.config.async_scheduling) {
        std::cerr << "async traces need the engine\n";
        return 1;
    }

    Scheduler scheduler(trace.config);
    std::vector<int> num_sampled(trace.requests.size(), 0);
    std::size_t next_arrival = 0;

    for (int iteration = 0; iteration < trace.max_iterations; ++iteration) {
        while (next_arrival < trace.requests.size() && trace.requests[next_arrival].arrival <= iteration) {
            const TraceRequest& request = trace.requests[next_arrival++];
            scheduler.add_request(request.id, request.prompt, request.max_tokens, kEosToken);
        }

        if (!scheduler.has_requests()) {
            if (next_arrival == trace.requests.size()) {
                break;
            }
            continue;
        }

        const SchedulerOutput output = scheduler.schedule();
        const std::vector<int> running_ids = scheduler.running_ids();
        const int free_after_schedule = scheduler.num_free_blocks();

        std::vector<std::optional<int>> sampled;
        for (const ScheduledRequest& request : output.scheduled) {
            if (request.samples_token) {
                sampled.push_back(trace_token(trace.requests[request.id], num_sampled[request.id]++));
            } else {
                sampled.push_back(std::nullopt);
            }
        }

        const std::vector<int> finished = scheduler.update_from_output(output.scheduled, sampled);
        print_step(std::cout, output, running_ids, finished, free_after_schedule, scheduler.num_free_blocks());
    }

    return 0;
}
