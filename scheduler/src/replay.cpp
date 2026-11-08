#include "engine.h"
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

void print_step(std::ostream& out, const StepRecord& step) {
    std::vector<int> preempted = step.output.preempted_ids;
    std::sort(preempted.begin(), preempted.end());

    out << '{';
    print_ids(out, "new", step.output.new_ids);
    print_ids(out, "hits", step.output.prefix_hits);
    print_ids(out, "running", step.running_ids);

    out << "\"scheduled\":[";
    const std::vector<ScheduledRequest>& scheduled = step.output.scheduled;
    for (std::size_t i = 0; i < scheduled.size(); ++i) {
        out << (i == 0 ? "" : ",") << '[' << scheduled[i].id << ',' << scheduled[i].num_tokens << ']';
    }
    out << "],";

    print_ids(out, "preempted", preempted);
    print_ids(out, "finished", step.finished_ids);
    out << "\"free_after_schedule\":" << step.free_after_schedule << ",\"free_after_update\":" << step.free_after_update
        << "}\n";
}

} // namespace

// Replay a trace through the engine and print one JSON line per step.
int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: replay <trace-file>\n";
        return 1;
    }

    const Trace trace = read_trace(argv[1]);
    Engine engine(trace.config,
                  [&trace](int id, int output_index) { return trace_token(trace.requests[id], output_index); });
    std::size_t next_arrival = 0;

    for (int iteration = 0; iteration < trace.max_iterations; ++iteration) {
        while (next_arrival < trace.requests.size() && trace.requests[next_arrival].arrival <= iteration) {
            const TraceRequest& request = trace.requests[next_arrival++];
            engine.scheduler().add_request(request.id, request.prompt, request.max_tokens, kEosToken);
        }

        if (!engine.has_work() && next_arrival == trace.requests.size()) {
            break;
        }

        if (const std::optional<StepRecord> step = engine.step()) {
            print_step(std::cout, *step);
        }
    }

    return 0;
}
