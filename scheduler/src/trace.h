#pragma once

#include "scheduler.h"

#include <string>
#include <vector>

inline constexpr int kEosToken = 50256;

struct TraceRequest {
    int id;
    int arrival;
    int max_tokens;
    int stop_after; // the output token that is EOS (1-based), or -1 for never
    int output_salt;
    std::vector<int> prompt;
};

struct Trace {
    SchedulerConfig config;
    int max_iterations = 0;
    std::vector<TraceRequest> requests;
};

Trace read_trace(const std::string& path);

// The fake model's output, a function of the request and how many tokens it has generated.
int trace_token(const TraceRequest& request, int output_index);
