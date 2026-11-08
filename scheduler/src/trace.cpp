#include "trace.h"

#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>

Trace read_trace(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("cannot open trace " + path);
    }

    Trace trace;
    SchedulerConfig& config = trace.config;
    int chunked_prefill = 1;
    int prefix_caching = 0;
    int async_scheduling = 0;

    const std::map<std::string, int*> int_fields = {
        {"block_size", &config.block_size},
        {"num_blocks", &config.num_blocks},
        {"max_tokens_per_step", &config.max_tokens_per_step},
        {"max_running_requests", &config.max_running_requests},
        {"max_model_len", &config.max_model_len},
        {"long_prefill_threshold", &config.long_prefill_threshold},
        {"prefix_caching", &prefix_caching},
        {"chunked_prefill", &chunked_prefill},
        {"async_scheduling", &async_scheduling},
        {"max_iterations", &trace.max_iterations},
    };

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream fields(line);
        std::string key;
        fields >> key;

        if (key == "request") {
            TraceRequest request;
            int num_prompt_tokens = 0;
            fields >> request.id >> request.arrival >> request.max_tokens >> request.stop_after >>
                request.output_salt >> num_prompt_tokens;
            request.prompt.resize(num_prompt_tokens);
            for (int& token : request.prompt) {
                fields >> token;
            }
            trace.requests.push_back(std::move(request));
        } else if (key == "watermark") {
            fields >> config.watermark;
        } else if (const auto field = int_fields.find(key); field != int_fields.end()) {
            fields >> *field->second;
        } else if (!key.empty()) {
            throw std::runtime_error("unknown trace field " + key);
        }
    }

    config.prefix_caching = prefix_caching != 0;
    config.chunked_prefill = chunked_prefill != 0;
    config.async_scheduling = async_scheduling != 0;
    return trace;
}

int trace_token(const TraceRequest& request, int output_index) {
    if (output_index + 1 == request.stop_after) {
        return kEosToken;
    }
    return 30001 + (request.output_salt * 31 + output_index * 7) % 9973;
}
