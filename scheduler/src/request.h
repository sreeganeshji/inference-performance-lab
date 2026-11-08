#pragma once

#include <cstdint>
#include <utility>
#include <vector>

using BlockHash = std::uint64_t;

enum class RequestStatus { Waiting, Running, Preempted, FinishedStopped, FinishedLengthCapped };

struct Request {
    Request(int id, std::vector<int> prompt, int max_tokens, int eos_token)
        : id(id), tokens(std::move(prompt)), num_prompt_tokens(static_cast<int>(tokens.size())), max_tokens(max_tokens),
          eos_token(eos_token) {}

    int num_tokens() const { return static_cast<int>(tokens.size()); }
    int num_output_tokens() const { return num_tokens() - num_prompt_tokens; }
    bool is_finished() const { return status >= RequestStatus::FinishedStopped; }

    int id;
    std::vector<int> tokens; // prompt, then generated output
    int num_prompt_tokens;
    int max_tokens;
    int eos_token;

    RequestStatus status = RequestStatus::Waiting;
    int num_computed_tokens = 0;

    // Async scheduling runs ahead of the GPU, so num_computed_tokens is optimistic:
    // placeholders are outputs scheduled but not returned, in-flight tokens are in batches
    // not returned, and stale tokens are the in-flight tokens from before a preemption.
    int num_output_placeholders = 0;
    int num_in_flight_tokens = 0;
    int num_stale_output_tokens = 0;

    std::vector<BlockHash> block_hashes;
    std::vector<int> blocks;
    int num_cached_blocks = 0;
};
