// Response writing (arch024 M6): a request's OpenAI-style response, streamed as SSE chunks while tokens arrive
// or written as one JSON body at the end. Runs on the engine thread (the decode cycle pushes tokens); needs from
// the engine only the model id, the tokenizer (token pieces) and the id generator (ResponseCtx).
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "engine_api.h"
#include "protocol.h"
#include "whirl/thread_annotations.h"
#include "whirl/tokenizer.h"

#include <cstdint>
#include <mutex>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace whirl::server {

// Response / tool-call ids and the default sampling seeds: one mt19937_64 stream seeded with
// EngineOptions::seed, shared by both (the order of draws decides every id and seed), any thread.
class IdGen {
public:
    explicit IdGen(std::uint64_t seed) : prng_(seed) {}
    // prefix + n random [A-Za-z0-9] characters (protocol.h randomId)
    std::string id(std::string_view prefix, std::size_t n);
    // the next raw 64-bit draw (a request's sampling seed when it gave none)
    std::uint64_t next();

private:
    whirl::Mutex mu_;
    std::mt19937_64 prng_ WH_GUARDED_BY(mu_);
};

// What the response writer reads from the engine (all outlive every StreamState).
struct ResponseCtx {
    std::string_view model_name;
    const Tokenizer& tok;
    IdGen& ids;
};

struct Final {
    std::string reason;
    std::size_t reason_len = 0;
    std::size_t content_len = 0;
    std::size_t n_tools = 0;
    std::optional<std::vector<ToolCall>> calls;
    std::string content;
    std::string reasoning;
    // FR-SRV P3-8: why a "length" ended early ("" = max_tokens / context; "kv_pool_exhausted")
    std::string cause;
};

struct StreamState {
    Job* job = nullptr;
    const ResponseCtx* cx = nullptr;
    bool gone = false;
    std::string id;
    std::int64_t created = 0;

    void begin();
    void fail();
    // *n_after: tokens of toks that come after a matched stop string (not counted as completion)
    bool push(Out& out, std::span<const std::uint32_t> toks, std::uint32_t* n_after = nullptr);
    Final finish(Out& out, const std::string& reason);
    void end(Out& out, const Final& f, const Timings& tim, std::uint32_t n_prompt);

private:
    void chunkBegin(std::string& b) const;
    static void chunkEnd(std::string& b, const char* fin_reason);
    void sendDelta(Out& out, const Out::View& v);
    void send(const std::string& b);
};

}  // namespace whirl::server
