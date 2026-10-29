// OpenAI-compatible request / response pieces of the server: sampling
// parameters, the output view (reasoning / content / tool-call region,
// stop strings), tool-call parsing, timings and HTTP / JSON text helpers.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "whirl/chat.h"
#include "whirl/json.h"
#include "whirl/json_fields.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace whirl::server {

using RequestError = chat::RequestError;  // message is the HTTP 400 text

// ---- text helpers

// (one copy with the chat template rendering: whirl/json_fields.h, arch025-N6 lane 1 leg 7)
using chat::trimWs;
using chat::trimLeftWs;
using chat::trimRightWs;
// Longest k in [1, pat.size()) such that text ends with pat[0..k).
std::size_t partialHold(std::string_view text, std::string_view pat);
// Length of the longest prefix of s that does not end inside a UTF-8
// sequence (stray bytes pass through).
std::size_t validUtf8Prefix(std::string_view s);
// Invalid UTF-8 replaced by U+FFFD.
std::string sanitizeUtf8(std::string_view s);
// Streaming: s grew since the complete-text end `from` (a value this function returned before, or 0).
// Returns the new end of the complete text: s up to it is what may be sent.
std::size_t sanitizeUtf8Prefix(std::string& s, std::size_t from);
// "..." with Python json.dumps(ensure_ascii=False) escaping.
void appendJsonStr(std::string& out, std::string_view s);
// Shortest round-trip decimal (no exponent); NaN / inf -> null.
void appendFloat(std::string& out, double v);

// ---- request parameters

struct Params {
    float temperature = 0.6f;
    float top_p = 0.95f;
    std::uint32_t top_k = 20;
    float min_p = 0;
    std::uint64_t seed = 0;
    bool seed_given = false;
    std::optional<std::uint32_t> max_tokens;
    std::vector<std::string> stop;
    float presence_penalty = 0;
    float frequency_penalty = 0;
    bool stream = false;
    bool include_usage = false;
    bool ignore_eos = false;    // llama.cpp extension
    bool cache_prompt = true;   // llama.cpp extension
};

// Stop strings per request and bytes per stop string (more -> HTTP 400)
constexpr std::size_t max_stop_strings = 16;
constexpr std::size_t max_stop_bytes = 256;

// Throws RequestError. `think` / `chat` select the model-card defaults for
// non-thinking chat (temperature 0.7, top_p 0.8).
Params parseParams(const json::Object& obj, bool think, bool chat);

// ---- output view

struct Out {
    bool chat = false;
    bool think_open = false;
    bool tools_on = false;
    std::vector<std::string> stops;
    std::string text;
    std::size_t valid = 0;
    std::size_t sent_reason = 0;
    std::size_t sent_content = 0;

    struct View {
        std::string_view reason;
        std::string_view content;
        std::optional<std::string_view> tool;
        bool stopped = false;
        std::size_t stop_end = 0;  // stopped: byte offset in text just past the matched stop string
    };
    View view(bool final) const;
};

struct ToolCall {
    std::string name;
    std::string args;  // JSON object text (compact)
};

// <tool_call> blocks in the Qwen3.x XML function format or as JSON objects
// {"name", "arguments"}. nullopt when nothing parses.
std::optional<std::vector<ToolCall>> parseToolCalls(std::string_view region, const json::Array& tools);

// The final message of a generation: tool calls (finish_reason tool_calls) or the tool region kept
// as text. reason_in: why the generation ended (stop / length; a matched stop string is "stop").
struct ToolResolution {
    std::optional<std::vector<ToolCall>> calls;
    std::string content;
    std::string reason;
};
ToolResolution resolveTools(const Out::View& v, std::string_view reason_in, const json::Array& tools);

// ---- timings / usage / HTTP

struct Timings {
    std::uint32_t cache_n = 0;
    std::uint32_t prompt_n = 0;
    double prompt_ms = 0;
    std::uint32_t predicted_n = 0;
    double predicted_ms = 0;
    std::uint32_t draft_n = 0;
    std::uint32_t draft_n_accepted = 0;
    void write(std::string& out) const;
};

void writeUsage(std::string& out, std::uint32_t n_prompt, const Timings& t);
const char* statusText(int status);
// Complete HTTP/1.1 response with Content-Length and Connection: close.
std::string httpResponse(int status, std::string_view ctype, std::string_view body);
// extra_headers: complete header lines (each ending in CRLF), e.g. "Retry-After: 5\r\n"
std::string errorResponse(int status, std::string_view msg, std::string_view extra_headers = {});

// Random id: prefix + n chars of [a-zA-Z0-9].
std::string randomId(std::string_view prefix, std::size_t n, std::uint64_t (*next)(void*), void* ctx);

}  // namespace whirl::server
