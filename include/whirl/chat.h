// Chat template rendering for the supported qwen35 checkpoints, hand-coded
// from their GGUF `tokenizer.chat_template` (no Jinja interpreter).
// SPDX-License-Identifier: Apache-2.0
//
// Variant `a`: Qwen3.8 / unsloth template (reasoning effort, merged leading
// system messages, preserve_thinking). Variant `b`: Ornith / Qwen3.6-style
// template. Deliberate differences from the Jinja templates:
//   - a system/developer message after the leading ones: `a` renders it as a
//     system turn (the template raises), `b` skips it (as the template does);
//   - tool_call.function.arguments given as a JSON string is parsed into an
//     object (template `a` raises);
//   - an unknown role is an error for both variants (template `b` skips it).
// String trimming uses the ASCII whitespace set " \t\n\v\f\r" like
// llama.cpp's Jinja engine (Python's str.strip would also remove Unicode
// spaces such as U+3000).

#pragma once

#include "whirl/json.h"

#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace whirl::chat {

enum class TemplateKind { a, b };

// `a` if the template text mentions reasoning_effort, otherwise `b`.
TemplateKind detectTemplate(std::string_view chat_template);
const char* templateName(TemplateKind k);

// Invalid request content; the message is suitable for an HTTP 400 reply.
class RequestError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct ChatOptions {
    bool think = true;                       // enable_thinking
    std::optional<std::string> effort;       // reasoning_effort (variant a)
    std::optional<bool> preserve_thinking;   // variant a
    bool allow_media = false;                // render image/video parts as vision placeholders
};

// Renders messages (+ tools) and the generation prompt.
std::string renderChat(TemplateKind kind, const json::Array& messages, const json::Array& tools,
                       const ChatOptions& opt);

// OpenAI-style request fields used for rendering.
struct ChatRequest {
    json::Array messages;
    json::Array tools;  // empty when tool_choice == "none"
    std::string tool_choice = "auto";
    ChatOptions options;
};

// Reads messages, tools, tool_choice, chat_template_kwargs {enable_thinking,
// reasoning_effort, preserve_thinking} and top-level enable_thinking /
// reasoning_effort (top level wins).
ChatRequest parseChatRequest(const json::Value& root);

}  // namespace whirl::chat
