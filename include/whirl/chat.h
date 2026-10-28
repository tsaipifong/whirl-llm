// Chat template rendering for the supported qwen35 checkpoints, hand-coded
// from their GGUF `tokenizer.chat_template` (no Jinja interpreter).
// SPDX-License-Identifier: Apache-2.0
//
// Variant `a`: Qwen3.8 / unsloth template (reasoning effort, one leading
// system message, preserve_thinking). Variant `b`: Ornith / Qwen3.6-style
// template (the product GGUF Ornith-1.5-35B-A3B-MXFP4). Variant `b_pyargs`:
// the same renderer for templates that write scalar tool-call arguments with
// `args_value | string` (Qwen3-Coder, Qwen3.5, the original ornith-ai
// Ornith-1.5 Q4_K_M / 9B GGUFs): bool / null arguments become Python `True` /
// `False` / `None`. In `a` and `b` every non-string argument is
// `args_value | tojson` (`true` / `false` / `null`; Python json.dumps
// spacing); containers are tojson in all three.
// Deliberate differences from the Jinja templates (each one is an
// `expect_diverge` case in tests/corpus/template_cases.json, checked by
// `whirl-parity template` against the GGUFs; table in docs/phase1_parity.md):
//   - a system/developer message after the leading ones: `a` renders it as a
//     system turn (the template raises), `b` / `b_pyargs` skip it (as the
//     original Ornith-1.5 template does; the Ornith MXFP4 template raises);
//   - tool_call.function.arguments given as a JSON string is parsed into an
//     object, an empty / whitespace string means no arguments (the product
//     templates raise or drop them);
//   - an unknown role is an error for all variants (the product templates
//     raise "Unexpected message role" too; the original Ornith-1.5 template,
//     `b_pyargs`, silently skips the message);
//   - more lenient than the product templates: for these inputs the official
//     Swift (`a`) / Ornith MXFP4 (`b`) template raises and WHIRL renders a
//     prompt instead (kept on purpose, so clients that send them still work):
//       * the developer role: treated as system (llama.cpp also maps developer
//         to system before rendering; the original Ornith-1.5 template accepts
//         it);
//       * several leading system / developer messages: `a` merges ALL leading
//         ones into one system turn, `b` / `b_pyargs` the first two, one
//         newline between texts, empty texts skipped in `a` (the product
//         templates raise "System message must be at the beginning" /
//         "Unexpected message role"; the original Ornith-1.5 template merges
//         the first two like WHIRL);
//       * reasoning_effort aliases such as "high" / "max" / "ultra" (-> xhigh),
//         "minimal" (-> low) and unknown values (default effort + one warning)
//         in `a` (the template raises "Unexpected reasoning effort");
//       * tool_call.function.arguments as a JSON string (parsed) or an empty /
//         whitespace string (no arguments) (above);
//     and stricter: a tool call without a name is rejected (the templates
//     render `<function=>`);
//   - an empty / whitespace / null leading system message follows each
//     template exactly (bug-emptysystem): `b` renders an empty system turn
//     `<|im_start|>system\n<|im_end|>\n` (the Ornith MXFP4 template, like the
//     Qwen3 / Qwen3.5 ones, does not check the content), `a` and `b_pyargs`
//     render none (their templates test `if content` / `if merged_system`);
//     with tools the empty text adds nothing in all three. Not a divergence;
//     the corpus cases empty_system / null_system / empty_string_system /
//     empty_system_parts / empty_system_tools / empty_system_no_think pin it;
//   - `b` / `b_pyargs` keep the <think> block (reasoning_content or the
//     <think> in content) of EVERY earlier assistant turn and ignore a
//     client's preserve_thinking, like llama.cpp's default
//     preserve_reasoning=true; the product Ornith MXFP4 template keeps it only
//     after the last user query unless preserve_thinking=true. Kept on purpose:
//     the history renders the same bytes the model generated, so the prefix
//     cache / COLD-RESUME is reused across turns (research:think-template-reuse,
//     research\pidle\2026-10-26-think-template-reuse.md). The original
//     ornith-ai Ornith-1.5 template also always keeps it.
// String trimming uses the ASCII whitespace set " \t\n\v\f\r" like
// llama.cpp's Jinja engine (Python's str.strip would also remove Unicode
// spaces such as U+3000).

#pragma once

#include "whirl/json.h"

#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace whirl::chat {

// The text one image stands for in a prompt (an image content part, and one per --image in `whirl chat`): one copy for
// the template rendering and the CLI (arch025 lane 1 leg 7, the vision CLI / server comparison; llama.cpp
// mtmd_default_marker is one string for mtmd-cli and the server too).
inline constexpr std::string_view k_image_placeholder = "<|vision_start|><|image_pad|><|vision_end|>";

enum class TemplateKind { a, b, b_pyargs };

// `a` if the template text mentions reasoning_effort; otherwise `b_pyargs` if
// it contains `args_value is mapping` (scalar arguments via `| string`), else `b`.
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

// Canonical reasoning effort for a client value (ASCII case-insensitive,
// surrounding whitespace ignored): "max" / "ultra" / "xhigh" / "high" -> "xhigh",
// "medium" -> "medium", "low" / "minimal" -> "low", "none" -> "none" (thinking
// off). Anything else -> nullopt.
std::optional<std::string> canonicalEffort(std::string_view value);

// OpenAI-style request fields used for rendering.
struct ChatRequest {
    json::Array messages;
    json::Array tools;  // empty when tool_choice == "none"
    std::string tool_choice = "auto";
    ChatOptions options;            // options.effort is canonical (xhigh / medium / low) when set
    std::vector<std::string> warnings;  // ignored values (e.g. an unknown reasoning effort), for the log
};

// Reads messages, tools, tool_choice and the thinking options, applied in this
// order (a later source overrides an earlier one):
//   chat_template_kwargs {enable_thinking, reasoning_effort, preserve_thinking},
//   reasoning {enabled, effort} (OpenRouter / OpenAI Responses style; a string
//   or boolean `reasoning` is read as the effort / enabled flag),
//   top-level enable_thinking, top-level reasoning_effort.
// Efforts go through canonicalEffort; "none" or reasoning.enabled == false turns
// thinking off, reasoning.enabled == true turns it on. An unknown effort is not
// an error: it is ignored (the default effort applies) and noted in `warnings`.
ChatRequest parseChatRequest(const json::Value& root);

}  // namespace whirl::chat
