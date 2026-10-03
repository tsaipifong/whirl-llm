// Chat-token facts the server needs from the vocabulary: end-of-turn ids,
// the think-open split of a thinking-mode prompt, the end of a leading
// system message, and the CR LF -> LF token map for n-gram drafting.
// Written from the Qwen3.x chat format (GGUF tokenizer.chat_template).
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "whirl/tokenizer.h"

#include <cstdint>
#include <span>
#include <vector>

namespace whirl::server {

class ChatTokens {
public:
    ChatTokens() = default;
    explicit ChatTokens(const Tokenizer& tok);

    static constexpr std::uint32_t none = 0xffffffffu;
    std::uint32_t im_start = none, im_end = none, endoftext = none, think = none, nl = none, system = none;

    // A thinking-mode generation prompt ends with "<think>" "\n": the prefill
    // stops before that last "\n" (the next turn re-renders the assistant
    // turn as "<think>" "\n\n" ... , so the common prefix ends after
    // "<think>"). Returns that position (N - 1), else 0.
    std::size_t thinkOpenSplit(std::span<const std::uint32_t> toks) const;

    // Prompt starting with "<|im_start|>" "system": the position after the
    // first "<|im_end|>" and the "\n" that follows it. 0 = none.
    std::size_t sysBoundary(std::span<const std::uint32_t> toks) const;

    // token id -> id of the same text with every CR LF as LF (itself when
    // there is no CR LF or no such token).
    static std::vector<std::uint32_t> crlfToLf(const Tokenizer& tok);
};

}  // namespace whirl::server
