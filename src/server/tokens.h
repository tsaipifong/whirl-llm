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
    // feat026-pidle-r2: "</think>", "\n\n" and "assistant" as single tokens (none when the vocabulary splits them)
    std::uint32_t think_end = none, nl2 = none, assistant = none;

    // A thinking-mode generation prompt ends with "<think>" "\n": the prefill
    // stops before that last "\n" (the next turn re-renders the assistant
    // turn as "<think>" "\n\n" ... , so the common prefix ends after
    // "<think>"). Returns that position (N - 1), else 0.
    std::size_t thinkOpenSplit(std::span<const std::uint32_t> toks) const;

    // feat026-pidle-r2: a thinking-mode prompt whose history shows that the client sends the reasoning back: an
    // earlier assistant turn ("<|im_start|>" "assistant" "\n" "<think>", before the generation prompt's own
    // "<think>" at N - 2) whose block is not empty (the empty rendering is "<think>" "\n\n" "</think>"). Then the
    // next prompt will hold this reply as generated (research/pidle/2026-10-26-think-template-reuse.md: FULL prefix).
    // false when the prompt does not end with "<think>" "\n", or any token id is missing.
    bool reasoningInHistory(std::span<const std::uint32_t> toks) const;

    // Prompt starting with "<|im_start|>" "system": the position after the
    // first "<|im_end|>" and the "\n" that follows it. 0 = none.
    std::size_t sysBoundary(std::span<const std::uint32_t> toks) const;
    // The start of the turn after the one starting at `from` (the position after its "<|im_end|>" and
    // newline, where the next "<|im_start|>" is). 0 = none.
    std::size_t nextTurn(std::span<const std::uint32_t> toks, std::size_t from) const;

    // token id -> id of the same text with every CR LF as LF (itself when
    // there is no CR LF or no such token).
    static std::vector<std::uint32_t> crlfToLf(const Tokenizer& tok);
};

}  // namespace whirl::server
