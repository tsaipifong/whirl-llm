// Chat-token facts of a vocabulary, shared by whirl.exe (src/cli) and whirl-server (arch025-N15c; was
// src/server/tokens.* and the CLI's own copy in src/cli/cli.h Tok): end-of-turn ids, the think-open
// split of a thinking-mode prompt, the end of a leading system message (the prefill plan's forced
// splits, include/whirl/prefill_schedule.h), the image-end id, and the CR LF -> LF token map for
// n-gram drafting. Written from the Qwen3.x chat format (GGUF tokenizer.chat_template).
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "whirl/tokenizer.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace whirl::chat {

class ChatTokens {
public:
    ChatTokens() = default;
    explicit ChatTokens(const Tokenizer& tok);

    static constexpr std::uint32_t none = 0xffffffffu;
    std::uint32_t im_start = none, im_end = none, endoftext = none, think = none, nl = none, system = none;
    std::uint32_t vision_end = none;  // "<|vision_end|>" (the CLI's image-end split; none = no vision tokens)

    // A thinking-mode generation prompt ends with "<think>" "\n": the prefill
    // stops before that last "\n" (the next turn re-renders the assistant
    // turn as "<think>" "\n\n" ... , so the common prefix ends after
    // "<think>"). Returns that position (N - 1), else 0.
    std::size_t thinkOpenSplit(std::span<const std::uint32_t> toks) const;

    // Prompt starting with "<|im_start|>" "system": the position after the
    // first "<|im_end|>" and the "\n" that follows it. 0 = none.
    std::size_t sysBoundary(std::span<const std::uint32_t> toks) const;
    // The start of the turn after the one starting at `from` (the position after its "<|im_end|>" and
    // newline, where the next "<|im_start|>" is). 0 = none.
    std::size_t nextTurn(std::span<const std::uint32_t> toks, std::size_t from) const;

    // token id -> id of the same text with every CR LF as LF (itself when
    // there is no CR LF or no such token). Normal tokens only: a control /
    // user-defined token never maps or is mapped to (the CLI's former table
    // also looked at those; on the Swift-1.5 and Ornith-1.5 vocabularies both
    // tables are equal row for row, scratch/a25_lane1/crlf_tables.py).
    static std::vector<std::uint32_t> crlfToLf(const Tokenizer& tok);
};

}  // namespace whirl::chat
