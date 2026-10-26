// SPDX-License-Identifier: Apache-2.0

#include "tokens.h"

#include "whirl/prefill_schedule.h"

#include <string>
#include <unordered_map>

namespace whirl::server {

namespace {
std::uint32_t idOf(const Tokenizer& tok, std::string_view text) {
    const TokenId t = tok.find(text);
    return t < 0 ? ChatTokens::none : static_cast<std::uint32_t>(t);
}
}  // namespace

ChatTokens::ChatTokens(const Tokenizer& tok) {
    im_start = idOf(tok, "<|im_start|>");
    im_end = idOf(tok, "<|im_end|>");
    endoftext = idOf(tok, "<|endoftext|>");
    think = idOf(tok, "<think>");
    think_end = idOf(tok, "</think>");
    // "\n" and "system" as single tokens (byte-level text of "\n" is U+010A)
    {
        const auto ids = tok.encode("\n", false);
        if (ids.size() == 1) nl = static_cast<std::uint32_t>(ids[0]);
    }
    {
        const auto ids = tok.encode("system", false);
        if (ids.size() == 1) system = static_cast<std::uint32_t>(ids[0]);
    }
    {
        const auto ids = tok.encode("\n\n", false);
        if (ids.size() == 1) nl2 = static_cast<std::uint32_t>(ids[0]);
    }
    {
        const auto ids = tok.encode("assistant", false);
        if (ids.size() == 1) assistant = static_cast<std::uint32_t>(ids[0]);
    }
}

std::size_t ChatTokens::thinkOpenSplit(std::span<const std::uint32_t> toks) const {
    const std::size_t n = toks.size();
    if (n < 2 || think == none || nl == none) return 0;
    if (toks[n - 2] == think && toks[n - 1] == nl) return n - 1;
    return 0;
}

bool ChatTokens::reasoningInHistory(std::span<const std::uint32_t> toks) const {
    if (thinkOpenSplit(toks) == 0) return false;
    if (im_start == none || assistant == none || think_end == none || nl2 == none) return false;
    const std::size_t gen = toks.size() - 2;  // the generation prompt's "<think>"
    for (std::size_t i = 0; i + 3 < gen; ++i) {
        if (toks[i] != im_start || toks[i + 1] != assistant || toks[i + 2] != nl || toks[i + 3] != think) continue;
        const std::size_t j = i + 3;
        const bool empty = j + 2 < toks.size() && toks[j + 1] == nl2 && toks[j + 2] == think_end;
        if (!empty) return true;
    }
    return false;
}

std::size_t ChatTokens::sysBoundary(std::span<const std::uint32_t> toks) const {
    // the turn ends at "<|im_end|>" "\n" "<|im_start|>": a "<|im_end|>" that the
    // system text itself mentions (parsed as the special token) is not the end
    // (shared with the CLI: include/whirl/prefill_schedule.h, CLI-SCHED)
    return sched::sysBoundary(toks, im_start, system, im_end, nl, none);
}

std::size_t ChatTokens::nextTurn(std::span<const std::uint32_t> toks, std::size_t from) const {
    if (im_start == none || im_end == none) return 0;
    for (std::size_t i = from + 1; i + 2 < toks.size(); ++i) {
        if (toks[i] == im_end && toks[i + 1] == nl && toks[i + 2] == im_start) return i + 2;
    }
    return 0;
}

std::vector<std::uint32_t> ChatTokens::crlfToLf(const Tokenizer& tok) {
    const std::size_t n = tok.size();
    std::vector<std::uint32_t> norm(n);
    std::unordered_map<std::string, std::uint32_t> by_piece;
    std::vector<std::string> pieces(n);
    for (std::size_t i = 0; i < n; ++i) {
        norm[i] = static_cast<std::uint32_t>(i);
        if (tok.type(static_cast<TokenId>(i)) != TokenType::normal) continue;
        pieces[i] = tok.piece(static_cast<TokenId>(i), false);
        by_piece.emplace(pieces[i], static_cast<std::uint32_t>(i));
    }
    for (std::size_t i = 0; i < n; ++i) {
        const std::string& p = pieces[i];
        if (p.find("\r\n") == std::string::npos) continue;
        std::string q;
        q.reserve(p.size());
        for (std::size_t k = 0; k < p.size(); ++k) {
            if (p[k] == '\r' && k + 1 < p.size() && p[k + 1] == '\n') continue;
            q.push_back(p[k]);
        }
        auto it = by_piece.find(q);
        if (it != by_piece.end()) norm[i] = it->second;
    }
    return norm;
}

}  // namespace whirl::server
