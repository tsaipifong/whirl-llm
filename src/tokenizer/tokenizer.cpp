// Byte-level BPE tokenizer (see include/whirl/tokenizer.h).
// SPDX-License-Identifier: Apache-2.0 (this file). The special-token
// partition and the rank-ordered merge loop reproduce llama.cpp's
// tokenizer_st_partition / llm_tokenizer_bpe_session behaviour and are
// adapted from llama.cpp (src/llama-vocab.cpp), MIT License, Copyright (c)
// 2023-2026 The ggml authors -- see THIRD_PARTY_NOTICES.md. The GPT-2
// byte-to-unicode mapping follows the published GPT-2 encoder definition.
// Parity with llama-tokenize is tested by `whirl-parity tokenizer` (tests/parity).

#include "whirl/tokenizer.h"

#include "whirl/common.h"
#include "whirl/gguf.h"
#include "whirl/unicode.h"

#include <algorithm>
#include <queue>

namespace whirl {

namespace {

// GPT-2 byte-to-unicode mapping: printable Latin-1 bytes map to themselves,
// the remaining 68 bytes to U+0100.. in byte order.
std::uint32_t byteToCodePoint(int b) {
    auto printable = [](int x) { return (x >= 0x21 && x <= 0x7E) || (x >= 0xA1 && x <= 0xAC) || (x >= 0xAE && x <= 0xFF); };
    if (printable(b)) return static_cast<std::uint32_t>(b);
    int n = 0;
    for (int x = 0; x < b; ++x)
        if (!printable(x)) ++n;
    return 0x100u + static_cast<std::uint32_t>(n);
}

// names that llama.cpp always treats as control tokens
// value of one hex digit, -1 if c is not [0-9a-fA-F]
int hexDigit(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

// byte value of a "<0xNN>" byte token, -1 if the text is not exactly that form
int byteTokenValue(std::string_view t) {
    if (t.size() != 6 || t.substr(0, 3) != "<0x" || t[5] != '>') return -1;
    const int hi = hexDigit(t[3]), lo = hexDigit(t[4]);
    return hi < 0 || lo < 0 ? -1 : hi * 16 + lo;
}

bool controlLookingName(std::string_view t) {
    static constexpr std::string_view names[] = {
        "<|eot_id|>", "<|im_end|>", "<|end|>", "<end_of_turn>", "<|endoftext|>", "<|end_of_text|>",
        "<EOT>", "_<EOT>", "<|eom_id|>", "<fim-prefix>", "<|fim_prefix|>", "<fim-suffix>", "<|fim_suffix|>",
        "<fim-middle>", "<|fim_middle|>", "<fim-pad>", "<|fim_pad|>", "<|repo_name|>", "<|file_sep|>",
    };
    for (auto n : names)
        if (t == n) return true;
    return false;
}

struct Symbol {
    std::uint32_t start;  // byte offset in the encoded word
    std::uint32_t len;    // bytes; 0 once merged into the left neighbour
    int prev, next;
    TokenId id;  // token of the symbol text, -1 if it is not a token
};

struct Bigram {
    int rank;
    int left, right;
    std::uint32_t left_len, right_len;  // symbol lengths when queued
};

struct BigramOrder {
    // lowest rank first, then leftmost
    bool operator()(const Bigram& a, const Bigram& b) const {
        return a.rank > b.rank || (a.rank == b.rank && a.left > b.left);
    }
};

}  // namespace

Tokenizer Tokenizer::fromGguf(const gguf::File& f) {
    const std::string_view model = f.getString("tokenizer.ggml.model");
    if (model != "gpt2") throw TokenizerError("unsupported tokenizer model '" + std::string(model) + "' (only gpt2)");
    Tokenizer t;
    t.pre_ = std::string(f.getStringOr("tokenizer.ggml.pre", "default"));
    if (t.pre_ != "qwen35") throw TokenizerError("unsupported pre-tokenizer '" + t.pre_ + "' (only qwen35)");

    const auto tokens = f.stringArray("tokenizer.ggml.tokens");
    if (tokens.empty() || tokens.size() > 0x7FFFFFFF) throw TokenizerError("bad vocabulary size");
    std::size_t total = 0;
    for (auto s : tokens) total += s.size();
    t.blob_.reserve(total);
    t.offsets_.reserve(tokens.size() + 1);
    for (auto s : tokens) {
        t.offsets_.push_back(static_cast<std::uint32_t>(t.blob_.size()));
        t.blob_.append(s);
    }
    t.offsets_.push_back(static_cast<std::uint32_t>(t.blob_.size()));

    t.types_.assign(tokens.size(), TokenType::normal);
    if (f.has("tokenizer.ggml.token_type")) {
        const auto types = f.intArray("tokenizer.ggml.token_type");
        if (types.size() != tokens.size()) throw TokenizerError("token_type length does not match tokens");
        for (std::size_t i = 0; i < types.size(); ++i) t.types_[i] = static_cast<TokenType>(types[i]);
    }

    t.by_text_.reserve(tokens.size() * 2);
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        // a later duplicate text does not replace the first id
        t.by_text_.emplace(t.text(static_cast<TokenId>(i)), static_cast<TokenId>(i));
        if (controlLookingName(t.text(static_cast<TokenId>(i))) && t.types_[i] != TokenType::control &&
            t.types_[i] != TokenType::unused)
            t.types_[i] = TokenType::control;
    }
    // byte tokens must be exactly "<0xNN>": piece() decodes them without further checks
    for (std::size_t i = 0; i < tokens.size(); ++i)
        if (t.types_[i] == TokenType::byte && byteTokenValue(tokens[i]) < 0)
            throw TokenizerError("malformed byte token at id " + std::to_string(i) + " (expected <0xNN>)");

    if (f.has("tokenizer.ggml.merges")) {
        const auto merges = f.stringArray("tokenizer.ggml.merges");
        t.n_merges_ = merges.size();
        t.merge_rank_.reserve(merges.size() * 2);
        for (std::size_t i = 0; i < merges.size(); ++i) {
            const std::string_view m = merges[i];
            const std::size_t sp = m.find(' ', 1);
            const std::string_view a = sp == std::string_view::npos ? std::string_view{} : m.substr(0, sp);
            const std::string_view b = sp == std::string_view::npos ? std::string_view{} : m.substr(sp + 1);
            const TokenId ia = t.find(a), ib = t.find(b);
            const int r = static_cast<int>(i);
            if (ia >= 0 && ib >= 0) {
                const std::uint64_t key = (static_cast<std::uint64_t>(ia) << 32) | static_cast<std::uint32_t>(ib);
                t.merge_rank_.emplace(key, r);  // first occurrence keeps its rank
            } else {
                std::string key(a);
                key.push_back(' ');
                key.append(b);
                t.merge_rank_text_.emplace(std::move(key), r);
            }
        }
    }

    for (int b = 0; b < 256; ++b) {
        const std::uint32_t cp = byteToCodePoint(b);
        t.byte_char_[b].clear();
        appendUtf8(t.byte_char_[b], cp);
        t.byte_token_[b] = t.find(t.byte_char_[b]);
        t.char_byte_.emplace(cp, static_cast<std::uint8_t>(b));
    }

    for (std::size_t i = 0; i < tokens.size(); ++i) {
        const TokenType ty = t.types_[i];
        if ((ty == TokenType::control || ty == TokenType::user_defined || ty == TokenType::unknown) &&
            !t.text(static_cast<TokenId>(i)).empty())
            t.specials_.push_back(static_cast<TokenId>(i));
    }
    std::stable_sort(t.specials_.begin(), t.specials_.end(),
                     [&](TokenId a, TokenId b) { return t.text(a).size() > t.text(b).size(); });

    auto optId = [&](const char* key) -> TokenId {
        if (!f.has(key)) return -1;
        const std::uint64_t v = f.getUint(key);
        return v < tokens.size() ? static_cast<TokenId>(v) : -1;
    };
    t.bos_ = optId("tokenizer.ggml.bos_token_id");
    t.eos_ = optId("tokenizer.ggml.eos_token_id");
    t.pad_ = optId("tokenizer.ggml.padding_token_id");
    t.add_bos_ = f.getBoolOpt("tokenizer.ggml.add_bos_token").value_or(false);
    return t;
}

std::string_view Tokenizer::text(TokenId id) const {
    const auto i = static_cast<std::size_t>(id);
    if (id < 0 || i + 1 >= offsets_.size()) throw TokenizerError("token id out of range: " + std::to_string(id));
    return std::string_view(blob_).substr(offsets_[i], offsets_[i + 1] - offsets_[i]);
}

TokenId Tokenizer::find(std::string_view s) const {
    const auto it = by_text_.find(s);
    return it == by_text_.end() ? -1 : it->second;
}

int Tokenizer::rank(TokenId left, TokenId right) const {
    const std::uint64_t key = (static_cast<std::uint64_t>(left) << 32) | static_cast<std::uint32_t>(right);
    const auto it = merge_rank_.find(key);
    return it == merge_rank_.end() ? -1 : it->second;
}

int Tokenizer::rankByText(std::string_view left, std::string_view right) const {
    if (merge_rank_text_.empty()) return -1;
    std::string key(left);
    key.push_back(' ');
    key.append(right);
    const auto it = merge_rank_text_.find(key);
    return it == merge_rank_text_.end() ? -1 : it->second;
}

// BPE over one pre-tokenized word given in byte-level encoding.
void Tokenizer::bpeWord(std::string_view w, std::vector<TokenId>& out) const {
    std::vector<Symbol> sym;
    sym.reserve(w.size());
    for (std::size_t off = 0; off < w.size();) {
        const std::size_t n = std::min(w.size() - off, utf8LenFromLead(static_cast<std::uint8_t>(w[off])));
        const int idx = static_cast<int>(sym.size());
        sym.push_back({static_cast<std::uint32_t>(off), static_cast<std::uint32_t>(n), idx - 1,
                       off + n == w.size() ? -1 : idx + 1, find(w.substr(off, n))});
        off += n;
    }

    std::priority_queue<Bigram, std::vector<Bigram>, BigramOrder> queue;
    auto textOf = [&](const Symbol& s) { return w.substr(s.start, s.len); };
    auto tryAdd = [&](int l, int r) {
        if (l < 0 || r < 0) return;
        const Symbol& a = sym[static_cast<std::size_t>(l)];
        const Symbol& b = sym[static_cast<std::size_t>(r)];
        const int rk = (a.id >= 0 && b.id >= 0) ? rank(a.id, b.id) : rankByText(textOf(a), textOf(b));
        if (rk < 0) return;
        queue.push({rk, l, r, a.len, b.len});
    };
    for (int i = 1; i < static_cast<int>(sym.size()); ++i) tryAdd(i - 1, i);

    while (!queue.empty()) {
        const Bigram bg = queue.top();
        queue.pop();
        Symbol& a = sym[static_cast<std::size_t>(bg.left)];
        Symbol& b = sym[static_cast<std::size_t>(bg.right)];
        // stale entry: one side has changed since it was queued
        if (a.len == 0 || b.len == 0 || a.len != bg.left_len || b.len != bg.right_len) continue;
        a.len += b.len;
        b.len = 0;
        a.next = b.next;
        if (b.next >= 0) sym[static_cast<std::size_t>(b.next)].prev = bg.left;
        a.id = find(textOf(a));
        tryAdd(a.prev, bg.left);
        tryAdd(bg.left, a.next);
    }

    for (const Symbol& s : sym) {
        if (s.len == 0) continue;
        if (s.id >= 0) {
            out.push_back(s.id);
            continue;
        }
        // not a token: fall back to the single bytes of the symbol text
        for (char c : textOf(s)) {
            const TokenId id = find(std::string_view(&c, 1));
            if (id >= 0) out.push_back(id);
        }
    }
}

void Tokenizer::encodeFragment(std::string_view text, std::vector<TokenId>& out) const {
    const std::vector<std::uint32_t> cps = unicode::decodeUtf8Lenient(text);
    const std::vector<std::size_t> words = unicode::splitQwen35(cps);
    std::size_t at = 0;
    std::string encoded;
    for (std::size_t len : words) {
        encoded.clear();
        for (std::size_t k = at; k < at + len; ++k) {
            const std::uint32_t cp = cps[k];
            if (cp > unicode::k_max_code_point)
                throw TokenizerError("invalid code point in input (lead byte F5..F7)");
            std::string utf8;
            appendUtf8(utf8, cp);
            for (unsigned char b : utf8) encoded += byte_char_[b];
        }
        at += len;
        bpeWord(encoded, out);
    }
}

std::vector<TokenId> Tokenizer::encode(std::string_view text, bool parse_special) const {
    std::vector<TokenId> out;
    if (text.empty()) return out;

    // fragments: [begin, end) of raw text, or a token id when token >= 0
    struct Frag {
        std::size_t begin, end;
        TokenId token;
    };
    std::vector<Frag> frags{{0, text.size(), -1}};
    for (TokenId sid : specials_) {
        const TokenType ty = types_[static_cast<std::size_t>(sid)];
        if (!parse_special && (ty == TokenType::control || ty == TokenType::unknown)) continue;
        const std::string_view st = this->text(sid);
        std::vector<Frag> next;
        next.reserve(frags.size() + 4);
        for (const Frag& fr : frags) {
            if (fr.token >= 0) {
                next.push_back(fr);
                continue;
            }
            std::size_t from = fr.begin;
            while (true) {
                const std::size_t m = text.substr(0, fr.end).find(st, from);
                if (m == std::string_view::npos) break;
                if (m > from) next.push_back({from, m, -1});
                next.push_back({0, 0, sid});
                from = m + st.size();
            }
            if (from < fr.end) next.push_back({from, fr.end, -1});
        }
        frags.swap(next);
    }

    for (const Frag& fr : frags) {
        if (fr.token >= 0)
            out.push_back(fr.token);
        else
            encodeFragment(text.substr(fr.begin, fr.end - fr.begin), out);
    }
    return out;
}

std::string Tokenizer::piece(TokenId id, bool special) const {
    const TokenType ty = type(id);
    const std::string_view t = text(id);
    switch (ty) {
        case TokenType::control:
        case TokenType::unknown: return special ? std::string(t) : std::string();
        case TokenType::user_defined: return std::string(t);
        case TokenType::normal: {
            std::string r;
            const auto cps = unicode::decodeUtf8Lenient(t);
            for (std::uint32_t cp : cps) {
                const auto it = char_byte_.find(cp);
                if (it != char_byte_.end()) {
                    r.push_back(static_cast<char>(it->second));
                } else {
                    // not a byte-level character: llama.cpp emits a marker
                    std::string u;
                    appendUtf8(u, cp);
                    r += "[UNK_BYTE_0x";
                    static const char* hex = "0123456789abcdef";
                    for (unsigned char c : u) {
                        r.push_back(hex[c >> 4]);
                        r.push_back(hex[c & 15]);
                    }
                    r += std::string(t) + "]";
                }
            }
            return r;
        }
        case TokenType::byte: {
            // <0xXX> byte tokens are not used by byte-level BPE vocabularies
            // (validated at load, so the fallback only guards hand-built tokenizers)
            const int v = byteTokenValue(t);
            return v < 0 ? std::string() : std::string(1, static_cast<char>(v));
        }
        default: return std::string();  // unused / undefined
    }
}

std::string Tokenizer::decode(std::span<const TokenId> ids, bool special) const {
    std::string out;
    for (TokenId id : ids) out += piece(id, special);
    return out;
}

}  // namespace whirl
