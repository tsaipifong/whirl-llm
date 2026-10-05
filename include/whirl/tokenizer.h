// Byte-level BPE tokenizer for the GGUF `gpt2` tokenizer model with the
// `qwen35` pre-tokenizer, producing exactly the token ids of llama.cpp's
// llama-tokenize (no BOS/EOS added).
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace whirl {

namespace gguf {
class File;
}

class TokenizerError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// GGUF tokenizer.ggml.token_type values.
enum class TokenType : std::int32_t {
    undefined = 0,
    normal = 1,
    unknown = 2,
    control = 3,
    user_defined = 4,
    unused = 5,
    byte = 6,
};

using TokenId = std::int32_t;

class Tokenizer {
public:
    // Loads vocab, types and merges from the GGUF metadata. Only
    // tokenizer.ggml.model = "gpt2" with pre = "qwen35" is supported.
    static Tokenizer fromGguf(const gguf::File& f);

    // parse_special = true (llama-tokenize default): CONTROL and
    // USER_DEFINED token texts in `text` become their ids. With false only
    // USER_DEFINED tokens (e.g. <think>) are still matched atomically.
    // Throws TokenizerError for a code point above U+10FFFF (llama.cpp
    // throws as well).
    std::vector<TokenId> encode(std::string_view text, bool parse_special = true) const;

    // Piece of one token: NORMAL tokens are byte-decoded, CONTROL/UNKNOWN
    // tokens give their text only when `special`, USER_DEFINED tokens always
    // give their text, UNUSED tokens give nothing.
    std::string piece(TokenId id, bool special = true) const;
    std::string decode(std::span<const TokenId> ids, bool special = true) const;

    std::size_t size() const { return offsets_.size() - 1; }
    std::string_view text(TokenId id) const;
    TokenType type(TokenId id) const { return types_.at(static_cast<std::size_t>(id)); }
    TokenId find(std::string_view text) const;  // -1 if absent
    TokenId bos() const { return bos_; }
    TokenId eos() const { return eos_; }
    TokenId pad() const { return pad_; }
    bool addBos() const { return add_bos_; }
    std::size_t mergeCount() const { return n_merges_; }
    std::size_t specialCount() const { return specials_.size(); }
    std::span<const TokenId> specials() const { return specials_; }
    TokenId byteToken(int b) const { return byte_token_[b & 255]; }  // -1 if none
    const std::string& pre() const { return pre_; }

private:
    void bpeWord(std::string_view encoded_word, std::vector<TokenId>& out) const;
    void encodeFragment(std::string_view text, std::vector<TokenId>& out) const;
    int rank(TokenId left, TokenId right) const;
    int rankByText(std::string_view left, std::string_view right) const;

    std::string blob_;                  // all token texts back to back
    std::vector<std::uint32_t> offsets_;  // size()+1 offsets into blob_
    std::vector<TokenType> types_;
    std::unordered_map<std::string_view, TokenId> by_text_;
    std::unordered_map<std::uint64_t, int> merge_rank_;  // (left id << 32 | right id) -> rank
    std::unordered_map<std::string, int> merge_rank_text_;  // merges whose halves are not tokens
    std::size_t n_merges_ = 0;
    std::vector<TokenId> specials_;  // CONTROL / USER_DEFINED / UNKNOWN, longest text first
    TokenId byte_token_[256] = {};   // token of each byte's byte-level character (-1 if none)
    std::string byte_char_[256];     // byte-level character (UTF-8) of each byte
    std::unordered_map<std::uint32_t, std::uint8_t> char_byte_;  // inverse, by code point
    TokenId bos_ = -1, eos_ = -1, pad_ = -1;
    bool add_bos_ = false;
    std::string pre_;
};

}  // namespace whirl
