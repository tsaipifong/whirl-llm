// Unicode code point properties and the qwen35 pre-tokenizer split.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace whirl::unicode {

// Category flags (one per code point) plus the White_Space bit.
enum Flag : std::uint16_t {
    UNDEFINED = 0x0001,  // Cn, and every value >= 0x110000
    NUMBER = 0x0002,     // \p{N}
    LETTER = 0x0004,     // \p{L}
    SEPARATOR = 0x0008,  // \p{Z}
    MARK = 0x0010,       // \p{M}
    PUNCT = 0x0020,      // \p{P}
    SYMBOL = 0x0040,     // \p{S}
    CONTROL = 0x0080,    // Cc Cf Co Cs
    WHITESPACE = 0x0100, // White_Space property
};

constexpr std::uint32_t k_max_code_point = 0x10FFFF;
constexpr std::uint32_t k_replacement = 0xFFFD;

std::uint16_t flags(std::uint32_t cp);
std::uint32_t toLower(std::uint32_t cp);  // simple lowercase mapping
const char* tableVersion();               // Unicode level of the tables

// Lenient UTF-8 decoding with the tokenizer's rules: the sequence length
// comes from the lead byte's high bits and only the continuation-byte
// pattern 10xxxxxx is checked (no overlong / surrogate / range checks). A
// lead byte that does not start a complete sequence becomes U+FFFD and
// consumes one byte. Lead bytes F5..F7 can yield values above U+10FFFF.
std::vector<std::uint32_t> decodeUtf8Lenient(std::string_view s);

// Pre-tokenizer split of one text fragment for `tokenizer.ggml.pre =
// qwen35`, equivalent to the regex
//   (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])
//   |[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
//   |\s*[\r\n]+|\s+(?!\S)|\s+
// with llama.cpp's exact edge-case behaviour. Returns the word lengths in
// code points (they sum to cps.size()).
std::vector<std::size_t> splitQwen35(const std::vector<std::uint32_t>& cps);

}  // namespace whirl::unicode
