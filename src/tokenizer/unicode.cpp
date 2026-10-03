// Unicode properties and the qwen35 pre-tokenizer (see include/whirl/unicode.h).
// SPDX-License-Identifier: Apache-2.0 (this file); the split routine
// reproduces llama.cpp's unicode_regex_split_custom_qwen35 behaviour and is
// adapted from llama.cpp (src/unicode.cpp), MIT License, Copyright (c)
// 2023-2026 The ggml authors -- see THIRD_PARTY_NOTICES.md.
//
// The property tables come from tools/gen-unicode-tables (own C++ generator,
// Unicode Character Database) and are verified against llama.cpp's
// unicode-data.cpp by `whirl-parity unicode` (tests/parity).

#include "whirl/unicode.h"

#include <algorithm>
#include <vector>

namespace whirl::unicode {

namespace {

struct FlagRange {
    std::uint32_t first;
    std::uint16_t flag;
};

struct CasePair {
    std::uint32_t cp;
    std::uint32_t lower;
};

#include "unicode_tables.inc"

constexpr std::uint32_t k_table_size = 0x110000;

const std::vector<std::uint16_t>& flagTable() {
    static const std::vector<std::uint16_t> table = [] {
        std::vector<std::uint16_t> t(k_table_size, UNDEFINED);
        constexpr std::size_t n = sizeof(k_flag_ranges) / sizeof(k_flag_ranges[0]);
        for (std::size_t i = 0; i + 1 < n; ++i)
            std::fill(t.begin() + k_flag_ranges[i].first, t.begin() + k_flag_ranges[i + 1].first,
                      k_flag_ranges[i].flag);
        for (std::uint32_t cp : k_whitespace) t[cp] |= WHITESPACE;
        return t;
    }();
    return table;
}

}  // namespace

std::uint16_t flags(std::uint32_t cp) {
    return cp < k_table_size ? flagTable()[cp] : static_cast<std::uint16_t>(UNDEFINED);
}

std::uint32_t toLower(std::uint32_t cp) {
    const auto* end = k_lowercase + sizeof(k_lowercase) / sizeof(k_lowercase[0]);
    const auto* it = std::lower_bound(k_lowercase, end, cp, [](const CasePair& p, std::uint32_t v) { return p.cp < v; });
    return (it != end && it->cp == cp) ? it->lower : cp;
}

const char* tableVersion() { return k_unicode_version; }

std::vector<std::uint32_t> decodeUtf8Lenient(std::string_view s) {
    std::vector<std::uint32_t> out;
    out.reserve(s.size());
    const auto* p = reinterpret_cast<const unsigned char*>(s.data());
    const std::size_t n = s.size();
    auto cont = [&](std::size_t i) { return i < n && (p[i] & 0xC0) == 0x80; };
    std::size_t i = 0;
    while (i < n) {
        const unsigned c = p[i];
        if (c < 0x80) {
            out.push_back(c);
            i += 1;
        } else if ((c & 0x40) == 0) {  // stray continuation byte
            out.push_back(k_replacement);
            i += 1;
        } else if ((c & 0x20) == 0) {
            if (cont(i + 1)) {
                out.push_back(((c & 0x1F) << 6) | (p[i + 1] & 0x3F));
                i += 2;
            } else {
                out.push_back(k_replacement);
                i += 1;
            }
        } else if ((c & 0x10) == 0) {
            if (cont(i + 1) && cont(i + 2)) {
                out.push_back(((c & 0x0F) << 12) | ((p[i + 1] & 0x3F) << 6) | (p[i + 2] & 0x3F));
                i += 3;
            } else {
                out.push_back(k_replacement);
                i += 1;
            }
        } else if ((c & 0x08) == 0) {
            if (cont(i + 1) && cont(i + 2) && cont(i + 3)) {
                out.push_back(((c & 0x07) << 18) | ((p[i + 1] & 0x3F) << 12) | ((p[i + 2] & 0x3F) << 6) |
                              (p[i + 3] & 0x3F));
                i += 4;
            } else {
                out.push_back(k_replacement);
                i += 1;
            }
        } else {  // F8..FF
            out.push_back(k_replacement);
            i += 1;
        }
    }
    return out;
}

std::vector<std::size_t> splitQwen35(const std::vector<std::uint32_t>& cps) {
    std::vector<std::size_t> words;
    const std::size_t end = cps.size();
    constexpr std::uint32_t k_none = 0xFFFFFFFF;  // position outside the fragment

    auto cpAt = [&](std::size_t pos) -> std::uint32_t { return pos < end ? cps[pos] : k_none; };
    // outside the fragment every flag (including UNDEFINED) is clear
    auto flAt = [&](std::size_t pos) -> std::uint16_t { return pos < end ? flags(cps[pos]) : 0; };
    auto isLM = [](std::uint16_t f) { return (f & (LETTER | MARK)) != 0; };

    std::size_t prev = 0;
    auto emit = [&](std::size_t to) -> std::size_t {
        const std::size_t len = to - prev;
        if (len > 0) words.push_back(len);
        prev = to;
        return len;
    };

    std::size_t pos = 0;
    while (pos < end) {
        const std::uint32_t cp = cps[pos];
        const std::uint16_t f = flags(cp);

        // contractions: '[sS] '[tT] '[mM] '[dD] '[rR][eE] '[vV][eE] '[lL][lL]
        if (cp == '\'' && pos + 1 < end) {
            const std::uint32_t c1 = toLower(cpAt(pos + 1));
            if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') {
                pos += emit(pos + 2);
                continue;
            }
            if (pos + 2 < end) {
                const std::uint32_t c2 = toLower(cpAt(pos + 2));
                if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) {
                    pos += emit(pos + 3);
                    continue;
                }
            }
        }

        // [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+  (the optional lead may itself be L/M)
        if (!(cp == '\r' || cp == '\n' || (f & NUMBER))) {
            if (isLM(f) || isLM(flAt(pos + 1))) {
                ++pos;
                while (isLM(flAt(pos))) ++pos;
                emit(pos);
                continue;
            }
        }

        // \p{N}
        if (f & NUMBER) {
            emit(++pos);
            continue;
        }

        // " ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*"
        constexpr std::uint16_t k_stop = WHITESPACE | LETTER | MARK | NUMBER;
        std::uint16_t f2 = (cp == ' ') ? flAt(pos + 1) : f;
        if (!(f2 & k_stop) && f != 0) {
            if (cp == ' ') ++pos;
            while (!(f2 & k_stop) && f2 != 0) f2 = flAt(++pos);
            while (cpAt(pos) == '\r' || cpAt(pos) == '\n') ++pos;
            emit(pos);
            continue;
        }

        std::size_t n_ws = 0;
        std::size_t last_nl_end = 0;
        while (flAt(pos + n_ws) & WHITESPACE) {
            const std::uint32_t c = cpAt(pos + n_ws);
            if (c == '\r' || c == '\n') last_nl_end = pos + n_ws + 1;
            ++n_ws;
        }

        // \s*[\r\n]+
        if (last_nl_end > 0) {
            pos = last_nl_end;
            emit(pos);
            continue;
        }

        // \s+(?!\S): keep the last space for the next word, unless the
        // whitespace run reaches the end of the fragment
        if (n_ws > 1 && cpAt(pos + n_ws) != k_none) {
            pos += n_ws - 1;
            emit(pos);
            continue;
        }

        // \s+
        if (n_ws > 0) {
            pos += n_ws;
            emit(pos);
            continue;
        }

        // no branch matched: a single code point
        emit(++pos);
    }
    return words;
}

}  // namespace whirl::unicode
