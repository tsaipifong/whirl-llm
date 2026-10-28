// WHIRL unit tests of module tokenizer (arch025 N7: moved verbatim from the single tests/unit/unit_tests.cpp; one file
// per module, built alone as whirl-unit-tokenizer-tests and together in whirl-tests).
// SPDX-License-Identifier: Apache-2.0

#include "../test_check.h"
#include "../test_env.h"
#include "whirl/unicode.h"
#include "whirl/tokenizer.h"
#include "whirl/gguf.h"
#include "gguf_builder.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <new>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <psapi.h>  // K32GetProcessMemoryInfo (arch025-surv testPlatformProcSurvivors)
#endif

using namespace whirl;
using whirl::test::g_fail;
using whirl::test::g_pass;
using whirl::test::throws;

namespace {
// ---------------------------------------------------------------------------
void testUnicode() {
    using unicode::decodeUtf8Lenient;
    CHECK(decodeUtf8Lenient("a\xC3\xA9") == (std::vector<std::uint32_t>{'a', 0xE9}));
    // stray continuation, truncated sequence, F8 lead
    CHECK(decodeUtf8Lenient("\x80" "a") == (std::vector<std::uint32_t>{0xFFFD, 'a'}));
    CHECK(decodeUtf8Lenient("\xE4\xB8") == (std::vector<std::uint32_t>{0xFFFD, 0xFFFD}));
    CHECK(decodeUtf8Lenient("\xF8\x80") == (std::vector<std::uint32_t>{0xFFFD, 0xFFFD}));
    // overlong and surrogate encodings are decoded as-is
    CHECK(decodeUtf8Lenient("\xC0\xAF") == (std::vector<std::uint32_t>{0x2F}));
    CHECK(decodeUtf8Lenient("\xED\xA0\x80") == (std::vector<std::uint32_t>{0xD800}));
    CHECK(decodeUtf8Lenient("\xF7\xBF\xBF\xBF") == (std::vector<std::uint32_t>{0x1FFFFF}));

    CHECK_EQ(unicode::flags('a') & unicode::LETTER, unicode::LETTER);
    CHECK_EQ(unicode::flags(' ') & unicode::WHITESPACE, unicode::WHITESPACE);
    CHECK_EQ(unicode::flags(0x3000) & unicode::WHITESPACE, unicode::WHITESPACE);
    CHECK_EQ(unicode::flags(0x4E2D) & unicode::LETTER, unicode::LETTER);
    CHECK_EQ(unicode::flags(0x0301) & unicode::MARK, unicode::MARK);
    CHECK_EQ(unicode::flags(0x0378), unicode::UNDEFINED);
    CHECK_EQ(unicode::flags(0x110000), unicode::UNDEFINED);
    CHECK_EQ(unicode::toLower('S'), static_cast<std::uint32_t>('s'));
    CHECK_EQ(unicode::toLower(0x0130), 0x69u);

    auto split = [](std::string_view s) {
        const auto cps = unicode::decodeUtf8Lenient(s);
        const auto lens = unicode::splitQwen35(cps);
        std::vector<std::string> words;
        std::size_t at = 0;
        for (std::size_t n : lens) {
            std::string w;
            for (std::size_t k = at; k < at + n; ++k) appendUtf8(w, cps[k]);
            words.push_back(w);
            at += n;
        }
        return words;
    };
    CHECK(split("Hello world") == (std::vector<std::string>{"Hello", " world"}));
    CHECK(split("I'm 123") == (std::vector<std::string>{"I", "'m", " ", "1", "2", "3"}));
    CHECK(split("    return x") == (std::vector<std::string>{"   ", " return", " x"}));
    CHECK(split("a  \n\n  b") == (std::vector<std::string>{"a", "  \n\n", " ", " b"}));
    CHECK(split("x = (1);\n") == (std::vector<std::string>{"x", " =", " (", "1", ");\n"}));
    CHECK(split("end  ") == (std::vector<std::string>{"end", "  "}));
}

// ---------------------------------------------------------------------------
// minimal tokenizer metadata (no tensors): vocab {"a", "b", byte_text} with byte_text typed as a byte token
std::vector<std::uint8_t> makeTokenizerGguf(std::string_view byte_text) {
    GgufBuilder g;
    g.put<std::uint32_t>(0x46554747);
    g.put<std::uint32_t>(3);
    g.put<std::uint64_t>(0);  // tensors
    g.put<std::uint64_t>(4);  // kv
    auto key = [&](std::string_view k, gguf::ValueType t) {
        g.str(k);
        g.put<std::uint32_t>(static_cast<std::uint32_t>(t));
    };
    key("tokenizer.ggml.model", gguf::ValueType::string), g.str("gpt2");
    key("tokenizer.ggml.pre", gguf::ValueType::string), g.str("qwen35");
    key("tokenizer.ggml.tokens", gguf::ValueType::array), g.put<std::uint32_t>(8), g.put<std::uint64_t>(3), g.str("a"),
        g.str("b"), g.str(byte_text);
    key("tokenizer.ggml.token_type", gguf::ValueType::array), g.put<std::uint32_t>(5), g.put<std::uint64_t>(3),
        g.put<std::int32_t>(1), g.put<std::int32_t>(1), g.put<std::int32_t>(6);
    while (g.b.size() % 32) g.b.push_back(0);
    return g.b;
}

void testTokenizerByteTokens() {
    auto load = [](std::string_view byte_text) {
        const auto bytes = makeTokenizerGguf(byte_text);
        const gguf::File f = gguf::File::parse(bytes);
        return Tokenizer::fromGguf(f);
    };
    {
        const Tokenizer t = load("<0x41>");
        CHECK_EQ(t.piece(2), std::string("A"));
        CHECK_EQ(t.piece(0), std::string("a"));
    }
    CHECK_EQ(load("<0xfF>").piece(2), std::string("\xFF"));
    CHECK_EQ(load("<0x00>").piece(2), std::string(1, '\0'));
    // malformed byte tokens are rejected at load with a TokenizerError (std::stoi used to throw
    // invalid_argument on decode, or silently mis-decode "<0x-1>" / "<0x1Z>")
    for (std::string_view bad : {"<0xZZ>", "<0x-1>", "<0x1Z>", "<0x+1>", "<0x 1>", "<0x41", "<0x41]", "<0x041>", "0x41>>", ""}) {
        bool tokenizer_error = false;
        try {
            (void)load(bad);
        } catch (const TokenizerError&) {
            tokenizer_error = true;
        } catch (...) {
        }
        CHECK(tokenizer_error);
    }
}

}  // namespace

// the old main()'s calls of this module, in their order (tests/unit/unit_tests.cpp ORDER lists all modules)
const whirl::test::Suite& unitSuite_tokenizer() {
    static const whirl::test::Suite s{"tokenizer",
                                      {
                                          {"testUnicode", [] { testUnicode(); }},
                                          {"testTokenizerByteTokens", [] { testTokenizerByteTokens(); }},
                                      },
                                      nullptr};
    return s;
}

#ifndef WHIRL_UNIT_ALL  // this module's own program (cmake/unit.cmake whirl-unit-*-tests, or an isolated build of this file)
int main(int argc, char** argv) {
    return whirl::test::runSuites({&unitSuite_tokenizer()}, argc, argv, [] { whirl::test::clearWhirlEnv(); });
}
#endif
