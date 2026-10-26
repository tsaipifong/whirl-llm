// WHIRL host-side unit tests (no GPU, no model files needed).
// SPDX-License-Identifier: Apache-2.0

#include "../test_env.h"
#include "whirl/chat.h"
#include "whirl/common.h"
#include "whirl/gguf.h"
#include "whirl/json.h"
#include "whirl/tokenizer.h"
#include "whirl/unicode.h"
#include "whirl/numerics.h"
#include "whirl/spec_sample.h"
#include "whirl/relax_accept.h"
#include "whirl/dflash_policy.h"
#include "whirl/device_mem.h"
#include "whirl/uma_mem.h"
#include "whirl/vram_limit.h"
#include "whirl/wddm_watch.h"
#include "whirl/pool_budget.h"
#include "whirl/startup_fit.h"
#include "whirl/gpu_select.h"
#include "whirl/mem_governor.h"
#include "whirl/value_scope.h"
#include "whirl/kernels_abi.h"
#include "hip/alloc_log.h"
#include "hip/launch_log.h"
#if WHIRL_TEST_DLL_DEPS
#include "release/dll_deps.h"
#endif

#include "../../kernels/gemm_small_addr.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <format>
#include <cstdio>
#include <cstring>
#include <optional>
#include <regex>
#include <stdexcept>
#include <functional>
#include <map>
#include <new>
#include <string>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

using namespace whirl;

namespace {

int g_fail = 0, g_pass = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (cond) {                                                              \
            ++g_pass;                                                            \
        } else {                                                                 \
            ++g_fail;                                                            \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);          \
        }                                                                        \
    } while (0)

#define CHECK_EQ(a, b)                                                           \
    do {                                                                         \
        const auto va_ = (a);                                                    \
        const auto vb_ = (b);                                                    \
        if (va_ == vb_) {                                                        \
            ++g_pass;                                                            \
        } else {                                                                 \
            ++g_fail;                                                            \
            std::printf("FAIL %s:%d: %s == %s\n", __FILE__, __LINE__, #a, #b);   \
        }                                                                        \
    } while (0)

template <class F>
bool throws(F f) {
    try {
        f();
    } catch (...) {
        return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// JSON test text with '~' standing for a backslash
std::string J(std::string s) {
    for (auto& c : s)
        if (c == '~') c = static_cast<char>(92);
    return s;
}

void testRelaxAccept() {
    using whirl::relax::Params;
    Params p;
    p.on = true;
    p.k = 2;
    p.alpha = 0.5f;
    const std::int32_t ids[3] = {7, 3, 9};
    const float lg[3] = {1.0f, 2.0f, 2.0f - 0.6f};  // ranks: 3, 9, 7; p9/p3 = e^-0.6 = 0.549
    CHECK(whirl::relax::acceptGreedy(p, ids, lg, 2.0f, 3));     // argmax
    CHECK(whirl::relax::acceptGreedy(p, ids, lg, 2.0f, 9));     // 2nd, ratio 0.549 >= 0.5
    CHECK(!whirl::relax::acceptGreedy(p, ids, lg, 2.0f, 7));    // 3rd
    CHECK(!whirl::relax::acceptGreedy(p, ids, lg, 2.0f, 42));   // not a candidate
    p.alpha = 0.6f;
    CHECK(!whirl::relax::acceptGreedy(p, ids, lg, 2.0f, 9));    // ratio below alpha
    p.alpha = 0.5f;
    p.k = 1;
    CHECK(!whirl::relax::acceptGreedy(p, ids, lg, 2.0f, 9) && whirl::relax::acceptGreedy(p, ids, lg, 2.0f, 3));
    const std::int32_t tid[2] = {5, 4};
    const float tl[2] = {1.0f, 1.0f};  // tie: lower id is the argmax
    CHECK(!whirl::relax::acceptGreedy(p, tid, tl, 1.0f, 5) && whirl::relax::acceptGreedy(p, tid, tl, 1.0f, 4));
    // repetition guard (6-gram)
    {
        const std::uint32_t h[8] = {1, 2, 3, 4, 5, 6, 9, 1};
        const std::uint32_t t1[5] = {2, 3, 4, 5, 6};  // ... 1 2 3 4 5 6 repeats
        const std::uint32_t t2[5] = {2, 3, 4, 5, 7};
        CHECK(whirl::relax::extendsRepeat(h, t1) && !whirl::relax::extendsRepeat(h, t2));
    }
    // typical: min(eps, delta * e^-H)
    CHECK(whirl::relax::acceptTypical(p, 0.1, 0.0) && !whirl::relax::acceptTypical(p, 0.08, 0.0));
    CHECK(whirl::relax::acceptTypical(p, 0.3 * std::exp(-3.0) + 1e-9, 3.0) && !whirl::relax::acceptTypical(p, 0.01, 3.0));
}

void testJson() {
    using json::pythonFloat;
    CHECK_EQ(pythonFloat(1.0), std::string("1.0"));
    CHECK_EQ(pythonFloat(0.1), std::string("0.1"));
    CHECK_EQ(pythonFloat(-2.5), std::string("-2.5"));
    CHECK_EQ(pythonFloat(1e16), std::string("1e+16"));
    CHECK_EQ(pythonFloat(1234567890123456.0), std::string("1234567890123456.0"));
    CHECK_EQ(pythonFloat(1e-4), std::string("0.0001"));
    CHECK_EQ(pythonFloat(1e-5), std::string("1e-05"));
    CHECK_EQ(pythonFloat(1.5e-7), std::string("1.5e-07"));
    CHECK_EQ(pythonFloat(1e100), std::string("1e+100"));
    CHECK_EQ(pythonFloat(123.456), std::string("123.456"));
    CHECK_EQ(pythonFloat(-0.0), std::string("-0.0"));
    CHECK_EQ(pythonFloat(3.14159e20), std::string("3.14159e+20"));

    const json::Value v = json::parse(J(R"( {"b": 1, "a": [true, null, 2.0, "x~n~u00e9~ud83d~ude00"], "c": {"z": -7, "y": 12345678901234567890}} )"));
    CHECK_EQ(json::dumpPython(v),
             std::string("{\"b\": 1, \"a\": [true, null, 2.0, \"x\\n\xC3\xA9\xF0\x9F\x98\x80\"], \"c\": {\"z\": -7, \"y\": 12345678901234567890}}"));
    CHECK_EQ(json::dumpPython(v, true), std::string("{\"b\":1,\"a\":[true,null,2.0,\"x\\n\xC3\xA9\xF0\x9F\x98\x80\"],\"c\":{\"z\":-7,\"y\":12345678901234567890}}"));
    // duplicate keys: first position, last value
    CHECK_EQ(json::dumpPython(json::parse(R"({"k": 1, "j": 2, "k": 3})")), std::string("{\"k\": 3, \"j\": 2}"));
    CHECK_EQ(json::dumpPython(json::parse(J(R"("~u001f~b~f~t~/~~")"))), J(R"("~u001f~b~f~t/~~")"));
    CHECK_EQ(json::parse(J(R"("~ud800x")")).asString(), json::parse(J(R"("~ufffdx")")).asString());
    CHECK(throws([] { json::parse("{\"a\": }"); }));
    CHECK(throws([] { json::parse("[1, 2,]"); }));
    CHECK(throws([] { json::parse("01"); }));
    CHECK(throws([] { json::parse("\"\xFF\""); }));
    CHECK(throws([] { json::parse("\"a\nb\""); }));
    CHECK(throws([] { json::parse("1 2"); }));
}

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
struct GgufBuilder {
    std::vector<std::uint8_t> b;
    template <class T>
    void put(T v) {
        const auto* p = reinterpret_cast<const std::uint8_t*>(&v);
        b.insert(b.end(), p, p + sizeof(T));
    }
    void str(std::string_view s) {
        put<std::uint64_t>(s.size());
        b.insert(b.end(), s.begin(), s.end());
    }
};

std::vector<std::uint8_t> makeGguf(std::uint32_t version, std::uint32_t alignment) {
    GgufBuilder g;
    g.put<std::uint32_t>(0x46554747);
    g.put<std::uint32_t>(version);
    g.put<std::uint64_t>(2);   // tensors
    g.put<std::uint64_t>(15);  // kv
    auto key = [&](std::string_view k, gguf::ValueType t) {
        g.str(k);
        g.put<std::uint32_t>(static_cast<std::uint32_t>(t));
    };
    key("u8", gguf::ValueType::u8), g.put<std::uint8_t>(200);
    key("i8", gguf::ValueType::i8), g.put<std::int8_t>(-5);
    key("u16", gguf::ValueType::u16), g.put<std::uint16_t>(65000);
    key("i16", gguf::ValueType::i16), g.put<std::int16_t>(-30000);
    key("u32", gguf::ValueType::u32), g.put<std::uint32_t>(4000000000u);
    key("i32", gguf::ValueType::i32), g.put<std::int32_t>(-2000000000);
    key("f32", gguf::ValueType::f32), g.put<float>(1.5f);
    key("bool", gguf::ValueType::boolean), g.put<std::uint8_t>(1);
    key("str", gguf::ValueType::string), g.str("hello \xE4\xB8\xAD");
    key("u64", gguf::ValueType::u64), g.put<std::uint64_t>(1ull << 40);
    key("i64", gguf::ValueType::i64), g.put<std::int64_t>(-(1ll << 40));
    key("f64", gguf::ValueType::f64), g.put<double>(-0.25);
    key("arr.str", gguf::ValueType::array), g.put<std::uint32_t>(8), g.put<std::uint64_t>(3), g.str("a"), g.str(""), g.str("ccc");
    key("arr.i32", gguf::ValueType::array), g.put<std::uint32_t>(5), g.put<std::uint64_t>(3), g.put<std::int32_t>(1),
        g.put<std::int32_t>(-2), g.put<std::int32_t>(3);
    key("general.alignment", gguf::ValueType::u32), g.put<std::uint32_t>(alignment);
    // tensors: f32 [4, 2] at 0, q8_0 [64] after it
    g.str("t.f32");
    g.put<std::uint32_t>(2), g.put<std::uint64_t>(4), g.put<std::uint64_t>(2);
    g.put<std::uint32_t>(0), g.put<std::uint64_t>(0);
    g.str("t.q8");
    g.put<std::uint32_t>(1), g.put<std::uint64_t>(64);
    g.put<std::uint32_t>(8), g.put<std::uint64_t>(alignment * ((32 + alignment - 1) / alignment));
    while (g.b.size() % alignment) g.b.push_back(0);
    const std::size_t data = g.b.size();
    for (int i = 0; i < 8; ++i) g.put<float>(static_cast<float>(i));
    while ((g.b.size() - data) % alignment) g.b.push_back(0);
    for (int i = 0; i < 68; ++i) g.b.push_back(static_cast<std::uint8_t>(i));
    return g.b;
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

void testGguf() {
    for (std::uint32_t version : {2u, 3u}) {
        for (std::uint32_t alignment : {32u, 64u}) {
            const auto bytes = makeGguf(version, alignment);
            const gguf::File f = gguf::File::parse(bytes);
            CHECK_EQ(f.version(), version);
            CHECK_EQ(f.alignment(), static_cast<std::uint64_t>(alignment));
            CHECK_EQ(f.dataOffset() % alignment, 0u);
            CHECK_EQ(f.getUint("u8"), 200u);
            CHECK_EQ(f.getInt("i8"), -5);
            CHECK_EQ(f.getUint("u16"), 65000u);
            CHECK_EQ(f.getInt("i16"), -30000);
            CHECK_EQ(f.getUint("u32"), 4000000000u);
            CHECK_EQ(f.getInt("i32"), -2000000000);
            CHECK_EQ(f.getFloat("f32"), 1.5);
            CHECK_EQ(f.getBool("bool"), true);
            CHECK_EQ(f.getString("str"), std::string_view("hello \xE4\xB8\xAD"));
            CHECK_EQ(f.getUint("u64"), 1ull << 40);
            CHECK_EQ(f.getInt("i64"), -(1ll << 40));
            CHECK_EQ(f.getFloat("f64"), -0.25);
            CHECK(f.stringArray("arr.str") == (std::vector<std::string_view>{"a", "", "ccc"}));
            CHECK(f.intArray("arr.i32") == (std::vector<std::int64_t>{1, -2, 3}));
            CHECK_EQ(f.kv().front().key, std::string_view("u8"));
            CHECK(f.find("missing") == nullptr);
            CHECK(throws([&] { (void)f.getString("u8"); }));
            const gguf::TensorInfo* t = f.tensor("t.f32");
            CHECK(t != nullptr);
            if (t) {
                CHECK_EQ(t->nbytes(), 32u);
                const auto d = f.tensorData(*t);
                float x3 = 0;
                std::memcpy(&x3, d.data() + 12, 4);
                CHECK_EQ(x3, 3.0f);
            }
            const gguf::TensorInfo* q = f.tensor("t.q8");
            CHECK(q != nullptr);
            if (q) {
                CHECK_EQ(q->nbytes(), 68u);
                CHECK_EQ(f.tensorData(*q)[67], 67);
            }
        }
    }
    auto bytes = makeGguf(3, 32);
    bytes[0] = 'X';
    CHECK(throws([&] { (void)gguf::File::parse(bytes); }));
    bytes = makeGguf(3, 32);
    bytes[4] = 4;  // version 4
    CHECK(throws([&] { (void)gguf::File::parse(bytes); }));
    bytes = makeGguf(3, 32);
    bytes.resize(bytes.size() - 10);  // q8 tensor truncated
    CHECK(throws([&] { (void)gguf::File::parse(bytes); }));
    bytes = makeGguf(3, 32);
    bytes.resize(100);  // truncated metadata
    CHECK(throws([&] { (void)gguf::File::parse(bytes); }));
    CHECK(gguf::typeTraits(gguf::GgmlType::q4_k)->type_size == 144);
    CHECK(gguf::typeTraits(gguf::GgmlType::mxfp4)->block_size == 32);
    CHECK(gguf::typeTraits(static_cast<gguf::GgmlType>(4)) == nullptr);
}

// One-tensor GGUF v3 image (no metadata): `ne` dims, `type`, `offset`, then
// `payload` zero bytes after the aligned data start. offset_pos receives the
// byte position of the offset field (for patching), data_off the data start.
std::vector<std::uint8_t> oneTensorGguf(std::vector<std::uint64_t> ne, gguf::GgmlType type, std::uint64_t offset, std::size_t payload,
                                        std::size_t* offset_pos = nullptr, std::size_t* data_off = nullptr) {
    GgufBuilder g;
    g.put<std::uint32_t>(0x46554747);
    g.put<std::uint32_t>(3);
    g.put<std::uint64_t>(1);  // tensors
    g.put<std::uint64_t>(0);  // kv
    g.str("t");
    g.put<std::uint32_t>(static_cast<std::uint32_t>(ne.size()));
    for (std::uint64_t n : ne) g.put<std::uint64_t>(n);
    g.put<std::uint32_t>(static_cast<std::uint32_t>(type));
    if (offset_pos) *offset_pos = g.b.size();
    g.put<std::uint64_t>(offset);
    while (g.b.size() % 32) g.b.push_back(0);
    if (data_off) *data_off = g.b.size();
    g.b.resize(g.b.size() + payload, 0);
    return g.b;
}

// error text of gguf::File::parse, "" if it parsed
std::string parseError(const std::vector<std::uint8_t>& bytes) {
    try {
        (void)gguf::File::parse(bytes);
    } catch (const gguf::Error& e) {
        return e.what();
    } catch (const std::exception& e) {
        return std::string("non-gguf exception: ") + e.what();
    }
    return "";
}

bool contains(const std::string& s, const char* sub) { return s.find(sub) != std::string::npos; }

// Malformed tensor directories (review SR-1 #1 / N7): overflowing offsets and
// shapes, header counts the file cannot hold. Each must be a gguf::Error.
void testGgufHardening() {
    using gguf::GgmlType;
    constexpr std::uint64_t u64max = ~0ull;
    // valid one-tensor file: f32 [16] = 64 bytes at offset 0, ending exactly at EOF
    {
        const auto ok = oneTensorGguf({16}, GgmlType::f32, 0, 64);
        CHECK_EQ(parseError(ok), std::string());
        const gguf::File f = gguf::File::parse(ok);
        CHECK(f.tensor("t") != nullptr && f.tensor("t")->nbytes() == 64u && f.tensorData(*f.tensor("t")).size() == 64u);
        // one byte short -> past the end
        auto shortf = ok;
        shortf.pop_back();
        CHECK(contains(parseError(shortf), "extends past the end"));
        // offset 32: last byte at size + 32 - 1
        const auto off32 = oneTensorGguf({16}, GgmlType::f32, 32, 64);
        CHECK(contains(parseError(off32), "extends past the end"));
        const auto off32ok = oneTensorGguf({16}, GgmlType::f32, 32, 96);
        CHECK_EQ(parseError(off32ok), std::string());
    }
    // offset close to 2^64: data_offset + offset + nb wraps to a small value
    {
        std::size_t pos = 0, data = 0;
        auto img = oneTensorGguf({16}, GgmlType::f32, 0, 64, &pos, &data);
        const std::uint64_t evil = 0ull - 32 - static_cast<std::uint64_t>(data);  // 2^64 - 32 - data_offset (32-aligned)
        std::memcpy(img.data() + pos, &evil, 8);
        CHECK(contains(parseError(img), "extends past the end"));
        const std::uint64_t evil2 = u64max - 31;  // 2^64 - 32
        std::memcpy(img.data() + pos, &evil2, 8);
        CHECK(contains(parseError(img), "extends past the end"));
    }
    // review case: q4_K [256, 2^62] (2^70 elements, 144 * 2^62 bytes)
    CHECK(contains(parseError(oneTensorGguf({256, 1ull << 62}, GgmlType::q4_k, 0, 0)), "overflows"));
    // byte size overflows although the element count fits: f32 [2^31, 2^32] = 2^63 elements, 2^65 bytes
    CHECK(contains(parseError(oneTensorGguf({1ull << 31, 1ull << 32}, GgmlType::f32, 0, 0)), "size overflows"));
    // element count overflows: f32 [2^40, 2^40]
    CHECK(contains(parseError(oneTensorGguf({1ull << 40, 1ull << 40}, GgmlType::f32, 0, 0)), "shape overflows"));
    // rows overflow: [32, 2^32, 2^32, 2]
    CHECK(contains(parseError(oneTensorGguf({32, 1ull << 32, 1ull << 32, 2}, GgmlType::f32, 0, 0)), "shape overflows"));
    // a dimension beyond int64
    CHECK(contains(parseError(oneTensorGguf({u64max}, GgmlType::f32, 0, 0)), "dimension that is too large"));
    // an unknown type with a huge shape is still rejected by the shape check
    CHECK(contains(parseError(oneTensorGguf({1ull << 40, 1ull << 40}, static_cast<GgmlType>(4), 0, 0)), "shape overflows"));
    // header counts the file cannot hold (old check: count <= size / 8)
    {
        auto img = makeGguf(3, 32);
        const std::uint64_t n_t = img.size() / 8;  // passes "size / 8", but a tensor entry takes >= 24 bytes
        std::memcpy(img.data() + 8, &n_t, 8);
        CHECK(contains(parseError(img), "implausible header counts"));
        img = makeGguf(3, 32);
        const std::uint64_t n_kv = img.size() / 9;  // a key/value entry takes >= 13 bytes
        std::memcpy(img.data() + 16, &n_kv, 8);
        CHECK(contains(parseError(img), "implausible header counts"));
        img = makeGguf(3, 32);
        const std::uint64_t huge = u64max / 2;
        std::memcpy(img.data() + 8, &huge, 8);
        CHECK(contains(parseError(img), "implausible header counts"));
    }
    // TensorInfo is public: an edited copy must not reach outside the mapping
    {
        const auto ok = oneTensorGguf({16}, GgmlType::f32, 0, 64);
        const gguf::File f = gguf::File::parse(ok);
        gguf::TensorInfo t = *f.tensor("t");
        t.offset = 0ull - 32 - f.dataOffset();
        CHECK(throws([&] { (void)f.tensorData(t); }));
        CHECK(!f.inBounds(t, 64));
        t.offset = 0;
        t.ne[0] = 1ull << 40, t.ne[1] = 1ull << 40;  // nbytes would wrap
        CHECK_EQ(t.nbytes(), 0u);
        CHECK(throws([&] { (void)f.tensorData(t); }));
        t.ne[0] = 16, t.ne[1] = 1;
        CHECK(f.inBounds(t, 64) && !f.inBounds(t, 65));
    }
}

// ---------------------------------------------------------------------------
void testChat() {
    auto render = [](chat::TemplateKind k, const char* req) {
        const auto r = chat::parseChatRequest(json::parse(req));
        return chat::renderChat(k, r.messages, r.tools, r.options);
    };
    CHECK_EQ(render(chat::TemplateKind::b, R"({"messages":[{"role":"user","content":" hi "}]})"),
             std::string("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n"));
    CHECK_EQ(render(chat::TemplateKind::a, R"({"messages":[{"role":"user","content":"hi"}],"enable_thinking":false})"),
             std::string("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    CHECK_EQ(render(chat::TemplateKind::a,
                    R"({"messages":[{"role":"system","content":"S"},{"role":"user","content":"q"}],"reasoning_effort":"medium"})"),
             std::string("<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n<think>\n"));
    // arch024-S11b P3: leading system messages are merged into one system block with one newline between them (the
    // mutation that dropped the newline survived every CPU test and the prompt token pin: no pinned request had two)
    CHECK_EQ(render(chat::TemplateKind::a,
                    R"({"messages":[{"role":"system","content":"A"},{"role":"system","content":"B"},{"role":"user","content":"q"}],"reasoning_effort":"medium"})"),
             std::string("<|im_start|>system\nA\nB<|im_end|>\n<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n<think>\n"));
    CHECK(throws([&] { render(chat::TemplateKind::a, R"({"messages":[]})"); }));
    // reasoning effort: aliases, the OpenRouter-style "reasoning" object, unknown values
    {
        const auto ra = [&](const std::string& extra) {
            return render(chat::TemplateKind::a, (R"({"messages":[{"role":"user","content":"q"}])" + extra + "}").c_str());
        };
        const std::string def = ra(""), xhigh = ra(R"(,"reasoning_effort":"xhigh")"), low = ra(R"(,"reasoning_effort":"low")"),
                          medium = ra(R"(,"reasoning_effort":"medium")"), off = ra(R"(,"enable_thinking":false)");
        CHECK(def == xhigh && low != xhigh && medium != xhigh && medium != low);
        CHECK(xhigh.find("Reasoning effort is set to xhigh") != std::string::npos);
        CHECK(low.find("Reasoning effort is set to low") != std::string::npos);
        for (const char* v : {"high", "max", "ultra", "XHigh", " Max "}) {
            const std::string extra = std::string(",\"reasoning_effort\":\"") + v + "\"";
            CHECK_EQ(ra(extra), xhigh);
        }
        CHECK_EQ(ra(R"(,"reasoning_effort":"minimal")"), low);
        CHECK_EQ(ra(R"(,"reasoning_effort":"MEDIUM")"), medium);
        CHECK_EQ(ra(R"(,"reasoning_effort":"none")"), off);
        CHECK_EQ(ra(R"(,"reasoning":{"effort":"low"})"), low);
        CHECK_EQ(ra(R"(,"reasoning":{"effort":"ultra"})"), xhigh);
        CHECK_EQ(ra(R"(,"reasoning":{"effort":"none"})"), off);
        CHECK_EQ(ra(R"(,"reasoning":{"enabled":false})"), off);
        CHECK_EQ(ra(R"(,"reasoning":{"enabled":true,"effort":"medium"})"), medium);
        CHECK_EQ(ra(R"(,"reasoning":"minimal")"), low);
        CHECK_EQ(ra(R"(,"chat_template_kwargs":{"reasoning_effort":"max"})"), xhigh);
        // later sources override earlier ones: kwargs < reasoning object < top level
        CHECK_EQ(ra(R"(,"chat_template_kwargs":{"reasoning_effort":"low"},"reasoning":{"effort":"medium"})"), medium);
        CHECK_EQ(ra(R"(,"reasoning":{"effort":"low"},"reasoning_effort":"medium")"), medium);
        CHECK_EQ(ra(R"(,"reasoning":{"enabled":false},"enable_thinking":true)"), def);
        // a valid effort does not turn thinking back on
        CHECK_EQ(ra(R"(,"enable_thinking":false,"reasoning_effort":"low")"), off);
        // unknown values: no error, default effort, one warning
        CHECK_EQ(ra(R"(,"reasoning_effort":"turbo")"), def);
        CHECK_EQ(ra(R"(,"reasoning":{"effort":"turbo"})"), def);
        CHECK_EQ(ra(R"(,"reasoning_effort":"low","reasoning":{"effort":"turbo"})"), low);
        const auto r1 = chat::parseChatRequest(json::parse(R"({"messages":[{"role":"user","content":"q"}],"reasoning_effort":"turbo"})"));
        CHECK(r1.warnings.size() == 1 && r1.warnings[0].find("turbo") != std::string::npos && !r1.options.effort);
        const auto r2 = chat::parseChatRequest(json::parse(R"({"messages":[{"role":"user","content":"q"}],"reasoning_effort":"Max"})"));
        CHECK(r2.warnings.empty() && r2.options.effort && *r2.options.effort == "xhigh");
        CHECK(chat::canonicalEffort("minimal") == std::optional<std::string>("low"));
        CHECK(!chat::canonicalEffort("") && !chat::canonicalEffort("hi"));
        // direct renderChat callers still get an error for an unknown value
        chat::ChatOptions bad_opt;
        bad_opt.effort = "turbo";
        const json::Array no_tools;
        CHECK(throws([&] { (void)chat::renderChat(chat::TemplateKind::a, r1.messages, no_tools, bad_opt); }));
    }
    CHECK(throws([&] { render(chat::TemplateKind::b, R"({"messages":[{"role":"robot","content":"q"}]})"); }));
    CHECK(chat::detectTemplate("{% if reasoning_effort %}") == chat::TemplateKind::a);
    CHECK(chat::detectTemplate("{% if x %}") == chat::TemplateKind::b);
}

void testVramLimit() {
    constexpr std::uint64_t MiB = 1ull << 20, GiB = 1ull << 30;
    // WHIRL_VRAM_LIMIT_MB parsing
    CHECK_EQ(vram::parseLimitMb("16384"), 16384ull);
    CHECK_EQ(vram::parseLimitMb(" 15360 "), 15360ull);
    CHECK_EQ(vram::parseLimitMb(""), 0ull);
    CHECK_EQ(vram::parseLimitMb("0"), 0ull);
    CHECK_EQ(vram::parseLimitMb("-1"), 0ull);
    CHECK_EQ(vram::parseLimitMb("16G"), 0ull);
    CHECK_EQ(vram::parseLimitMb("99999999999999"), 0ull);
    CHECK_EQ(vram::parseLimitMb(nullptr), 0ull);

    // clamp: the simulated card keeps other programs' use, loses the missing VRAM
    const std::uint64_t total = 32 * GiB - 300 * MiB, free = total - 1 * GiB;
    const auto c = vram::clampMemInfo(free, total, 16 * GiB);
    CHECK_EQ(c.total, 16 * GiB);
    CHECK_EQ(c.free, 15 * GiB);
    CHECK_EQ(c.total - c.free, total - free);  // used stays the same
    const auto n = vram::clampMemInfo(free, total, 0);
    CHECK(n.free == free && n.total == total);  // no limit: unchanged
    const auto big = vram::clampMemInfo(free, total, 64 * GiB);
    CHECK(big.free == free && big.total == total);  // limit above the card: unchanged
    CHECK_EQ(vram::clampMemInfo(2 * GiB, total, 1 * GiB).free, 0ull);  // floor at 0
    // WDDM budget: lowered by the VRAM the simulated card lacks
    const std::uint64_t budget = 31 * GiB;
    CHECK_EQ(vram::clampBudget(budget, total, 16 * GiB), budget - (total - 16 * GiB));
    CHECK_EQ(vram::clampBudget(budget, total, 0), budget);
    CHECK_EQ(vram::clampBudget(1 * GiB, total, 1 * GiB), 0ull);
    // allocation cap of the process: the simulated card's free VRAM at the first allocation
    CHECK_EQ(vram::processCap(free, total, 16 * GiB), 15 * GiB);

    // counter: over-cap allocations are refused and leave the count unchanged
    vram::Counter k;
    k.setCap(10 * MiB);
    CHECK(k.tryReserve(4 * MiB));
    CHECK(k.tryReserve(6 * MiB));
    CHECK_EQ(k.live(), 10 * MiB);
    CHECK(!k.tryReserve(1));
    CHECK_EQ(k.live(), 10 * MiB);
    k.release(6 * MiB);
    CHECK(!k.tryReserve(7 * MiB));
    CHECK(k.tryReserve(6 * MiB));
    CHECK(!k.tryReserve(~0ull));  // no wrap-around
    k.release(100 * MiB);         // over-release clamps at 0
    CHECK_EQ(k.live(), 0ull);
    vram::Counter u;  // cap 0 = unlimited
    CHECK(u.tryReserve(64 * GiB) && u.live() == 64 * GiB);

    // limit note / override
    vram::setLimitMb(16384);
    CHECK_EQ(vram::limitBytes(), 16384 * MiB);
    CHECK(vram::limitNote() == "VRAM limit 16384 MiB (WHIRL_VRAM_LIMIT_MB)");
    vram::setLimitMb(0);
    CHECK(vram::limitNote().empty());

    // auto-shrink decision. 27B Q3_K_S-like on 16 GB: weights 11.35 GiB, prefill buffers
    // ~0.56 MiB per row (2.3 GiB at 4096), checkpoints 150.6 MiB, q8h 36.1 KiB/token
    vram::FitInput fi;
    fi.avail = 13 * GiB + 700 * MiB;
    fi.weights = 11 * GiB + 358 * MiB;
    fi.fixed = 512 * MiB;
    fi.buf_per_row = 576 * 1024;
    fi.buf_fixed = 64 * MiB;
    fi.ckpt_bytes = 150 * MiB;
    fi.parallel = 1;
    fi.kv_per_token = 36 * 1024;
    fi.pool_min_tokens = 16384;
    fi.batch = 4096;
    fi.n_ck = 4;
    fi.n_spe = 2;
    CHECK_EQ(vram::fixedNeed(fi, 4096, 4, 2), fi.weights + fi.fixed + fi.buf_fixed + fi.buf_per_row * 4096 + 6 * fi.ckpt_bytes);
    const vram::FitPlan p = vram::planFit(fi);
    CHECK(p.reduced && p.fits);
    CHECK_EQ(p.batch, 1280u);  // VRAM-01: never below one 1024 chunk + the 256-row merged tail
    CHECK_EQ(p.n_ck, 2u);
    CHECK_EQ(p.n_spe, 1u);
    CHECK(p.pool_tokens >= 16384);
    CHECK(p.note == "prefill batch 4096 -> 1280, checkpoints per slot 4 -> 2, shared checkpoints 2 -> 1");
    // the same model on 32 GB: nothing changes
    vram::FitInput roomy = fi;
    roomy.avail = 29 * GiB;
    const vram::FitPlan pr = vram::planFit(roomy);
    CHECK(!pr.reduced && pr.fits && pr.batch == 4096 && pr.n_ck == 4 && pr.n_spe == 2 && pr.note.empty());
    // just short of the wanted pool: only the first step (batch 2048)
    vram::FitInput step1 = fi;
    step1.avail = vram::fixedNeed(fi, 4096, 4, 2) + 16000ull * fi.kv_per_token;
    const vram::FitPlan p1 = vram::planFit(step1);
    CHECK(p1.reduced && p1.batch == 2048 && p1.n_ck == 4 && p1.n_spe == 2);
    // user-set batch: only the checkpoints move
    vram::FitInput fixb = fi;
    fixb.batch_fixed = true;
    const vram::FitPlan pb = vram::planFit(fixb);
    CHECK(pb.batch == 4096 && pb.n_ck == 1 && pb.n_spe == 0);
    // both fixed: no change, and it does not fit
    vram::FitInput fixall = fixb;
    fixall.ck_fixed = true;
    fixall.avail = 12 * GiB;
    const vram::FitPlan pf = vram::planFit(fixall);
    CHECK(!pf.reduced && !pf.fits && pf.pool_tokens < 4096);
    // hopeless (weights alone over the budget): every step taken, still does not fit
    vram::FitInput hopeless = fi;
    hopeless.avail = 10 * GiB;
    const vram::FitPlan ph = vram::planFit(hopeless);
    CHECK(ph.reduced && !ph.fits && ph.batch == 1280 && ph.n_ck == 1 && ph.n_spe == 0 && ph.pool_tokens == 0);
    // prefix cache off (0 checkpoints): checkpoint steps change nothing
    vram::FitInput nopc = fi;
    nopc.n_ck = 0;
    nopc.n_spe = 0;
    const vram::FitPlan pn = vram::planFit(nopc);
    CHECK(pn.n_ck == 0 && pn.n_spe == 0 && pn.batch <= 2048);
    // VRAM-01 sweep: whatever the budget, the planned batch is never below the scheduler's largest
    // forward (1024-row chunk + 256-row tail), so a long prompt cannot hit BatchTooLarge
    for (std::uint32_t req : {4096u, 2048u, 1536u, 1280u}) {
        for (std::uint64_t av = 8 * GiB; av <= 30 * GiB; av += 256 * MiB) {
            vram::FitInput sw = fi;
            sw.batch = req;
            sw.avail = av;
            const vram::FitPlan ps = vram::planFit(sw);
            CHECK(ps.batch >= 1280u && ps.batch <= req);
        }
    }
    CHECK_EQ(vram::min_prefill_batch, 1280u);
    // the same floor for a user-set WHIRL_PREFILL_BATCH (REVIEW-RC4-C P2-3)
    CHECK_EQ(vram::prefillBatchFloor(1u), 1280u);
    CHECK_EQ(vram::prefillBatchFloor(512u), 1280u);
    CHECK_EQ(vram::prefillBatchFloor(1279u), 1280u);
    CHECK_EQ(vram::prefillBatchFloor(1280u), 1280u);
    CHECK_EQ(vram::prefillBatchFloor(4096u), 4096u);
}

// server defaults by card size (vram::cardDefaults)
void testCardDefaults() {
    const std::uint64_t GiB = 1ull << 30, MiB = 1ull << 20;
    // R9700 32 GB / RX 7900 XTX 24 GB: unchanged (4 slots, f16 when it fits, no headroom, no log line)
    for (std::uint64_t t : {32 * GiB - 140 * MiB, 24 * GiB, 20 * GiB}) {
        vram::CardInput in;
        in.total = t;
        const vram::CardDefaults d = vram::cardDefaults(in);
        CHECK(!d.small && d.parallel == 4 && !d.parallel_auto && d.headroom == 0 && d.note.empty());
    }
    // 16 GB card (or WHIRL_VRAM_LIMIT_MB=16384 / 14336): 1 slot; no KV preference (BAL-Q8); no extra headroom since
    // 0.2.2.1 (the WDDM margin, 3 GiB with a display attached, covers the desktop)
    for (std::uint64_t t : {16 * GiB, 14336 * MiB, 20 * GiB - 1}) {
        vram::CardInput in;
        in.total = t;
        const vram::CardDefaults d = vram::cardDefaults(in);
        CHECK(d.small && d.parallel == 1 && d.parallel_auto);
        CHECK_EQ(d.headroom, 0ull);
        CHECK(d.note.find("small card") != std::string::npos && d.note.find("--parallel 1") != std::string::npos &&
              d.note.find("KV") == std::string::npos && d.note.find("WDDM budget margin") != std::string::npos);
    }
    vram::CardInput s16;
    s16.total = 16 * GiB;
    // --parallel given: kept (also --parallel 4 explicitly)
    {
        vram::CardInput in = s16;
        in.parallel_arg = 4;
        const vram::CardDefaults d = vram::cardDefaults(in);
        CHECK(d.small && d.parallel == 4 && !d.parallel_auto && d.note.find("--parallel 4 (given)") != std::string::npos);
        in.parallel_arg = 2;
        CHECK_EQ(vram::cardDefaults(in).parallel, 2u);
    }
    // WHIRL_VRAM_HEADROOM_MB overrides, on small and large cards; 0 = none
    {
        vram::CardInput in = s16;
        in.headroom_mb_env = 0;
        CHECK_EQ(vram::cardDefaults(in).headroom, 0ull);
        in.headroom_mb_env = 3072;
        CHECK_EQ(vram::cardDefaults(in).headroom, 3072 * MiB);
        vram::CardInput big;
        big.total = 32 * GiB;
        big.headroom_mb_env = 1024;
        const vram::CardDefaults d = vram::cardDefaults(big);
        CHECK(!d.small && d.parallel == 4 && d.headroom == 1024 * MiB && d.note.find("1024 MiB") != std::string::npos);
    }
    // explicit WHIRL_POOL_RESERVE_MB: no default headroom (WHIRL_VRAM_HEADROOM_MB still adds one)
    {
        vram::CardInput in = s16;
        in.reserve_explicit = true;
        CHECK_EQ(vram::cardDefaults(in).headroom, 0ull);
        in.headroom_mb_env = 512;
        CHECK_EQ(vram::cardDefaults(in).headroom, 512 * MiB);
    }
    // UMA iGPU (8060S) and an unknown size are never "small"
    {
        vram::CardInput in = s16;
        in.uma = true;
        const vram::CardDefaults d = vram::cardDefaults(in);
        CHECK(!d.small && d.parallel == 4 && d.headroom == 0);
        vram::CardInput z;
        CHECK(!vram::cardDefaults(z).small);
    }
}


// numerics modes: parsing, capability table, overrides, precise KV decisions (whirl/numerics.h)
void testNumerics() {
    namespace nu = whirl::numerics;
    using nu::Item;
    using nu::Mode;
    // parsing and resolution
    CHECK(nu::resolve(std::nullopt, std::nullopt).mode == Mode::balance);  // default: balance
    CHECK(nu::resolve(std::nullopt, std::nullopt).items == nu::balance_items);
    CHECK(nu::resolve(std::nullopt, std::nullopt).source == "default");
    // FIX-PS: selftest loads in the default mode (balance) and Model::checkPreciseDecode turns the
    // kernel-switching balance items off itself (fp8_prefill, moe_fp8, out_h16). A new balance
    // item fails here: add its switch to checkPreciseDecode (or the precise selftest measures it).
    CHECK(nu::balance_items ==
          (nu::bit(Item::fp8) | nu::bit(Item::moefp8) | nu::bit(Item::gdnwmma) | nu::bit(Item::h16) | nu::bit(Item::kvq8) | nu::bit(Item::specsample)));
    CHECK(nu::resolve(std::nullopt, std::string("")).mode == Mode::balance);
    CHECK(nu::resolve(std::string("precise"), std::nullopt).mode == Mode::precise);
    CHECK(nu::resolve(std::nullopt, std::string("balance")).mode == Mode::balance);
    CHECK(nu::resolve(std::nullopt, std::string("Balanced")).mode == Mode::balance);
    CHECK(nu::resolve(std::string("precise"), std::string("fast")).mode == Mode::precise);  // command line first
    {
        const nu::Request r = nu::parseMode("balance", "t");
        CHECK(r.items == nu::balance_items && !r.custom && r.has(Item::fp8) && r.has(Item::kvq8) && !r.has(Item::kvq4));
        const nu::Request f = nu::parseMode("fast", "t");
        CHECK(f.items == (nu::balance_items | (nu::fast_only_items & ~nu::fast_optin_items)) && !f.has(Item::i8qk));
        // feat027-i8qk: opt-in fast item (explicit list or all); never in balance
        CHECK(nu::parseMode("fast:kvq4,i8qk", "t").has(Item::i8qk) && nu::parseMode("fast:all", "t").has(Item::i8qk));
        CHECK(!nu::parseMode("balance", "t").has(Item::i8qk));
        const nu::Request c = nu::parseMode("balance:fp8, kvq8", "t");
        CHECK(c.custom && c.items == (nu::bit(Item::fp8) | nu::bit(Item::kvq8)));
        const nu::Request c2 = nu::parseMode("fast:kvq4", "t");
        CHECK(c2.mode == Mode::fast && c2.items == nu::bit(Item::kvq4));
        CHECK(nu::parseMode("precise", "t").items == 0);
    }
    auto throws = [](const char* s) {
        try {
            (void)nu::parseMode(s, "t");
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    CHECK(throws("turbo"));
    CHECK(throws("balance:fp9"));
    CHECK(throws("precise:fp8"));
    CHECK(throws("balance:kvq4"));  // a fast-only item in balance
    // command-line flags
    {
        std::string sp;
        bool nx = false;
        CHECK(nu::modeFlag("--precise", &sp, &nx) && sp == "precise" && !nx);
        CHECK(nu::modeFlag("--balance", &sp, &nx) && sp == "balance");
        CHECK(nu::modeFlag("--balance=fp8,h16", &sp, &nx) && sp == "balance:fp8,h16");
        CHECK(nu::modeFlag("--fast", &sp, &nx) && sp == "fast");
        CHECK(nu::modeFlag("--mode", &sp, &nx) && nx);
        CHECK(nu::modeFlag("--mode=balanced", &sp, &nx) && sp == "balanced" && !nx);
        CHECK(!nu::modeFlag("--fastest", &sp, &nx) && !nu::modeFlag("--ctx", &sp, &nx));
        CHECK(nu::parseMode(sp, "t").mode == Mode::balance);
    }
    // capability table: R9700 (gfx1201) MXFP4 dense / MXFP4 MoE / Q4_K_M dense, 8060S
    nu::Target r97;
    r97.k_gemm8 = r97.k_gemm8_moe = r97.k_gdn_wmma = r97.k_kv_q8v = r97.k_kv_q8h = true;
    nu::Target swift = r97;
    swift.has_mxfp4 = true;
    nu::Target ornith = swift;
    ornith.moe = ornith.moe_mxfp4 = true;
    nu::Target q4 = r97;
    nu::Target s8060;
    s8060.gfx1151 = true;
    s8060.has_mxfp4 = true;
    const nu::Request bal = nu::parseMode("balance", "t"), pre{}, fast = nu::parseMode("fast", "t");
    {
        const nu::Plan p = nu::plan(pre, swift);
        for (const nu::ItemState& st : p.items) CHECK(!st.enabled && !st.requested);
        CHECK(nu::logLine(p).find("numerics: precise - f16/f32") == 0);
        CHECK(nu::logLine(p).find("q8dec") == std::string::npos && nu::logLine(p).find("f16 decode / verify activations") != std::string::npos);
        CHECK(!p.q8dec && nu::propsJson(p, "f16").find("\"decode\":\"f16\",\"always_on\":[]") != std::string::npos);
        CHECK(nu::modeLabel(p) == "precise");
        nu::Plan po = p;
        nu::setQ8dec(po, true, "1");  // user-requested int8 decode in precise: lossy override
        CHECK(po.q8dec && nu::modeLabel(po) == "precise+overrides" && nu::logLine(po).find("WHIRL_Q8DEC=1: q8dec on") != std::string::npos);
        nu::Plan pn = p;
        nu::setQ8dec(pn, false, "0");  // same as the default: no override
        CHECK(pn.overrides.empty() && nu::modeLabel(pn) == "precise");
        const nu::Plan pb = nu::plan(nu::parseMode("balance", "t"), swift);
        CHECK(pb.q8dec && nu::logLine(pb).find("decode: q8dec") != std::string::npos);
    }
    {
        const nu::Plan p = nu::plan(bal, swift);
        CHECK(p.on(Item::fp8) && !p.on(Item::moefp8) && p.on(Item::gdnwmma) && p.on(Item::h16) && p.on(Item::kvq8));
        CHECK(p.items[static_cast<std::size_t>(Item::moefp8)].note == "dense model");
        const nu::Plan po = nu::plan(bal, ornith);  // MODES-MXFP4: MoE balance = fp8 + moefp8 + gdnwmma + h16 + q8dec, f16 KV
        CHECK(po.on(Item::fp8) && po.on(Item::moefp8) && po.on(Item::gdnwmma) && po.on(Item::h16) && !po.on(Item::kvq8));
        CHECK(po.on(Item::specsample) && po.q8dec && po.q8dec_default);
        const nu::Plan pq = nu::plan(bal, q4);
        CHECK(!pq.on(Item::fp8) && pq.on(Item::gdnwmma) && pq.on(Item::h16) && pq.on(Item::kvq8));  // KG-1: dense Q4_K_M on the R9700
        CHECK(pq.items[static_cast<std::size_t>(Item::fp8)].note.find("no MXFP4") != std::string::npos);
        CHECK(!nu::plan(pre, q4).on(Item::gdnwmma) && !nu::plan(pre, q4).on(Item::h16));
        nu::Target q4moe = q4;  // MoE with non-MXFP4 experts: unchanged (f32 paths)
        q4moe.moe = true;
        const nu::Plan pqm = nu::plan(bal, q4moe);
        CHECK(!pqm.on(Item::gdnwmma) && !pqm.on(Item::h16) && pqm.items[static_cast<std::size_t>(Item::h16)].note.find("MoE") != std::string::npos);
        nu::Target q4s = q4;  // Radeon 8060S Q4_K_M: unchanged (no WMMA DeltaNet, no h16 for quantized weights)
        q4s.gfx1151 = true;
        q4s.k_gemm8 = q4s.k_gemm8_moe = q4s.k_gdn_wmma = false;
        const nu::Plan pqs = nu::plan(bal, q4s);
        CHECK(!pqs.on(Item::gdnwmma) && !pqs.on(Item::h16));
        const nu::Plan p8 = nu::plan(bal, s8060);  // never an error: skipped with a reason
        CHECK(!p8.on(Item::fp8) && !p8.on(Item::moefp8) && !p8.on(Item::gdnwmma) && p8.on(Item::h16) && p8.on(Item::kvq8));
        CHECK(p8.items[static_cast<std::size_t>(Item::fp8)].note.find("gfx1151") != std::string::npos);
        CHECK(p8.items[static_cast<std::size_t>(Item::kvq8)].note.rfind("q8", 0) == 0);
        const std::string l = nu::logLine(p8);
        CHECK(l.find("numerics: balance - enabled: h16") == 0 && l.find("skipped: fp8 (") != std::string::npos);
    }
    {
        const nu::Plan p = nu::plan(fast, swift);  // R9700: no q4 KV kernels, the rest not implemented
        CHECK(p.on(Item::fp8) && !p.on(Item::kvq4) && !p.on(Item::a4));
        CHECK(p.items[static_cast<std::size_t>(Item::kvq4)].note.find("8060S only") != std::string::npos);
        CHECK(p.items[static_cast<std::size_t>(Item::a4)].note.find("not yet implemented") != std::string::npos);
        CHECK(p.on(Item::relaxacc) && nu::logLine(p).find("fast runs as balance") == std::string::npos);
        CHECK(!nu::plan(bal, swift).on(Item::relaxacc) && !nu::plan(pre, swift).on(Item::relaxacc));
        CHECK(!nu::plan(fast, ornith).on(Item::relaxacc));  // MoE: skipped (no measured gain)
        {
            // FIX-NUM P3-7: relaxacc needs an MTP head; a run that cannot use it reports it skipped
            nu::Target nomtp = swift;
            nomtp.mtp = false;
            const nu::Plan pn = nu::plan(fast, nomtp);
            CHECK(!pn.on(Item::relaxacc));
            CHECK(pn.items[static_cast<std::size_t>(Item::relaxacc)].note.find("no MTP head") != std::string::npos);
            CHECK(nu::logLine(pn).find("fast runs as balance") != std::string::npos);
            nu::Plan pd = nu::plan(fast, swift);
            nu::skipItem(pd, Item::relaxacc, "--draft dflash");
            CHECK(!pd.on(Item::relaxacc) && nu::logLine(pd).find("relaxacc (--draft dflash)") != std::string::npos);
            CHECK(nu::logLine(pd).find("fast runs as balance") != std::string::npos);
            // FIX-NUM P3-5: kernel-path A/B switches are recorded as overrides
            nu::Plan ps = nu::plan(bal, swift);
            nu::noteSwitches(ps, [](const char*) -> std::optional<std::string> { return std::nullopt; });
            CHECK(ps.overrides.empty() && nu::modeLabel(ps) == "balance");
            nu::noteSwitches(ps, [](const char* n) -> std::optional<std::string> {
                if (std::string(n) == "GEMV_MAX") return std::string("8");
                if (std::string(n) == "FD_SPLITS") return std::string("128");
                if (std::string(n) == "NO_FUSE") return std::string("");
                return std::nullopt;
            });
            CHECK(ps.overrides.size() == 2 && nu::modeLabel(ps) == "balance+overrides");
            CHECK(nu::logLine(ps).find("WHIRL_GEMV_MAX=8") != std::string::npos && nu::logLine(ps).find("WHIRL_FD_SPLITS=128") != std::string::npos);
        }
        nu::Request fk = nu::parseMode("fast:kvq8", "t");  // custom fast list without relaxacc / kvq4
        CHECK(nu::logLine(nu::plan(fk, swift)).find("fast runs as balance") != std::string::npos);
        nu::Target t8 = s8060;
        t8.k_kv_q4 = true;
        const nu::Plan p8 = nu::plan(fast, t8);  // 8060S: kvq4 on
        CHECK(p8.on(Item::kvq4) && p8.on(Item::kvq8) && nu::logLine(p8).find("fast runs as balance") == std::string::npos);
        t8.kv_explicit = true;
        CHECK(!nu::plan(fast, t8).on(Item::kvq4));
        t8.kv_explicit = false;
        CHECK(!nu::plan(bal, t8).on(Item::kvq4));
    }
    // specsample: balance / fast item, off in precise, skipped without an MTP head or the kernel
    {
        CHECK(nu::plan(bal, swift).on(Item::specsample) && nu::plan(bal, ornith).on(Item::specsample));
        CHECK(nu::plan(fast, s8060).on(Item::specsample) && !nu::plan(pre, swift).on(Item::specsample));
        nu::Target nomtp = swift;
        nomtp.mtp = false;
        CHECK(!nu::plan(bal, nomtp).on(Item::specsample) &&
              nu::plan(bal, nomtp).items[static_cast<std::size_t>(Item::specsample)].note == "no MTP head");
        nu::Target nok = swift;
        nok.k_spec_sample = false;
        CHECK(!nu::plan(bal, nok).on(Item::specsample));
        CHECK(nu::parseMode("balance:specsample", "t").items == nu::bit(Item::specsample));
        nu::Plan pp = nu::plan(pre, swift);
        nu::applyOverride(pp, Item::specsample, "SPEC_SAMPLE", "1", true, true);
        CHECK(pp.on(Item::specsample) && pp.lossy_override);
        nu::Plan pb = nu::plan(bal, swift);
        nu::applyOverride(pb, Item::specsample, "SPEC_SAMPLE", "0", false, true);
        CHECK(!pb.on(Item::specsample) && !pb.lossy_override);
    }
    // MODES-MXFP4 mode table (owner 2026-10-16, supersedes MOE-BAL 2026-10-13): MoE balance has the
    // dense balance items (fp8 + moefp8 back); forModel moves nothing; dense unchanged
    {
        CHECK(nu::moe_fast_items == 0);
        const nu::Request rb = nu::forModel(bal, true);
        CHECK(rb.mode == Mode::balance && rb.items == nu::balance_items && rb.moved == 0);
        CHECK(rb.has(Item::fp8) && rb.has(Item::moefp8) && rb.has(Item::gdnwmma) && rb.has(Item::h16) && rb.has(Item::specsample));
        const nu::Request rb2 = nu::forModel(rb, true);  // idempotent
        CHECK(rb2.items == rb.items && rb2.moved == rb.moved);
        CHECK(nu::forModel(bal, false).items == nu::balance_items && nu::forModel(bal, false).moved == 0);  // dense: unchanged
        CHECK(nu::forModel(fast, true).items == fast.items && nu::forModel(fast, true).moved == 0);       // fast: unchanged
        CHECK(nu::forModel(pre, true).items == 0 && nu::forModel(pre, true).moved == 0);
        const nu::Request cust = nu::parseMode("balance:gdnwmma,h16", "t");  // an explicit list is honoured (the old MOE-BAL table)
        CHECK(nu::forModel(cust, true).items == cust.items && nu::forModel(cust, true).moved == 0);
        // R9700 Ornith (MXFP4 MoE): balance enables fp8 / moefp8, nothing is "MoE: fast only"
        const nu::Plan pb = nu::plan(bal, ornith);
        const nu::ItemState& f8 = pb.items[static_cast<std::size_t>(Item::fp8)];
        CHECK(f8.enabled && f8.requested && !f8.moved && pb.on(Item::moefp8));
        for (const nu::ItemState& st : pb.items) CHECK(!st.moved && st.note.find("fast only") == std::string::npos);
        const std::string l = nu::logLine(pb);
        CHECK(l.find("numerics: balance - enabled: fp8, moefp8, gdnwmma, h16") == 0 && l.find("fast only") == std::string::npos &&
              l.find("decode: q8dec") != std::string::npos && l.find("f16 decode / verify") == std::string::npos);
        const std::string j = nu::propsJson(pb, "f16");
        CHECK(j.find("\"enabled\":[\"fp8\",\"moefp8\",\"gdnwmma\",\"h16\",\"specsample\"]") != std::string::npos &&
              j.find("fast only") == std::string::npos);
        // MoE balance vs fast on the R9700: the same items except q4 KV
        nu::Target ornith_q4 = ornith;
        ornith_q4.k_kv_q4 = true;  // R9700 since FAST-1c
        const nu::Plan pf = nu::plan(fast, ornith_q4);
        CHECK(pf.on(Item::fp8) && pf.on(Item::moefp8) && pf.on(Item::gdnwmma) && pf.on(Item::h16) && pf.on(Item::kvq4));
        CHECK(!nu::plan(cust, ornith).on(Item::fp8) && !nu::plan(cust, ornith).on(Item::moefp8));
        // KV: MoE balance f16, MoE fast q4 (where the kernels exist), dense balance q8h
        CHECK(nu::chooseKv(rb, true, nu::KvCaps{true, true, true}, std::nullopt).kv == nu::KvKind::f16);
        CHECK(nu::chooseKv(nu::forModel(fast, true), true, nu::KvCaps{true, true, true}, std::nullopt).kv == nu::KvKind::q4);
        CHECK(nu::chooseKv(nu::forModel(bal, false), false, nu::KvCaps{true, true, true}, std::nullopt).kv == nu::KvKind::q8h);
        // a user override turning fp8 off in MoE balance is a (non-lossy) override; turning it on adds nothing
        nu::Plan po = nu::plan(bal, ornith);
        nu::applyOverride(po, Item::fp8, "FP8", "0", false, true);
        CHECK(!po.on(Item::fp8) && !po.lossy_override);
        // dense: unchanged (MXFP4 dense keeps fp8 in balance; Q4_K_M has no fp8)
        CHECK(nu::plan(bal, swift).on(Item::fp8) && !nu::plan(bal, swift).items[static_cast<std::size_t>(Item::fp8)].moved);
        const nu::Plan pq = nu::plan(bal, q4);
        CHECK(!pq.on(Item::fp8) && pq.on(Item::gdnwmma) && pq.on(Item::h16) && pq.on(Item::kvq8) && !pq.items[0].moved);
        // Radeon 8060S MoE: no fp8 WMMA there, so fp8 / moefp8 stay skipped with the hardware reason and
        // the enabled set is the same as under MOE-BAL (h16, specsample; f16 KV): 8060S MoE balance unchanged
        nu::Target s8m = s8060;
        s8m.moe = s8m.moe_mxfp4 = true;
        const nu::Plan p8 = nu::plan(bal, s8m);
        CHECK(!p8.on(Item::fp8) && !p8.on(Item::moefp8) && !p8.on(Item::gdnwmma) && p8.on(Item::h16) && !p8.on(Item::kvq8) && p8.on(Item::specsample));
        CHECK(p8.items[static_cast<std::size_t>(Item::fp8)].note.find("gfx1151") != std::string::npos);
        CHECK(p8.items[static_cast<std::size_t>(Item::moefp8)].note.find("gfx1151") != std::string::npos);
        std::uint32_t en_old = 0, en_new = 0;  // old = the MOE-BAL table (gdnwmma, h16, specsample)
        const nu::Plan p8o = nu::plan(nu::parseMode("balance:gdnwmma,h16,specsample", "t"), s8m);
        for (std::size_t i = 0; i < nu::n_items; ++i) {
            if (p8o.items[i].enabled) en_old |= 1u << i;
            if (p8.items[i].enabled) en_new |= 1u << i;
        }
        CHECK(en_old == en_new);
        CHECK(p8.q8dec && p8o.q8dec);  // MOE-BAL-3: q8dec on in 8060S MoE balance too
    }
    // MOE-BAL-3 (2026-10-15): q8dec back in MoE balance (MOE-BAL-2 had made it fast-only); the
    // decode activation form depends only on the mode, never on the model type
    {
        CHECK(!nu::q8decDefault(pre, false) && !nu::q8decDefault(pre, true));
        CHECK(nu::q8decDefault(bal, false) && nu::q8decDefault(bal, true));
        CHECK(nu::q8decDefault(fast, false) && nu::q8decDefault(fast, true));
        CHECK(nu::q8decDefault(nu::forModel(bal, true), true));  // after the loader's forModel too
        const nu::Request cust = nu::parseMode("balance:fp8,moefp8,gdnwmma,h16,specsample", "t");
        CHECK(nu::q8decDefault(cust, true));
        nu::Target s8m = s8060;
        s8m.moe = s8m.moe_mxfp4 = true;
        // dense and MoE, R9700 and 8060S, balance and fast: int8 decode, nothing about q8dec skipped
        for (const nu::Target& t : {swift, q4, s8060, ornith, s8m}) {
            for (const nu::Request& r : {bal, fast}) {
                const nu::Plan p = nu::plan(r, t);
                CHECK(p.q8dec && p.q8dec_default && nu::logLine(p).find("decode: q8dec") != std::string::npos);
                CHECK(nu::logLine(p).find("q8dec (MoE") == std::string::npos && nu::logLine(p).find("f16 decode / verify") == std::string::npos);
                const std::string j = nu::propsJson(p, t.moe ? "f16" : "q8h");
                CHECK(j.find("\"decode\":\"q8dec\",\"always_on\":[{\"item\":\"q8dec\"") != std::string::npos &&
                      j.find("{\"item\":\"q8dec\",\"reason\"") == std::string::npos);
                CHECK(json::parse(j).isObject());
                CHECK(nu::modeLabel(p) == (r.mode == nu::Mode::balance ? "balance" : "fast"));
            }
            const nu::Plan pp = nu::plan(pre, t);
            CHECK(!pp.q8dec && !pp.q8dec_default);
        }
        for (const nu::Target& t : {ornith, s8m}) {
            const nu::Plan pb = nu::plan(bal, t);
            // WHIRL_Q8DEC=0 in MoE balance: f16 decode, a (non-lossy) override
            nu::Plan p0 = pb;
            nu::setQ8dec(p0, false, "0");
            CHECK(!p0.q8dec && !p0.lossy_override && p0.overrides.size() == 1 && nu::modeLabel(p0) == "balance");
            CHECK(nu::logLine(p0).find("WHIRL_Q8DEC=0") != std::string::npos);
            nu::Plan p1 = pb;
            nu::setQ8dec(p1, true, "1");  // same as the default: no override
            CHECK(p1.overrides.empty() && nu::modeLabel(p1) == "balance");
        }
        // dense balance WHIRL_Q8DEC=0: off, not lossy (unchanged)
        nu::Plan pd = nu::plan(bal, swift);
        nu::setQ8dec(pd, false, "0");
        CHECK(!pd.q8dec && !pd.lossy_override && pd.overrides.size() == 1 && nu::logLine(pd).find("q8dec (MoE") == std::string::npos);
    }
    // feat027-i8qk-r1: i8qk (int8 attention) is fast-only by the mode standard - precise and balance never run it,
    // not by the mode, an item list or WHIRL_ATTN_I8QK (tune.cpp: applyFastOnlyOverride + mode == fast)
    {
        nu::Target ti = r97;
        ti.has_mxfp4 = true;
        ti.k_attn_i8qk = true;
        const nu::Request rp = nu::parseMode("precise", "t"), rb = nu::parseMode("balance", "t"), rf = nu::parseMode("fast", "t"),
                          ra = nu::parseMode("fast:all", "t");
        // the mode alone
        CHECK(!nu::plan(rp, ti).on(Item::i8qk) && !nu::plan(rb, ti).on(Item::i8qk) && !nu::plan(rf, ti).on(Item::i8qk));
        CHECK(nu::plan(ra, ti).on(Item::i8qk));
        // item lists: precise takes none, balance rejects fast items (incl. all)
        bool thr_p = false, thr_b = false, thr_ba = false;
        try { (void)nu::parseMode("precise:i8qk", "t"); } catch (const std::invalid_argument&) { thr_p = true; }
        try { (void)nu::parseMode("balance:i8qk", "t"); } catch (const std::invalid_argument&) { thr_b = true; }
        try { (void)nu::parseMode("balance:all", "t"); } catch (const std::invalid_argument&) { thr_ba = true; }
        CHECK(thr_p && thr_b && thr_ba);
        // WHIRL_ATTN_I8QK=1 in precise / balance (r2: the moeshx semantics of applyFastOnlyOverride): ignored, the plan
        // is unchanged (item off, no override, label / lossy flag / numerics line identical) and the returned note
        // goes to the load log (tune.cpp)
        for (const nu::Request* r : {&rp, &rb}) {
            nu::Plan pl = nu::plan(*r, ti);
            const std::string before = nu::logLine(pl), label0 = nu::modeLabel(pl);
            const bool lossy0 = pl.lossy_override;
            const std::string ign = nu::applyFastOnlyOverride(pl, Item::i8qk, "ATTN_I8QK", "1", true, true);
            CHECK(!pl.on(Item::i8qk) && pl.lossy_override == lossy0 && nu::modeLabel(pl) == label0);
            CHECK(pl.overrides.empty() && nu::logLine(pl) == before);
            CHECK(ign == std::string("WHIRL_ATTN_I8QK=1 ignored: i8qk is a fast-only item (") + (r == &rp ? "precise" : "balance") +
                             " never takes it)");
            nu::Plan p0 = nu::plan(*r, ti);  // =0: nothing to note
            CHECK(nu::applyFastOnlyOverride(p0, Item::i8qk, "ATTN_I8QK", "0", false, true).empty());
            CHECK(!p0.on(Item::i8qk) && p0.overrides.empty());
        }
        // fast: the switch works both ways (plain fast + =1 -> on; fast:all + =0 -> off); without kernels: off; no note
        nu::Plan pf = nu::plan(rf, ti);
        CHECK(nu::applyFastOnlyOverride(pf, Item::i8qk, "ATTN_I8QK", "1", true, true).empty());
        CHECK(pf.on(Item::i8qk) && pf.overrides.size() == 1 && pf.overrides[0].find("i8qk on") != std::string::npos);
        nu::Plan pa = nu::plan(ra, ti);
        CHECK(nu::applyFastOnlyOverride(pa, Item::i8qk, "ATTN_I8QK", "0", false, true).empty());
        CHECK(!pa.on(Item::i8qk));
        nu::Plan pn = nu::plan(rf, ti);
        CHECK(nu::applyFastOnlyOverride(pn, Item::i8qk, "ATTN_I8QK", "1", true, false).empty());
        CHECK(!pn.on(Item::i8qk));
        // r2 (review N1): i8qk is a fast item, so a fast plan with only i8qk on does not say "fast runs as balance";
        // with i8qk off (plain fast on a target without kvq4 / relaxacc) it still does
        nu::Plan pi = nu::plan(nu::parseMode("fast:i8qk", "t"), ti);
        CHECK(pi.on(Item::i8qk) && !pi.on(Item::kvq4) && !pi.on(Item::relaxacc));
        CHECK(nu::logLine(pi).find("fast runs as balance") == std::string::npos);
        nu::Plan pi0 = nu::plan(nu::parseMode("fast:i8qk", "t"), ti);
        CHECK(nu::applyFastOnlyOverride(pi0, Item::i8qk, "ATTN_I8QK", "0", false, true).empty() && !pi0.on(Item::i8qk));
        CHECK(nu::logLine(pi0).find("fast runs as balance") != std::string::npos);
    }
    // feat027-i8qk-r2 (review N2): strict unsigned parse for WHIRL_ATTN_I8QK_MIN / _MIN_WORK - a typo keeps the
    // default (tune.cpp warns) instead of strtoul's 0 / prefix, which would run int8 at any length
    {
        const std::uint64_t u32 = 0xffffffffull, u64 = ~0ull;
        CHECK(parseUnsigned("8192", u32) == std::optional<std::uint64_t>(8192));
        CHECK(parseUnsigned("0", u32) == std::optional<std::uint64_t>(0));  // an explicit 0 is allowed (any size)
        CHECK(parseUnsigned("4294967295", u32) == std::optional<std::uint64_t>(u32));
        CHECK(!parseUnsigned("4294967296", u32));  // > uint32: was silently truncated
        CHECK(parseUnsigned("18446744073709551615", u64) == std::optional<std::uint64_t>(u64));
        CHECK(!parseUnsigned("18446744073709551616", u64));  // from_chars result_out_of_range
        for (const char* bad : {"", "8k", "abc", " 8192", "8192 ", "+8192", "-1", "1e4", "0x2000", "8,192", "81 92"})
            CHECK(!parseUnsigned(bad, u64));
    }
    // per-item environment overrides
    {
        nu::Plan p = nu::plan(pre, swift);
        nu::applyOverride(p, Item::fp8, "FP8", "1", true, true);
        CHECK(p.on(Item::fp8) && p.lossy_override && p.overrides.size() == 1 && nu::modeLabel(p) == "precise+overrides");
        CHECK(p.overrides[0].find("lossy, user-requested") != std::string::npos);
        nu::Plan pq = nu::plan(pre, q4);
        nu::applyOverride(pq, Item::fp8, "FP8", "1", true, false);  // not applicable: no change
        CHECK(!pq.on(Item::fp8) && pq.overrides.empty() && !pq.lossy_override);
        nu::Plan pb = nu::plan(bal, swift);
        nu::applyOverride(pb, Item::fp8, "FP8", "0", false, true);
        CHECK(!pb.on(Item::fp8) && !pb.lossy_override && pb.overrides.size() == 1 && nu::modeLabel(pb) == "balance");
        nu::applyOverride(pb, Item::gdnwmma, "GDN_WMMA", "1", true, true);  // already on: nothing
        CHECK(pb.overrides.size() == 1);
        const std::string j = nu::propsJson(pb, "q8v");
        CHECK(j.find("\"mode\":\"balance\"") != std::string::npos && j.find("\"kv\":\"q8v\"") != std::string::npos);
        CHECK(j.find("\"enabled\":[\"gdnwmma\",\"h16\",\"kvq8\",\"specsample\"]") != std::string::npos);
        CHECK(j.find("{\"item\":\"fp8\",\"reason\":\"off by WHIRL_FP8=0\"}") != std::string::npos && j.find("{\"item\":\"moefp8\",\"reason\":\"dense model\"}") != std::string::npos);
        const json::Value v = json::parse(nu::propsJson(nu::plan(fast, s8060), "f16"));
        CHECK(v.isObject());
    }
    // precise KV, CLI: fits / shrinks (implicit) / refuses (explicit or below the minimum)
    {
        const std::uint64_t GiB = 1ull << 30, pt = 65536;  // 64 KiB/token
        nu::CtxFit a = nu::cliCtxFit(8192, false, 100, 10 * GiB, GiB, pt, 256);
        CHECK(!a.shrunk && !a.refuse && a.ctx == 8192);
        nu::CtxFit b = nu::cliCtxFit(131072, false, 1000, 5 * GiB, GiB, pt, 256);  // 4 GiB -> 65536 tokens
        CHECK(b.shrunk && !b.refuse && b.ctx == 65536 && b.msg.find("--balance") != std::string::npos);
        nu::CtxFit c = nu::cliCtxFit(131072, true, 1000, 5 * GiB, GiB, pt, 256);
        CHECK(c.refuse && c.msg.find("--balance") != std::string::npos && c.msg.find("--ctx") != std::string::npos);
        nu::CtxFit d = nu::cliCtxFit(131072, false, 70000, 5 * GiB, GiB, pt, 256);  // shrink would cut the prompt
        CHECK(d.refuse);
    }
    // precise KV, server: f16 pool, context shrink / refusal
    {
        const std::uint64_t GiB = 1ull << 30, pt = 69632;  // ~68 KiB/token (27B f16)
        // R9700 32 GB Swift: ~11.24 GiB -> ~173k tokens >= one 131072 request (no shrink)
        nu::PoolFit a = nu::serverPoolFit(11 * GiB + GiB / 4, pt, std::nullopt, false, 131072, false, 256, 8 * 262144);
        CHECK(!a.refuse && !a.shrunk_ctx && a.slot_ctx == 131072 && a.pool >= 131072 && a.pool % 256 == 0 && a.msg.empty());
        // 16 GB card: 4 GiB -> 61k tokens: per-request context lowered (not given) / refused (given)
        nu::PoolFit b = nu::serverPoolFit(4 * GiB, pt, std::nullopt, false, 131072, false, 256, 8 * 262144);
        CHECK(!b.refuse && b.shrunk_ctx && b.slot_ctx == b.pool && b.pool == 61440 && b.msg.find("--balance") != std::string::npos);
        nu::PoolFit c = nu::serverPoolFit(4 * GiB, pt, std::nullopt, false, 131072, true, 256, 8 * 262144);
        CHECK(c.refuse && c.msg.find("--ctx-per-slot") != std::string::npos);
        // explicit --ctx that does not fit: refused; the UMA default pool: shrunk
        nu::PoolFit d = nu::serverPoolFit(4 * GiB, pt, 131072u, true, 131072, false, 256, 8 * 262144);
        CHECK(d.refuse && d.msg.find("--ctx") != std::string::npos);
        nu::PoolFit e = nu::serverPoolFit(4 * GiB, pt, 262144u, false, 131072, false, 256, 8 * 262144);
        CHECK(!e.refuse && e.shrunk_pool && e.shrunk_ctx && e.pool == 61440 && e.slot_ctx == 61440);
        nu::PoolFit f = nu::serverPoolFit(4 * GiB, pt, 32768u, true, 32768, false, 256, 8 * 262144);  // fits
        CHECK(!f.refuse && f.pool == 32768 && f.slot_ctx == 32768 && f.msg.empty());
        nu::PoolFit g = nu::serverPoolFit(GiB / 8, pt, std::nullopt, false, 131072, false, 256, 8 * 262144);  // < 4096 tokens
        CHECK(g.refuse && g.msg.find("--balance") != std::string::npos);
    }
}

// KV format (BAL-Q8): one fixed format per mode x model type, never by what fits
void testKvChoice() {
    namespace nu = whirl::numerics;
    using K = nu::KvKind;
    const nu::Request pre = nu::parseMode("precise", "t"), bal = nu::parseMode("balance", "t"), fast = nu::parseMode("fast", "t");
    const nu::KvCaps noq4{true, true, false};   // a code object without q4 (gfx1201 before FAST-1c)
    const nu::KvCaps r9700{true, true, true};   // gfx1201: q8h / q8v / q4 (FAST-1c)
    const nu::KvCaps s8060{true, true, true};   // gfx1151: q4 too
    for (const nu::KvCaps& c : {noq4, r9700, s8060}) {
        for (bool moe : {false, true}) {
            CHECK(nu::chooseKv(pre, moe, c, std::nullopt).kv == K::f16);
            CHECK(nu::chooseKv(bal, moe, c, std::nullopt).kv == (moe ? K::f16 : K::q8h));
            // fast: q4 where the kernels exist; else f16 (no int8 stand-in)
            CHECK(nu::chooseKv(fast, moe, c, std::nullopt).kv == (c.q4 ? K::q4 : K::f16));
            CHECK(!nu::chooseKv(bal, moe, c, std::nullopt).debug);
        }
    }
    CHECK(nu::chooseKv(fast, false, noq4, std::nullopt).why.find("no q4 KV kernels") != std::string::npos);
    CHECK(nu::chooseKv(fast, false, r9700, std::nullopt).kv == K::q4);
    CHECK(nu::chooseKv(bal, false, r9700, std::nullopt).why == "balance, dense model: q8h");
    // a code object without q8h: plain q8 (never q8v)
    CHECK(nu::chooseKv(bal, false, nu::KvCaps{false, true, false}, std::nullopt).kv == K::q8);
    // custom item lists: kvq8 decides for balance, kvq4 for fast
    CHECK(nu::chooseKv(nu::parseMode("balance:fp8", "t"), false, r9700, std::nullopt).kv == K::f16);
    CHECK(nu::chooseKv(nu::parseMode("fast:kvq8", "t"), false, s8060, std::nullopt).kv == K::q8h);
    // RC6: gfx1151 (fast_q8h) fast on a dense model -> q8h; MoE keeps q4; the R9700 keeps q4; WHIRL_KV still wins
    {
        const nu::KvCaps s8060f{true, true, true, true};
        CHECK(nu::chooseKv(fast, false, s8060f, std::nullopt).kv == K::q8h);
        CHECK(nu::chooseKv(fast, false, s8060f, std::nullopt).why == "fast, dense model: q8h (Radeon 8060S: faster decode than q4)");
        CHECK(nu::chooseKv(fast, true, s8060f, std::nullopt).kv == K::q4);
        CHECK(nu::chooseKv(bal, false, s8060f, std::nullopt).kv == K::q8h);
        CHECK(nu::chooseKv(pre, false, s8060f, std::nullopt).kv == K::f16);
        CHECK(nu::chooseKv(fast, false, s8060f, K::q4).kv == K::q4);
        CHECK(nu::chooseKv(fast, false, nu::KvCaps{false, true, true, true}, std::nullopt).kv == K::q4);  // no q8h kernels: q4
        nu::Target t8;
        t8.gfx1151 = true;
        t8.k_kv_q8h = true;
        t8.k_kv_q4 = true;
        t8.fast_kv_q8h = true;
        const nu::Plan pd = nu::plan(fast, t8);
        CHECK(!pd.on(nu::Item::kvq4));
        CHECK(nu::logLine(pd).find("kvq4 (Radeon 8060S dense models: q8h KV") != std::string::npos);
        t8.moe = true;
        CHECK(nu::plan(nu::forModel(fast, true), t8).on(nu::Item::kvq4));
        // GDN-8S default: gdnwmma reported enabled when the strix prefill runs
        nu::Target tg;
        tg.gfx1151 = true;
        tg.gdn_strix = true;
        CHECK(nu::plan(bal, tg).on(nu::Item::gdnwmma));
        tg.gdn_strix = false;
        CHECK(!nu::plan(bal, tg).on(nu::Item::gdnwmma));
    }
    // WHIRL_KV: debug override, any mode
    const nu::KvChoice o = nu::chooseKv(pre, false, r9700, K::q8v);
    CHECK(o.kv == K::q8v && o.debug && o.why.find("WHIRL_KV=q8v") != std::string::npos);
    CHECK(nu::kvKindFromName("q8h") == K::q8h && !nu::kvKindFromName("auto") && !nu::kvKindFromName("q5"));

    // does not fit: the chosen format only, shrink (implicit) or refuse (explicit); 32 GB vs 16 GB (simulated)
    const std::uint64_t GiB = 1ull << 30;
    const std::uint64_t f16_pt = 65536, q8h_pt = 34816;  // ~64 / 34 KiB/token
    const nu::KvFitLabel lb_bal = nu::kvFitLabel(bal, false, K::q8h);
    CHECK(lb_bal.mode == "balance" && lb_bal.fmt == "q8h" && lb_bal.alt.empty());
    CHECK(nu::kvFitLabel(pre, false, K::f16).alt.find("--balance") != std::string::npos);
    CHECK(nu::kvFitLabel(pre, true, K::f16).alt.empty());  // MoE: balance is f16 as well
    {
        // CLI: 32 GB card (~12 GiB left): 128k q8h fits, unchanged
        const nu::CtxFit a = nu::cliCtxFit(131072, false, 1000, 13 * GiB, GiB, q8h_pt, 256, lb_bal);
        CHECK(!a.shrunk && !a.refuse && a.ctx == 131072);
        // 16 GB card (~4 GiB left): shrunk in q8h (not given) / refused (given), never a lower format
        const nu::CtxFit b = nu::cliCtxFit(131072, false, 1000, 5 * GiB, GiB, q8h_pt, 256, lb_bal);
        CHECK(b.shrunk && b.ctx == 4 * GiB / q8h_pt / 256 * 256 && b.msg.find("balance mode keeps the KV cache in q8h") == 0);
        CHECK(b.msg.find("--balance") == std::string::npos);
        const nu::CtxFit c = nu::cliCtxFit(131072, true, 1000, 5 * GiB, GiB, q8h_pt, 256, lb_bal);
        CHECK(c.refuse && c.msg.find("q8h") != std::string::npos && c.msg.find("--ctx") != std::string::npos);
        // MoE balance (f16) on 16 GB
        const nu::CtxFit d = nu::cliCtxFit(131072, false, 1000, 5 * GiB, GiB, f16_pt, 256, nu::kvFitLabel(bal, true, K::f16));
        CHECK(d.shrunk && d.ctx == 65536 && d.msg.find("balance mode keeps the KV cache in f16") == 0);
    }
    {
        // server: 32 GB (~11 GiB) holds a 131072 request in q8h; 16 GB (~4 GiB): per-request ctx lowered / refused
        const nu::PoolFit a = nu::serverPoolFit(11 * GiB, q8h_pt, std::nullopt, false, 131072, false, 256, 8 * 262144, lb_bal);
        CHECK(!a.refuse && !a.shrunk_ctx && a.pool >= 131072 && a.msg.empty());
        const nu::PoolFit b = nu::serverPoolFit(4 * GiB, q8h_pt, std::nullopt, false, 131072, false, 256, 8 * 262144, lb_bal);
        CHECK(!b.refuse && b.shrunk_ctx && b.slot_ctx == b.pool && b.pool == 4 * GiB / q8h_pt / 256 * 256);
        CHECK(b.msg.find("balance mode keeps the KV cache in q8h") == 0);
        const nu::PoolFit c = nu::serverPoolFit(4 * GiB, q8h_pt, std::nullopt, false, 131072, true, 256, 8 * 262144, lb_bal);
        CHECK(c.refuse && c.msg.find("--ctx-per-slot") != std::string::npos);
        const nu::PoolFit d = nu::serverPoolFit(4 * GiB, q8h_pt, 262144u, true, 131072, false, 256, 8 * 262144, lb_bal);
        CHECK(d.refuse && d.msg.find("q8h") != std::string::npos);
        const nu::PoolFit e = nu::serverPoolFit(GiB / 16, q8h_pt, std::nullopt, false, 131072, false, 256, 8 * 262144, lb_bal);
        CHECK(e.refuse && e.msg.find("q8h KV cache (balance mode)") != std::string::npos);
    }
    // the card's defaults carry no KV choice: 16 GB and 32 GB pick the same format
    {
        vram::CardInput in;
        in.total = 16 * GiB;
        CHECK(vram::cardDefaults(in).note.find("KV") == std::string::npos);
    }
}


// ---------------------------------------------------------------------------
// Kernel fetch bounds (FIX-OVR): replay the W-stage addressing of the raw-stage GEMMs
// (gfx1201 gemms_impl / gemmsd_impl / gemvh_impl, gfx1151 gemms11_impl) and assert that no
// 16-byte load starts at or past the end of the matrix and none ends past the end rounded
// up to 16 bytes (a 16-byte aligned block holding a matrix byte never crosses a page).
// The pre-fix addressing (every chunk loaded) must be caught by the same check.
struct FetchBound {
    std::uint64_t max_start = 0, max_end = 0;
};
// s: super-block stages per round (gemvh_impl S = 2 when ncols % 512 == 0: SB = 2 * sbb, nst = ncols / 512)
FetchBound replayStageFetch(int sb, std::uint64_t row_bytes, int nrows, int ncols, bool guarded, int s = 1) {
    const int ch = gemms_stage_chunks(sb), nst = ncols / (256 * s);
    FetchBound b;
    // rows past nrows are clamped to nrows - 1 by every kernel; the pattern only depends on
    // the row start, so the first rows, the last two and the clamped row cover every case
    std::vector<int> rows = {0, 1, nrows - 2, nrows - 1, nrows + 5};
    for (int r0 : rows) {
        if (r0 < 0) continue;
        const int row = std::min(r0, nrows - 1);
        for (int js = 0; js < nst; ++js) {
            const std::uint64_t st = static_cast<std::uint64_t>(row) * row_bytes + static_cast<std::uint64_t>(js) * sb;
            const std::uint32_t st32 = static_cast<std::uint32_t>(st);  // gemmsd_impl's 32-bit offsets
            for (int c = 0; c < ch; ++c) {
                const bool need = gemms_chunk_needed(st, c, sb);
                if (gemms_chunk_needed(st32, c, sb) != need) b.max_end = ~0ull;  // the 32-bit form must agree
                if (guarded && !need) continue;
                const std::uint64_t a = gemms_chunk_addr(st, c);
                b.max_start = std::max(b.max_start, a);
                b.max_end = std::max(b.max_end, a + 16);
            }
        }
    }
    return b;
}

void testKernelFetchBounds() {
    struct T {
        const char* name;
        int blk, bytes;
    };
    const T types[] = {{"f16", 8, 16},     {"q8_0", 32, 34},    {"iq4_nl", 32, 18}, {"q3_k", 256, 110}, {"q4_k", 256, 144},
                       {"q5_k", 256, 176}, {"q6_k", 256, 210},  {"iq3_s", 256, 110}, {"iq4_xs", 256, 136}, {"mxfp4", 256, 136}};
    int old_caught = 0, s2_cases = 0, s2_old_caught = 0;
    for (const T& t : types)
        for (int st : {1, 2}) {  // FIX-K97 (FR-K97 P3-1): also gemvh_impl's S = 2 stage (2 super-blocks per LDS round)
            const int sb = st * (256 / t.blk * t.bytes);
            CHECK(gemms_stage_chunks(sb) * 16 >= sb + 15);  // CH covers the worst misalignment
            for (int ncols : {256, 512, 1536, 2048, 4096, 5120, 8192, 12288, 17408})
                for (int nrows : {1, 17, 37, 40, 47, 48, 128, 2048, 4097}) {
                    if (ncols % (256 * st) != 0) continue;  // S = 2 only when ncols % 512 == 0 (gemvh_<T> / gemvh2_<T>)
                    const std::uint64_t rb = static_cast<std::uint64_t>(ncols) / t.blk * t.bytes;
                    const std::uint64_t total = rb * nrows, total16 = (total + 15) & ~15ull;
                    const FetchBound b = replayStageFetch(sb, rb, nrows, ncols, true, st);
                    const bool ok = b.max_start < total && b.max_end <= total16;
                    if (!ok)
                        std::printf("FAIL fetch bound %s S=%d %dx%d: max start %llu end %llu, matrix %llu bytes\n", t.name, st, nrows, ncols,
                                    static_cast<unsigned long long>(b.max_start), static_cast<unsigned long long>(b.max_end),
                                    static_cast<unsigned long long>(total));
                    CHECK(ok);
                    // also within the allocation tail pad for any matrix size
                    CHECK(b.max_end <= total + 256);
                    const FetchBound o = replayStageFetch(sb, rb, nrows, ncols, false, st);
                    if (o.max_end > total16) ++(st == 1 ? old_caught : s2_old_caught);
                    s2_cases += st == 2;
                }
        }
    CHECK(old_caught > 0);     // the check does see the pre-fix over-read (every type reads >= 16 B past)
    CHECK(s2_cases > 0);       // the S = 2 addressing was replayed
    CHECK(s2_old_caught > 0);  // and the check bites there too
}

// ---------------------------------------------------------------------------
// DFlash2 host policy (whirl/dflash_policy.h): draft-source choice, environment, capture /
// ring sizing, draft limit, the server slot's context range and ingest plan
void testDflashPolicy() {
    namespace d = whirl::dflash;
    // --draft parsing
    CHECK(d::parseSource("mtp") == d::Source::mtp);
    CHECK(d::parseSource("dflash") == d::Source::dflash);
    CHECK(!d::parseSource("DFlash") && !d::parseSource("") && !d::parseSource("eagle"));
    CHECK_EQ(std::string(d::sourceName(d::Source::dflash)), std::string("dflash"));
    // choice: mtp stays mtp (a set WHIRL_DRAFT_MODEL is reported, not loaded)
    const std::optional<std::string> dm = std::string("D:/m/dflash2-q8_0.gguf"), none;
    {
        const d::Choice c = d::choose(d::Source::mtp, none, "", true);
        CHECK(c.source == d::Source::mtp && c.warning.empty());
        const d::Choice c2 = d::choose(d::Source::mtp, dm, "", true);
        CHECK(c2.source == d::Source::mtp && c2.warning.find("not loaded") != std::string::npos);
    }
    // dflash: every fallback is mtp with a reason; all conditions met -> dflash, silent
    CHECK(d::choose(d::Source::dflash, dm, "", true).source == d::Source::dflash);
    CHECK(d::choose(d::Source::dflash, dm, "", true).warning.empty());
    {
        const d::Choice c = d::choose(d::Source::dflash, none, "", true);
        CHECK(c.source == d::Source::mtp && c.warning.find("WHIRL_DRAFT_MODEL") != std::string::npos);
        const d::Choice c2 = d::choose(d::Source::dflash, dm, "hidden size 2048 != target 5120", true);
        CHECK(c2.source == d::Source::mtp && c2.warning.find("hidden size 2048") != std::string::npos &&
              c2.warning.find("dflash2-q8_0.gguf") != std::string::npos);
        const d::Choice c3 = d::choose(d::Source::dflash, dm, "", false);
        CHECK(c3.source == d::Source::mtp && c3.warning.find("dflash_") != std::string::npos);
        // an incompatible drafter is reported before missing kernels
        CHECK(d::choose(d::Source::dflash, dm, "x", false).warning.find("does not fit") != std::string::npos);
    }
    // --draft-model FILE vs WHIRL_DRAFT_MODEL: the command line wins, the environment is the fallback
    // (a debug override when no --draft-model is given); empty = not set (REVIEW-RC4-C P3-8)
    {
        const std::optional<std::string> cli = std::string("a.gguf"), env = std::string("b.gguf"), empty = std::string();
        CHECK(*d::resolveDraftModel(cli, none) == "a.gguf");
        CHECK(*d::resolveDraftModel(none, env) == "b.gguf");
        CHECK(*d::resolveDraftModel(cli, env) == "a.gguf");
        CHECK(*d::resolveDraftModel(cli, empty) == "a.gguf");
        CHECK(!d::resolveDraftModel(none, none) && !d::resolveDraftModel(empty, empty) && !d::resolveDraftModel(none, empty));
        // the log line names the source
        CHECK(d::draftModelNote(cli, env).find("WHIRL_DRAFT_MODEL=b.gguf ignored") != std::string::npos);
        CHECK(d::draftModelNote(none, env).find("from WHIRL_DRAFT_MODEL") != std::string::npos);
        CHECK(d::draftModelNote(cli, none).empty() && d::draftModelNote(none, none).empty() && d::draftModelNote(cli, cli).empty());
    }
    // relaxacc: MTP drafts only (DFlash2 keeps exact acceptance in fast mode)
    CHECK(d::relaxApplies(d::Source::mtp, true) && !d::relaxApplies(d::Source::mtp, false));
    CHECK(!d::relaxApplies(d::Source::dflash, true) && !d::relaxApplies(d::Source::dflash, false));
    // FR-DFL P2-3: the engine's per-slot rule (draft source codes 0 MTP, 1 n-gram, 2 DFlash2)
    CHECK(d::relaxAppliesTo(d::DraftKind::mtp, true) && !d::relaxAppliesTo(d::DraftKind::mtp, false));
    CHECK(!d::relaxAppliesTo(d::DraftKind::ngram, true) && !d::relaxAppliesTo(d::DraftKind::dflash, true));
    CHECK(static_cast<int>(d::DraftKind::mtp) == 0 && static_cast<int>(d::DraftKind::ngram) == 1 && static_cast<int>(d::DraftKind::dflash) == 2);
    // environment
    {
        std::map<std::string, std::string> envm;
        auto env = [&](const char* n) -> std::optional<std::string> {
            auto it = envm.find(n);
            if (it == envm.end()) return std::nullopt;
            return it->second;
        };
        std::vector<std::string> w;
        d::Params p = d::parseParams(env, w);
        CHECK(w.empty() && p.p_min == 0.0f && p.n_min == 0 && p.warm == 64 && !p.capture_rows);
        envm = {{"DFLASH_PMIN", "0.25"}, {"DFLASH_NMIN", "2"}, {"DFLASH_WARM", "0"}, {"DFLASH_CAPTURE_ROWS", "512"}};
        p = d::parseParams(env, w);
        CHECK(w.empty() && std::fabs(p.p_min - 0.25f) < 1e-7f && p.n_min == 2 && p.warm == 0 && p.capture_rows && *p.capture_rows == 512);
        envm = {{"DFLASH_PMIN", "1.5"}, {"DFLASH_NMIN", "-1"}, {"DFLASH_WARM", "6x"}, {"DFLASH_CAPTURE_ROWS", "0"}};
        p = d::parseParams(env, w);
        CHECK_EQ(w.size(), std::size_t{4});
        CHECK(p.p_min == 0.0f && p.n_min == 0 && p.warm == 64 && !p.capture_rows);
        w.clear();
        // FR-DFL P2-2: fewer capture rows than one verify of a full block (16) are refused
        for (const char* bad : {"4", "7", "15"}) {
            envm = {{"DFLASH_CAPTURE_ROWS", bad}};
            p = d::parseParams(env, w);
            CHECK(!p.capture_rows);
            CHECK(w.size() == 1 && w[0].find(">= 16") != std::string::npos);
            w.clear();
        }
        envm = {{"DFLASH_CAPTURE_ROWS", "16"}};
        p = d::parseParams(env, w);
        CHECK(w.empty() && p.capture_rows && *p.capture_rows == d::kMinCaptureRows);
        w.clear();
        envm = {{"DFLASH_PMIN", "1"}, {"DFLASH_NMIN", "99999999999"}, {"DFLASH_PMIN_X", "z"}};
        p = d::parseParams(env, w);
        CHECK(p.p_min == 1.0f && w.size() == 1 && w[0].find("DFLASH_NMIN") != std::string::npos);
        w.clear();
        envm = {{"DFLASH_PMIN", ".5"}};
        CHECK(d::parseParams(env, w).p_min == 0.5f && w.empty());
        envm = {{"DFLASH_PMIN", "."}};
        CHECK(d::parseParams(env, w).p_min == 0.0f && w.size() == 1);
        w.clear();
        envm = {{"DFLASH_PMIN", "0.1.2"}};
        CHECK(d::parseParams(env, w).p_min == 0.0f && w.size() == 1);
    }
    // capture rows / ring: the window (2048 without one), the batch caps the capture
    {
        d::Params p;
        CHECK_EQ(d::captureRows(p, 2048, 4096), 2048u);
        CHECK_EQ(d::captureRows(p, 0, 4096), 2048u);
        CHECK_EQ(d::captureRows(p, 4096, 1024), 1024u);
        p.capture_rows = 300;
        CHECK_EQ(d::captureRows(p, 2048, 4096), 300u);
        CHECK_EQ(d::ringRows(0), 2048u);
        CHECK_EQ(d::ringRows(4096), 4096u);
    }
    // ring pieces: wrap at the end, never more than the ring
    {
        d::RingPieces r = d::ringPieces(10, 5, 2048);
        CHECK(r.start == 10 && r.first == 5 && r.second == 0);
        r = d::ringPieces(2046, 5, 2048);
        CHECK(r.start == 2046 && r.first == 2 && r.second == 3);
        r = d::ringPieces(4096, 7, 2048);
        CHECK(r.start == 0 && r.first == 7 && r.second == 0);
        r = d::ringPieces(5, 3000, 2048);
        CHECK(r.start == 5 && r.first == 2043 && r.second == 5 && r.first + r.second == 2048);
        r = d::ringPieces(5, 0, 2048);
        CHECK(r.first == 0 && r.second == 0);
    }
    // draft limit: block 8 -> 7 drafts solo; shared verify rows / snapshots per slot; unfused 1
    {
        CHECK_EQ(d::draftLimit(8, 15, 15, 16, 1, true), 7u);
        CHECK_EQ(d::draftLimit(16, 15, 15, 16, 1, true), 15u);
        CHECK_EQ(d::draftLimit(8, 15, 15, 16, 2, true), 7u);
        CHECK_EQ(d::draftLimit(8, 15, 15, 16, 4, true), 3u);   // 16 rows / 4 slots - own row
        CHECK_EQ(d::draftLimit(8, 15, 15, 32, 4, true), 3u);   // snapshots 15 / 4
        CHECK_EQ(d::draftLimit(8, 15, 32, 32, 4, true), 7u);   // replay: snap_cap = vrows
        CHECK_EQ(d::draftLimit(8, 15, 15, 16, 1, false), 1u);
        CHECK_EQ(d::draftLimit(1, 15, 15, 16, 1, true), 0u);   // no drafter
        CHECK_EQ(d::draftLimit(8, 15, 15, 16, 0, true), 0u);
        CHECK_EQ(d::draftLimit(8, 15, 15, 16, 17, true), 0u);
    }
    // slot range + ingest plan (Engine::dflIngest / dflReady)
    {
        d::Range r;
        // prefill 0..99 in one forward, all captured
        d::Ingest g = d::planIngest(r, 100, 100, 0, 100, 0);
        CHECK(g.any && g.cap_row == 0 && g.n == 100 && g.pos == 0 && g.gap == 0 && r.lo == 0 && r.hi == 100);
        CHECK(d::ready(r, 100, 64) && !d::ready(r, 101, 64));
        // verify of 8 rows (slot rows 0..7), kept 3: positions 100..102
        g = d::planIngest(r, 8, 8, 0, 3, 100);
        CHECK(g.any && g.cap_row == 0 && g.n == 3 && g.pos == 100 && r.hi == 103 && r.lo == 0);
        // batched verify: this slot is rows 8..15 of 24, the capture holds the last 16 rows (8..23)
        g = d::planIngest(r, 16, 24, 8, 2, 103);
        CHECK(g.any && g.cap_row == 0 && g.n == 2 && g.pos == 103 && r.hi == 105);
        // capture of only the last 4 rows of a 2048-row chunk: a gap, the run restarts
        g = d::planIngest(r, 4, 2048, 0, 2048, 105);
        CHECK(g.any && g.cap_row == 0 && g.n == 4 && g.pos == 105 + 2044 && g.gap == 2044 && r.lo == 2149 && r.hi == 2153);
        CHECK(!d::ready(r, 2153, 64) && d::ready(r, 2153, 4) && d::ready(r, 2153, 0));
        // rows entirely outside the capture: gap counted, range untouched
        const d::Range before = r;
        g = d::planIngest(r, 4, 16, 0, 8, 2153);
        CHECK(!g.any && g.gap == 8 && r.lo == before.lo && r.hi == before.hi);
        // restore point (dflReset at 500): ready after warm rows
        r = d::Range{500, 500};
        CHECK(!d::ready(r, 500, 64));
        g = d::planIngest(r, 64, 64, 0, 64, 500);
        CHECK(d::ready(r, 564, 64) && r.lo == 500);
        d::Range r0{0, 10};
        CHECK(d::ready(r0, 10, 64));  // short prompt: min(pos, warm)
        CHECK_EQ(d::planIngest(r0, 8, 8, 0, 0, 10).n, 0u);
    }
    // FR-DFL P3-3: prefix reuse trims the run to positions < reuse that the ring still holds
    {
        d::Trim t = d::trimForReuse(d::Range{0, 5000}, 2000, 2048);  // ring holds 2952..4999: nothing below 2000
        CHECK(t.reset && t.r.lo == 2000 && t.r.hi == 2000);
        t = d::trimForReuse(d::Range{0, 3000}, 2000, 2048);  // ring holds 952..2999: [952, 2000) stays
        CHECK(!t.reset && t.r.lo == 952 && t.r.hi == 2000);
        t = d::trimForReuse(d::Range{0, 1500}, 2000, 2048);  // all in the ring, below reuse: unchanged
        CHECK(!t.reset && t.r.lo == 0 && t.r.hi == 1500);
        t = d::trimForReuse(d::Range{100, 5000}, 4000, 0);  // no ring limit: the old rule
        CHECK(!t.reset && t.r.lo == 100 && t.r.hi == 4000);
        t = d::trimForReuse(d::Range{3500, 5000}, 3000, 2048);  // run started after reuse: nothing left
        CHECK(t.reset && t.r.lo == 3000 && t.r.hi == 3000);
        t = d::trimForReuse(d::Range{0, 0}, 500, 2048);  // empty run: no reset (the next ingest counts the gap)
        CHECK(!t.reset && t.r.lo == 0 && t.r.hi == 0);
    }
    // FR-DFL P2-2: whirl chat's ingest after a verify (verifyIngest): capture rows < verify rows never
    // index before the capture buffer (was cap_n - (nd + 1): 4 - 8 wrapped to 4294967292)
    {
        d::Ingest g = d::verifyIngest(8, 8, 3, 100);  // full capture: rows 0..2 kept
        CHECK(g.any && g.cap_row == 0 && g.n == 3 && g.pos == 100 && g.gap == 0);
        g = d::verifyIngest(4, 8, 8, 100);  // capture holds verify rows 4..7, all kept
        CHECK(g.any && g.cap_row == 0 && g.n == 4 && g.pos == 104 && g.gap == 4 && g.cap_row + g.n <= 4);
        g = d::verifyIngest(4, 8, 6, 100);  // kept rows 0..5: rows 4, 5 are capture rows 0, 1
        CHECK(g.any && g.cap_row == 0 && g.n == 2 && g.pos == 104);
        g = d::verifyIngest(4, 8, 2, 100);  // kept rows 0..1 are not in the capture
        CHECK(!g.any && g.n == 0);
        g = d::verifyIngest(2048, 8, 8, 7);  // capture larger than the verify (cap_n = rows of the last forward)
        CHECK(g.any && g.cap_row == 0 && g.n == 8 && g.pos == 7);
    }
}

}  // namespace

// speculative sampling (whirl/spec_sample.h): race-coupled draws; the race is an exact draw from p
// (chi-square over many trials, closed and open targets), the draft matches it with probability
// close to sum min(p, q), and whole replies are identical for a seed whatever the drafts / draft
// counts / draft-head state
namespace specsample_test {
namespace sp = whirl::spec;

// Wilson-Hilferty upper quantile of chi-square(df) at z standard deviations
double chi2Crit(double df, double z) {
    const double a = 2.0 / (9.0 * df);
    return df * std::pow(1.0 - a + z * std::sqrt(a), 3.0);
}

// filtered distribution over a row of logits (ids = token id of each logit), the target filter rule
sp::QDist filtered(const std::vector<float>& lg, const std::vector<std::uint32_t>& ids, float inv_t, std::uint32_t top_k, float top_p,
                   float min_p) {
    float m = -1e30f;
    for (float v : lg) m = std::max(m, v);
    float sum = 0;
    for (float v : lg) sum += std::exp((v - m) * inv_t);
    std::vector<std::size_t> ord(lg.size());
    for (std::size_t i = 0; i < ord.size(); ++i) ord[i] = i;
    std::sort(ord.begin(), ord.end(), [&](std::size_t a, std::size_t b) { return lg[a] != lg[b] ? lg[a] > lg[b] : ids[a] < ids[b]; });
    const std::size_t K = std::min<std::size_t>(sp::draftCands(top_k), ord.size());
    std::vector<std::uint32_t> ci(K);
    std::vector<float> cl(K);
    for (std::size_t i = 0; i < K; ++i) {
        ci[i] = ids[ord[i]];
        cl[i] = lg[ord[i]];
    }
    return sp::draftDist(ci, cl, m, sum, inv_t, top_k, top_p, min_p);
}

// target race draw over a closed distribution, as Engine::sampleRace
std::uint32_t raceDraw(const sp::QDist& pt, std::uint64_t seed, std::uint64_t pos) { return sp::sampleRace(pt, sp::raceBase(seed, pos)); }

// one case: race target against p (chi-square), race-coupled draft acceptance against sum min(p, q)
// and against the greedy draft p(argmax q)
void runCase(const char* name, const std::vector<float>& tl, const std::vector<float>& dl, const std::vector<std::uint32_t>& dsub,
             float inv_t, std::uint32_t top_k, float top_p, float min_p, int trials) {
    std::vector<std::uint32_t> all(tl.size());
    for (std::size_t i = 0; i < all.size(); ++i) all[i] = static_cast<std::uint32_t>(i);
    const sp::QDist pt = filtered(tl, all, inv_t, top_k, top_p, min_p);  // target p (closed)
    const sp::QDist q = filtered(dl, dsub, inv_t, top_k, top_p, min_p);  // draft q over the subset
    std::vector<double> cnt(tl.size(), 0);
    double acc = 0, exact = 0;
    const std::uint32_t qtop = q.id[0];
    for (int i = 0; i < trials; ++i) {
        const std::uint64_t seed = 0x1234567ull + static_cast<std::uint64_t>(i) * 7919ull;
        const std::uint32_t y = raceDraw(pt, seed, 5);  // target (emitted)
        const std::uint32_t d = raceDraw(q, seed, 5);   // device draft
        cnt[y] += 1;
        acc += d == y;
        exact += y == qtop;
    }
    double chi = 0;
    for (std::size_t t = 0; t < tl.size(); ++t) {
        const double e = pt.of(static_cast<std::uint32_t>(t)) * trials;
        if (e <= 0) {
            CHECK(cnt[t] == 0);  // nothing outside the target's support
            continue;
        }
        chi += (cnt[t] - e) * (cnt[t] - e) / e;
    }
    const double df = std::max(1.0, static_cast<double>(pt.n) - 1.0);
    const double crit = chi2Crit(df, 3.719);  // upper tail ~1e-4
    double ov = 0;
    for (std::uint32_t i = 0; i < q.n; ++i) ov += std::min(static_cast<double>(pt.of(q.id[i])), static_cast<double>(q.p[i]));
    std::printf("  specsample %-22s support p %2u q %2u  chi2 %7.2f (crit %.1f, df %.0f)  accept race %.3f, sum min(p,q) %.3f, greedy draft %.3f\n",
                name, pt.n, q.n, chi, crit, df, acc / trials, ov, exact / trials);
    CHECK(chi < crit);
    CHECK(acc / trials <= ov + 4.0 * std::sqrt(ov * (1 - ov) / trials) + 1e-3);  // no coupling beats sum min(p, q)
    CHECK(acc / trials >= 0.85 * ov);                                              // close to it
    CHECK(acc >= exact);                                                           // better than the greedy draft
}

// open target (Engine::sampleRace): the first `nc` tokens are the candidates, the rest a tail; u < S
// races over the candidates, else the inverse-CDF walk over the tail
void runOpen(const std::vector<float>& tl, float inv_t, std::size_t nc, int trials) {
    std::vector<std::uint32_t> all(tl.size());
    for (std::size_t i = 0; i < all.size(); ++i) all[i] = static_cast<std::uint32_t>(i);
    const sp::QDist pt = filtered(tl, all, inv_t, 0, 1.f, 0.f);  // sorted, whole vocab
    double S = 0;
    for (std::size_t i = 0; i < nc; ++i) S += pt.p[i];
    sp::QDist cand = pt;
    cand.n = static_cast<std::uint32_t>(nc);
    std::vector<double> cnt(tl.size(), 0);
    for (int i = 0; i < trials; ++i) {
        const std::uint64_t seed = 0xBEEFull + static_cast<std::uint64_t>(i) * 104729ull;
        const double u = sp::uniform(seed, 9, sp::u_sample);
        std::uint32_t y = pt.id[pt.n - 1];
        if (u < S) {
            y = raceDraw(cand, seed, 9);
        } else {
            double cum = S;
            for (std::uint32_t j = static_cast<std::uint32_t>(nc); j < pt.n; ++j) {
                cum += pt.p[j];
                if (cum > u) {
                    y = pt.id[j];
                    break;
                }
            }
        }
        cnt[y] += 1;
    }
    double chi = 0;
    for (std::uint32_t j = 0; j < pt.n; ++j) {
        const double e = pt.p[j] * trials;
        chi += (cnt[pt.id[j]] - e) * (cnt[pt.id[j]] - e) / e;
    }
    const double crit = chi2Crit(static_cast<double>(pt.n) - 1.0, 3.719);
    std::printf("  specsample open target, %zu candidates (mass %.3f): chi2 %7.2f (crit %.1f)\n", nc, S, chi, crit);
    CHECK(chi < crit);
}

// Toy LM for the reply test: target logits of the next token from a hash of the prefix; the draft
// head = target + a perturbation keyed by `dstate` (stands for the draft head's cache / history
// state, which differs between runs in the server).
std::uint64_t mix(std::uint64_t z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
std::vector<float> toyRow(const std::vector<std::uint32_t>& pre, std::uint32_t dstate, bool draft) {
    std::uint64_t h = 0x51ED27ull;
    for (std::uint32_t t : pre) h = mix(h ^ (t + 0x9E3779B97F4A7C15ull));
    std::vector<float> lg(24);
    for (std::uint32_t v = 0; v < 24; ++v) {
        lg[v] = static_cast<float>(mix(h + v) >> 40) * (5.0f / 16777216.0f);  // [0, 5)
        if (draft) lg[v] += static_cast<float>(mix(h * 31 + v * 7 + dstate * 1000003ull) >> 40) * (1.5f / 16777216.0f);
    }
    return lg;
}

// One reply of n tokens. nd_mode < 0: no drafts; otherwise speculative cycles whose draft count comes
// from a generator keyed by nd_mode (0..4 drafts, like the timing-trained allocation), drafts raced
// from the draft head (dstate) on the drafted prefix, accepted iff equal to the target's race draw.
std::vector<std::uint32_t> toyReply(std::uint64_t seed, int nd_mode, std::uint32_t dstate, float inv_t, std::uint32_t top_k,
                                    float top_p, float min_p, std::size_t n, double* acc_rate) {
    std::vector<std::uint32_t> all(24), out;
    for (std::uint32_t i = 0; i < 24; ++i) all[i] = i;
    std::uint64_t cyc = 0;
    double drafted = 0, accepted = 0;
    while (out.size() < n) {
        const std::uint32_t nd =
            nd_mode < 0 ? 0 : static_cast<std::uint32_t>(mix(cyc++ * 977 + static_cast<std::uint64_t>(nd_mode)) % 5);
        const std::uint64_t n_gen = out.size();
        std::vector<std::uint32_t> dr, pre = out;
        for (std::uint32_t r = 0; r < nd; ++r) {
            const sp::QDist q = filtered(toyRow(pre, dstate, true), all, inv_t, top_k, top_p, min_p);
            dr.push_back(raceDraw(q, seed, n_gen + r));
            pre.push_back(dr.back());
        }
        drafted += nd;
        for (std::uint32_t acc = 0;; ++acc) {
            const sp::QDist pt = filtered(toyRow(out, 0, false), all, inv_t, top_k, top_p, min_p);
            const std::uint32_t t = raceDraw(pt, seed, n_gen + acc);
            out.push_back(t);
            if (acc < nd && t == dr[acc]) {
                accepted += 1;
                continue;
            }
            break;
        }
    }
    out.resize(n);
    if (acc_rate) *acc_rate = drafted > 0 ? accepted / drafted : 0;
    return out;
}

void testReplies() {
    struct F {
        float inv_t;
        std::uint32_t top_k;
        float top_p, min_p;
    };
    const F fs[] = {{1.f / 0.7f, 0, 1.f, 0.f}, {1.f / 0.7f, 20, 0.95f, 0.f}, {1.f, 0, 0.8f, 0.05f}};
    int same = 0, total = 0, seed_diff = 0;
    double acc_sum = 0;
    for (const F& f : fs)
        for (std::uint64_t seed : {1234ull, 42ull, 987654321ull}) {
            const std::vector<std::uint32_t> plain = toyReply(seed, -1, 0, f.inv_t, f.top_k, f.top_p, f.min_p, 96, nullptr);
            for (int nd_mode = 0; nd_mode < 3; ++nd_mode)
                for (std::uint32_t dstate = 1; dstate <= 3; ++dstate) {
                    double ar = 0;
                    const std::vector<std::uint32_t> o = toyReply(seed, nd_mode, dstate, f.inv_t, f.top_k, f.top_p, f.min_p, 96, &ar);
                    same += o == plain;
                    ++total;
                    acc_sum += ar;
                }
            seed_diff += toyReply(seed + 1, -1, 0, f.inv_t, f.top_k, f.top_p, f.min_p, 96, nullptr) != plain;
        }
    std::printf("  specsample replies: %d / %d (3 draft-count patterns x 3 draft states x 3 seeds x 3 filters) == plain, mean acceptance %.3f\n",
                same, total, acc_sum / total);
    CHECK(same == total);
    CHECK(seed_diff == 9);         // the seed does matter
    CHECK(acc_sum / total > 0.3);  // drafts are accepted
}

void testSpecSample() {
    // stream 0 = the server sampler's Sampler::uniform
    {
        const std::uint64_t seed = 987654321ull, pos = 41;
        std::uint64_t z = seed + (pos + 1) * 0x9E3779B97F4A7C15ull;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        z ^= z >> 31;
        CHECK(sp::uniform(seed, pos, sp::u_sample) == static_cast<double>(z >> 11) * (1.0 / 9007199254740992.0));
        CHECK(sp::uniform(seed, pos, 1) != sp::uniform(seed, pos, sp::u_sample));
    }
    // q filter: sorted (logit desc, id asc), top-k closes, min-p cuts, renormalized; buffer decoding
    {
        const std::vector<std::uint32_t> ids = {7, 3, 9, 1};
        const std::vector<float> lg = {1.f, 3.f, 2.f, 2.f};
        const float m = 3.f;
        float sum = 0;
        for (float v : lg) sum += std::exp(v - m);
        sum += 0.5f;  // mass outside the candidates
        const sp::QDist a = sp::draftDist(ids, lg, m, sum, 1.f, 0, 1.f, 0.f);
        CHECK(a.n == 4 && a.id[0] == 3 && a.id[1] == 1 && a.id[2] == 9 && a.id[3] == 7);
        float z = 0;
        for (std::uint32_t i = 0; i < a.n; ++i) z += a.p[i];
        CHECK(std::fabs(z - 1.f) < 1e-6f && a.of(42) == 0.f);
        const sp::QDist b = sp::draftDist(ids, lg, m, sum, 1.f, 2, 1.f, 0.f);
        CHECK(b.n == 2 && b.id[0] == 3 && b.id[1] == 1);
        const sp::QDist c = sp::draftDist(ids, lg, m, sum, 1.f, 0, 1.f, 0.5f);  // e^-1 < 0.5: the top token only
        CHECK(c.n == 1 && c.id[0] == 3 && c.p[0] == 1.f);
        std::int32_t w[sp::q_words] = {};
        w[0] = static_cast<std::int32_t>(b.n);
        for (std::uint32_t i = 0; i < b.n; ++i) {
            w[sp::q_off_ids + i] = static_cast<std::int32_t>(b.id[i]);
            std::memcpy(&w[sp::q_off_p + i], &b.p[i], 4);
        }
        const sp::QDist r = sp::qFromWords(w);
        CHECK(r.n == b.n && r.id[1] == b.id[1] && r.p[1] == b.p[1]);
    }
    // distribution equality on toy rows: V = 24, draft head = perturbed target, draft vocabulary
    // subset = even tokens plus 1 and 3 (q = 0 on the other odd tokens)
    std::vector<float> tl(24), dl, dlfull(24);
    for (int i = 0; i < 24; ++i) {
        tl[i] = 2.5f * std::sin(0.7f * i) + 0.1f * i;
        dlfull[i] = tl[i] + 0.8f * std::cos(1.3f * i);
    }
    std::vector<std::uint32_t> sub, allv(24);
    for (std::uint32_t i = 0; i < 24; ++i) {
        allv[i] = i;
        if (i % 2 == 0 || i == 1 || i == 3) {
            sub.push_back(i);
            dl.push_back(dlfull[i]);
        }
    }
    const int N = 200000;
    runCase("t0.7 top-k20 top-p0.95", tl, dl, sub, 1.f / 0.7f, 20, 0.95f, 0.f, N);
    runCase("t0.7 top-k5", tl, dl, sub, 1.f / 0.7f, 5, 1.f, 0.f, N);
    runCase("t1.0 top-p0.8", tl, dl, sub, 1.f, 0, 0.8f, 0.f, N);
    runCase("t1.0 min-p0.1", tl, dl, sub, 1.f, 0, 1.f, 0.1f, N);
    runCase("t1.5 open, full vocab", tl, dlfull, allv, 1.f / 1.5f, 0, 1.f, 0.f, N);
    runCase("t1.5 open, subset", tl, dl, sub, 1.f / 1.5f, 0, 1.f, 0.f, N);
    runOpen(tl, 1.f, 6, N);
    runOpen(tl, 1.f / 1.5f, 3, N);
    testReplies();
}


}  // namespace specsample_test

// REVIEW-ARCH4 P3-2 / arch6e: the launch-log argument digest must be the same for the same argument values,
// whatever memory the argument struct was built in (two runs of one build differed: the GEMV launches' GvArgs
// were built on stack memory that held different bytes), and must still change when an argument changes.
void testLaunchDigest() {
    using whirl::kernels::GvArgs;
    auto build = [](unsigned char fill, unsigned char* buf) {
        std::memset(buf, fill, sizeof(GvArgs));  // what the stack held before (differs from run to run)
        GvArgs* g = new (buf) GvArgs;            // as forward.cpp's gemvKernels builds it: `GvArgs g;` + fields
        g->w[0] = 0x358f40000;
        g->w[1] = 0x359f40000;
        g->y[0] = 0x7dddd0000;
        g->y[1] = 0x7ddde0000;
        g->rb[0] = g->rb[1] = 0x440;
        g->n[0] = g->n[1] = 512;
        g->b1 = 64;
        return g;
    };
    alignas(GvArgs) unsigned char a[sizeof(GvArgs)], b[sizeof(GvArgs)];
    GvArgs* ga = build(0xAA, a);
    GvArgs* gb = build(0x55, b);
    auto digest = [](const GvArgs* g) {
        const std::int32_t ncols = 2048;  // the scalar arguments that follow it in the real launch
        return whirl::hip::launchlog::launchArgDigest(*g, ncols);
    };
    const std::uint64_t da = digest(ga), db = digest(gb);
    CHECK_EQ(da, db);  // same argument values -> same digest
    CHECK(std::memcmp(a, b, sizeof(GvArgs)) == 0);  // ... because the struct carries no undefined bytes
    // still sensitive to a real argument change (one field of one matrix)
    GvArgs* gc = build(0xAA, b);
    gc->n[1] = 513;
    CHECK(digest(gc) != da);
    gc = build(0xAA, b);
    gc->b1 = 65;
    CHECK(digest(gc) != da);
    gc = build(0xAA, b);
    gc->b2 = 96;
    CHECK(digest(gc) != da);
}

// arch6f (REVIEW-ARCH4 P3-2): the launch-log digest of the same argument values must not depend on where the
// device allocations happen to lie. The golden runs E_r97 / E2_r97 (same build) differed in 2 of 8 steps: int32
// pairs of all-integer argument structs (AwGroups first[0..1] = 0,5 -> word 0x500000000, Tok16, RowIdx, the
// row0/nrows of GdnSegs, KvArgs tab0/win) were read as "inside allocation 597" in one run and as a value in the other.
void testLaunchDigestLayout() {
    using namespace whirl::kernels;
    namespace ll = whirl::hip::launchlog;
    struct Layout {
        std::uint64_t base[3];
    };
    // A: the bases put 0x500000000, 0x800000ae0 and 0x4000000000 inside allocations 0..2 (as in the failing runs);
    // B: nowhere near them
    const Layout la{{0x4FE000000ull, 0x7FF000000ull, 0x3F00000000ull}};
    const Layout lb{{0x2000000000ull, 0x2100000000ull, 0x6000000000ull}};
    const std::uint64_t bytes[3] = {0x4000000ull, 0x2000000ull, 0x200000000ull};
    auto with = [&](const Layout& l, auto&& f) {
        ll::testClearAllocs();
        for (int i = 0; i < 3; ++i) ll::testAlloc(l.base[i], bytes[i]);
        const std::uint64_t d = f(l);
        ll::testClearAllocs();
        return d;
    };
    AwGroups aw;
    aw.first[1] = 5;
    aw.count[0] = aw.count[1] = 5;
    Tok16 tk;
    tk.t[4] = 0xae0;
    tk.t[5] = 8;
    RowIdx ri;
    ri.v[1] = 0x40;
    auto dAw = [&](const AwGroups& g) {
        return [&g](const Layout&) { return ll::launchArgDigest(g, std::int32_t{4}); };
    };
    auto dKv = [](std::int32_t win, std::uint64_t koff, int kalloc) {
        return [=](const Layout& l) {
            KvArgs kv;
            kv.k = l.base[kalloc] + koff;
            kv.v = l.base[1] + 0x40;
            kv.tab0 = 0;
            kv.win = win;
            return ll::launchArgDigest(kv, DevPtr{l.base[0] + 0x2000}, std::int32_t{7});
        };
    };
    auto dGdn = [](std::int32_t nrows0) {
        return [=](const Layout& l) {
            GdnSegs s;
            s.s[0].state = l.base[2] + 0x80;
            s.s[0].row0 = 0;
            s.s[0].nrows = nrows0;
            s.s[1].state = l.base[2] + 0x1080;
            s.s[1].row0 = 5;
            s.s[1].nrows = 5;
            return ll::launchArgDigest(s);
        };
    };
    // same values -> same digest in both layouts
    CHECK_EQ(with(la, dAw(aw)), with(lb, dAw(aw)));
    CHECK_EQ(with(la, [&](const Layout&) { return ll::launchArgDigest(tk); }), with(lb, [&](const Layout&) { return ll::launchArgDigest(tk); }));
    CHECK_EQ(with(la, [&](const Layout&) { return ll::launchArgDigest(ri); }), with(lb, [&](const Layout&) { return ll::launchArgDigest(ri); }));
    CHECK_EQ(with(la, dKv(5, 0x100, 0)), with(lb, dKv(5, 0x100, 0)));
    CHECK_EQ(with(la, dGdn(5)), with(lb, dGdn(5)));
    // still sensitive to a real argument change: an integer field, a pointer offset, the allocation pointed into
    AwGroups aw2 = aw;
    aw2.first[1] = 6;
    CHECK(with(la, dAw(aw2)) != with(la, dAw(aw)));
    CHECK(with(lb, dAw(aw2)) != with(lb, dAw(aw)));
    Tok16 tk2 = tk;
    tk2.t[5] = 9;
    CHECK(with(la, [&](const Layout&) { return ll::launchArgDigest(tk2); }) != with(la, [&](const Layout&) { return ll::launchArgDigest(tk); }));
    CHECK(with(la, dKv(6, 0x100, 0)) != with(la, dKv(5, 0x100, 0)));
    CHECK(with(la, dKv(5, 0x110, 0)) != with(la, dKv(5, 0x100, 0)));
    CHECK(with(lb, dKv(5, 0x110, 0)) != with(lb, dKv(5, 0x100, 0)));
    CHECK(with(la, dKv(5, 0x100, 1)) != with(la, dKv(5, 0x100, 0)));
    CHECK(with(la, dGdn(4)) != with(la, dGdn(5)));
    // a pointer is still normalized: the same (allocation, offset) in both layouts gives the same digest although
    // the raw addresses differ (checked above through KvArgs / GdnSegs), and a raw address change alone does not
    // count as an argument change
}

// arch024-S0a (REVIEW-ARCH6-code C-1): a pointer word whose value is 0 (an unused pointer: single-matrix GvArgs
// w[1..2] / y[1..2], the f16 pool's KvArgs ks / vs) and a small u64 scalar (forward.cpp w.row_bytes) are values. Before
// the fix they fell into the "below the next allocation" rule whenever the lowest allocation was at least as large as
// its base address (one allocation of ~12 GB or more), and were then digested as (ordinal, base - v), which depends on
// where that allocation lies. Also pins the adjacent-allocation case: a pointer inside an allocation that touches the
// next one stays "inside" its own allocation in both layouts.
void testLaunchDigestZeroAndAdjacent() {
    namespace ll = whirl::hip::launchlog;
    using whirl::kernels::KvArgs;
    const auto with = [&](std::initializer_list<std::pair<std::uint64_t, std::uint64_t>> al, auto&& f) {
        ll::testClearAllocs();
        for (const auto& [b, n] : al) ll::testAlloc(b, n);
        const std::uint64_t d = f();
        ll::testClearAllocs();
        return d;
    };
    // the lowest allocation is larger than its base address in both layouts, at different bases
    const std::uint64_t big = 0x400000000ull;  // 16 GiB
    const auto la = {std::pair{0x300000000ull, big}, std::pair{0x800000000ull, 0x1000000ull}};
    const auto lb = {std::pair{0x380000000ull, big}, std::pair{0x900000000ull, 0x1000000ull}};
    const auto zero = [] { return ll::launchArgDigest(std::uint64_t{0}, std::int32_t{3}); };
    CHECK_EQ(with(la, zero), with(lb, zero));
    const auto small = [] { return ll::launchArgDigest(std::uint64_t{0x440}); };
    CHECK_EQ(with(la, small), with(lb, small));
    const auto kv0 = [](std::uint64_t kbase) {
        return [=] {
            KvArgs kv;
            kv.k = kbase + 0x100;
            kv.v = kbase + 0x200;
            kv.ks = 0;  // f16 pool: no scale arrays
            kv.vs = 0;
            return ll::launchArgDigest(kv);
        };
    };
    CHECK_EQ(with(la, kv0(0x300000000ull)), with(lb, kv0(0x380000000ull)));
    // a value of 0 still differs from a real pointer, and from another value
    CHECK(with(la, zero) != with(la, [] { return ll::launchArgDigest(std::uint64_t{0x300000040ull}, std::int32_t{3}); }));
    CHECK(with(la, small) != with(la, [] { return ll::launchArgDigest(std::uint64_t{0x448}); }));
    // adjacent allocations (A ends where B starts) vs a gap between them: a pointer inside A near its end, and one
    // inside B, give the same digest in both layouts
    const auto adj = {std::pair{0x10000000ull, 0x100000ull}, std::pair{0x10100000ull, 0x100000ull}};
    const auto gap = {std::pair{0x20000000ull, 0x100000ull}, std::pair{0x28000000ull, 0x100000ull}};
    const auto two = [](std::uint64_t a, std::uint64_t b) {
        return [=] { return ll::launchArgDigest(std::uint64_t{a + 0xFFF00}, std::uint64_t{b + 0x80}); };
    };
    CHECK_EQ(with(adj, two(0x10000000ull, 0x10100000ull)), with(gap, two(0x20000000ull, 0x28000000ull)));
}

// arch024-S0a (REVIEW-ARCH6-code C-2): the RAW dump keeps at most max_lines lines and counts the rest; the server
// turns it off unless WHIRL_LAUNCH_LOG_RAW=server.
void testLaunchRawCap() {
    namespace ll = whirl::hip::launchlog;
    const ll::RawStats before = ll::rawStats();
    ll::testRaw(true, 3);
    for (int i = 0; i < 5; ++i) ll::testRawPush("x");
    ll::RawStats s = ll::rawStats();
    CHECK(s.on);
    CHECK_EQ(s.kept, std::size_t{3});
    CHECK_EQ(s.dropped, std::uint64_t{2});
    CHECK_EQ(s.max_lines, std::size_t{3});
    // the default cap (no WHIRL_LAUNCH_LOG_RAW_MAX in the test environment) is finite
    CHECK(before.max_lines > 0 && before.max_lines <= 1000000);
    // server process: off unless WHIRL_LAUNCH_LOG_RAW=server (the unit tests run without the variable, or with
    // WHIRL_LAUNCH_LOG_RAW=1 from a gate shell: both are "not server")
    ll::testRaw(true, 3);
    ll::serverProcess();
    const char* rv = std::getenv("WHIRL_LAUNCH_LOG_RAW");
    const bool want_on = rv != nullptr && std::string(rv) == "server";
    CHECK_EQ(ll::rawStats().on, want_on);
    ll::testRaw(before.on, before.max_lines);
}

// arch025-S0a: the allocation log records sizes, kinds and ordinals (no addresses): two runs whose allocations land
// at different addresses give the same text; a size or order change gives a different seqhash; the list is capped;
// whirl-server keeps only the summary unless WHIRL_ALLOC_LOG_RAW=server; off records nothing.
void testAllocLog() {
    namespace al = whirl::hip::alloclog;
    const bool was_on = al::g_on;
    auto run = [](std::uint64_t base, bool swap) {
        al::testEnable(true, true);
        al::onDevAlloc(base + 0x1000, 1u << 20);
        al::onHostAlloc(base + 0x900000, 4096);
        al::onDevAlloc(base + 0x200000, swap ? 512 : 256);
        al::onDevFree(base + 0x1000);
        al::onHostFree(base + 0x900000);
        al::onDevFree(base + 0x777);  // never allocated
        return al::testText();
    };
    const std::string a = run(0x7f0000000000ull, false);
    const std::string b = run(0x100000000ull, false);  // other addresses, same sizes and order
    CHECK(a == b);
    CHECK(a.find("# events 6 seqhash ") == 0);
    CHECK(a.find("0 dev 1048576 #0\n") != std::string::npos);
    CHECK(a.find("1 host 4096 #0\n") != std::string::npos);
    CHECK(a.find("2 dev 256 #1\n") != std::string::npos);
    CHECK(a.find("3 free #0 1048576\n") != std::string::npos);
    CHECK(a.find("4 hostfree #0 4096\n") != std::string::npos);
    CHECK(a.find("5 free #? 0\n") != std::string::npos);
    CHECK(a.find("# unknown frees 1\n") != std::string::npos);
    CHECK(a.find("peak 1048832 bytes") != std::string::npos);
    const std::string c = run(0x7f0000000000ull, true);  // one size differs
    CHECK(c.substr(0, 40) != a.substr(0, 40));          // the seqhash line
    // summary only (the server default): same seqhash, no event lines
    al::testEnable(true, false);
    al::onDevAlloc(0x7f0000001000ull, 1u << 20);
    al::onHostAlloc(0x7f0000900000ull, 4096);
    al::onDevAlloc(0x7f0000200000ull, 256);
    al::onDevFree(0x7f0000001000ull);
    al::onHostFree(0x7f0000900000ull);
    al::onDevFree(0x7f0000000777ull);
    const std::string d = al::testText();
    CHECK(d.substr(0, d.find('\n')) == a.substr(0, a.find('\n')));
    CHECK(d.find(" dev 1048576 #0") == std::string::npos);
    CHECK(d.find("# list off\n") != std::string::npos);
    // cap
    al::testEnable(true, true, 2);
    for (int i = 0; i < 5; ++i) al::onDevAlloc(0x10000ull * (i + 1), 64);
    const std::string e = al::testText();
    CHECK(e.find("# events 5 ") == 0);
    CHECK(e.find("1 dev 64 #1\n") != std::string::npos);
    CHECK(e.find("2 dev 64 #2\n") == std::string::npos);
    CHECK(e.find("# lines dropped 3 (kept the first 2)\n") != std::string::npos);
    // server process: list off unless WHIRL_ALLOC_LOG_RAW=server
    al::testEnable(true, true);
    al::serverProcess();
    al::onDevAlloc(0x10000ull, 64);
    const char* rv = std::getenv("WHIRL_ALLOC_LOG_RAW");
    const bool want_list = rv != nullptr && std::string(rv) == "server";
    CHECK_EQ(al::testText().find("0 dev 64 #0\n") != std::string::npos, want_list);
    // off: nothing recorded
    al::testEnable(false, true);
    al::onDevAlloc(0x10000ull, 64);
    al::onDevFree(0x10000ull);
    CHECK(al::testText().find("# events 0 ") == 0);
    al::testEnable(was_on, true);
}

// OVH-C1: the MTP draft window (Model::attn_win) must be restored when the draft attention throws,
// otherwise the next request's trunk attention would inherit the window and drop keys.
void testValueScope() {
    std::int32_t win = 0;
    {
        whirl::ValueScope<std::int32_t> s(win, 0x4005);
        CHECK_EQ(win, 0x4005);
    }
    CHECK_EQ(win, 0);
    bool threw = false;
    try {
        whirl::ValueScope<std::int32_t> s(win, 0x4005);
        throw std::runtime_error("attnBlock failed");
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
    CHECK_EQ(win, 0);  // restored by the destructor during unwinding
    win = 7;           // a nested scope restores the previous value, not blindly 0
    {
        whirl::ValueScope<std::int32_t> s(win, 9);
        CHECK_EQ(win, 9);
    }
    CHECK_EQ(win, 7);
}

// VIS-8S-2: device-local memory taken, per device class (numbers from the 8060S / R9700 probes:
// pinned 887 MiB moves hipMemGetInfo by 2 x 887 on the integrated GPU, 0 on the discrete one).
void testLocalMemTaken() {
    using whirl::hip::LocalMemSnap;
    using whirl::hip::MemClass;
    constexpr std::uint64_t M = 1ull << 20;
    auto snap = [&](MemClass c, std::uint64_t fr, bool pdh, std::uint64_t ded, std::uint64_t sh) {
        LocalMemSnap s;
        s.cls = c;
        s.hip_free = fr * M;
        s.pdh_ok = pdh;
        s.dedicated = ded * M;
        s.shared = sh * M;
        return s;
    };
    // integrated: pinned mmproj copy (system RAM) -> 0 taken although hipMemGetInfo dropped 1774 MiB
    auto t = whirl::hip::localMemTaken(snap(MemClass::integrated, 50000, true, 166, 10), snap(MemClass::integrated, 48226, true, 166, 897));
    CHECK_EQ(t.mib, 0.0);
    CHECK_EQ(t.hip_free_mib, 1774.0);
    CHECK_EQ(t.shared_mib, 887.0);
    // integrated: a 16 MiB device allocation (carve-out) is caught
    t = whirl::hip::localMemTaken(snap(MemClass::integrated, 50000, true, 166, 10), snap(MemClass::integrated, 47210, true, 182, 897));
    CHECK_EQ(t.mib, 16.0);
    // integrated, counters unreadable: falls back to hipMemGetInfo (over-reports: fails, never passes wrongly)
    t = whirl::hip::localMemTaken(snap(MemClass::integrated, 50000, false, 0, 0), snap(MemClass::integrated, 48226, false, 0, 0));
    CHECK_EQ(t.mib, 1774.0);
    t = whirl::hip::localMemTaken(snap(MemClass::integrated, 50000, true, 166, 0), snap(MemClass::integrated, 48226, false, 0, 0));
    CHECK_EQ(t.mib, 1774.0);
    // discrete: hipMemGetInfo, whatever the counters say
    t = whirl::hip::localMemTaken(snap(MemClass::discrete, 9000, true, 185, 7), snap(MemClass::discrete, 9000, true, 185, 894));
    CHECK_EQ(t.mib, 0.0);
    CHECK_EQ(t.shared_mib, 887.0);
    t = whirl::hip::localMemTaken(snap(MemClass::discrete, 9000, true, 185, 7), snap(MemClass::discrete, 8984, true, 185, 7));
    CHECK_EQ(t.mib, 16.0);
    CHECK(std::string(t.how).find("discrete") != std::string::npos);
    // given back: negative
    t = whirl::hip::localMemTaken(snap(MemClass::discrete, 8984, false, 0, 0), snap(MemClass::discrete, 9000, false, 0, 0));
    CHECK_EQ(t.mib, -16.0);
}

// UMA-1: the integrated-GPU carve-out check (pure parts of whirl/uma_mem.h)
void testUmaCheck() {
    using whirl::hip::UmaMem;
    using whirl::hip::UmaNeed;
    constexpr std::uint64_t GiB = 1ull << 30;
    UmaMem u;
    u.integrated = true;
    u.known = true;
    u.dedicated = 64 * GiB;
    u.shared = 48 * GiB;
    CHECK_EQ(whirl::hip::umaSizeText(u), std::string("dedicated 64.0 GiB (+ shared 48.0 GiB)"));
    UmaNeed n;
    n.weights = 16 * GiB;
    n.fixed = 20 * GiB;
    n.kv_per_token = 64 * 1024;
    n.min_tokens = 131072;   // 8 GiB of KV
    n.want_tokens = 262144;  // 16 GiB of KV
    // 64 GiB: fits the full pool, no refusal, no cap
    CHECK(whirl::hip::umaRefusal(u, n, "AMD Radeon(TM) 8060S Graphics").empty());
    CHECK(!whirl::hip::umaPoolCap(u, n, 256).has_value());
    // 32 GiB: one request fits (20 + 8 + 1.5), the full pool does not -> pool capped to the room
    u.dedicated = 32 * GiB;
    CHECK(whirl::hip::umaRefusal(u, n, "x").empty());
    const auto cap = whirl::hip::umaPoolCap(u, n, 256);
    CHECK(cap.has_value());
    if (cap) {
        CHECK_EQ(*cap % 256, 0u);
        CHECK(*cap >= n.min_tokens && *cap < n.want_tokens);
        CHECK(n.fixed + static_cast<std::uint64_t>(*cap) * n.kv_per_token <= whirl::hip::umaUsable(u.dedicated));
    }
    // 0.5 GiB (issue #7): refused, the message names the setting and the size to set
    u.dedicated = GiB / 2;
    const std::string why = whirl::hip::umaRefusal(u, n, "AMD Radeon(TM) 8060S Graphics");
    CHECK(!why.empty());
    CHECK(why.find("0.5 GB of dedicated GPU memory") != std::string::npos);
    CHECK(why.find("Armoury Crate") != std::string::npos);
    CHECK(why.find("BIOS") != std::string::npos);
    CHECK(why.find("at least 30 GB") != std::string::npos);   // 20 + 8 + 1.5 = 29.5 -> 30
    CHECK(why.find("38 GB for the full 262144-token") != std::string::npos);  // 20 + 16 + 1.5 = 37.5 -> 38
    CHECK(why.find("WHIRL_UMA_CHECK=0") != std::string::npos);
    CHECK(why.find("simulated") == std::string::npos);
    u.fake = true;
    CHECK(whirl::hip::umaRefusal(u, n, "x").find("simulated: WHIRL_FAKE_DEDICATED_GB") != std::string::npos);
    // a discrete GPU or an unknown carve-out: never refused, never capped, no 719 hint
    UmaMem d;
    d.dedicated = 0;
    CHECK(whirl::hip::umaRefusal(d, n, "x").empty());
    CHECK(!whirl::hip::umaPoolCap(d, n, 256).has_value());
    CHECK(whirl::hip::uma719Hint(&d).empty());
    CHECK(whirl::hip::uma719Hint(nullptr).empty());
    UmaMem unk;
    unk.integrated = true;
    CHECK(whirl::hip::umaRefusal(unk, n, "x").empty());
    // hipError 719 on the iGPU: the hint names the carve-out size and the setting
    const std::string h = whirl::hip::uma719Hint(&u);
    CHECK(h.find("0.5 GB") != std::string::npos);
    CHECK(h.find("Armoury Crate") != std::string::npos);
}

#if WHIRL_TEST_DLL_DEPS
// Issue #2: amdhip64_7.dll present but its import rocm_kpack.dll missing (error 126) must not be
// reported as "amdhip64_7.dll was not found". Optional real case: WHIRL_TEST_HIP3684 = path of a copy
// of the ROCm 10.1 driver's amdhip64_7.dll (10.0.3684) in a folder without rocm_kpack.dll.
void testDllDeps() {
    namespace dd = whirl::app::dlldeps;
    auto has = [](const std::vector<std::string>& v, const char* name) {
        for (const std::string& s : v)
            if (_stricmp(s.c_str(), name) == 0) return true;
        return false;
    };
    wchar_t self[MAX_PATH];
    const DWORD n = GetModuleFileNameW(nullptr, self, MAX_PATH);
    CHECK(n > 0 && n < MAX_PATH);
    const std::vector<std::string> imp = dd::peImports(self);
    CHECK(has(imp, "KERNEL32.dll"));
    CHECK(dd::peImports(L"C:\\no\\such\\file_whirl_test.dll").empty());
    CHECK(!dd::findDll(L"kernel32.dll").empty());
    CHECK(dd::findDll(L"no_such_dll_whirl_test.dll").empty());

    // the DLL itself is missing: the old wording, with the driver advice
    dd::Diagnosis none = dd::diagnose(L"no_such_dll_whirl_test.dll");
    CHECK(none.found_at.empty());
    std::string m = dd::message126("amdhip64_7.dll", none, "ADVICE\n");
    CHECK(m.find("amdhip64_7.dll) was not found") != std::string::npos);
    CHECK(m.find("ADVICE") != std::string::npos);

    // present, rocm_kpack.dll missing: names it and says where to put it, no driver advice
    dd::Diagnosis k;
    k.found_at = L"C:\\Windows\\System32\\amdhip64_7.dll";
    k.missing = {"rocm_kpack.dll"};
    m = dd::message126("amdhip64_7.dll", k, "ADVICE\n");
    CHECK(m.find("was found at C:\\Windows\\System32\\amdhip64_7.dll") != std::string::npos);
    CHECK(m.find("missing: rocm_kpack.dll") != std::string::npos);
    CHECK(m.find("next to whirl.exe") != std::string::npos);
    CHECK(m.find("was not found") == std::string::npos);
    CHECK(m.find("ADVICE") == std::string::npos);

    // rocm_kpack.dll present, but the Visual C++ runtime it needs is missing
    dd::Diagnosis v;
    v.found_at = L"C:\\Windows\\System32\\amdhip64_7.dll";
    v.missing = {"VCRUNTIME140.dll", "MSVCP140.dll"};
    m = dd::message126("amdhip64_7.dll", v, "ADVICE\n");
    CHECK(m.find("missing: VCRUNTIME140.dll, MSVCP140.dll") != std::string::npos);
    CHECK(m.find("from the WHIRL zip") != std::string::npos);  // 0.2.2 ships them app-locally
    CHECK(m.find("vc_redist.x64.exe") != std::string::npos);

    // present, nothing identifiable: still says "found", plus the driver advice
    dd::Diagnosis u;
    u.found_at = L"C:\\x\\amdhip64_7.dll";
    m = dd::message126("amdhip64_7.dll", u, "ADVICE\n");
    CHECK(m.find("was found at C:\\x\\amdhip64_7.dll") != std::string::npos);
    CHECK(m.find("ADVICE") != std::string::npos);

    // a present DLL whose imports all load: nothing missing
    dd::Diagnosis ok = dd::diagnose(self);
    CHECK(!ok.found_at.empty());
    CHECK(ok.missing.empty());

    if (const char* p = std::getenv("WHIRL_TEST_HIP3684"); p && *p) {
        const std::wstring wp(p, p + std::strlen(p));
        CHECK(has(dd::peImports(wp), "rocm_kpack.dll"));
        const dd::Diagnosis r = dd::diagnose(wp);
        CHECK(!r.found_at.empty());
        CHECK(has(r.missing, "rocm_kpack.dll"));
        std::printf("dll_deps: real 3684 case checked (%s)\n", p);
    }

    // ---- error 127 (0.2.2): a shipped DLL shadows the version a (future) driver expects.
    // The message must say which file to rename/remove or to use the matching WHIRL release,
    // and must never tell the user to update or install the driver.
    auto no_driver_advice = [](const std::string& s) {
        return s.find("update the driver") == std::string::npos && s.find("Install AMD Software") == std::string::npos &&
               s.find("newer driver") == std::string::npos;
    };
    dd::Shadow127 s;
    s.found_at = L"C:\\Windows\\System32\\amdhip64_7.dll";
    s.app_dir = L"C:\\whirl";
    s.bundled = {"rocm_kpack.dll", "msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll"};
    s.other_kpack = L"C:\\Windows\\System32\\rocm_kpack.dll";
    m = dd::message127("amdhip64_7.dll", s);
    CHECK(m.find("error 127") != std::string::npos);
    CHECK(m.find("at C:\\Windows\\System32\\amdhip64_7.dll") != std::string::npos);
    CHECK(m.find("C:\\whirl\\rocm_kpack.dll") != std::string::npos);
    CHECK(m.find("rename or remove rocm_kpack.dll in C:\\whirl") != std::string::npos);
    CHECK(m.find("WHIRL release that matches your driver") != std::string::npos);
    CHECK(m.find("own copy: C:\\Windows\\System32\\rocm_kpack.dll") != std::string::npos);
    CHECK(m.find("also rename msvcp140.dll, vcruntime140.dll, vcruntime140_1.dll") != std::string::npos);
    CHECK(no_driver_advice(m));
    // kpack shipped, no driver copy found, no VC runtime next to it
    s.bundled = {"rocm_kpack.dll"};
    s.other_kpack.clear();
    m = dd::message127("amdhip64_7.dll", s);
    CHECK(m.find("rename or remove rocm_kpack.dll") != std::string::npos);
    CHECK(m.find("own copy") == std::string::npos);
    CHECK(m.find("also rename") == std::string::npos);
    CHECK(no_driver_advice(m));
    // only the VC++ runtime shipped (kpack removed by the user): names those files
    s.bundled = {"msvcp140.dll", "vcruntime140.dll"};
    m = dd::message127("amdhip64_7.dll", s);
    CHECK(m.find("rename or remove msvcp140.dll, vcruntime140.dll in C:\\whirl") != std::string::npos);
    CHECK(m.find("WHIRL release that matches your driver") != std::string::npos);
    CHECK(no_driver_advice(m));
    // nothing shipped present
    s.bundled.clear();
    m = dd::message127("amdhip64_7.dll", s);
    CHECK(m.find("None of the DLLs WHIRL ships") != std::string::npos);
    CHECK(m.find("WHIRL release that matches your driver") != std::string::npos);
    CHECK(no_driver_advice(m));

    // inspect127 on a real folder: finds the shipped files that exist, ignores the rest
    {
        wchar_t tmp[MAX_PATH];
        CHECK(GetTempPathW(MAX_PATH, tmp) > 0);
        const std::wstring dir = std::wstring(tmp) + L"whirl_test_127_" + std::to_wstring(GetCurrentProcessId());
        CHECK(CreateDirectoryW(dir.c_str(), nullptr) != 0);
        const std::wstring files[] = {dir + L"\\rocm_kpack.dll", dir + L"\\vcruntime140.dll", dir + L"\\amdhip64_7.dll"};
        for (const std::wstring& f : files) {
            HANDLE h = CreateFileW(f.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
            CHECK(h != INVALID_HANDLE_VALUE);
            if (h != INVALID_HANDLE_VALUE) CloseHandle(h);
        }
        const dd::Shadow127 r = dd::inspect127(dir + L"\\amdhip64_7.dll", dir + L"\\");
        CHECK(r.found_at == dir + L"\\amdhip64_7.dll");
        CHECK(r.app_dir == dir);
        CHECK(r.bundled.size() == 2 && has(r.bundled, "rocm_kpack.dll") && has(r.bundled, "vcruntime140.dll"));
        // the runtime's own folder is the app folder here, so its kpack is ours, not "the driver's copy"
        CHECK(r.other_kpack.find(dir) == std::wstring::npos);
        for (const std::wstring& f : files) DeleteFileW(f.c_str());
        RemoveDirectoryW(dir.c_str());
    }
    CHECK(!dd::exeDir().empty());

    // Optional real case: WHIRL_TEST_KPACK127_DIR = folder with the ROCm 10.1 driver's amdhip64_7.dll
    // and a rocm_kpack.dll that lacks its exports (a stand-in for a mismatched kpack). The loader must
    // report 127 and the message must point at that folder's rocm_kpack.dll.
    if (const char* p = std::getenv("WHIRL_TEST_KPACK127_DIR"); p && *p) {
        const std::wstring dir(p, p + std::strlen(p));
        const std::wstring hip = dir + L"\\amdhip64_7.dll";
        const UINT old_mode = SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOOPENFILEERRORBOX);
        HMODULE h = LoadLibraryExW(hip.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
        const DWORD e = h ? 0 : GetLastError();
        SetErrorMode(old_mode);
        CHECK(h == nullptr && e == ERROR_PROC_NOT_FOUND);
        if (h) FreeLibrary(h);
        const dd::Shadow127 r = dd::inspect127(hip, dir);
        CHECK(has(r.bundled, "rocm_kpack.dll"));
        const std::string msg = dd::message127("amdhip64_7.dll", r);
        CHECK(msg.find("rename or remove rocm_kpack.dll in " + std::string(p)) != std::string::npos);
        CHECK(no_driver_advice(msg));
        std::printf("dll_deps: real 127 case checked (%s, LoadLibrary error %lu)\n%s", p, e, msg.c_str());
    }
}
#endif

// ---------------------------------------------------------------------------
// WDDM-LOG: runtime video memory watch (formatting, spill thresholds, rate limit, recovery, decode drop)
void testWddmWatch() {
    using namespace whirl::wddmwatch;
    constexpr std::uint64_t MiB = 1ull << 20;
    auto smp = [&](std::uint64_t used_mib, std::uint64_t budget_mib, std::uint64_t shared_mib, std::uint64_t pinned_mib = 0) {
        Sample s;
        s.ok = true;
        s.local_usage = used_mib * MiB;
        s.local_budget = budget_mib * MiB;
        s.nonlocal_usage = shared_mib * MiB;
        s.pinned = pinned_mib * MiB;
        return s;
    };
    auto has = [](const std::optional<Event>& e, const char* t) { return e && e->text.find(t) != std::string::npos; };
    const Sample base = smp(29000, 31232, 400);
    // formatting
    CHECK_EQ(format(smp(29798, 31232, 412), base), std::string("vram local 29.10 / budget 30.50 GiB, shared 412 MiB (+12 MiB since start)"));
    CHECK_EQ(format(base, base), std::string("vram local 28.32 / budget 30.50 GiB, shared 400 MiB (+0 MiB since start)"));
    CHECK_EQ(format(smp(29000, 31232, 397), base), std::string("vram local 28.32 / budget 30.50 GiB, shared 397 MiB (-3 MiB since start)"));
    CHECK_EQ(format(smp(29000, 31232, 400), Sample{}), std::string("vram local 28.32 / budget 30.50 GiB, shared 400 MiB"));
    CHECK_EQ(format(Sample{}, base), std::string());
    {
        Sample r = smp(29000, 31232, 400, 9318);
        r.rss = 18636ull * MiB;
        CHECK_EQ(format(r, base), std::string("vram local 28.32 / budget 30.50 GiB, shared 400 MiB (+0 MiB since start), ram 18.20 GiB (pinned 9.10 GiB)"));
        r.rss = 0;
        CHECK_EQ(format(r, Sample{}), std::string("vram local 28.32 / budget 30.50 GiB, shared 400 MiB, pinned ram 9.10 GiB"));
    }

    // no WDDM numbers: never warns
    {
        Watch w(Sample{});
        CHECK(!w.check(smp(40000, 31232, 9000), 0.0));
        Watch w2(base);
        CHECK(!w2.check(Sample{}, 0.0));
    }
    // under the budget, growth up to 256 MiB: quiet
    {
        Watch w(base);
        CHECK(!w.check(base, 0.0));
        CHECK(!w.check(smp(31232, 31232, 400 + 256), 10.0));  // at the budget, exactly +256: no spill
        CHECK(!w.spilling());
        CHECK_EQ(w.warnings(), 0u);
    }
    // pinned host memory WHIRL allocated later (vision encoder, checkpoints) is not a spill
    {
        Watch w(smp(29000, 31232, 400, 2000));
        CHECK(!w.check(smp(29000, 31232, 400 + 1000, 3000), 0.0));
        CHECK(w.check(smp(29000, 31232, 400 + 1300, 3000), 0.0).has_value());  // 300 MiB beyond the new pinned memory
    }
    // shared growth: warning at the start (all four parts), repeated at most once a minute while it lasts,
    // INFO when it is over
    {
        Watch w(base);
        const auto a = w.check(smp(29000, 31232, 400 + 300), 100.0);
        CHECK(a && a->warn);
        CHECK(a && a->text.rfind("GPU memory spill: WHIRL's shared GPU memory grew by 300 MiB since startup: Windows has likely "
                                 "moved part of WHIRL's GPU buffers to system RAM", 0) == 0);
        CHECK(has(a, "Generation slows down if those buffers are used at every step: decode can become several times slower."));
        CHECK(has(a, "What you can do: close other programs that use the GPU (browsers, games, other AI tools); or restart the "
                     "server, which sizes the KV pool from the budget at startup, and set WHIRL_POOL_BUDGET_MARGIN_MB=4096 (or "
                     "more) to keep more room free for other programs."));
        // review P2-1 / P3-1: never the variable that turns the budget rule off, never --parallel 1
        CHECK(!has(a, "WHIRL_POOL_RESERVE_MB"));
        CHECK(!has(a, "--parallel"));
        CHECK(has(a, "Now: vram local 28.32 / budget 30.50 GiB, shared 700 MiB (+300 MiB since start)"));
        CHECK(!has(a, "over the Windows"));
        CHECK(w.spilling());
        CHECK(!w.check(smp(29000, 31232, 400 + 300), 130.0));  // within the minute
        CHECK(!w.check(smp(29000, 31232, 400 + 900), 150.0));  // grew, still within the minute
        const auto b = w.check(smp(29000, 31232, 400 + 900), 160.0);
        CHECK(b && b->warn);
        CHECK(b && b->text.rfind("GPU memory spill continues (60 s so far): WHIRL's shared GPU memory grew by 900 MiB", 0) == 0);
        const auto c = w.check(smp(29000, 31232, 400 + 900), 400.0);
        CHECK(c && c->text.rfind("GPU memory spill continues (5 min so far): ", 0) == 0);
        const auto d = w.check(smp(29000, 31232, 400 + 100), 410.0);
        CHECK(d && !d->warn);
        CHECK(d && d->text.rfind("GPU memory spill is over (it lasted 5 min): WHIRL's GPU memory is back under the Windows video "
                                 "memory budget and its shared GPU memory is back within 256 MiB of startup, so decode is no "
                                 "longer slowed by it. Now: vram local 28.32",
                                 0) == 0);
        CHECK(!w.spilling());
        CHECK(!w.check(smp(29000, 31232, 400 + 100), 420.0));  // one INFO only
        CHECK_EQ(w.warnings(), 3u);
    }
    // local over the budget: named with the amount; a new spill within the minute after the last warning
    // waits for the minute; one that ends before any warning gets no INFO line
    {
        Watch w(base);
        const auto a = w.check(smp(31300, 31232, 400), 0.0);
        CHECK(a && a->warn);
        CHECK(a && a->text.rfind("GPU memory spill: WHIRL's GPU memory is 0.07 GiB over the Windows video memory (WDDM) budget, "
                                 "so Windows can page part of WHIRL's GPU buffers out of VRAM (this can happen before the shared "
                                 "GPU memory number grows).",
                                 0) == 0);
        CHECK(has(a, "Generation is likely slower now: decode can become several times slower."));
        CHECK(has(w.check(smp(30000, 31232, 400), 10.0), "GPU memory spill is over (it lasted 10 s)"));
        CHECK(!w.check(smp(31300, 31232, 400), 20.0));  // new spill, 20 s after the last warning: held
        CHECK(!w.check(smp(30000, 31232, 400), 30.0));  // over before it was reported: no INFO
        CHECK(!w.check(smp(31300, 31232, 400), 40.0));
        const auto b = w.check(smp(31300, 31232, 400), 61.0);
        CHECK(b && b->text.rfind("GPU memory spill: ", 0) == 0);  // a new spill, not "continues"
        CHECK_EQ(w.warnings(), 2u);
    }
    // over the budget and grown: both in one sentence
    {
        Watch w(base);
        const auto a = w.check(smp(31300, 31232, 1000), 0.0);
        CHECK(has(a, "GPU memory spill: WHIRL's GPU memory is 0.07 GiB over the Windows video memory (WDDM) budget, and Windows "
                     "has likely moved part of WHIRL's GPU buffers to system RAM (shared GPU memory +600 MiB since startup)."));
        CHECK(has(a, "Generation is likely slower now"));
    }
    // the KV pool size in the advice (review P3-3), other programs' VRAM note
    {
        Config c;
        c.pool_tokens = 334080;
        Watch w(base, c);
        const auto a = w.check(smp(31300, 31232, 400), 0.0);
        CHECK(has(a, "or restart the server, which sizes the KV pool (the KV pool is now 334080 tokens) from the budget at startup"));
        CHECK_EQ(othersNote(Others{3328 * MiB, 205 * MiB}),
                 std::string(" Other programs use 3.25 GiB of VRAM on this GPU now (0.20 GiB when the server started)."));
        CHECK_EQ(othersNote(Others{1024 * MiB, std::nullopt}), std::string(" Other programs use 1.00 GiB of VRAM on this GPU now."));
    }
    // a spill already present at startup is seen against the sizing-time baseline (review P3-4): the server passes
    // the read the pool was sized from, not a sample taken after everything was allocated
    {
        const Sample sized = smp(17640, 31760, 13692, 13576);       // before the pool: embedding + RAM tier pinned
        Watch w(sized);
        CHECK(w.check(smp(30260, 31760, 13692 + 600, 13576), 0.0).has_value());  // 600 MiB demoted while starting
        Watch ok(sized);
        CHECK(!ok.check(smp(30260, 31760, 13692 + 300, 13576 + 300), 0.0));     // 300 MiB pinned checkpoints (WHIRL_CKPT_HOST=1)
    }
    // config defaults (the server uses them: 256 MiB, 60 s)
    {
        Config c;
        CHECK_EQ(c.grow_warn, 256 * MiB);
        CHECK(c.min_gap_s == 60.0);
    }

    // decode speed drop within a request
    {
        DecodeWatch d;
        CHECK(!d.report(100, 1000.0, 25, false));  // before the early window (200 tokens)
        CHECK(!d.report(200, 2000.0, 50, false));  // early: 100 tok/s, 40 ms per cycle
        CHECK(!d.report(300, 3000.0, 75, false));  // 1 s window: too short to judge, keeps growing
        CHECK(!d.report(400, 4500.0, 100, false));  // 2.5 s window at 80 tok/s: -20%, fine
        const auto a = d.report(500, 7000.0, 125, true);  // 100 tok in 2.5 s = 40 tok/s, 100 ms per cycle, spill seen
        CHECK(a && a->warn);
        CHECK(a && a->text == "decode slowed from 100.0 to 40.0 tok/s (60% slower than early in this request; 40.0 -> 100.0 ms "
                              "per decode cycle), likely because of the GPU memory spill above");
        CHECK(!d.report(600, 10000.0, 150, true));  // once per request
        d.reset();
        CHECK(!d.report(200, 2000.0, 50, false));
        const auto b = d.report(300, 6000.0, 150, false);  // 25 tok/s at the same 40 ms per cycle: fewer tokens per cycle
        CHECK(b && !b->warn);
        CHECK(b && b->text == "decode slowed from 100.0 to 25.0 tok/s (75% slower than early in this request; 40.0 -> 40.0 ms per "
                              "decode cycle); each cycle is not slower: fewer tokens per cycle (fewer drafts accepted)");
        DecodeWatch e;
        CHECK(!e.report(200, 2000.0, 0, false));
        const auto c = e.report(300, 5000.0, 0, false);  // no cycle numbers
        CHECK(c && c->text == "decode slowed from 100.0 to 33.3 tok/s (67% slower than early in this request); no GPU memory spill is seen");
        DecodeWatch f;
        CHECK(!f.report(200, 2000.0, 50, false));
        CHECK(!f.report(300, 3600.0, 75, false));  // 62.5 tok/s: -37.5%, under the 40% line
        // review P3-5: a spill while each cycle is NOT slower is not blamed (draft acceptance fell)
        DecodeWatch g;
        CHECK(!g.report(200, 2000.0, 50, true));
        const auto gs = g.report(300, 6000.0, 150, true);
        CHECK(gs && !gs->warn);
        CHECK(gs && gs->text == "decode slowed from 100.0 to 25.0 tok/s (75% slower than early in this request; 40.0 -> 40.0 ms per "
                                "decode cycle); each cycle is not slower: fewer tokens per cycle (fewer drafts accepted), not the "
                                "GPU memory spill");
        // another request prefilled in the window (--parallel >= 2), no spill: named as the likely cause
        DecodeWatch q;
        CHECK(!q.report(200, 2000.0, 50, false, 1000));
        const auto qs = q.report(300, 7000.0, 75, false, 9192);  // 100 tok in 5 s, 200 ms per cycle; 8192 prefill tokens since
        CHECK(qs && !qs->warn);
        CHECK(qs && qs->text == "decode slowed from 100.0 to 20.0 tok/s (80% slower than early in this request; 40.0 -> 200.0 ms per "
                                "decode cycle); no GPU memory spill is seen; another request's prefill shared the GPU in this window "
                                "(with --parallel 2 or more, prefill chunks run between decode cycles)");
        // both: the spill is named, the prefill too
        DecodeWatch r;
        CHECK(!r.report(200, 2000.0, 50, true, 0));
        const auto rs = r.report(300, 7000.0, 75, true, 4096);
        CHECK(rs && rs->warn);
        CHECK(rs && rs->text.find(", likely because of the GPU memory spill above; also, another request's prefill shared the GPU") !=
                        std::string::npos);
    }
}

// ---------------------------------------------------------------------------
// issue #5: KV pool re-fit to the WDDM budget read after the pool is allocated
void testPoolBudget() {
    using whirl::poolbudget::fitPool;
    constexpr std::uint64_t MiB = 1ull << 20;
    constexpr double GiB = 1073741824.0;
    auto gib = [&](double g) { return static_cast<std::uint64_t>(g * GiB); };
    const std::uint64_t per_tok = 36966;  // 36.1 KiB/token (q8h KV, Swift-1.5 27B), as in the issue #5 log
    const std::uint32_t page = 256;
    const std::uint64_t margin = 768 * MiB;
    // the user's numbers: sized with budget 31.02 GiB, used 17.64 GiB -> room 12.63 GiB -> pool 366592 tokens
    const std::uint64_t room = gib(31.02) - gib(17.64) - margin;
    const std::uint32_t pool0 = static_cast<std::uint32_t>(room / per_tok / page * page);
    CHECK(pool0 >= 366080 && pool0 <= 367104);  // within two pages of the log's 366592 (rounded GiB figures)
    // with the pool allocated: local 29.92 GiB, budget 29.55 GiB (after pinning) -> must shrink
    {
        const auto t = fitPool(pool0, gib(29.92), gib(29.55), margin, per_tok, page, 4096);
        CHECK(t.shrink);
        CHECK(!t.at_floor);
        CHECK_EQ(t.pool_tokens % page, 0u);
        CHECK(t.over == gib(29.92) + margin - gib(29.55));
        // after freeing the dropped pages, local + margin fits under the post-pin budget
        const std::uint64_t freed = static_cast<std::uint64_t>(pool0 - t.pool_tokens) * per_tok;
        CHECK(gib(29.92) - freed + margin <= gib(29.55));
        // and not more than one page beyond what is needed
        CHECK(gib(29.92) - freed + margin + page * per_tok > gib(29.55));
        std::printf("pool budget: issue #5 numbers: pool %u -> %u tokens (%.2f -> %.2f GiB)\n", pool0, t.pool_tokens,
                    pool0 * static_cast<double>(per_tok) / GiB, t.pool_tokens * static_cast<double>(per_tok) / GiB);
    }
    // fits: no change (also exactly at the line)
    {
        const auto t = fitPool(pool0, gib(29.00), gib(31.02), margin, per_tok, page, 4096);
        CHECK(!t.shrink);
        CHECK_EQ(t.over, 0u);
        CHECK_EQ(t.pool_tokens, pool0);
        const auto e = fitPool(pool0, gib(29.00), gib(29.00) + margin, margin, per_tok, page, 4096);
        CHECK(!e.shrink);
    }
    // budget unknown: no change
    CHECK(!fitPool(pool0, gib(40.0), 0, margin, per_tok, page, 4096).shrink);
    // far over: kept at the smallest pool, flagged
    {
        const auto t = fitPool(65536, gib(40.0), gib(20.0), margin, per_tok, page, 4096);
        CHECK(t.shrink);
        CHECK(t.at_floor);
        CHECK_EQ(t.pool_tokens, 4096u);
        const auto u = fitPool(4096, gib(40.0), gib(20.0), margin, per_tok, page, 4096);  // already there
        CHECK(!u.shrink);
        CHECK(u.at_floor);
        CHECK(u.over > 0);
    }
    // one byte over: one page less
    {
        const auto t = fitPool(8192, gib(10.0) + 1, gib(10.0) + margin, margin, per_tok, page, 4096);
        CHECK_EQ(t.pool_tokens, 8192u - page);
    }
    // the shrink loop guard: another round only when the last free really lowered the local usage
    {
        using whirl::poolbudget::usageFell;
        const std::uint64_t freed = gib(1.0);
        CHECK(usageFell(gib(30.0), gib(29.0), freed));
        CHECK(usageFell(gib(30.0), gib(29.5), freed));      // half: enough (WDDM counts in its own granularity)
        CHECK(!usageFell(gib(30.0), gib(29.6), freed));     // stale counter: stop
        CHECK(!usageFell(gib(30.0), gib(30.0), freed));
        CHECK(!usageFell(gib(30.0), gib(30.2), freed));
    }
}

// v0.2.2.1 default WDDM budget margin (owner 2026-10-21): other processes' VRAM the budget does not show yet + room
void testPoolMargin() {
    using namespace whirl::poolbudget;
    constexpr std::uint64_t MiB = 1ull << 20;
    auto m = [&](std::uint64_t budget_mib, std::uint64_t start_mib, bool display, bool known, std::uint64_t others_mib) {
        MarginIn in;
        in.local_budget = budget_mib * MiB;
        in.budget_start = start_mib * MiB;
        in.display = display;
        in.others_known = known;
        in.others = others_mib * MiB;
        return defaultMargin(in);
    };
    // R9700 headless, nothing else on it: budget 31.02 GiB, 6% = 1906 MiB < the 2048 floor
    {
        const Margin r = m(31764, 31764, false, true, 0);
        CHECK_EQ(r.bytes, 2048 * MiB);
        CHECK_EQ(r.room, 2048 * MiB);
        CHECK_EQ(r.pending, 0ull);
        CHECK(r.why.rfind("2048 MiB room (no display on this GPU)", 0) == 0);
    }
    // the hog test: another process holds 3307 MiB, the budget does not show it yet -> counted, plus the room
    {
        const Margin r = m(31764, 31764, false, true, 3307);
        CHECK_EQ(r.pending, 3307 * MiB);
        CHECK_EQ(r.bytes, (3307 + 2048) * MiB);
        CHECK(r.why.find("+ 3307 MiB of the 3307 MiB other processes use that the budget does not show yet") != std::string::npos);
    }
    // ... after the pool was allocated the budget dropped 2.94 GiB (31.02 -> 28.08): only the rest is still pending
    {
        const Margin r = m(28754, 31764, false, true, 3307);
        CHECK_EQ(r.pending, (3307 - (31764 - 28754)) * MiB);
        CHECK(r.why.find("(it dropped 3010 MiB since startup)") != std::string::npos);
        // dropped more than the others used: never negative, never counted twice
        CHECK_EQ(m(24000, 31764, false, true, 3307).pending, 0ull);
    }
    // a monitor on the card: 3072 MiB room
    CHECK_EQ(m(31764, 31764, true, true, 0).room, 3072 * MiB);
    CHECK(m(31764, 31764, true, true, 0).why.find("display is attached") != std::string::npos);
    // a large card: 6% of the budget (48 GiB budget: 2949.12 -> 2950 MiB)
    {
        const Margin r = m(49152, 49152, false, true, 0);
        CHECK_EQ(r.room, 2950 * MiB);
        CHECK(r.why.rfind("2950 MiB room (6% of the 49152 MiB budget)", 0) == 0);
    }
    // counters unreadable: the budget-only rule, said so
    {
        const Margin r = m(31764, 31764, false, false, 99999);
        CHECK_EQ(r.bytes, 2048 * MiB);
        CHECK(r.why.find("other processes' VRAM unknown") != std::string::npos);
    }
    // budget at start unknown: treated as now (nothing dropped)
    CHECK_EQ(m(31764, 0, false, true, 1000).pending, 1000 * MiB);
    // WHIRL_POOL_BUDGET_MARGIN_MB sets the room; pending is still counted
    {
        MarginIn in;
        in.local_budget = 31764 * MiB;
        in.budget_start = 31764 * MiB;
        in.others_known = true;
        in.others = 500 * MiB;
        in.room_override = 4096 * MiB;
        const Margin r = defaultMargin(in);
        CHECK_EQ(r.room, 4096 * MiB);
        CHECK_EQ(r.bytes, (4096 + 500) * MiB);
        CHECK(r.why.rfind("4096 MiB room (WHIRL_POOL_BUDGET_MARGIN_MB)", 0) == 0);
    }
    // whole MiB
    CHECK_EQ(m(31764, 31764, false, true, 1).bytes % MiB, 0ull);
}

// v0.2.2.2 runtime memory governor: decisions with injected budgets (R9700-like numbers: Swift balance 36966 B/token,
// pool 330240 tokens = 11.37 GiB, margin 2048 MiB)
void testMemGovernor() {
    using namespace whirl::memgov;
    constexpr std::uint64_t MiB = 1ull << 20, GiB = 1ull << 30;
    Config c;
    c.per_tok = 36966;
    c.page = 256;
    c.pool_start = 330240;
    c.pool_min = 4096;
    c.ctx = 131072;
    c.recover_s = 30;
    c.hold_s = 60;
    const std::uint64_t M = 2048 * MiB;
    const std::uint64_t B0 = 31764 * MiB;          // budget at startup
    const std::uint64_t L0 = B0 - M - 300 * MiB;   // sized with the margin (+ rounding)
    auto in = [&](double t, std::uint64_t local, std::uint64_t budget, std::uint32_t pool, bool idle, std::uint64_t parked = 0,
                  std::uint64_t freeable = 0) {
        Input x;
        x.now_s = t;
        x.local = local;
        x.budget = budget;
        x.margin = M;
        x.pool = pool;
        x.all_idle = idle;
        x.parked = parked;
        x.freeable = freeable;
        return x;
    };
    // normal runs: the budget wiggles by up to 1 GiB: never an action
    {
        Governor g(c);
        for (int i = 0; i < 200; ++i) {
            const std::uint64_t b = B0 - static_cast<std::uint64_t>((i * 37) % 1000) * MiB;
            const Plan p = g.tick(in(i * 5.0, L0, b, 330240, i % 3 == 0));
            CHECK(!p.free_cache && !p.pool_target && !p.hold_admission && !p.warn_slow && !p.episode_start);
            CHECK_EQ(p.ctx_new, 131072u);
        }
        CHECK(!g.reduced());
    }
    // drop mid-run: another program takes 4 GiB while a request runs
    Governor g(c);
    {
        const std::uint64_t B1 = B0 - 4 * GiB;
        Plan p = g.tick(in(100, L0, B1, 330240, false, 0, 600 * MiB));
        CHECK(p.episode_start && p.free_cache && p.ram_tier_note && p.hold_admission && p.shrink && p.warn_slow);
        CHECK(p.pool_target.has_value());
        // the deficit (local + margin - budget = 4096 - 300 MiB) in whole pages
        const std::uint64_t def = L0 + M - B1;
        const std::uint32_t pages = static_cast<std::uint32_t>((def + 36966ull * 256 - 1) / (36966ull * 256));
        CHECK_EQ(*p.pool_target, 330240u - pages * 256);
        CHECK_EQ(p.ctx_new, 131072u);  // the smaller pool still holds 131072
        // still running, 600 MiB of checkpoint buffers parked by step 1: same target (the parked buffers come back after)
        p = g.tick(in(105, L0 - 600 * MiB, B1, 330240, false, 600 * MiB));
        CHECK(!p.episode_start && !p.ram_tier_note && p.hold_admission);
        CHECK_EQ(*p.pool_target, 330240u - pages * 256);
        // the request ends: every slot idle -> the target is applied by the engine
        p = g.tick(in(110, L0 - 600 * MiB, B1, 330240, true, 600 * MiB));
        CHECK(p.shrink && !p.hold_admission && !p.warn_slow);
        const std::uint32_t small = *p.pool_target;
        g.resized(small, false, 110);
        CHECK_EQ(g.shrinks(), 1u);
        // after the re-allocation and the parked buffers' return: the full margin free, no further action
        const std::uint64_t L1 = L0 - static_cast<std::uint64_t>(330240 - small) * 36966;
        for (int i = 0; i < 20; ++i) {
            p = g.tick(in(115 + i * 5.0, L1, B1, small, i % 2 == 0));
            CHECK(!p.pool_target && !p.free_cache && !p.hold_admission);
        }
        CHECK(B1 - L1 >= M);
        CHECK(g.reduced());
        // recovery: the other program exits; nothing before recover_s (30 s) of room and hold (60 s after the last trigger)
        const std::uint64_t B2 = B0;
        bool grew = false;
        double t_grow = 0;
        for (double t = 220; t < 400 && !grew; t += 5) {
            p = g.tick(in(t, L1, B2, small, true));
            if (p.pool_target) {
                grew = true;
                t_grow = t;
                CHECK(*p.pool_target > small);
                CHECK(*p.pool_target <= 330240u);
                g.resized(*p.pool_target, true, t);
            }
        }
        CHECK(grew);
        CHECK(t_grow >= 220 + 30);
        // grown back to the startup pool: the episode ends at the next sample
        p = g.tick(in(t_grow + 5, L0, B2, 330240, true));
        CHECK(p.episode_end);
        CHECK(!g.reduced());
        CHECK_EQ(g.grows(), 1u);
        // no grow while a request runs
        Governor h(c);
        h.tick(in(0, L0, B1, 330240, true));
        h.resized(small, false, 0);
        for (double t = 5; t < 300; t += 5) CHECK(!h.tick(in(t, L1, B2, small, false)).pool_target.has_value());
    }
    // the context for new requests follows a pool smaller than ctx (8 GiB taken from a 5 GiB pool's server)
    {
        Config c2 = c;
        c2.pool_start = 147456;  // 5.08 GiB
        Governor g2(c2);
        const std::uint64_t L = B0 - M - 100 * MiB;
        const Plan p = g2.tick(in(0, L, B0 - 4 * GiB, 147456, false));
        CHECK(p.shrink && p.ctx_new == *p.pool_target && p.ctx_new < 131072u);
        CHECK(p.ctx_new >= 4096u);
        // more than the pool can give: the smallest pool, never below pool_min
        const Plan q = g2.tick(in(5, L, B0 - 12 * GiB, 147456, true));
        CHECK(q.shrink && *q.pool_target == 4096u);
    }
    // repeated oscillation (another program starts and exits every 40 s for an hour): the hysteresis and the doubling
    // hold keep re-allocations rare
    {
        Governor g3(c);
        std::uint32_t pool = 330240;
        std::uint32_t resizes = 0;
        for (double t = 0; t < 3600; t += 5) {
            const bool hog = static_cast<int>(t / 40) % 2 == 1;
            const std::uint64_t b = hog ? B0 - 4 * GiB : B0;
            const std::uint64_t local = L0 - static_cast<std::uint64_t>(330240 - pool) * 36966;
            const Plan p = g3.tick(in(t, local, b, pool, true));
            if (p.pool_target) {
                g3.resized(*p.pool_target, *p.pool_target > pool, t);
                pool = *p.pool_target;
                ++resizes;
            }
        }
        // every 40 s an episode would be 90 changes; the hold (60 s, doubling up to 15 min) caps it
        CHECK(resizes <= 20u);
        CHECK(g3.hold() > 60.0);
        CHECK(g3.shrinks() >= 1u);
        // REVIEW-GOV-1 P3-1: after quiet_reset_s (30 min) without an over-budget sample the doubled hold goes back to hold_s
        for (double t = 3600; t < 3600 + 1900; t += 5) {
            const Plan p = g3.tick(in(t, L0 - static_cast<std::uint64_t>(330240 - pool) * 36966, B0, pool, true));
            if (p.pool_target) {
                g3.resized(*p.pool_target, *p.pool_target > pool, t);
                pool = *p.pool_target;
            }
        }
        CHECK_EQ(pool, 330240u);
        g3.tick(in(5600, L0, B0 - 4 * GiB, 330240, true));
        CHECK(g3.hold() <= 60.0);
    }
    // lagging WDDM counters after a re-allocation (R9700 S1, 2026-10-21: a 4.8 GiB grow first read +8.1 GiB, so the governor
    // shrank right back): within trust_s the usage counts as at most the expected one; afterwards the measured one
    {
        Governor g5(c);
        const std::uint64_t B1 = B0 - 4 * GiB;
        Plan p = g5.tick(in(0, L0, B1, 330240, true));
        CHECK(p.shrink);
        const std::uint32_t small = *p.pool_target;
        const std::uint64_t L1 = L0 - static_cast<std::uint64_t>(330240 - small) * 36966;
        g5.resized(small, false, 0, L1);
        // the free shows only partly (1.8 GiB still counted): no second shrink while the counters catch up
        for (double t = 5; t < 120; t += 5) CHECK(!g5.tick(in(t, L1 + 1843 * MiB, B1, small, true)).pool_target.has_value());
        // a real further drop after the window (trust_s, 120 s) is still acted on
        p = g5.tick(in(125, L1 + 1843 * MiB, B1, small, true));
        CHECK(p.shrink);
        // a grow that first reads too high (stale old pool) but leaves more than half the margin free is not shrunk back
        // within the window (REVIEW-GOV-2 C-1 / REVIEW-GOV-3 C-2: a reading inside the trigger band after a grow is a real
        // overshoot and is acted on, see testMemGovernorGrowOvershoot; this case used L0 + 3 GiB, 0.7 GiB over the budget,
        // before the C-1 fix, and L0 + 2 GiB, inside the trigger band, before the C-2 fix)
        Governor g6(c);
        g6.tick(in(0, L0, B1, 330240, true));
        g6.resized(small, false, 0, L1);
        g6.resized(330240, true, 200, L0);
        for (double t = 205; t < 220; t += 5) CHECK(!g6.tick(in(t, L0 + 1 * GiB, B0, 330240, true)).pool_target.has_value());
        CHECK(L0 + 1 * GiB + M / 2 < B0);  // 1 GiB high, but more than half the margin still free (outside the trigger band)
        // growDenied restarts the recovery time
        Governor g7(c);
        g7.tick(in(0, L0, B1, 330240, true));
        g7.resized(small, false, 0, L1);
        bool asked = false;
        double t_ask = 0;
        for (double t = 100; t < 300 && !asked; t += 5)
            if (g7.tick(in(t, L1, B0, small, true)).pool_target) {
                asked = true;
                t_ask = t;
            }
        CHECK(asked);
        g7.growDenied(t_ask);
        CHECK(!g7.tick(in(t_ask + 5, L1, B0, small, true)).pool_target.has_value());
        CHECK(g7.tick(in(t_ask + 35, L1, B0, small, true)).pool_target.has_value());
    }
    // a grow within trust_s is sized from the larger of measured and expected usage (R9700 S1c: sized from the expected
    // one, 0.86 GiB under the measured, it overshot and was shrunk back)
    {
        const std::uint64_t B1 = B0 - 4 * GiB;
        auto growTarget = [&](std::uint64_t measured_extra) {
            Governor g8(c);
            const Plan p0 = g8.tick(in(0, L0, B1, 330240, true));
            const std::uint32_t small = *p0.pool_target;
            const std::uint64_t L1 = L0 - static_cast<std::uint64_t>(330240 - small) * 36966;
            g8.resized(small, false, 0, L1);
            for (double t = 5; t < 110; t += 5) {
                const Plan p = g8.tick(in(t, L1 + measured_extra, B0, small, true));
                if (p.pool_target) return *p.pool_target - small;
            }
            return 0u;
        };
        const std::uint32_t a = growTarget(0), b = growTarget(880 * MiB);
        CHECK(a > 0u && b > 0u && b < a);
        CHECK(static_cast<std::uint64_t>(a - b) * 36966 >= 800 * MiB);
    }
    // REVIEW-GOV-1 P2-1: with the budget back at its startup value the episode really ends: the pool grows back to its startup
    // size and new requests get the startup context again. Driven by the governor's own targets (no hand-set pool), with
    // the R9700 S2 numbers (Ornith, budget 31.02 GiB, margin 2068 MiB, startup sized with exactly the margin free) and the
    // 0.19 GiB that usage outside the pool grew by after the episode in those runs
    {
        Config r;
        r.per_tok = 22528;
        r.page = 256;
        r.pool_start = 393984;
        r.pool_min = 4096;
        r.ctx = 393984;  // --parallel 1 on a card where the context per request is the whole pool
        const std::uint64_t Mr = 2068 * MiB, Br = 31764 * MiB, Lr0 = Br - Mr, drift = 195 * MiB;
        auto inr = [&](double t, std::uint64_t local, std::uint64_t budget, std::uint32_t pool) {
            Input x;
            x.now_s = t;
            x.local = local;
            x.budget = budget;
            x.margin = Mr;
            x.pool = pool;
            x.all_idle = true;
            return x;
        };
        auto usage = [&](std::uint32_t pool, std::uint64_t extra) {
            return Lr0 - static_cast<std::uint64_t>(393984 - pool) * 22528 + extra;
        };
        // run the episode: budget drop of 4 GiB at t=0, back to Br at t_back; returns the pool at the end
        auto episode = [&](Governor& g, std::uint64_t b_back, double t_back, double t_end, bool& ended) {
            std::uint32_t pool = 393984;
            ended = false;
            for (double t = 0; t < t_end; t += 5) {
                const std::uint64_t b = t < t_back ? Br - 4 * GiB : b_back;
                const Plan p = g.tick(inr(t, usage(pool, t >= 5 ? drift : 0), b, pool));
                if (p.episode_end) ended = true;
                if (p.pool_target) {
                    const std::uint32_t np = *p.pool_target;
                    g.resized(np, np > pool, t, usage(np, 0));
                    pool = np;
                }
            }
            return pool;
        };
        Governor g(r);
        bool ended = false;
        const std::uint32_t pool = episode(g, Br, 300, 1200, ended);
        CHECK_EQ(g.shrinks(), 1u);
        CHECK_EQ(g.grows(), 1u);  // one grow, straight back
        CHECK_EQ(pool, 393984u);
        CHECK(ended);
        CHECK(!g.reduced());
        CHECK_EQ(g.ctxNew(), 393984u);
        // stable afterwards: an hour of samples at the startup budget with the drift: no action
        for (double t = 1200; t < 4800; t += 5) {
            const Plan p = g.tick(inr(t, usage(393984, drift), Br, 393984));
            CHECK(!p.pool_target && !p.free_cache && !p.episode_start);
        }
        // the budget comes back only partly (1 GiB still taken): the startup pool would leave less than endMargin free, so
        // the pool grows only part way and new requests keep the smaller context (no spill risk taken)
        Governor h(r);
        bool ended_h = false;
        const std::uint32_t pool_h = episode(h, Br - 1 * GiB, 300, 1200, ended_h);
        CHECK(!ended_h);
        CHECK(h.reduced());
        CHECK(pool_h < 393984u);
        CHECK_EQ(h.ctxNew(), pool_h);
        CHECK(Br - 1 * GiB >= usage(pool_h, drift) + Mr);  // the full margin stays free
    }
    // unknown budget: nothing
    {
        Governor g4(c);
        CHECK(!g4.tick(in(0, L0, 0, 330240, true)).free_cache);
    }
    // the status object
    {
        Status st;
        st.on = true;
        st.reduced = true;
        st.pool_tokens = 218112;
        st.pool_tokens_start = 330240;
        st.ctx_per_request = 131072;
        st.slots = 1;
        st.slots_admitting = 1;
        st.last_action = "pool 330240 -> \"218112\"";
        const std::string j = statusJson(st);
        CHECK(j.find("{\"on\":true,\"reduced\":true,\"pool_tokens\":218112,\"pool_tokens_start\":330240,\"ctx_per_request\":131072,") == 0);
        CHECK(j.find("\"last_action\":\"pool 330240 -> \\\"218112\\\"\"}") != std::string::npos);
    }
}

// REVIEW-GOV-2 C-1: a grow that costs more than expected (R9700 Swift R2b_nolim1 19:14:48: WHIRL 25.87 -> 32.50 GiB with
// the budget at 31.02 GiB, the HIP resource cache kept the freed pool) leaves WHIRL over the budget. The governor must not
// report the end of the episode ("back to the startup settings") while the measured usage is over the budget, and must act
// on it at once instead of trusting the expected usage for trust_s (120 s: the next episode started 115 s later).
// Written from the review's description, before the fix.
void testMemGovernorGrowOvershoot() {
    using namespace whirl::memgov;
    constexpr std::uint64_t MiB = 1ull << 20, GiB = 1ull << 30;
    Config r;
    r.per_tok = 22528;
    r.page = 256;
    r.pool_start = 393984;
    r.pool_min = 4096;
    r.ctx = 393984;
    const std::uint64_t Mr = 2068 * MiB, Br = 31764 * MiB, Lr0 = Br - Mr;
    auto usage = [&](std::uint32_t pool) { return Lr0 - static_cast<std::uint64_t>(393984 - pool) * 22528; };
    auto inr = [&](double t, std::uint64_t local, std::uint64_t budget, std::uint32_t pool) {
        Input x;
        x.now_s = t;
        x.local = local;
        x.budget = budget;
        x.margin = Mr;
        x.pool = pool;
        x.all_idle = true;
        return x;
    };
    // overshoot: what the grow costs above the expected usage, kept until the next re-allocation
    double t_band_end = -1;  // episode_end while inside the trigger band (less than margin/2 free), REVIEW-GOV-3 C-2
    auto run = [&](std::uint64_t overshoot, double& t_grow, double& t_act, double& t_bad_end, std::uint64_t& measured_after) {
        Governor g(r);
        std::uint32_t pool = 393984;
        bool grown = false;
        t_grow = t_act = t_bad_end = t_band_end = -1;
        measured_after = 0;
        for (double t = 0; t < 1200; t += 5) {
            const std::uint64_t b = t < 300 ? Br - 4 * GiB : Br;
            const std::uint64_t measured = usage(pool) + (grown ? overshoot : 0);
            const Plan p = g.tick(inr(t, measured, b, pool));
            if (grown) {
                if (p.episode_end && measured > b && t_bad_end < 0) t_bad_end = t;
                if (p.episode_end && measured + Mr / 2 > b && t_band_end < 0) t_band_end = t;
                if ((p.free_cache || p.shrink || p.episode_start) && t_act < 0) t_act = t;
                if (t_act >= 0) break;
            }
            if (p.pool_target) {
                const std::uint32_t np = *p.pool_target;
                g.resized(np, np > pool, t, usage(np));
                if (np > pool && !grown) {
                    grown = true;
                    t_grow = t;
                    measured_after = usage(np) + overshoot;
                }
                pool = np;
            }
        }
    };
    // the measured R2b_nolim1 case: after the grow WHIRL is 1.48 GiB over the budget
    {
        const std::uint64_t over = Mr + 1515 * MiB;  // expected leaves the margin free; measured = budget + 1.48 GiB
        double t_grow, t_act, t_bad_end;
        std::uint64_t m;
        run(over, t_grow, t_act, t_bad_end, m);
        std::printf("C-1 overshoot: grow at t=%.0f, measured after it %.2f GiB vs budget %.2f GiB; episode_end while over the "
                    "budget at t=%.0f; first action after the grow at t=%.0f (%.0f s after it)\n",
                    t_grow, m / double(GiB), Br / double(GiB), t_bad_end, t_act, t_act >= 0 ? t_act - t_grow : -1.0);
        CHECK(t_grow > 0);
        CHECK(m > Br);
        CHECK(t_bad_end < 0);                                   // no "back to the startup settings" while over the budget
        CHECK(t_act > 0 && t_act - t_grow <= r.settle_s + 10);  // acted on within the next samples, not after trust_s
    }
    // REVIEW-GOV-3 C-2: a grow that costs 1.52 GiB more than expected lands under the budget but inside the trigger band
    // (500 MiB free, the trigger is at margin/2 = 1034 MiB): it must be acted on at once too, not min()-masked for trust_s
    // with the episode end logged and a new episode ~117 s later (the review's probe: episode_end t=233, episode_start t=350)
    {
        const std::uint64_t over = Mr - 500 * MiB;  // measured 500 MiB under the budget
        double t_grow, t_act, t_bad_end;
        std::uint64_t m;
        run(over, t_grow, t_act, t_bad_end, m);
        std::printf("C-2 overshoot in the trigger band: grow at t=%.0f, measured after it %.2f GiB vs budget %.2f GiB "
                    "(trigger at %.2f GiB); episode_end inside the band at t=%.0f; first action after the grow at t=%.0f "
                    "(%.0f s after it)\n",
                    t_grow, m / double(GiB), Br / double(GiB), (Br - Mr / 2) / double(GiB), t_band_end, t_act,
                    t_act >= 0 ? t_act - t_grow : -1.0);
        CHECK(t_grow > 0);
        CHECK(m < Br && m + Mr / 2 > Br);                       // under the budget, inside the trigger band
        CHECK(t_band_end < 0);                                  // no "back to the startup settings" inside the band
        CHECK(t_act > 0 && t_act - t_grow <= r.settle_s + 10);  // acted on within the next samples, not after trust_s
    }
    // control: a grow whose reading is high but leaves more than half the margin free (counters catching up) is not acted
    // on within trust_s (this control read 300 MiB under the budget, inside the trigger band, before the C-2 fix)
    {
        const std::uint64_t over = Mr - 1300 * MiB;  // measured 1300 MiB under the budget, 266 MiB outside the band
        double t_grow, t_act, t_bad_end;
        std::uint64_t m;
        run(over, t_grow, t_act, t_bad_end, m);
        CHECK(t_grow > 0);
        CHECK(m + Mr / 2 < Br);
        CHECK(t_bad_end < 0);
        CHECK(t_act < 0 || t_act - t_grow >= r.trust_s - 5);  // not shrunk back while the counters may lag
    }
}

// REVIEW-GOV-3 question 4 (R9700 Ornith --mmproj, gov3 orn_new3): startup left exactly the margin free (29696 MiB local,
// budget 31767 MiB, margin 2068 MiB); the first image added 873 MiB (encoder weights, 860.5 MiB pinned, made resident) and a
// 1920x1080 image 134 MiB more (activations that did not fit the lent prefill scratch), so 30 MiB were left before the
// governor's trigger (margin/2 free) and a false episode could never end. With --mmproj the startup KV pool must leave room
// for what the encoder adds; without it (text only) the pool must be exactly what it was. Written before the fix.
void testVisionPoolReserve() {
    using namespace whirl;
    namespace nu = whirl::numerics;
    constexpr std::uint64_t KiB = 1ull << 10, MiB = 1ull << 20;
    const std::uint64_t per_tok = 22528;                   // Ornith MXFP4 KV bytes per token
    const std::uint64_t avail = 8468 * MiB;                // orn_new3: room 8.27 GiB (budget - used - margin)
    const std::uint64_t added = 873 * MiB + 134 * MiB;     // measured local usage the encoder added (orn_new3 health)
    poolbudget::VisionVram vv;
    vv.weights = 860 * MiB + 512 * KiB;                    // the mmproj's weights (the startup log's pinned MiB)
    vv.act_extra = 134 * MiB;                              // beyond the lent scratch (the 1920x1080 step)
    const std::uint64_t r = poolbudget::visionReserve(vv);
    const nu::PoolFit text = nu::serverPoolFit(avail, per_tok, std::nullopt, false, 131072, false, 256, 8 * 262144);
    const nu::PoolFit text2 =
        nu::serverPoolFit(poolbudget::kvAvail(avail, poolbudget::visionReserve({})), per_tok, std::nullopt, false, 131072, false, 256, 8 * 262144);
    const nu::PoolFit vis =
        nu::serverPoolFit(poolbudget::kvAvail(avail, r), per_tok, std::nullopt, false, 131072, false, 256, 8 * 262144);
    std::printf("vision pool reserve: reserve %llu MiB (encoder added %llu MiB); KV pool text-only %u tokens, no-mmproj via "
                "kvAvail %u, --mmproj %u tokens (%lld tokens less)\n",
                static_cast<unsigned long long>(r / MiB), static_cast<unsigned long long>(added / MiB), text.pool, text2.pool,
                vis.pool, static_cast<long long>(text.pool) - static_cast<long long>(vis.pool));
    // text only: exactly the same pool (no reserve, kvAvail is the identity)
    CHECK_EQ(poolbudget::visionReserve({}), 0ull);
    CHECK_EQ(text2.pool, text.pool);
    for (std::uint64_t a : {0ull, 1ull, 4096ull * MiB, 8468ull * MiB, 31767ull * MiB}) CHECK_EQ(poolbudget::kvAvail(a, 0), a);
    // --mmproj: the reserve covers what the encoder really added, without reserving much more
    CHECK(r >= added);
    CHECK(r <= added + 64 * MiB);
    // and the pool is smaller by at least that much, so after the encoder and the 1920x1080 image the margin is still free
    CHECK(vis.pool < text.pool);
    CHECK((text.pool - vis.pool) * per_tok >= added);
    CHECK(vis.pool >= 131072);  // one 131072-token request still fits
}

// v0.2.2.1 startup fit of explicit settings (owner 2026-10-21: adjust, log, refuse only below the minimum)
void testStartupFit() {
    using namespace whirl;
    constexpr std::uint64_t MiB = 1ull << 20, GiB = 1ull << 30;
    // 1. RAM tier (ramTierFit)
    // explicit: kept when within half of the available RAM and within the 20480 MiB cap
    CHECK(!startfit::ramTierFit(12288, 48 * GiB, 16384, true).has_value());
    CHECK(!startfit::ramTierFit(20480, 48 * GiB, 16384, true).has_value());
    // the hipError 719 repro (REVIEW-0221 C1): 24064 MiB on a box with 49247 MiB available is within half, but capped
    {
        const auto r = startfit::ramTierFit(24064, 49247 * MiB, 16384, true);
        CHECK(r.has_value() && r->mb == 20480ull && r->safe && !r->half);
    }
    // v0.2.2.2: the cap first, then half of the available RAM on the capped size: 65536 with 48955 MiB available -> the cap
    // (20480 is within half), no longer the default size
    {
        const auto r = startfit::ramTierFit(65536, 48955 * MiB, 16384, true);
        CHECK(r.has_value() && r->mb == 20480ull && !r->half && r->safe);
    }
    {
        const auto r = startfit::ramTierFit(65536, 64 * GiB, 0, true);
        CHECK(r.has_value() && r->mb == 20480ull && !r->half && r->safe);
    }
    // the cap, then half: 30 GiB available -> 15360
    {
        const auto r = startfit::ramTierFit(65536, 30 * GiB, 8192, true);
        CHECK(r.has_value() && r->mb == 15360ull && r->half && r->safe);
    }
    CHECK_EQ(startfit::ramTierFit(9000, 17000 * MiB, 0, true)->mb, 8192ull);  // half = 8500 MiB -> 8192 (512 steps)
    CHECK_EQ(startfit::ramTierFit(9000, 12 * GiB, 8192, true)->mb, 6144ull);  // half
    // monotonic (0.2.2.2 item 1; 0.2.2.1 gave 22528 -> 20480 but 24064 -> 16384 with 47348 MiB available): asking for more
    // never gives less, for any available RAM
    for (const std::uint64_t avail_mb : {8192ull, 17000ull, 30720ull, 47348ull, 49247ull, 65536ull, 131072ull}) {
        std::uint64_t prev = 0;
        bool mono = true;
        for (std::uint64_t ask = 512; ask <= 65536; ask += 512) {
            const auto r = startfit::ramTierFit(ask, avail_mb * MiB, 16384, true);
            const std::uint64_t got = r ? r->mb : ask;
            mono = mono && got >= prev;
            prev = got;
        }
        CHECK(mono);
    }
    CHECK_EQ(startfit::ramTierFit(22528, 47348 * MiB, 16384, true)->mb, 20480ull);
    CHECK_EQ(startfit::ramTierFit(24064, 47348 * MiB, 16384, true)->mb, 20480ull);
    // the automatic size (a PC with 128 GB RAM: 32 GiB by the 1/4 rule) is capped too, never the half rule
    {
        const auto r = startfit::ramTierFit(32768, 100 * GiB, 32768, false);
        CHECK(r.has_value() && r->mb == 20480ull && r->safe && !r->half);
        CHECK(!startfit::ramTierFit(16384, 20 * GiB, 16384, false).has_value());  // auto within the cap: kept (half rule not applied)
    }
    CHECK(!startfit::ramTierFit(0, 8 * GiB, 8192, true).has_value());  // 0 = tiers off: kept
    CHECK(!startfit::ramTierFit(4096, 0, 8192, true).has_value());     // RAM unknown, small: kept
    CHECK_EQ(startfit::ramTierFit(30000, 0, 0, true)->mb, 20480ull);    // RAM unknown, large: capped
    // s8-719 repro (scratch\s8_719\results.txt, 2026-10-22 23:34-23:43): the 8060S at a 96 GB carve-out (32407 MiB RAM,
    // SharedSystemMemory 16203 MiB on every adapter), the R9700 also a HIP GPU (DXGI non-local budget 15435 MiB), the
    // embedding (1288 MiB) pinned, Swift precise server. --kv-ram-mb 4096 answered, 8192 failed with hipError 719 on the
    // first request (0.2.3 and v0.2.2.2); with only the 8060S visible to HIP, 8192 answered.
    {
        startfit::PinnedState s;
        s.total_phys = 32407 * MiB;
        s.pinned_max = 1441 * MiB;  // 8060S adapter Shared Usage with the embedding pinned (case s23_ram0)
        s.gpus = {{0, 0, 16203 * MiB}, {15435 * MiB, 1288 * MiB, 16203 * MiB}};  // 8060S (integrated: budget 0), R9700
        const std::uint64_t room = startfit::pinnedRoomMb(s);
        CHECK(room >= 4096ull && room < 8192ull);
        const auto r = startfit::ramTierFit(8192, 19374 * MiB, 0, true, room);  // 19374 MiB available (case g23_swift)
        CHECK(r.has_value() && r->mb < 8192ull && r->mb >= 4096ull && r->pinned);
        CHECK(!startfit::ramTierFit(4096, 19374 * MiB, 0, true, room).has_value());  // 4096 answered: kept
        // only the 8060S visible to HIP (HIP_VISIBLE_DEVICES): 8192 answered -> kept
        s.gpus = {{0, 0, 16203 * MiB}};
        CHECK(!startfit::ramTierFit(8192, 19430 * MiB, 0, true, startfit::pinnedRoomMb(s)).has_value());
    }
    // kvram719 (the same PC at a 64/64 carve-out: 65175 MiB RAM, R9700 non-local budget 48026 MiB, the 8060S's
    // SharedSystemMemory 32587 MiB, both HIP GPUs, process on the R9700): 24064 failed, 20480 worked
    {
        startfit::PinnedState s;
        s.total_phys = 65175 * MiB;
        s.pinned_max = 1288 * MiB;
        s.gpus = {{48026 * MiB, 1288 * MiB, 32587 * MiB}, {0, 0, 32587 * MiB}};
        const std::uint64_t room = startfit::pinnedRoomMb(s);
        CHECK(room >= 20480ull && room < 24064ull);
        CHECK(!startfit::ramTierFit(20480, 55 * GiB, 16384, true, room).has_value());
        s.pinned_max = (10890 + 1288) * MiB;  // another server had 10890 MiB pinned: 20480 failed (kvram719 10:41)
        CHECK(startfit::pinnedRoomMb(s) < 20480ull);
    }
    CHECK_EQ(startfit::pinnedRoomMb(startfit::PinnedState{}), startfit::pinned_room_unknown);  // nothing known
    // everything pinned by other programs: room 0 -> the tier is off rather than crashing
    {
        startfit::PinnedState s;
        s.total_phys = 64 * GiB;
        s.pinned_max = 30 * GiB;
        const auto r = startfit::ramTierFit(16384, 30 * GiB, 16384, false, startfit::pinnedRoomMb(s));
        CHECK(r.has_value() && r->mb == 0ull && r->pinned);
    }
    // REVIEW-S8-719 A1: the reason text of an adjusted (healthy) server must match none of the error patterns the gates and
    // repro scripts scan server logs for (kvram719_tier.ps1:50 "hipError|\| failed:", golden.ps1:377/393/397, fix719.ps1:190;
    // PowerShell -match / Select-String are case-insensitive), and it still names the error the user would have seen (719)
    {
        const std::regex bad(R"(hipError|hip.*\b719\b|\b719\b.*hip|\| failed:|PAL failed|Failed PAL|result:-5|unspecified launch)",
                             std::regex::icase);
        for (const std::uint64_t room : {0ull, 512ull, 4096ull, 8192ull, 12288ull, 20480ull})
            for (const std::size_t n : {std::size_t{1}, std::size_t{2}, std::size_t{3}}) {
                const std::string why = startfit::pinnedWhy(room, n);
                const std::string line = "2026-10-23 00:05:55.955 W adjusted: --kv-ram-mb 8192 -> " + std::to_string(room) + " MiB (" + why + ")";
                CHECK(!std::regex_search(line, bad));
                CHECK(why.find("719") != std::string::npos);
            }
    }
    // REVIEW-S8-719 A2: every measured working size stays admitted, and single-GPU defaults with no failure on record keep
    // their tier (the machine reserve 6144 MiB cut them)
    {
        // 8060S only visible (HIP_VISIBLE_DEVICES), 96 GB carve-out, case s23_ram12288_vis (scratch\s8_719\results.txt 23:39):
        // --kv-ram-mb 12288 was lowered to 8704 by the half rule (17643 MiB available) and answered (8060S Shared Usage 10161 MiB)
        startfit::PinnedState s;
        s.total_phys = 32407 * MiB;
        s.pinned_max = 1441 * MiB;
        for (const std::uint64_t own : {0ull, 1441ull}) {  // DXGI reports 0 for the iGPU; PDH 1441 (REVIEW C4)
            s.gpus = {{0, own * MiB, 16203 * MiB}};
            const auto r = startfit::ramTierFit(12288, 17643 * MiB, 0, true, startfit::pinnedRoomMb(s));
            CHECK(r.has_value() && r->mb == 8704ull && r->half && !r->pinned);
            // golden's 8192 on the 8060S alone with other programs holding 1000 MiB more shared GPU memory: kept
            startfit::PinnedState s2 = s;
            s2.pinned_max = (1441 + 1000) * MiB;
            CHECK(!startfit::ramTierFit(8192, 18930 * MiB, 0, true, startfit::pinnedRoomMb(s2)).has_value());
        }
        // single R9700 on a 32 GB-RAM host (DXGI non-local budget as measured, 15435 MiB), other programs 700 MiB of shared
        // GPU memory: the automatic tier (1/4 of RAM, 8192) and one max-context entry (9216) are kept
        startfit::PinnedState r9;
        r9.total_phys = 32407 * MiB;
        r9.pinned_max = (1294 + 700) * MiB;
        r9.gpus = {{15435 * MiB, 1294 * MiB, 16203 * MiB}};
        CHECK(!startfit::ramTierFit(8192, 20000 * MiB, 8192, false, startfit::pinnedRoomMb(r9), 9216).has_value());
        CHECK(!startfit::ramTierFit(9216, 20000 * MiB, 9216, false, startfit::pinnedRoomMb(r9), 9216).has_value());
        // single R9700 at 64 GB (kvram719 numbers), automatic 16384: kept
        r9.total_phys = 65175 * MiB;
        r9.pinned_max = 1300 * MiB;
        r9.gpus = {{48026 * MiB, 1294 * MiB, 32587 * MiB}};
        CHECK(!startfit::ramTierFit(16384, 50000 * MiB, 16384, false, startfit::pinnedRoomMb(r9), 9216).has_value());
        // single 8060S at 64 GB, automatic 8192: kept
        startfit::PinnedState i8;
        i8.total_phys = 65175 * MiB;
        i8.pinned_max = 1441 * MiB;
        i8.gpus = {{0, 1441 * MiB, 32587 * MiB}};
        CHECK(!startfit::ramTierFit(8192, 50000 * MiB, 8192, false, startfit::pinnedRoomMb(i8), 9216).has_value());
        // R9700 + 8060S at 64/64: process on the 8060S (golden 8192), on the R9700 (automatic 16384, explicit 20480): kept
        startfit::PinnedState both;
        both.total_phys = 65175 * MiB;
        both.pinned_max = 1441 * MiB;
        both.gpus = {{0, 1441 * MiB, 32587 * MiB}, {48026 * MiB, 1294 * MiB, 32587 * MiB}};
        const std::uint64_t rb = startfit::pinnedRoomMb(both);
        CHECK(!startfit::ramTierFit(8192, 50000 * MiB, 0, true, rb, 9216).has_value());
        CHECK(!startfit::ramTierFit(16384, 50000 * MiB, 16384, false, rb, 9216).has_value());
        CHECK(!startfit::ramTierFit(20480, 50000 * MiB, 16384, true, rb, 9216).has_value());
    }
    // REVIEW-S8-719 C2 (FR-CACHE P3-4, src/tier/ram_size.h): a RAM tier the pinned room lowers below one max-context entry
    // is off (the longest session could never be kept), not a tier smaller than one entry
    {
        // REVIEW-S8-719-2 X1/X2: only the automatic size; an explicit size is clamped to the room and used as given
        // (ram_size.h), so golden's 8192 on two GPUs at 96 GB gets 4096 (round 1, 709d575: gate 8/8), not off
        const auto e = startfit::ramTierFit(8192, 19374 * MiB, 0, true, 4096, 9216);  // explicit, room 4096 < entry 9216
        CHECK(e.has_value() && e->mb == 4096ull && e->pinned && !e->below_entry);
        const auto a = startfit::ramTierFit(9216, 19374 * MiB, 9216, false, 4096, 9216);  // automatic = one entry
        CHECK(a.has_value() && a->mb == 0ull && a->below_entry);
        const auto k = startfit::ramTierFit(8192, 19374 * MiB, 0, true, 4096, 4096);  // room holds one entry: 4096
        CHECK(k.has_value() && k->mb == 4096ull && k->pinned && !k->below_entry);
        CHECK(!startfit::ramTierFit(4096, 19374 * MiB, 0, true, startfit::pinned_room_unknown, 9216).has_value());  // user's own size
    }
    // REVIEW-S8-719-2 X2: an explicit size is monotonic in the request (asking for more never gives a smaller tier), and a
    // room > 0 never turns an explicit tier off. Sweep 0..20480 step 512 over room / entry / available-RAM configurations,
    // including the measured two-GPU room 4096 with entry 10240 (8060S 96 GB + R9700, golden slot).
    {
        struct Cfg { std::uint64_t room, entry, avail_mb; };
        const Cfg cfgs[] = {{4096, 10240, 19032}, {4096, 9216, 19374}, {4096, 4096, 19374}, {0, 10240, 19032},
                            {12288, 10240, 17643}, {12288, 0, 17643}, {20480, 9216, 50000}, {19456, 10240, 50000},
                            {8192, 16384, 12000}, {startfit::pinned_room_unknown, 10240, 30000}, {512, 9216, 40000}};
        for (const Cfg& c : cfgs) {
            std::uint64_t prev = 0;
            bool mono = true, never_off = true, within = true;
            for (std::uint64_t ask = 0; ask <= 20480; ask += 512) {
                const auto r = startfit::ramTierFit(ask, c.avail_mb * MiB, 0, true, c.room, c.entry);
                const std::uint64_t got = r ? r->mb : ask;
                if (got < prev) mono = false;
                if (ask > 0 && c.room != startfit::pinned_room_unknown && c.room > 0 && got == 0) never_off = false;
                if (got > ask || (c.room != startfit::pinned_room_unknown && got > c.room)) within = false;
                if (r && r->below_entry) never_off = false;  // the entry rule is for the automatic size only
                prev = got;
            }
            CHECK(mono);
            CHECK(never_off);
            CHECK(within);
        }
    }
    // REVIEW-S8-719-2 C-1: when the pinned room turns the tier off, the text says the SSD tier is off with it (both forms),
    // and still matches none of the log-scan error patterns
    {
        const std::regex bad(R"(hipError|hip.*\b719\b|\b719\b.*hip|\| failed:|PAL failed|Failed PAL|result:-5|unspecified launch)",
                             std::regex::icase);
        for (const bool be : {true, false}) {
            const std::string off = startfit::pinnedOffWhy(be, 10240);
            CHECK(off.find("SSD") != std::string::npos);
            CHECK(off.find("RAM") != std::string::npos);
            const std::string line = "2026-10-23 00:21:26.686 W kv tier: RAM tier 0 MiB (the automatic size would be 10240 MiB): " +
                                     startfit::pinnedWhy(4096, 2) + off;
            CHECK(!std::regex_search(line, bad));
        }
    }
    // REVIEW-S8-719-2 independent tests (scratch\s8_719\review2\indep_test.cpp, written from the problem statement), ported:
    // measured inputs of fixb_case_g23_swift 00:21:26 (8060S 96 GB + R9700) and the 64 GB / 64/64 configurations
    {
        const auto two96 = [&](std::uint64_t own_r, std::uint64_t own_i, std::uint64_t pmax) {
            startfit::PinnedState s;
            s.total_phys = 33982058496ull;
            s.pinned_max = pmax * MiB;
            s.gpus = {{0, own_i * MiB, 16203 * MiB}, {15435 * MiB, own_r * MiB, 0}};
            return s;
        };
        const auto fit = [&](std::uint64_t mb, std::uint64_t avail_mb, bool expl, std::uint64_t room, std::uint64_t entry) {
            const auto r = startfit::ramTierFit(mb, avail_mb * MiB, 0, expl, room, entry);
            return r ? r->mb : mb;
        };
        const std::uint64_t room2 = startfit::pinnedRoomMb(two96(1294, 1381, 1453));
        // T1/T1b/T1c: explicit 8192 lowered, > 0, at most the measured-working 4096 (also with own usage reported doubled)
        CHECK(fit(8192, 19032, true, room2, 0) > 0ull && fit(8192, 19032, true, room2, 0) <= 4096ull);
        CHECK(fit(8192, 19032, true, startfit::pinnedRoomMb(two96(2588, 1381, 1453)), 0) <= 4096ull);
        // T2 (lead judgment): explicit 8192 with entry 10240 -> 4096, not off; T2b: automatic one entry (10240) -> off
        CHECK_EQ(fit(8192, 19032, true, room2, 10240), 4096ull);
        {
            const auto a = startfit::ramTierFit(10240, 19032 * MiB, 0, false, room2, 10240);
            CHECK(a.has_value() && a->mb == 0ull && a->below_entry);
        }
        // T3: explicit 4096 on two GPUs kept
        CHECK_EQ(fit(4096, 19032, true, room2, 0), 4096ull);
        // T4/T5/T5b: 8060S only visible: 8192 kept, 12288 -> 8704 by the half rule (not the pinned room), also with entry 10240
        startfit::PinnedState one;
        one.total_phys = 33982058496ull;
        one.pinned_max = 1453 * MiB;
        one.gpus = {{0, 1381 * MiB, 16203 * MiB}};
        const std::uint64_t room1 = startfit::pinnedRoomMb(one);
        CHECK_EQ(fit(8192, 17643, true, room1, 0), 8192ull);
        {
            const auto r = startfit::ramTierFit(12288, 17643 * MiB, 0, true, room1, 0);
            CHECK(r.has_value() && r->mb == 8704ull && r->half && !r->pinned);
        }
        CHECK_EQ(fit(8192, 17643, true, room1, 10240), 8192ull);
        // T6/T6b/T6c: single R9700 (64 GB host: auto 16384, explicit 20480; 32 GB desktop: auto 10240) kept
        startfit::PinnedState r64;
        r64.total_phys = 65174 * MiB;
        r64.pinned_max = 1500 * MiB;
        r64.gpus = {{48026 * MiB, 1294 * MiB, 0}};
        CHECK_EQ(fit(16384, 40000, false, startfit::pinnedRoomMb(r64), 9216), 16384ull);
        CHECK_EQ(fit(20480, 50000, true, startfit::pinnedRoomMb(r64), 9216), 20480ull);
        startfit::PinnedState d32;
        d32.total_phys = 32407 * MiB;
        d32.pinned_max = (1294 + 600) * MiB;
        d32.gpus = {{15435 * MiB, 1294 * MiB, 0}};
        CHECK_EQ(fit(10240, 22000, false, startfit::pinnedRoomMb(d32), 10240), 10240ull);
        // T7/T7b: single 8060S 64 GB: auto 10240 and golden 8192 kept
        startfit::PinnedState i64;
        i64.total_phys = 65174 * MiB;
        i64.pinned_max = 1453 * MiB;
        i64.gpus = {{0, 1381 * MiB, 32587 * MiB}};
        CHECK_EQ(fit(10240, 40000, false, startfit::pinnedRoomMb(i64), 10240), 10240ull);
        CHECK_EQ(fit(8192, 40000, true, startfit::pinnedRoomMb(i64), 10240), 8192ull);
        // T8/T8b/T8c: 64/64 two GPUs: golden 8192, auto 16384, explicit 20480 kept (T8d, the 197 MiB margin, is noted only)
        startfit::PinnedState t64 = i64;
        t64.gpus.push_back({48026 * MiB, 1294 * MiB, 0});
        const std::uint64_t rt = startfit::pinnedRoomMb(t64);
        CHECK_EQ(fit(8192, 40000, true, rt, 10240), 8192ull);
        CHECK_EQ(fit(16384, 40000, false, rt, 9216), 16384ull);
        CHECK_EQ(fit(20480, 50000, true, rt, 9216), 20480ull);
        // T10: explicit sizes monotonic at the measured two-GPU room (512..24576)
        std::uint64_t prev = 0;
        bool mono = true;
        for (std::uint64_t ask = 512; ask <= 24576; ask += 512) {
            const std::uint64_t got = fit(ask, 19032, true, room2, 10240);
            if (got < prev) mono = false;
            prev = got;
        }
        CHECK(mono);
        // T11/T11b: nothing known -> room unknown, explicit 8192 unchanged
        CHECK_EQ(startfit::pinnedRoomMb(startfit::PinnedState{}), startfit::pinned_room_unknown);
        CHECK(!startfit::ramTierFit(8192, 40000 * MiB, 0, true, startfit::pinned_room_unknown, 10240).has_value());
    }
    // 2b. slots after the weights: the C5 case (15 slots left 13312 tokens for a 16384-token request) -> fewer slots
    {
        const std::uint64_t per_tok = 36966, per_slot = 380 * MiB, shared = 556 * MiB;
        const std::uint64_t avail = shared + 15 * per_slot + 13312 * per_tok;  // 15 slots: exactly 13312 tokens
        const std::uint32_t p = startfit::slotsThatFit(15, avail, per_slot, shared, per_tok, 16384);
        CHECK(p < 15u && p >= 1u);
        CHECK(avail - shared - per_slot * p >= 16384 * per_tok);
        CHECK(avail - shared - per_slot * (p + 1) < 16384 * per_tok);  // the least reduction
        CHECK_EQ(startfit::slotsThatFit(4, 40 * GiB, per_slot, shared, per_tok, 16384), 4u);  // fits: unchanged
        CHECK_EQ(startfit::slotsThatFit(8, 1 * GiB, per_slot, shared, per_tok, 16384), 1u);   // never below 1
    }
    // the startup JSON (REVIEW-0221 A1: kv_ram_mb is the RAM tier in use)
    {
        startfit::StartupInfo si;
        si.ctx_per_request = 13312;
        si.pool_tokens = 13312;
        si.parallel = 15;
        si.kv_ram_mb = 18432;
        si.wddm_margin_mib = 9468;
        si.other_processes_mib = 7412;
        si.display_attached = false;
        si.adjusted = {"--parallel 16 -> 15", "say \"hi\" \\ ok"};
        CHECK_EQ(startfit::startupJson(si),
                 std::string("{\"ctx_per_request\":13312,\"pool_tokens\":13312,\"parallel\":15,\"kv_ram_mb\":18432,\"wddm_margin_mib\":9468,"
                             "\"other_processes_mib\":7412,\"display_attached\":false,\"adjusted\":[\"--parallel 16 -> 15\",\"say \\\"hi\\\" "
                             "\\\\ ok\"]}"));
        CHECK_EQ(startfit::startupJson(startfit::StartupInfo{}),
                 std::string("{\"ctx_per_request\":0,\"pool_tokens\":0,\"parallel\":0,\"kv_ram_mb\":0,\"wddm_margin_mib\":0,"
                             "\"other_processes_mib\":0,\"display_attached\":false,\"adjusted\":[]}"));
    }
    // a Swift-like card: 16 GiB weights, 0.75 GiB fixed per slot, 36 KiB / token, ~11 GiB left with 1 slot
    vram::FitInput fi;
    fi.avail = 28 * GiB;
    fi.weights = 16 * GiB;
    fi.fixed = 512 * MiB;
    fi.buf_per_row = 64 * 1024;
    fi.buf_fixed = 256 * MiB;
    fi.ckpt_bytes = 150 * MiB;
    fi.kv_per_token = 36966;
    fi.pool_min_tokens = 16384;
    fi.batch = 4096;
    fi.n_ck = 4;
    fi.n_spe = 2;
    const std::uint64_t per_slot = 150 * MiB;
    // 2. --parallel 4 fits: unchanged
    {
        vram::FitInput a = fi;
        a.parallel = 4;
        a.fixed += 4 * per_slot;
        const auto r = startfit::fitParallel(a, per_slot, true);
        CHECK_EQ(r.parallel, 4u);
        CHECK(r.plan.pool_tokens >= startfit::wantTokens(a));
    }
    // --parallel 16 on a card with less room (another program holds VRAM): lowered until one 16k request fits
    {
        vram::FitInput a = fi;
        a.avail = 24 * GiB;
        a.parallel = 16;
        a.fixed += 16 * per_slot;
        a.ck_fixed = true;  // keep the checkpoint plan out of it
        a.batch_fixed = true;
        const auto r = startfit::fitParallel(a, per_slot, true);
        CHECK(r.parallel < 16u && r.parallel >= 1u);
        CHECK(r.pool_requested < startfit::wantTokens(a));
        CHECK(r.plan.pool_tokens >= startfit::wantTokens(r.fi));
        // one slot more would not fit (the least reduction)
        vram::FitInput b = r.fi;
        b.parallel = r.parallel + 1;
        b.fixed += per_slot;
        CHECK(vram::planFit(b).pool_tokens < startfit::wantTokens(b));
        // not given (a default): never lowered here
        const auto d = startfit::fitParallel(a, per_slot, false);
        CHECK_EQ(d.parallel, 16u);
    }
    // even one slot does not fit: kept at 1, still short (the server then refuses at sizing)
    {
        vram::FitInput a = fi;
        a.avail = 17 * GiB;
        a.parallel = 4;
        a.fixed += 4 * per_slot;
        a.batch_fixed = a.ck_fixed = true;
        const auto r = startfit::fitParallel(a, per_slot, true);
        CHECK_EQ(r.parallel, 1u);
        CHECK(r.plan.pool_tokens < startfit::wantTokens(r.fi));
    }
    // 3. user-set prefill batch / checkpoints: lowered only when short, and only if it helps
    {
        vram::FitInput a = fi;
        a.avail = 16 * GiB + 1600 * MiB;  // tight
        a.parallel = 1;
        a.fixed += per_slot;
        a.batch_fixed = a.ck_fixed = true;
        const vram::FitPlan p = vram::planFit(a);
        CHECK(p.pool_tokens < startfit::wantTokens(a));
        const auto r = startfit::fitFixed(a, p);
        CHECK(r.changed);
        CHECK(r.batch_changed || r.ck_changed);
        CHECK(r.plan.pool_tokens > p.pool_tokens);
        CHECK(!r.fi.batch_fixed && !r.fi.ck_fixed);
        // with room: nothing changes
        vram::FitInput roomy = a;
        roomy.avail = 30 * GiB;
        const auto q = startfit::fitFixed(roomy, vram::planFit(roomy));
        CHECK(!q.changed);
        // nothing user-set: nothing to lower
        vram::FitInput autoset = a;
        autoset.batch_fixed = autoset.ck_fixed = false;
        CHECK(!startfit::fitFixed(autoset, vram::planFit(autoset)).changed);
    }
    // 4. / 5. -c and --ctx-per-slot: serverPoolFit as "not explicit" lowers instead of refusing
    {
        const std::uint64_t per_tok = 36966;
        const std::uint64_t avail = 11 * GiB;
        const auto refused = numerics::serverPoolFit(avail, per_tok, 400000u, true, 131072, true, 256, 8 * 262144);
        CHECK(refused.refuse);  // the old contract (still used where nothing is adjusted)
        const auto r = numerics::serverPoolFit(avail, per_tok, 400000u, false, 131072, false, 256, 8 * 262144);
        CHECK(!r.refuse);
        CHECK(r.pool < 400000u && r.pool >= 4096u && r.pool % 256 == 0);
        CHECK(static_cast<std::uint64_t>(r.pool) * per_tok <= avail);
        CHECK_EQ(r.slot_ctx, 131072u);  // still fits one 131072-token request
        const auto s = numerics::serverPoolFit(3 * GiB, per_tok, 400000u, false, 131072, false, 256, 8 * 262144);
        CHECK(!s.refuse);
        CHECK(s.slot_ctx == s.pool && s.slot_ctx < 131072u);  // context per request capped to the pool
        // below the minimum pool: refused with a clear message
        const auto z = numerics::serverPoolFit(100 * MiB, per_tok, 400000u, false, 131072, false, 256, 8 * 262144);
        CHECK(z.refuse);
        CHECK(z.msg.find("at least 4096 needed") != std::string::npos);
    }
}

// ---------------------------------------------------------------------------
// arch024-B0 (0.2.4 bug-fix line): REVIEW-S8-719 C3 / C5, REVIEW-S8-719-3 N3-1 / N3-2 / N3-4 / N3-6 (+ the N3-5 blind spot).
// Each block is written from the review finding, before its fix (progress\arch024-B0.log has the before/after runs).
void testB0SizingFollowUps() {
    using namespace whirl;
    constexpr std::uint64_t MiB = 1ull << 20, GiB = 1ull << 30;
    const std::regex bad(R"(hipError|hip.*\b719\b|\b719\b.*hip|\| failed:|PAL failed|Failed PAL|result:-5|unspecified launch)",
                         std::regex::icase);
    // C5 (REVIEW-S8-719:77-80): a desktop R9700 host whose Ryzen iGPU (gfx1036, no WHIRL kernels) is a HIP GPU. What makes
    // n = 1 is restrictToSelected (gpu_restrict.cpp, since 0.2.3): HIP_VISIBLE_DEVICES names the chosen GPU alone, whichever
    // GPU that is (REVIEW-B0 K2); on this host the chosen one happens to be the R9700 because the iGPU has no kernels. So
    // the pinned room sees n = 1 and the 32 GB host keeps v0.2.2.2's automatic 8192 (C5 feared n = 2 -> about 4096). Either order.
    {
        using namespace whirl::gpusel;
        auto rec = [](int i, const char* name, const char* arch, int bus, bool integ, bool sup, std::uint64_t mib) {
            GpuRecord r;
            r.index = i;
            r.name = name;
            r.arch = arch;
            r.pci_bus = bus;
            r.luid = std::format("{:016x}", 0x2a000 + 0x1000 * i);
            r.integrated = integ;
            r.supported = sup;
            r.total_mib = mib;
            r.compute_units = integ ? 2 : 64;
            r.uuid = std::format("{:08x}-0000-0000-0000-{:012x}", 0xc0 + i, 0xfeed00 + bus);
            return r;
        };
        for (const bool igpu_first : {false, true}) {
            const int ri = igpu_first ? 1 : 0;
            std::vector<GpuRecord> g = {rec(ri, "AMD Radeon AI PRO R9700", "gfx1201", 0x03, false, true, 32624),
                                        rec(1 - ri, "AMD Radeon(TM) Graphics", "gfx1036", 0x7c, true, false, 16300)};
            if (igpu_first) std::swap(g[0], g[1]);
            CHECK_EQ(defaultGpu(g).index, ri);
            const std::vector<int> sel = selectGpus(g, "");
            CHECK(sel.size() == 1 && g[sel[0]].arch == "gfx1201");
            CHECK_EQ(visibleDevicesValue({g[sel[0]].index}), std::to_string(ri));  // one ordinal: HIP sees one GPU
        }
        startfit::PinnedState one;
        one.total_phys = 32407 * MiB;
        one.pinned_max = (1294 + 300) * MiB;
        one.gpus = {{15435 * MiB, 1294 * MiB, 16203 * MiB}};
        CHECK(!startfit::ramTierFit(8192, 20000 * MiB, 8192, false, startfit::pinnedRoomMb(one), 8192).has_value());
    }
    // REVIEW-B0 K2: the owner's host, 8060S + R9700 (scratch\s8_719\devices_all.txt: device 0 8060S gfx1151 integrated,
    // 95.8 GiB carve-out, pci:196:0; device 1 R9700 gfx1201 31.9 GiB, pci:71:0). The default is the 8060S (the most GPU
    // memory, owner 2026-10-23), --device pci:71:0 the R9700; either way ONE ordinal is made visible, and the enumeration
    // HIP then shows (the chosen GPU alone, as ordinal 0) passes restrictToSelected's identity check with n = 1. (The
    // runtime call hipGetDeviceCount itself is GPU-only: s8_719_root.ps1's "1 HIP GPU(s)" and release_gate srv_gpu_select.)
    {
        using namespace whirl::gpusel;
        auto rec = [](int i, const char* name, const char* arch, int bus, bool integ, std::uint64_t total, std::uint64_t fit,
                      int cu) {
            GpuRecord r;
            r.index = i;
            r.name = name;
            r.arch = arch;
            r.pci_bus = bus;
            r.luid = std::format("{:016x}", i == 0 ? 0x182faull : 0x1aa96ull);
            r.integrated = integ;
            r.supported = true;
            r.total_mib = total;
            r.fit_mib = fit;
            r.compute_units = cu;
            r.uuid = std::format("{:08x}-0000-0000-0000-{:012x}", 0xd0 + i, 0xbeef00 + bus);
            return r;
        };
        const std::vector<GpuRecord> host = {
            rec(0, "AMD Radeon(TM) 8060S Graphics", "gfx1151", 196, true, (958 + 158) * 1024 / 10, 958 * 1024 / 10, 20),
            rec(1, "AMD Radeon AI PRO R9700", "gfx1201", 71, false, 319 * 1024 / 10, 319 * 1024 / 10, 32)};
        CHECK_EQ(defaultGpu(host).index, 0);  // the 8060S, not the R9700
        for (const char* spec : {"", "pci:196:0", "pci:71:0"}) {
            const std::vector<int> sel = selectGpus(host, spec);
            CHECK_EQ(sel.size(), 1u);
            const GpuRecord& chosen = host[sel[0]];
            CHECK_EQ(chosen.arch, std::string(std::string_view(spec) == "pci:71:0" ? "gfx1201" : "gfx1151"));
            CHECK_EQ(visibleDevicesValue({chosen.index}), std::to_string(chosen.index));  // one ordinal
            GpuRecord seen = chosen;  // what HIP enumerates after HIP_VISIBLE_DEVICES: that GPU alone, renumbered 0
            seen.index = 0;
            const std::vector<GpuRecord> now = {seen};
            CHECK(now.size() == sel.size() && sameGpu(now[0], chosen));
            CHECK(!sameGpu(now[0], host[1 - sel[0]]));  // and the other GPU would not pass for it
        }
    }
    // C3 (REVIEW-S8-719:65-69): the per-GPU term must count every process's pinned memory on that adapter, not only this
    // process's. 8060S at 96 GB (32407 MiB RAM), this server on the R9700 alone (n = 1, non-local budget 15435), another
    // server holding 6000 MiB pinned on the R9700 adapter: the room + all of that adapter's Shared Usage + the reserve must
    // stay within the budget (the machine term, half of the RAM, is looser here by 768 MiB).
    {
        startfit::PinnedState s;
        s.total_phys = 32407 * MiB;
        s.pinned_max = (1294 + 6000) * MiB;
        s.gpus = {{15435 * MiB, 1294 * MiB, 0, (1294 + 6000) * MiB}};
        const std::uint64_t room = startfit::pinnedRoomMb(s);
        CHECK(room * MiB + (1294 + 6000) * MiB + startfit::pinned_reserve_mb * MiB <= 15435 * MiB);
        CHECK(room > 0ull);
        // the adapter's usage unknown (0): this process only, as in 0.2.3
        s.gpus = {{15435 * MiB, 1294 * MiB, 0, 0}};
        s.pinned_max = 1294 * MiB;
        CHECK_EQ(startfit::pinnedRoomMb(s), 11776ull);  // 15435 - 1294 - 2048 = 12093 -> 11776
        // integrated GPU (C4): a SharedSystemMemory below half of the RAM, other programs 6000 MiB on that adapter
        startfit::PinnedState i;
        i.total_phys = 65536 * MiB;
        i.pinned_max = (1441 + 6000) * MiB;
        i.gpus = {{0, 1441 * MiB, 16384 * MiB, (1441 + 6000) * MiB}};
        const std::uint64_t ri = startfit::pinnedRoomMb(i);
        CHECK(ri * MiB + (1441 + 6000) * MiB + startfit::pinned_reserve_mb * MiB <= 16384 * MiB);
    }
    // N3-1 (REVIEW-S8-719-3:55-57): an explicit size is monotonic in the request at every MiB, not only at multiples of 512
    // (fixb s23_ram12288_vis: 17643 MiB available, ask 8800 -> 8800 but 8822 -> 8704), and never above the ask
    for (const std::uint64_t avail_mb : {17643ull, 17000ull, 19032ull, 8191ull, 40001ull})
        for (const std::uint64_t room : {startfit::pinned_room_unknown, 4096ull, 12288ull}) {
            std::uint64_t prev = 0;
            bool mono = true, within = true;
            for (std::uint64_t ask = 1; ask <= 20480; ++ask) {
                const auto r = startfit::ramTierFit(ask, avail_mb * MiB, 0, true, room, 0);
                const std::uint64_t got = r ? r->mb : ask;
                if (got < prev) mono = false;
                if (got > ask) within = false;
                prev = got;
            }
            CHECK(mono);
            CHECK(within);
        }
    {
        const auto r = startfit::ramTierFit(12288, 17643 * MiB, 0, true);  // the measured case keeps its 8704
        CHECK(r.has_value() && r->mb == 8704ull && r->half);
    }
    // N3-4 (REVIEW-S8-719-3:61, REVIEW-S8-719 N2): pinned memory this process allocates after the measurement (the
    // discrete-GPU mmproj, 887 MiB in s23_ram8192_mm) must come out of the room, the same as if it were pinned already
    {
        startfit::PinnedState s;
        s.total_phys = 32407 * MiB;
        s.pinned_max = 1294 * MiB;
        s.gpus = {{15435 * MiB, 1294 * MiB, 0, 1294 * MiB}};
        startfit::PinnedState l = s;
        l.later = 887 * MiB;
        startfit::PinnedState now = s;
        now.pinned_max += 887 * MiB;
        now.gpus[0].own_nonlocal += 887 * MiB;
        now.gpus[0].adapter_shared += 887 * MiB;
        CHECK_EQ(startfit::pinnedRoomMb(l), startfit::pinnedRoomMb(now));
        CHECK(startfit::pinnedRoomMb(l) < startfit::pinnedRoomMb(s));
        startfit::PinnedState ig;  // the integrated GPU and the machine term too
        ig.total_phys = 32407 * MiB;
        ig.pinned_max = 1441 * MiB;
        ig.gpus = {{0, 1441 * MiB, 16203 * MiB, 1441 * MiB}};
        startfit::PinnedState igl = ig;
        igl.later = 887 * MiB;
        CHECK(startfit::pinnedRoomMb(igl) * MiB + (1441 + 887) * MiB + startfit::pinned_reserve_mb * MiB <= 16203 * MiB);
        // REVIEW-B0 N1: the mmproj file size is not an upper bound of what Vision::load pins. MEASURED: Swift's
        // mmproj-Swift-1.5-Qwen3.8-27B-F16.gguf is 927607584 B (884.6 MiB) and pinned 887 MiB (REVIEW-S8-719-3 N3-4)
        constexpr std::uint64_t swift_mmproj = 927607584;
        CHECK(startfit::mmprojPinnedBytes(swift_mmproj) >= 887 * MiB);
        CHECK(startfit::mmprojPinnedBytes(swift_mmproj) <= swift_mmproj + 128 * MiB);  // a bound, not a guess of 2x
        CHECK(startfit::mmprojPinnedBytes(0) == 0);  // no mmproj: nothing later
    }
    // N3-2 (REVIEW-S8-719-3:59): when the half-of-available-RAM rule (not the pinned room) turns an explicit tier off, the
    // texts must not say "no room left"; they must name the half rule. A room of 0 still says "no room left".
    {
        const std::string pw = startfit::pinnedWhy(4096, 1);
        const auto h = startfit::ramTierFit(8192, 700 * MiB, 0, true, 4096, 0);  // room 4096, then half of 700 MiB -> 0
        CHECK(h.has_value() && h->mb == 0ull && h->half);
        const auto th = startfit::ramTierText(h, 8192, true, "--kv-ram-mb", "--kv-ram-mb 8192", pw, 0, 700);
        CHECK(th.adjusted_why.find("no room left") == std::string::npos);
        CHECK(th.off_reason.find("no room left") == std::string::npos);
        CHECK(th.off_reason.find("half") != std::string::npos);
        CHECK(th.off_reason.find("SSD") != std::string::npos);
        const auto h2 = startfit::ramTierFit(4096, 700 * MiB, 0, true);  // the half rule alone
        const auto th2 = startfit::ramTierText(h2, 4096, true, "--kv-ram-mb", "--kv-ram-mb 4096", "", 0, 700);
        CHECK(th2.off_reason.find("half") != std::string::npos);
        const auto z = startfit::ramTierFit(8192, 40000 * MiB, 0, true, 0, 0);  // room 0
        CHECK(z.has_value() && z->mb == 0ull && z->pinned);
        const auto tz = startfit::ramTierText(z, 8192, true, "--kv-ram-mb", "--kv-ram-mb 8192", startfit::pinnedWhy(0, 1), 0, 40000);
        CHECK(tz.off_reason.find("no room left") != std::string::npos);
    }
    // N3-6 (REVIEW-S8-719-3:63): "kv tier: RAM tier size 4096 MiB (WHIRL_KV_RAM_MB 8192)" shows the asked size next to the
    // clamped one; a lowered size must say it was lowered (explicit and automatic). Unchanged: the reason as before.
    {
        const auto e = startfit::ramTierFit(8192, 19374 * MiB, 0, true, 4096, 9216);
        const auto te = startfit::ramTierText(e, 8192, true, "WHIRL_KV_RAM_MB", "WHIRL_KV_RAM_MB 8192", startfit::pinnedWhy(4096, 1), 9216, 19374);
        CHECK(te.size_reason.find("lowered") != std::string::npos);
        const auto a = startfit::ramTierFit(16384, 30 * GiB, 16384, false, 12288, 9216);
        CHECK(a.has_value() && a->mb == 12288ull);
        const auto ta = startfit::ramTierText(a, 16384, false, "--kv-ram-mb", "auto: 1/4 of 64 GiB RAM", startfit::pinnedWhy(12288, 1), 9216, 30720);
        CHECK(ta.size_reason.find("lowered") != std::string::npos);
        const auto tu = startfit::ramTierText(std::nullopt, 8192, true, "--kv-ram-mb", "--kv-ram-mb 8192", "", 9216, 30720);
        CHECK_EQ(tu.size_reason, std::string("--kv-ram-mb 8192"));
        CHECK_EQ(tu.off_reason, std::string("--kv-ram-mb 8192"));
    }
    // N3-5 blind spot: every line form the server builds from these texts ("adjusted:", "RAM tier off (...)", "RAM tier
    // size N MiB (...)", the automatic W line) matches none of the log-scan error patterns, over a grid of fits
    {
        int lines = 0;
        bool clean = true;
        for (const std::uint64_t ask : {0ull, 4096ull, 8192ull, 8822ull, 24064ull})
            for (const std::uint64_t avail : {700ull, 17643ull, 50000ull})
                for (const std::uint64_t room : {startfit::pinned_room_unknown, 0ull, 4096ull})
                    for (const bool ex : {true, false}) {
                        const auto f = startfit::ramTierFit(ask, avail * MiB, ask, ex, room, 9216);
                        const auto t = startfit::ramTierText(f, ask, ex, "--kv-ram-mb", ex ? "--kv-ram-mb N" : "auto: 1/4 of 64 GiB RAM",
                                                             room == startfit::pinned_room_unknown ? "" : startfit::pinnedWhy(room, 1), 9216, avail);
                        const std::uint64_t mb = f ? f->mb : ask;
                        for (const std::string& l :
                             {std::format("2026-10-23 18:00:00.000 W adjusted: --kv-ram-mb {} -> {} MiB ({})", ask, mb, t.adjusted_why),
                              std::format("2026-10-23 18:00:00.000 I kv tier: RAM tier off ({})", t.off_reason),
                              std::format("2026-10-23 18:00:00.000 I kv tier: RAM tier size {} MiB ({})", mb, t.size_reason),
                              std::format("2026-10-23 18:00:00.000 W kv tier: RAM tier {} MiB (the automatic size would be {} MiB): {}", mb, ask,
                                          t.pinned_why)}) {
                            ++lines;
                            if (std::regex_search(l, bad)) clean = false;
                        }
                    }
        CHECK(clean);
        CHECK(lines > 0);
    }
}

// ---------------------------------------------------------------------------
// s8-719 root fix: GPU selection (--device / -dev list), the HIP_VISIBLE_DEVICES value, user variables,
// the identity check, the probe protocol, the stable lock name, the device list (written from the bug
// description and the owner / lead requirements of 2026-10-23: a process that uses one GPU must make HIP
// see only that GPU; the user's own visibility variable wins; the option is a list; identical cards and
// same-architecture cards are never resolved by "first match")
void testGpuSelect() {
    using namespace whirl::gpusel;
    auto rec = [](int i, const char* name, const char* arch, int bus, bool integ, bool sup, std::uint64_t mib) {
        GpuRecord r;
        r.index = i;
        r.name = name;
        r.arch = arch;
        r.pci_bus = bus;
        r.luid = std::format("{:016x}", 0x182fa + 0x1000 * i);
        r.integrated = integ;
        r.supported = sup;
        r.total_mib = mib;
        r.compute_units = integ ? 20 : 32;
        r.uuid = std::format("{:08x}-0000-0000-0000-{:012x}", 0xa0 + i, 0xbeef00 + bus);  // per card
        return r;
    };
    const char* k8060 = "AMD Radeon(TM) 8060S Graphics";
    const char* kR97 = "AMD Radeon AI PRO R9700";
    const char* k9070 = "AMD Radeon RX 9070 XT";
    // this machine (scratch\s8_719\devices_all.txt): 0 = 8060S, 1 = R9700
    const std::vector<GpuRecord> two = {rec(0, k8060, "gfx1151", 0xc3, true, true, 110490), rec(1, kR97, "gfx1201", 0x63, false, true, 32624)};
    const std::vector<GpuRecord> igpu_only = {rec(0, k8060, "gfx1151", 0xc3, true, true, 110490)};
    const std::vector<GpuRecord> r9700_only = {rec(0, kR97, "gfx1201", 0x03, false, true, 32624)};
    const std::vector<GpuRecord> r9700_first = {rec(0, kR97, "gfx1201", 0x63, false, true, 32624), rec(1, k8060, "gfx1151", 0xc3, true, true, 110490)};
    const std::vector<GpuRecord> unsupported_first = {rec(0, "AMD Radeon RX 6800", "gfx1030", 0x10, false, false, 16368),
                                                      rec(1, k8060, "gfx1151", 0xc3, true, true, 110490)};
    // owner's 4-card host: 2x R9700 + 2x RX 9070 XT (all gfx1201), distinct PCI buses
    const std::vector<GpuRecord> four = {rec(0, kR97, "gfx1201", 0x03, false, true, 32624), rec(1, kR97, "gfx1201", 0x04, false, true, 32624),
                                         rec(2, k9070, "gfx1201", 0x07, false, true, 16304), rec(3, k9070, "gfx1201", 0x0a, false, true, 16304)};
    auto errOf = [](const std::vector<GpuRecord>& g, const char* spec) {
        try {
            (void)selectGpus(g, spec);
        } catch (const SelectError& e) {
            return std::string(e.what());
        } catch (...) {
            return std::string("other exception");
        }
        return std::string("no error");
    };
    auto has = [](const std::string& s, const std::string& part) { return s.find(part) != std::string::npos; };

    // list parsing (llama.cpp -dev <dev1,dev2,..>)
    CHECK(parseDeviceList("").empty());
    CHECK(parseDeviceList("   ").empty());
    CHECK(parseDeviceList("8060s") == std::vector<std::string>{"8060s"});
    CHECK(parseDeviceList(" r9700 , 8060s ") == (std::vector<std::string>{"r9700", "8060s"}));
    CHECK(throws([] { parseDeviceList("r9700,"); }));
    CHECK(throws([] { parseDeviceList(",r9700"); }));
    CHECK(throws([] { parseDeviceList("0,,1"); }));

    // single choices: alias, index, llama.cpp-style name, arch, name part, the full name the device list prints
    CHECK(selectGpus(two, "8060s") == std::vector<int>{0});
    CHECK(selectGpus(two, "8060S") == std::vector<int>{0});
    CHECK(selectGpus(two, "igpu") == std::vector<int>{0});
    CHECK(selectGpus(two, "strix") == std::vector<int>{0});
    CHECK(selectGpus(two, "r9700") == std::vector<int>{1});
    CHECK(selectGpus(two, "R9700") == std::vector<int>{1});
    CHECK(selectGpus(two, "1") == std::vector<int>{1});
    CHECK(selectGpus(two, "0") == std::vector<int>{0});
    CHECK(selectGpus(two, "ROCm1") == std::vector<int>{1});
    CHECK(selectGpus(two, "rocm0") == std::vector<int>{0});
    CHECK(selectGpus(two, "HIP1") == std::vector<int>{1});
    CHECK(selectGpus(two, "gfx1201") == std::vector<int>{1});
    CHECK(selectGpus(two, "gfx1151") == std::vector<int>{0});
    CHECK(selectGpus(two, kR97) == std::vector<int>{1});
    CHECK(selectGpus(two, k8060) == std::vector<int>{0});
    CHECK(selectGpus(two, " r9700 ") == std::vector<int>{1});
    // PCI address forms (hex), stable when cards are added / removed
    // (decimal, as Task Manager > Performance > GPU > Physical location and Device Manager show it:
    // "PCI bus 99, device 0, function 0" -> pci:99:0.0; never hex)
    CHECK(selectGpus(two, "pci:99:0") == std::vector<int>{1});
    CHECK(selectGpus(two, "pci:99:0.0") == std::vector<int>{1});  // a pasted full form works too
    CHECK(selectGpus(two, "PCI:195:0") == std::vector<int>{0});
    CHECK(has(errOf(two, "pci:99"), "pci:99:0"));  // bus alone: error that says how to write it
    CHECK(has(errOf(two, "pci:100:0"), "pci:100:0"));
    CHECK(has(errOf(two, "pci:63:0"), "no GPU matches"));  // hex 0x63 = 99 is not accepted as 63
    CHECK_EQ(pciText(two[1]), std::string("99:0.0"));
    CHECK_EQ(pciLongText(two[1]), std::string("PCI bus 99, device 0, function 0"));
    {
        GpuRecord fn = two[1];
        fn.pci_device = 1;
        fn.pci_function = 2;
        CHECK_EQ(pciText(fn), std::string("99:1.2"));
        CHECK_EQ(pciLongText(fn), std::string("PCI bus 99, device 1, function 2"));
        const std::vector<GpuRecord> same_bus = {two[1], fn};
        CHECK(selectGpus(same_bus, "pci:99:1.2") == std::vector<int>{1});
        // the bus number is what users see and type; the full form only where two GPUs share a bus
        CHECK_EQ(pciId(two, two[1]), std::string("99:0"));
        CHECK_EQ(pciIdLong(two, two[1]), std::string("PCI bus 99, device 0"));
        CHECK_EQ(pciId(same_bus, same_bus[0]), std::string("99:0"));
        CHECK_EQ(pciId(same_bus, same_bus[1]), std::string("99:1"));
        CHECK_EQ(pciIdLong(same_bus, same_bus[1]), std::string("PCI bus 99, device 1"));
        CHECK(has(listingText(same_bus, 0), "pci:99:1 (PCI bus 99, device 1)"));
        CHECK(selectGpus(same_bus, "pci:99:1") == std::vector<int>{1});  // bus:device, as the list prints it
        CHECK(selectGpus(same_bus, "pci:99:0") == std::vector<int>{0});
        GpuRecord dom = two[1];
        dom.pci_domain = 1;
        CHECK_EQ(pciText(dom), std::string("1:99:0.0"));  // a non-zero PCI domain is shown in front
    }

    // default (no option; owner 2026-10-23 「預設用vram最大的卡」「不管是不是獨顯」): among the GPUs this build
    // has kernels for, the most GPU memory (an integrated GPU: its dedicated carve-out), discrete or
    // integrated alike; a tie -> the lowest index. On this host that is now the 8060S (was the R9700).
    CHECK(selectGpus(two, "") == std::vector<int>{0});
    CHECK(selectGpus(r9700_first, "") == std::vector<int>{1});
    CHECK(selectGpus(igpu_only, "") == std::vector<int>{0});
    CHECK(selectGpus(r9700_only, "") == std::vector<int>{0});
    CHECK(selectGpus(unsupported_first, "") == std::vector<int>{1});
    CHECK(selectGpus(four, "") == std::vector<int>{0});
    {
        const std::vector<GpuRecord> small_first = {rec(0, k9070, "gfx1201", 0x07, false, true, 16304), rec(1, kR97, "gfx1201", 0x03, false, true, 32624)};
        CHECK(selectGpus(small_first, "") == std::vector<int>{1});  // most memory, not index 0
        const std::vector<GpuRecord> twins = {rec(0, kR97, "gfx1201", 0x09, false, true, 32624), rec(1, kR97, "gfx1201", 0x03, false, true, 32624)};
        CHECK(selectGpus(twins, "") == std::vector<int>{0});  // identical cards: lowest index
    }
    CHECK_EQ(autoPick(two), 0);
    CHECK_EQ(autoPick(std::vector<GpuRecord>{rec(0, "AMD Radeon RX 6800", "gfx1030", 1, false, false, 16368)}), -1);
    CHECK(throws([&] { selectGpus(std::vector<GpuRecord>{rec(0, "AMD Radeon RX 6800", "gfx1030", 1, false, false, 16368)}, ""); }));
    CHECK(throws([&] { selectGpus(std::vector<GpuRecord>{}, ""); }));

    // the alias r9700 is the card's NAME, not gfx1201: an RX 9070 XT is gfx1201 too
    {
        const std::vector<GpuRecord> r97_and_9070 = {rec(0, k9070, "gfx1201", 0x07, false, true, 16304), rec(1, kR97, "gfx1201", 0x03, false, true, 32624)};
        CHECK(selectGpus(r97_and_9070, "r9700") == std::vector<int>{1});
        CHECK(selectGpus(r97_and_9070, "9070") == std::vector<int>{0});
        const std::string e = errOf(r97_and_9070, "gfx1201");  // a raw architecture is ambiguous here
        CHECK(has(e, "device 0") && has(e, "device 1") && has(e, "more than one GPU"));
        const std::vector<GpuRecord> only9070 = {rec(0, k9070, "gfx1201", 0x07, false, true, 16304)};
        CHECK(has(errOf(only9070, "r9700"), "no GPU matches"));
    }
    // the 4-card host
    {
        const std::string e = errOf(four, "r9700");
        CHECK(has(e, "more than one GPU"));
        CHECK(has(e, "device 0: AMD Radeon AI PRO R9700 (gfx1201), pci:3:0, 31.9 GiB"));
        CHECK(has(e, "device 1: AMD Radeon AI PRO R9700 (gfx1201), pci:4:0, 31.9 GiB"));
        CHECK(!has(e, "device 2:") && !has(e, "device 3:"));
        CHECK(has(e, "--device 0") && has(e, "--device pci:3:0"));
        const std::string e2 = errOf(four, "9070");
        CHECK(has(e2, "more than one GPU"));
        CHECK(has(e2, "device 2: AMD Radeon RX 9070 XT (gfx1201), pci:7:0, 15.9 GiB"));
        CHECK(has(e2, "device 3: AMD Radeon RX 9070 XT (gfx1201), pci:10:0, 15.9 GiB"));
        CHECK(!has(e2, "device 0:") && !has(e2, "device 1:"));
        CHECK(selectGpus(four, "2") == std::vector<int>{2});
        CHECK(selectGpus(four, "pci:10:0.0") == std::vector<int>{3});
        CHECK(selectGpus(four, "ROCm3") == std::vector<int>{3});
        CHECK(has(errOf(four, "gfx1201"), "more than one GPU"));
        // the list
        const std::string l = listingText(four, autoPick(four));
        CHECK(has(l, std::string("device 0: AMD Radeon AI PRO R9700 (gfx1201), 31.9 GiB, 32 CUs, pci:3:0 (PCI bus 3, device 0), uuid " + four[0].uuid + "  [supported] [default]")));
        CHECK(has(l, std::string("device 3: AMD Radeon RX 9070 XT (gfx1201), 15.9 GiB, 32 CUs, pci:10:0 (PCI bus 10, device 0), uuid " + four[3].uuid + "  [supported]\n")));
        CHECK(has(l, "can change") && has(l, "pci:") && has(l, "Physical location"));
        const std::string l2 = listingText(unsupported_first, autoPick(unsupported_first));
        CHECK(has(l2, std::string("device 0: AMD Radeon RX 6800 (gfx1030), 16.0 GiB, 32 CUs, pci:16:0 (PCI bus 16, device 0), uuid " + unsupported_first[0].uuid + "  [no GPU kernels for this architecture in this build]")));
        CHECK(has(l2, "[integrated: memory shared with the CPU] [default]"));
        // the PCI function number is never part of what users read (one card = one bus number)
        CHECK(!has(l, "function") && !has(errOf(four, "r9700"), "function") && !has(restrictionNote(four, {2}, ""), "function"));
        CHECK(!has(errOf(four, "rtx4090"), "function") && !has(l, ":0.0"));
    }

    // daisy-chained USB4 eGPUs: two identical R9700 (bus numbers given at connect time) + the 8060S
    {
        const std::vector<GpuRecord> chain = {rec(0, k8060, "gfx1151", 196, true, true, 110490), rec(1, kR97, "gfx1201", 71, false, true, 32624),
                                              rec(2, kR97, "gfx1201", 74, false, true, 32624)};
        const std::string e = errOf(chain, "r9700");
        CHECK(has(e, "more than one GPU") && has(e, "pci:71:0") && has(e, "pci:74:0") && !has(e, "pci:196:0"));
        CHECK(has(errOf(chain, "R9700"), "Physical location"));
        CHECK(selectGpus(chain, "pci:71:0") == std::vector<int>{1});
        CHECK(selectGpus(chain, "pci:74:0") == std::vector<int>{2});
        CHECK(selectGpus(chain, "pci:196:0") == std::vector<int>{0});
        CHECK(selectGpus(chain, "8060s") == std::vector<int>{0});
        CHECK(selectGpus(chain, "") == std::vector<int>{0});  // the 8060S: most memory
        CHECK(has(errOf(chain, "pci:74"), "pci:74:0"));
        const std::string l = listingText(chain, autoPick(chain));
        CHECK(has(l, "pci:71:0 (PCI bus 71, device 0)") && has(l, "pci:74:0 (PCI bus 74, device 0)") && has(l, "pci:196:0 (PCI bus 196, device 0)"));
    }

    // the default rule in one function (defaultGpu): most GPU memory, tie -> lowest index, kernels required
    {
        auto fit = [](std::vector<GpuRecord> g, int i, std::uint64_t mib) {
            g[i].fit_mib = mib;
            return g;
        };
        const std::vector<GpuRecord> host = fit(two, 0, 98099);  // this machine: 8060S dedicated 95.8 GiB, R9700 31.9 GiB
        CHECK_EQ(defaultGpu(host).index, 0);
        CHECK_EQ(defaultGpu(fit(two, 0, 16 * 1024)).index, 1);  // an 8060S with a 16 GiB carve-out: the R9700 has more
        CHECK_EQ(defaultGpu(r9700_only).index, 0);
        CHECK_EQ(defaultGpu(igpu_only).index, 0);
        CHECK_EQ(defaultGpu(four).index, 0);
        {
            const std::vector<GpuRecord> eq = {rec(0, k9070, "gfx1201", 9, false, true, 16304), rec(1, k9070, "gfx1201", 3, false, true, 16304)};
            CHECK_EQ(defaultGpu(eq).index, 0);  // equal memory: lowest index
            const std::vector<GpuRecord> big_unsup = {rec(0, "AMD Instinct MI300X", "gfx942", 5, false, false, 196608), rec(1, kR97, "gfx1201", 3, false, true, 32624)};
            CHECK_EQ(defaultGpu(big_unsup).index, 1);  // a larger card without kernels in this build is ignored
            CHECK_EQ(defaultGpu(std::vector<GpuRecord>{rec(0, "AMD Instinct MI300X", "gfx942", 5, false, false, 196608)}).index, -1);
        }
        // the startup line (owner 2026-10-23: usage only, no cause): "GPU pci:196:0 (NAME): most GPU memory (95.8 GiB);
        // use --device pci:<bus>:<device> to choose another (see --list-devices)"
        const std::vector<GpuRecord> h2 = fit({rec(0, k8060, "gfx1151", 196, true, true, 110490), rec(1, kR97, "gfx1201", 71, false, true, 32624)}, 0, 98099);
        const AutoChoice c = defaultGpu(h2);
        CHECK_EQ(c.index, 0);
        CHECK(has(c.reason, "most GPU memory (95.8 GiB)"));
        const std::string n = restrictionNote(h2, {0}, c.reason);
        CHECK_EQ(n, std::string("GPU pci:196:0 (AMD Radeon(TM) 8060S Graphics): most GPU memory (95.8 GiB); use --device "
                                "pci:<bus>:<device> to choose another (see --list-devices)"));
        CHECK(!has(n, "function"));
        const std::string ne = restrictionNote(h2, {1}, "");  // chosen with --device: no automatic reason
        CHECK_EQ(ne, std::string("GPU pci:71:0 (AMD Radeon AI PRO R9700): chosen with --device or WHIRL_DEVICE; use --device "
                                 "pci:<bus>:<device> to choose another (see --list-devices)"));
        // the list marks the same default and shows the dedicated memory of the iGPU
        CHECK(has(listingText(h2, defaultGpu(h2).index), "[integrated: memory shared with the CPU] [default]"));
        // the probe carries the usable memory
        const std::vector<GpuRecord> back = parseRecords(formatRecord(host[0]));
        CHECK(back.size() == 1 && back[0].fit_mib == host[0].fit_mib);
    }
    // a list of two: a clear error, never a silent fallback to one of them
    // owner 2026-10-23: WHIRL always used one GPU, so the error says only that (no "in this version", no "planned")
    for (const char* lst : {"r9700,8060s", "0,1"}) {
        const std::string msg = errOf(two, lst);
        CHECK(has(msg, "only one GPU can be given with --device"));
        CHECK(!has(msg, "in this version"));
        CHECK(!has(msg, "planned"));
        CHECK(!has(msg, "splitting"));
    }
    CHECK(throws([&] { selectGpus(two, "1,1"); }));
    // unknown name / index out of range: the error names the value and lists the GPUs present
    for (const char* bad : {"rtx4090", "7", "ROCm5", "gfx90a"}) {
        const std::string msg = errOf(two, bad);
        CHECK(has(msg, bad));
        CHECK(has(msg, "device 0: AMD Radeon(TM) 8060S Graphics (gfx1151)"));
        CHECK(has(msg, "device 1: AMD Radeon AI PRO R9700 (gfx1201)"));
    }

    // HIP_VISIBLE_DEVICES value (a list later)
    CHECK_EQ(visibleDevicesValue({1}), std::string("1"));
    CHECK_EQ(visibleDevicesValue({0}), std::string("0"));
    CHECK_EQ(visibleDevicesValue({0, 1}), std::string("0,1"));

    // a visibility variable the user set wins (never overridden); an empty value counts as set
    {
        std::map<std::string, std::string> envm;
        auto ge = [&](const char* k) -> const char* {
            auto it = envm.find(k);
            return it == envm.end() ? nullptr : it->second.c_str();
        };
        CHECK_EQ(userVisibilityVar(ge), std::string());
        envm["WHIRL_DEVICE"] = "8060s";
        CHECK_EQ(userVisibilityVar(ge), std::string());
        for (const char* k : {"HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"}) {
            envm.clear();
            envm[k] = "0";
            CHECK_EQ(userVisibilityVar(ge), std::string(k));
        }
        envm.clear();
        envm["HIP_VISIBLE_DEVICES"] = "";
        CHECK_EQ(userVisibilityVar(ge), std::string("HIP_VISIBLE_DEVICES"));
    }

    // identity: the GPU left after the restriction must be the one chosen (never trust the ordinal alone)
    {
        GpuRecord after = two[1];
        after.index = 0;  // re-numbered by the restriction
        CHECK(sameGpu(two[1], after));
        CHECK(!sameGpu(two[0], after));
        GpuRecord bus = two[1];
        bus.pci_bus = 0x64;
        CHECK(!sameGpu(two[1], bus));
        GpuRecord luid = two[1];
        luid.luid = "0000000000000001";
        CHECK(!sameGpu(two[1], luid));
        GpuRecord noluid = two[1];
        noluid.luid = "";
        CHECK(sameGpu(two[1], noluid));  // LUID unknown on one side: the PCI address + name + arch decide
        GpuRecord nm = two[1];
        nm.name = k9070;
        CHECK(!sameGpu(two[1], nm));
        CHECK(!sameGpu(four[0], four[1]));  // identical cards differ by PCI bus
    }
    // uuid (stable per card when the runtime reports one): selection by full id or a unique prefix
    {
        CHECK(selectGpus(four, std::string("uuid:" + four[3].uuid).c_str()) == std::vector<int>{3});
        CHECK(selectGpus(four, "uuid:000000a2") == std::vector<int>{2});
        CHECK(selectGpus(four, "UUID:000000A1") == std::vector<int>{1});
        CHECK(has(errOf(four, "uuid:000000a"), "more than one GPU"));  // prefix of all four
        CHECK(has(errOf(four, "uuid:ffff"), "no GPU matches"));
        std::vector<GpuRecord> nouuid = two;
        for (auto& g : nouuid) g.uuid.clear();
        CHECK(has(errOf(nouuid, "uuid:000000a0"), "no GPU matches"));
        const std::string l = listingText(four, 0);
        CHECK(has(l, "uuid " + four[0].uuid));
        CHECK(!has(listingText(nouuid, 1), "uuid"));
        CHECK(!has(restrictionNote(four, {1}, ""), "uuid"));  // the startup line stays short (owner 2026-10-23)
    }
    // the GPU lock is keyed by the PCI address: index 0 after the restriction is not "GPU 0"
    {
        GpuRecord a0 = four[1];
        a0.index = 0;  // the same card as seen by a restricted process
        CHECK_EQ(lockName(a0), lockName(four[1]));
        CHECK(lockName(four[0]) != lockName(four[1]));
        GpuRecord b0 = four[2];
        b0.index = 0;
        CHECK(lockName(a0) != lockName(b0));  // two restricted processes on two GPUs do not block each other
        CHECK(has(lockName(four[1]), four[1].uuid));  // uuid first when reported
        GpuRecord np = four[1];
        np.uuid.clear();
        CHECK(has(lockName(np), "pci-4-0-0"));  // else the PCI address
        GpuRecord np0 = np;
        np0.index = 0;
        CHECK_EQ(lockName(np0), lockName(np));
    }

    // probe protocol round trip (names with spaces and parentheses), other lines ignored
    {
        const std::string text =
            "some runtime chatter\n" + formatRecord(two[0]) + "\n" + formatRecord(two[1]) + "\r\nwhirl-gpu\tbad\n";
        const std::vector<GpuRecord> back = parseRecords(text);
        CHECK_EQ(back.size(), std::size_t(2));
        if (back.size() == 2) {
            for (std::size_t i = 0; i < 2; ++i) {
                CHECK_EQ(back[i].index, two[i].index);
                CHECK_EQ(back[i].name, two[i].name);
                CHECK_EQ(back[i].arch, two[i].arch);
                CHECK_EQ(back[i].pci_bus, two[i].pci_bus);
                CHECK_EQ(back[i].luid, two[i].luid);
                CHECK_EQ(back[i].integrated, two[i].integrated);
                CHECK_EQ(back[i].supported, two[i].supported);
                CHECK_EQ(back[i].total_mib, two[i].total_mib);
                CHECK_EQ(back[i].compute_units, two[i].compute_units);
                CHECK_EQ(back[i].uuid, two[i].uuid);
                CHECK(sameGpu(back[i], two[i]));
            }
        }
        CHECK(parseRecords("").empty());
    }

    // the log line names the GPU used (name, pci) and how to choose another; owner 2026-10-23: usage only, the cause
    // is not public (no "hidden from HIP", no "pinned host memory", no HIP_VISIBLE_DEVICES)
    {
        const std::string n = restrictionNote(two, {1}, "");
        CHECK(has(n, kR97) && has(n, "GPU pci:99:0 (") && !has(n, "pci:99:0."));
        CHECK(has(n, "use --device pci:<bus>:<device> to choose another (see --list-devices)"));
        const std::string n4 = restrictionNote(four, {2}, "");
        for (const std::string& t : {n, n4}) {
            CHECK(!has(t, "hidden") && !has(t, "HIP") && !has(t, "pinned") && !has(t, "other GPU"));
        }
        // must not look like a HIP error to the log scanners (golden.ps1 / kvram719_tier.ps1 patterns)
        CHECK(!std::regex_search(
            n + n4, std::regex("hipError|hip.*\\b719\\b|\\b719\\b.*hip|\\| failed:|PAL failed|Failed PAL|result:-5|unspecified launch",
                               std::regex::icase)));
        const std::string one = restrictionNote(igpu_only, {0}, "");
        CHECK(!has(one, "hidden"));  // nothing to hide
        CHECK(has(one, k8060) && has(one, "the only GPU"));
        const std::string u = userVarNote("HIP_VISIBLE_DEVICES", "1");
        CHECK(has(u, "HIP_VISIBLE_DEVICES=1") && has(u, "visible"));
    }
}

// s8-719 review C1 (written from the finding, 2026-10-23): the GPU probe is a hidden helper process, so
// (a) a crash in it must never show a Windows error dialog, (b) the parent must not hand it every inheritable
// handle (log file, listening socket), only the pipe and NUL, and (c) it must give up after 10 s, not 60 s.
// The child is this test program itself (helper modes in main()).
#if WHIRL_WITH_HIP && defined(_WIN32)
void testProbeChild(const wchar_t* self) {
    using namespace whirl::gpusel;
    CHECK_EQ(k_probe_timeout_ms, 10000u);
    const std::wstring exe = self;
    const std::wstring q = L"\"" + exe + L"\" ";
    // (a) error mode of the child: SEM_NOGPFAULTERRORBOX even when the parent has none
    const UINT old = SetErrorMode(0);
    ChildRun em = runChild(exe, q + L"--child-error-mode", 10000);
    SetErrorMode(old);
    CHECK(em.started && em.exit_code == 0);
    const unsigned long mode = em.out.empty() ? 0 : std::stoul(em.out);
    CHECK((mode & SEM_NOGPFAULTERRORBOX) != 0);
    // (b) an inheritable handle of the parent must not reach the child
    SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
    HANDLE ev = CreateEventW(&sa, TRUE, TRUE, nullptr);  // inheritable, manual reset, signaled
    CHECK(ev != nullptr);
    ChildRun hi = runChild(exe, q + L"--child-has-handle " + std::to_wstring(reinterpret_cast<std::uintptr_t>(ev)), 10000);
    CHECK(hi.started && hi.exit_code == 0);
    CHECK(hi.out.rfind("0", 0) == 0);  // "0" = not inherited
    CloseHandle(ev);
    // (c) a child that hangs is killed after the bound
    const ULONGLONG t0 = GetTickCount64();
    ChildRun hang = runChild(exe, q + L"--child-sleep-ms 30000", 1000);
    const ULONGLONG dt = GetTickCount64() - t0;
    CHECK(hang.timed_out);
    CHECK(hang.out.empty());
    CHECK(dt < 8000);
    // a normal child: output captured
    ChildRun ok = runChild(exe, q + L"--child-echo", 10000);
    CHECK(ok.started && ok.exit_code == 0 && ok.why.empty());
    CHECK(ok.out.rfind("echo-ok", 0) == 0);
}
#endif

int main(int argc, char** argv) {
#if WHIRL_WITH_HIP && defined(_WIN32)
    if (argc >= 2) {  // helper modes for testProbeChild (the test runs this program as the child)
        const std::string m = argv[1];
        if (m == "--child-error-mode") { std::printf("%u", GetErrorMode()); return 0; }
        if (m == "--child-has-handle" && argc == 3) {
            HANDLE h = reinterpret_cast<HANDLE>(static_cast<std::uintptr_t>(std::stoull(argv[2])));
            DWORD f = 0;
            const bool live = GetHandleInformation(h, &f) && WaitForSingleObject(h, 0) == WAIT_OBJECT_0;
            std::printf("%d", live ? 1 : 0);
            return 0;
        }
        if (m == "--child-sleep-ms" && argc == 3) { Sleep(static_cast<DWORD>(std::stoul(argv[2]))); return 0; }
        if (m == "--child-echo") { std::printf("echo-ok"); return 0; }
    }
#endif
    whirl::test::clearWhirlEnv();  // FR-TEST P3-11
#if WHIRL_TEST_DLL_DEPS
    testDllDeps();
#endif
    testRelaxAccept();
    testJson();
    testUnicode();
    testGguf();
    testGgufHardening();
    testTokenizerByteTokens();
    testChat();
    testVramLimit();
    testCardDefaults();
    testNumerics();
    testKvChoice();
    testKernelFetchBounds();
    testValueScope();
    testLaunchDigest();
    testLaunchDigestLayout();
    testLaunchDigestZeroAndAdjacent();
    testLaunchRawCap();
    testAllocLog();
    specsample_test::testSpecSample();
    testDflashPolicy();
    testLocalMemTaken();
    testUmaCheck();
    testWddmWatch();
    testPoolBudget();
    testPoolMargin();
    testStartupFit();
    testB0SizingFollowUps();
    testVisionPoolReserve();
    testMemGovernor();
    testMemGovernorGrowOvershoot();
    testGpuSelect();
#if WHIRL_WITH_HIP && defined(_WIN32)
    {
        wchar_t self[1024];
        const DWORD n = GetModuleFileNameW(nullptr, self, 1024);
        if (n > 0 && n < 1024) testProbeChild(self);
        else CHECK(false);
    }
#endif
    std::printf("unit tests: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
