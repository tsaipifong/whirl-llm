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
#include "whirl/value_scope.h"
#if WHIRL_TEST_DLL_DEPS
#include "release/dll_deps.h"
#endif

#include "../../kernels/gemm_small_addr.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <functional>
#include <map>
#include <string>
#include <vector>

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
    // 16 GB card (or WHIRL_VRAM_LIMIT_MB=16384 / 14336): 1 slot, 1536 MiB headroom; no KV preference (BAL-Q8)
    for (std::uint64_t t : {16 * GiB, 14336 * MiB, 20 * GiB - 1}) {
        vram::CardInput in;
        in.total = t;
        const vram::CardDefaults d = vram::cardDefaults(in);
        CHECK(d.small && d.parallel == 1 && d.parallel_auto);
        CHECK_EQ(d.headroom, 1536 * MiB);
        CHECK(d.note.find("small card") != std::string::npos && d.note.find("--parallel 1") != std::string::npos &&
              d.note.find("KV") == std::string::npos && d.note.find("1536 MiB") != std::string::npos);
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
        CHECK(f.items == (nu::balance_items | nu::fast_only_items));
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

int main() {
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
    specsample_test::testSpecSample();
    testDflashPolicy();
    testLocalMemTaken();
    testUmaCheck();
    std::printf("unit tests: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
