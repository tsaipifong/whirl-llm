// WHIRL host-side unit tests (no GPU, no model files needed).
// SPDX-License-Identifier: Apache-2.0

#include "whirl/chat.h"
#include "whirl/common.h"
#include "whirl/gguf.h"
#include "whirl/json.h"
#include "whirl/tokenizer.h"
#include "whirl/unicode.h"

#include <cstdio>
#include <cstring>
#include <functional>
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

}  // namespace

int main() {
    testJson();
    testUnicode();
    testGguf();
    testGgufHardening();
    testTokenizerByteTokens();
    testChat();
    std::printf("unit tests: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
