// WHIRL unit tests of module gguf (arch025 N7: moved verbatim from the single tests/unit/unit_tests.cpp; one file
// per module, built alone as whirl-unit-gguf-tests and together in whirl-tests).
// SPDX-License-Identifier: Apache-2.0

#include "../test_check.h"
#include "../test_env.h"
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

}  // namespace

// the old main()'s calls of this module, in their order (tests/unit/unit_tests.cpp ORDER lists all modules)
const whirl::test::Suite& unitSuite_gguf() {
    static const whirl::test::Suite s{"gguf",
                                      {
                                          {"testGguf", [] { testGguf(); }},
                                          {"testGgufHardening", [] { testGgufHardening(); }},
                                      },
                                      nullptr};
    return s;
}

#ifndef WHIRL_UNIT_ALL  // this module's own program (cmake/unit.cmake whirl-unit-*-tests, or an isolated build of this file)
int main(int argc, char** argv) {
    return whirl::test::runSuites({&unitSuite_gguf()}, argc, argv, [] { whirl::test::clearWhirlEnv(); });
}
#endif
