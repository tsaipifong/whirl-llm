// whirl-iso-model-weights-tests: src/model/qwen35/weights.cpp built alone (ARCH-1 8.5, arch025-lane2 leg 3) with its
// declared dependencies: the GGUF reader (src/gguf/gguf.cpp), the kernel ABI (kernels_abi.cpp: the MXFP4 row
// repack), lib whirl's util and the platform layer it uses (fs / env / args). The device side is what Model::load
// injects (WeightLoader::Device: alloc + upload callbacks), so no HIP layer and no class Model: the tests run the
// loader on a small GGUF file written here and record what it would put on the device.
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "model/qwen35/model_error.h"
#include "model/qwen35/weight_loader.h"
#include "whirl/gguf.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace q = whirl::qwen35;
namespace gg = whirl::gguf;

// a GGUF v3 image: no metadata, tensors (name, dims, type, payload) at 32-byte aligned offsets
struct T {
    std::string name;
    std::vector<std::uint64_t> ne;
    std::uint32_t type;
    std::vector<std::uint8_t> data;
};

template <class V>
void put(std::vector<std::uint8_t>& b, V v) {
    for (std::size_t i = 0; i < sizeof(V); ++i) b.push_back(static_cast<std::uint8_t>(static_cast<std::uint64_t>(v) >> (8 * i)));
}

std::vector<std::uint8_t> ggufImage(const std::vector<T>& ts) {
    std::vector<std::uint8_t> b;
    put<std::uint32_t>(b, 0x46554747u);
    put<std::uint32_t>(b, 3);
    put<std::uint64_t>(b, ts.size());
    put<std::uint64_t>(b, 0);
    std::uint64_t off = 0;
    std::vector<std::uint64_t> offs;
    for (const T& t : ts) {
        put<std::uint64_t>(b, t.name.size());
        b.insert(b.end(), t.name.begin(), t.name.end());
        put<std::uint32_t>(b, static_cast<std::uint32_t>(t.ne.size()));
        for (std::uint64_t n : t.ne) put<std::uint64_t>(b, n);
        put<std::uint32_t>(b, t.type);
        put<std::uint64_t>(b, off);
        offs.push_back(off);
        off = (off + t.data.size() + 31) / 32 * 32;
    }
    b.resize((b.size() + 31) / 32 * 32, 0);
    const std::size_t data0 = b.size();
    for (std::size_t i = 0; i < ts.size(); ++i) {
        b.resize(data0 + offs[i], 0);
        b.insert(b.end(), ts[i].data.begin(), ts[i].data.end());
    }
    b.resize(data0 + off, 0);
    return b;
}

struct TempFile {
    std::filesystem::path p;
    explicit TempFile(const std::vector<std::uint8_t>& bytes) {
        p = std::filesystem::temp_directory_path() /
            ("whirl_iso_weights_" + std::to_string(reinterpret_cast<std::uintptr_t>(this)) + "_" + std::to_string(bytes.size()) + ".gguf");
        std::FILE* f = std::fopen(p.string().c_str(), "wb");
        if (!f) throw std::runtime_error("cannot write " + p.string());
        std::fwrite(bytes.data(), 1, bytes.size(), f);
        std::fclose(f);
    }
    ~TempFile() {
        std::error_code ec;
        std::filesystem::remove(p, ec);
    }
};

std::vector<std::uint8_t> f32s(std::initializer_list<float> v) {
    std::vector<std::uint8_t> out(v.size() * 4);
    std::memcpy(out.data(), v.begin(), out.size());
    return out;
}

template <class F>
std::string codeOf(F&& f) {
    try {
        f();
    } catch (const q::ModelError& e) {
        return e.code() + "|" + e.what();
    }
    return "";
}

// what the loader puts on the "device"
struct Dev {
    std::vector<std::uint64_t> allocs;
    std::vector<std::pair<q::DevPtr, std::vector<std::uint8_t>>> ups;
    q::DevPtr next = 0x100000;
    q::WeightLoader::Device dev() {
        q::WeightLoader::Device d;
        d.alloc = [this](std::uint64_t n) {
            allocs.push_back(n);
            const q::DevPtr p = next;
            next += (n + 255) / 256 * 256;
            return p;
        };
        d.upload = [this](q::DevPtr dst, const void* src, std::size_t n) {
            const auto* s = static_cast<const std::uint8_t*>(src);
            ups.emplace_back(dst, std::vector<std::uint8_t>(s, s + n));
        };
        return d;
    }
};

void testBlockNames() {
    ISO_CHECK(q::WeightLoader::bn(0, "attn_q.weight") == "blk.0.attn_q.weight");
    ISO_CHECK(q::WeightLoader::bn(63, "ffn_down.weight") == "blk.63.ffn_down.weight");
}

void testReaderReadsAtOffset() {
    TempFile tf({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    q::Reader r(tf.p.string());
    std::uint8_t b[3] = {};
    r.read(b, 3, 6);
    ISO_CHECK(b[0] == 6 && b[1] == 7 && b[2] == 8);
    ISO_CHECK(codeOf([&] { r.read(b, 3, 9); }).rfind("Truncated|", 0) == 0);  // short read
    ISO_CHECK(codeOf([&] { q::Reader x((tf.p.string() + ".missing")); }).rfind("OpenFailed|", 0) == 0);
}

void testVecUploadsThroughStaging() {
    // a 5-float f32 vector through an 8-byte staging buffer: one allocation of 20 bytes, uploads of 8 + 8 + 4 at
    // consecutive device offsets, the bytes of the file; the load statistics and per-type byte counts follow
    const auto v = f32s({1.0f, -2.0f, 3.5f, 0.25f, 1e6f});
    TempFile tf(ggufImage({{"output_norm.weight", {5}, 0, v}, {"pad", {8}, 0, f32s({0, 0, 0, 0, 0, 0, 0, 0})}}));
    const gg::File f = gg::File::open(tf.p.string());
    q::Config cfg;
    whirl::kernels::KernelTable k{};
    q::Weights w;
    q::LoadStats st;
    Dev d;
    std::uint8_t staging[8];
    q::WeightLoader L(f, cfg, k, w, st, d.dev(), staging, sizeof staging);
    const q::DevPtr p = L.vec("output_norm.weight");
    ISO_CHECK(d.allocs.size() == 1 && d.allocs[0] == 20 && p == 0x100000);
    ISO_CHECK(d.ups.size() == 3);
    std::vector<std::uint8_t> got;
    for (std::size_t i = 0; i < d.ups.size(); ++i) {
        ISO_CHECK(d.ups[i].first == p + got.size());
        got.insert(got.end(), d.ups[i].second.begin(), d.ups[i].second.end());
    }
    ISO_CHECK(d.ups[0].second.size() == 8 && d.ups[1].second.size() == 8 && d.ups[2].second.size() == 4);
    ISO_CHECK(got == v);
    ISO_CHECK(st.bytes == 20 && st.tensors == 1);
    ISO_CHECK(w.type_bytes[0] == 20);  // GgmlType::f32
}

void testVecRejects() {
    TempFile tf(ggufImage({{"x.weight", {4}, 1 /* f16 */, std::vector<std::uint8_t>(8, 0)}}));
    const gg::File f = gg::File::open(tf.p.string());
    q::Config cfg;
    whirl::kernels::KernelTable k{};
    q::Weights w;
    q::LoadStats st;
    Dev d;
    std::uint8_t staging[64];
    q::WeightLoader L(f, cfg, k, w, st, d.dev(), staging, sizeof staging);
    ISO_CHECK(codeOf([&] { L.vec("nope.weight"); }) == "MissingTensor|MissingTensor: nope.weight");
    ISO_CHECK(codeOf([&] { L.vec("x.weight"); }) == "UnsupportedTensorType|UnsupportedTensorType: x.weight: expected f32, got f16");
    ISO_CHECK(d.allocs.empty() && d.ups.empty() && st.tensors == 0);  // nothing reached the device
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("model.weights",
                    {{"block_names", testBlockNames},
                     {"reader_reads_at_offset", testReaderReadsAtOffset},
                     {"vec_uploads_through_staging", testVecUploadsThroughStaging},
                     {"vec_rejects", testVecRejects}},
                    argc, argv);
}
