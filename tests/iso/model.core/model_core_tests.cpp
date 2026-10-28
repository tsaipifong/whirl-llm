// whirl-iso-model-core-tests: model.core (class qwen35::Model and its non-forward members: src/model/*.cpp and
// src/model/qwen35/model_*.cpp, layer_subset.cpp) built alone (ARCH-1 8.5, arch025-lane2 leg 4) with its declared
// dependencies: every model module below it (forward.state + forward.*, model.weights / kv / scratch / tune, the
// architecture registry with the product table), env_snapshot, the low-level contracts, the op planners, the GGUF reader,
// lib whirl's util / VRAM limit / GPU choice text and the platform layer. The HIP layer is a link double
// (tests/iso/fake_hip.cpp, tests/iso/model.core/fake_gpu_restrict.cpp): nothing here reaches a GPU. No server, no engine,
// no server log, no KV host tier. The tests are the host-only parts: the GGUF config check of Model::load's first step,
// the embedded draft vocabulary decode, the tune cache text of a Model with no layers, a Model built and destroyed
// without a device.
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "model/arch_registry.h"
#include "whirl/gguf.h"
#include "whirl/model.h"

#include <cstdint>
#include <cstdio>
#include <memory>
#include <span>
#include <string>
#include <vector>

// the embedded draft vocabulary (whirl-bin2c output in the product build): three ids, little-endian u32
extern const unsigned char whirl_draft_vocab_64k[] = {0x01, 0x00, 0x00, 0x00, 0x10, 0x27, 0x00, 0x00, 0x78, 0x56, 0x34, 0x12};
extern const unsigned long long whirl_draft_vocab_64k_size = sizeof(whirl_draft_vocab_64k);

namespace {

namespace q = whirl::qwen35;
namespace gg = whirl::gguf;

template <class V>
void put(std::vector<std::uint8_t>& b, V v) {
    for (std::size_t i = 0; i < sizeof(V); ++i) b.push_back(static_cast<std::uint8_t>(static_cast<std::uint64_t>(v) >> (8 * i)));
}

void putStr(std::vector<std::uint8_t>& b, const std::string& s) {
    put<std::uint64_t>(b, s.size());
    b.insert(b.end(), s.begin(), s.end());
}

// a GGUF v3 image with no tensors: general.architecture = arch (type 8, string) and, when block_count > 0,
// <arch>.block_count (type 4, u32)
std::vector<std::uint8_t> ggufImage(const std::string& arch, std::uint32_t block_count) {
    std::vector<std::uint8_t> b;
    put<std::uint32_t>(b, 0x46554747u);
    put<std::uint32_t>(b, 3);
    put<std::uint64_t>(b, 0);                         // tensors
    put<std::uint64_t>(b, block_count > 0 ? 2 : 1);   // metadata
    putStr(b, "general.architecture");
    put<std::uint32_t>(b, 8);
    putStr(b, arch);
    if (block_count > 0) {
        putStr(b, arch + ".block_count");
        put<std::uint32_t>(b, 4);
        put<std::uint32_t>(b, block_count);
    }
    b.resize((b.size() + 31) / 32 * 32, 0);
    return b;
}

template <class F>
std::string codeOf(F&& f) {
    try {
        f();
    } catch (const q::ModelError& e) {
        return e.code() + "|" + e.what();  // what(): "<code>: <detail>"
    }
    return "";
}

// the two strings, both printed when they differ
bool same(const std::string& got, const std::string& want) {
    if (got != want) std::fprintf(stderr, "  got  '%s'\n  want '%s'\n", got.c_str(), want.c_str());
    return got == want;
}

void testConfigArchCheck() {
    // Model::load's first step: the architecture name against the product table (module.cpp's row through
    // arch_registry), the v0.2.4 message
    const auto llama = ggufImage("llama", 0);
    const gg::File fl = gg::File::parse(llama);
    ISO_CHECK(same(codeOf([&] { (void)q::Config::fromGguf(fl); }),
                   "UnsupportedArch|UnsupportedArch: general.architecture = 'llama' (WHIRL supports qwen35 and qwen35moe only)"));
    const auto none = ggufImage("", 0);
    const gg::File fn = gg::File::parse(none);
    ISO_CHECK(same(codeOf([&] { (void)q::Config::fromGguf(fn); }),
                   "UnsupportedArch|UnsupportedArch: general.architecture = '' (WHIRL supports qwen35 and qwen35moe only)"));
    // a supported name gets past the check: the next missing piece is the token embedding
    for (const char* a : {"qwen35", "qwen35moe"}) {
        const auto img = ggufImage(a, 4);
        const gg::File f = gg::File::parse(img);
        ISO_CHECK(same(codeOf([&] { (void)q::Config::fromGguf(f); }),
                   "MissingTensor|MissingTensor: token_embd.weight"));
    }
    ISO_CHECK(whirl::model::findArchName("qwen35moe") != nullptr && whirl::model::findArchName("qwen35moe")->moe);
    ISO_CHECK(whirl::model::findArchName("qwen3") == nullptr);
}

void testEmbeddedDraftVocab() {
    const std::vector<std::uint32_t> ids = q::embeddedDraftVocab64k();
    ISO_CHECK(ids.size() == 3 && ids[0] == 1 && ids[1] == 10000 && ids[2] == 0x12345678u);
}

void testEmptyModel() {
    // a Model with nothing loaded: built and destroyed without a device (~Model releases no allocation, stream, graph
    // or code object; the HIP double would throw / terminate on one), and its tune cache text is the empty table's
    auto m = std::make_unique<q::Model>();
    ISO_CHECK(!m->module.loaded() && m->layers.empty());
    ISO_CHECK(m->writeTune() == q::tuneText({}));
    const std::string t = m->writeTune();
    ISO_CHECK(m->readTune(t, 4) == q::applyTuneText(t, 4, {}));
    m.reset();
    ISO_CHECK(m == nullptr);
}

void testLoadRefusesBeforeTheDevice() {
    // Model::load checks the GGUF header before its first device call (UMA-1: the carve-out check, then allocations):
    // an unsupported or incomplete file is a ModelError here, never a call into the HIP layer (the double's
    // std::logic_error)
    q::LoadStats stats;
    const auto llama = ggufImage("llama", 0);
    const gg::File fl = gg::File::parse(llama);
    ISO_CHECK(same(codeOf([&] { (void)q::Model::load(fl, 0, stats); }),
                   "UnsupportedArch|UnsupportedArch: general.architecture = 'llama' (WHIRL supports qwen35 and qwen35moe only)"));
    const auto img = ggufImage("qwen35", 4);
    const gg::File fq = gg::File::parse(img);
    ISO_CHECK(same(codeOf([&] { (void)q::Model::load(fq, 4096, stats); }), "MissingTensor|MissingTensor: token_embd.weight"));
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("model.core",
                    {{"config_arch_check", testConfigArchCheck},
                     {"embedded_draft_vocab", testEmbeddedDraftVocab},
                     {"empty_model", testEmptyModel},
                     {"load_refuses_before_device", testLoadRefusesBeforeTheDevice}},
                    argc, argv);
}
