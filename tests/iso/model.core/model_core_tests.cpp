// whirl-iso-model-core-tests: model.core (class qwen35::Model and its non-forward members: src/model/*.cpp and
// src/model/qwen35/model_*.cpp, layer_subset.cpp) built alone (ARCH-1 8.5, arch025-lane2 leg 4) with its declared
// dependencies: every model module below it (forward.state + forward.*, model.weights / kv / scratch / tune, the
// architecture registry; the product table, module.cpp, is above it: arch025-lane2 leg 5), env_snapshot, the low-level contracts, the op planners, the GGUF reader,
// lib whirl's util / VRAM limit / GPU choice text and the platform layer. The HIP layer is a link double
// (tests/iso/fake_hip.cpp, tests/iso/model.core/fake_gpu_restrict.cpp): nothing here reaches a GPU. No server, no engine,
// no server log, no KV host tier. The tests are the host-only parts: the GGUF config check of Model::load's first step,
// the embedded draft vocabulary decode, the tune cache text of a Model with no layers, a Model built and destroyed
// without a device.
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "model/arch_registry.h"
#include "model/qwen35/arch_names.h"
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

// arch025-gmsurv: the opt-in allocation recorder of the HIP double (tests/iso/fake_hip.cpp)
namespace whirl::hip::isofake {
void recordAllocs(bool on);
std::vector<std::uint64_t> recordedSizes();
}  // namespace whirl::hip::isofake

// arch025-gmsurv: the test seam of class Model (friend struct ModelTestAccess in include/whirl/model.h; defined only here)
namespace whirl::qwen35 {
struct ModelTestAccess {
    static void allocScratch(Model& m, std::uint32_t fd_splits) { m.allocScratch(fd_splits); }
};
}  // namespace whirl::qwen35

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
    // Model::load's first step: the architecture name against the qwen35 family's names (arch_names.h, the array the
    // product table's row in module.cpp lists; arch025-lane2 leg 5), the v0.2.4 message
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
    const whirl::model::ArchTable names(q::arch_names_table);
    ISO_CHECK(whirl::model::findArchName("qwen35moe", names) != nullptr && whirl::model::findArchName("qwen35moe", names)->moe);
    ISO_CHECK(whirl::model::findArchName("qwen35", names) != nullptr && !whirl::model::findArchName("qwen35", names)->moe);
    ISO_CHECK(whirl::model::findArchName("qwen3", names) == nullptr);
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

// arch025-gmsurv (R9700 gpumut-A survivor m3-scratch-fd; Model::allocScratch was reachable only through Model::load on a
// GPU): allocScratch on a small dense config with no layers, through the HIP double's allocation recorder. Pinned:
// m.fd_splits is the argument, and the first 25 allocations (x .. out_tok, then part_ml / part_acc) have the sizes of
// their definition, part_ml / part_acc sized for that split count. Run for 64 (f16 / q8 default) and 256 (q4 KV).
void testAllocScratchFdSplits() {
    for (const std::uint32_t splits : {64u, 128u, 256u}) {
        std::vector<std::uint64_t> got;
        std::uint32_t fd = 0;
        {
            auto m = std::make_unique<q::Model>();
            q::Config& c = m->cfg;
            c.n_embd = 64, c.n_head = 4, c.n_head_kv = 2, c.head_dim = 16, c.n_ff = 128, c.n_vocab = 100;
            c.d_conv = 4, c.d_state = 8, c.n_k_heads = 2, c.n_v_heads = 2, c.d_inner = 32;
            m->max_batch = 32;
            whirl::hip::isofake::recordAllocs(true);
            q::ModelTestAccess::allocScratch(*m, splits);
            whirl::hip::isofake::recordAllocs(false);
            got = whirl::hip::isofake::recordedSizes();
            fd = m->fd_splits;
        }  // ~Model releases the recorded ledger through the double
        const std::uint64_t B = 32, f4 = 4, E = 64, H = 4, HKV = 2, HD = 16, FF = 128, V = 100, CC = 2 * 2 * 8 + 32, DI = 32, NV = 2;
        const std::uint64_t R = q::max_verify_rows, S = splits;
        const std::vector<std::uint64_t> want = {B * E * f4,      B * E * f4,  B * H * HD * 2 * f4, B * HKV * HD * f4, B * HKV * HD * f4,
                                                 B * H * HD * f4, B * CC * f4, B * CC * f4,         B * DI * f4,       B * NV * f4,
                                                 B * NV * f4,     B * 2 * NV * f4, B * DI * f4,     B * FF * f4,       B * FF * f4,
                                                 R * V * f4,      B * E * f4,  B * 2 * E * f4,      q::max_small_batch * E * f4,
                                                 B * 4,           B * 4,       B * 4,               q::ctl_words * 4,
                                                 R * S * H * 2 * f4, R * S * H * HD * f4};
        ISO_CHECK(fd == splits);
        ISO_CHECK(got.size() > want.size());
        bool same_prefix = got.size() > want.size();
        for (std::size_t i = 0; same_prefix && i < want.size(); ++i) same_prefix = got[i] == want[i];
        if (!same_prefix) {
            std::string s;
            for (std::size_t i = 0; i < got.size() && i < want.size(); ++i)
                s += std::to_string(i) + ":" + std::to_string(got[i]) + "/" + std::to_string(want[i]) + " ";
            std::printf("  alloc sizes (got/want), splits %u: %s\n", splits, s.c_str());
        }
        ISO_CHECK(same_prefix);
    }
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("model.core",
                    {{"config_arch_check", testConfigArchCheck},
                     {"embedded_draft_vocab", testEmbeddedDraftVocab},
                     {"empty_model", testEmptyModel},
                     {"load_refuses_before_device", testLoadRefusesBeforeTheDevice},
                     {"alloc_scratch_fd_splits", testAllocScratchFdSplits}},
                    argc, argv);
}
