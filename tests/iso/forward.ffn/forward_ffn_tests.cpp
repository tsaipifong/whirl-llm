// whirl-iso-forward-ffn-tests: the module forward.ffn (src/model/qwen35/forward/ffn.cpp: the dense SwiGLU FFN ffnBlock and the
// MoE FFN moeBlock) built over forward.dispatch (dispatch.cpp) and forward.state (context.cpp), WITHOUT any other forward source
// (ARCH-1 8.5 isolation, arch025-n8b3, PLAN N8 batch 3): a call from ffn.cpp into run / mtp_verify / prefill / attn / gdn /
// vision_rows, class Model, the engine or the server log is LNK2019 here. ffn.cpp only launches: the cases read what it launches
// through the launch recorder of tests/iso/fake_hip.cpp (kernel handle, grid, block; never a GPU result).
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "model/qwen35/context.h"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

// Link doubles for this test program only (as tests/iso/forward.state): Context's TuneTable member defaults (model.core).
namespace whirl::qwen35 {
GemvR defaultGemvR() { return {}; }
GemvW defaultGemvW() { return {}; }
GemvR defaultGemvWHead() { return {}; }
}  // namespace whirl::qwen35

// The launch recorder of the HIP link double (tests/iso/fake_hip.cpp).
namespace whirl::hip::isofake {
struct Launch {
    Function f;
    Dim3 grid, block;
};
void recordLaunches(bool on);
std::vector<Launch> recordedLaunches();
}  // namespace whirl::hip::isofake

namespace {

namespace q = whirl::qwen35;
namespace fake = whirl::hip::isofake;

struct FfnProbe : q::Context {
    using q::Context::ffnBlock;
    using q::Context::mm_class;  // Scratch (a protected base of Context)
};

whirl::hip::Function fn(std::uintptr_t i) { return reinterpret_cast<whirl::hip::Function>(i); }

std::unique_ptr<FfnProbe> small() {
    auto c = std::make_unique<FfnProbe>();
    c->cfg.n_embd = 256;
    c->cfg.n_ff = 512;
    c->k.rmsnorm_q8 = fn(0x10);
    c->k.silu_mul_q8 = fn(0x20);
    return c;
}

std::vector<fake::Launch> runFfn(FfnProbe& c, std::uint32_t n, bool& threw) {
    q::Mat g, u, d;
    g.ncols = u.ncols = 256, g.nrows = u.nrows = 512, d.ncols = 512, d.nrows = 256;
    fake::recordLaunches(true);
    threw = false;
    try {
        c.ffnBlock(0x100, g, u, d, n);
    } catch (const std::exception&) {
        threw = true;  // a matmul past the norm may refuse (no kernels in this table); what ran is recorded
    }
    auto l = fake::recordedLaunches();
    fake::recordLaunches(false);
    return l;
}

void testFusedNormFirst() {
    // a decode-sized batch (fused rows): the post-attention norm writes the int8 activations (rmsnorm_q8) first
    auto c = small();
    bool threw = false;
    const auto l = runFfn(*c, 1, threw);
    ISO_CHECK(!l.empty() && l[0].f == fn(0x10) && l[0].grid.x == 1 && l[0].block.x == 1024);
}

void testClassReset() {
    // the GEMM class of the FFN (gate / up 2, down 4) is reset to the default 1 on every exit, also when a matmul throws
    for (const std::uint32_t n : {1u, 2u}) {
        auto c = small();
        c->mm_class = 1;
        bool threw = false;
        const auto l = runFfn(*c, n, threw);
        ISO_CHECK(!l.empty());  // got past the norm (the class scope is open)
        ISO_CHECK(c->mm_class == 1);
    }
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("forward.ffn", {{"fused_norm_first", testFusedNormFirst}, {"class_reset", testClassReset}}, argc, argv);
}
