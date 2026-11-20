// whirl-iso-forward-dispatch-tests: the module forward.dispatch (src/model/qwen35/forward/dispatch.cpp: profiler marks, GEMV /
// GEMM launches, the activation-input problems, the fused-path predicates, rmsnorm / elementwise helpers) built over forward.state
// (context.cpp) and its op / kernel planners, WITHOUT any other forward source (ARCH-1 8.5 isolation, arch025-n8b3, PLAN N8
// batch 3). dispatch.cpp became a leaf when Context::step (a one-line wrapper over run.cpp's forward) moved to run.cpp
// (progress/arch025-n8b3.PRIOR_ART.md design 1): a call from dispatch.cpp into run.cpp or any other forward file, class Model,
// the engine or the server log is LNK2019 here. Host-only cases on tests/iso/fake_hip.cpp (its launch recorder).
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "model/qwen35/context.h"
#include "whirl/gguf.h"
#include "whirl/ops/act.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
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
// v0252-f1arg3: the stream and the registered 8-byte argument word of each recorded launch
struct LaunchCtx {
    Stream s;
    std::uint64_t word;
};
void recordArgWord(Function f, int index);
std::vector<LaunchCtx> recordedLaunchCtx();
}  // namespace whirl::hip::isofake

namespace {

namespace q = whirl::qwen35;
namespace fake = whirl::hip::isofake;
using whirl::gguf::GgmlType;

struct DispatchProbe : q::Context {
    using q::Context::elementwise;
    using q::Context::gemmF16;
    using q::Context::w16;  // Scratch (a protected base of Context)
    using q::Context::fused;
    using q::Context::matShape;
    using q::Context::rmsnorm;
    using q::Context::rmsnormQ8;
    using q::Context::mm_class;  // Scratch (a protected base of Context)
    using q::Context::x16_src;  // Scratch (a protected base of Context)
    using q::Context::xq_n;
    using q::Context::xq_src;
};

whirl::hip::Function fn(std::uintptr_t i) { return reinterpret_cast<whirl::hip::Function>(i); }

void testFusedRows() {
    auto c = std::make_unique<DispatchProbe>();
    for (const std::uint32_t sm : {4u, 16u})
        for (const bool fg : {false, true})
            for (const bool nof : {false, true})
                for (const std::uint32_t n : {1u, 4u, 5u, 16u, 17u}) {
                    c->small_max = sm;
                    c->num_plan.flags.float_gemv = fg;
                    c->num_plan.flags.no_fuse = nof;
                    ISO_CHECK(c->fused(n) == whirl::ops::fusedRows(n, sm, fg, nof));
                }
}

void testWideNeedsReplay() {
    auto c = std::make_unique<DispatchProbe>();  // no kernels: not wide-capable
    ISO_CHECK(!c->wideCapable());
    for (const bool replay : {false, true}) {
        c->gdn_replay = replay;
        ISO_CHECK(!c->wideOk());
    }
}

void testMatShape() {
    auto c = std::make_unique<DispatchProbe>();
    c->output.ptr = 0x5000;
    q::Mat w;
    w.ty = GgmlType::q8_0;
    w.nrows = 7;
    w.ncols = 512;
    w.row_bytes = 544;
    w.ptr = 0x1000;
    auto s = c->matShape(w);
    ISO_CHECK(s.ty == GgmlType::q8_0 && s.nrows == 7 && s.ncols == 512 && s.row_bytes == 544);
    ISO_CHECK(!s.has_ref && !s.head);
    w.ref = 0x2000;
    w.ptr = 0x5000;
    s = c->matShape(w);
    ISO_CHECK(s.has_ref && s.head);
}

void testMatmulFlags() {
    auto c = std::make_unique<DispatchProbe>();
    c->num_plan.flags.prec_dec = true;
    c->num_plan.flags.fp8_prefill = true;
    c->mm_class = 4;
    c->small_max = 32;  // raised above max_small_batch: gvMax = small_max
    const auto f = c->matmulFlags();
    ISO_CHECK(f.prec_dec && !f.float_gemv && f.fp8_prefill && f.mm_class == 4 && f.gv_max == 32);
    c->out_h16 = true;
    const auto g = c->gemmFlags();
    ISO_CHECK(g.prec_dec && g.out_h16 && g.gv_max == 32);
}

void testPrecMissing() {
    auto c = std::make_unique<DispatchProbe>();
    ISO_CHECK(c->precMissing() == "f32_to_f16");
    c->k.f32_to_f16 = fn(0x10);
    ISO_CHECK(c->precMissing().empty());  // no layer, no head weights: complete
    c->output.ptr = 0x1000;
    c->output.ty = GgmlType::q8_0;
    c->output.ncols = 100;  // not a multiple of 256: the small-batch GEMMs do not apply
    ISO_CHECK(c->precMissing() == "the f16 GEMM for " + whirl::gguf::typeName(GgmlType::q8_0) + " weights (output head)");
}

void testHasTiledFp8() {
    auto c = std::make_unique<DispatchProbe>();
    ISO_CHECK(!c->hasTiledFp8());
    for (std::size_t i = 0; i < c->k.gemm8t.size(); ++i) c->k.gemm8t[i] = c->k.gemm8th[i] = fn(0x100 + i);
    c->k.qact_fp8t = c->k.rmsnorm_x8t = c->k.silu_mul_x8t = c->k.silu_mul_x8ht = c->k.gated_norm_x8t = fn(0x20);
    ISO_CHECK(!c->hasTiledFp8());  // gated_norm_x8ht still missing
    c->k.gated_norm_x8ht = fn(0x30);
    ISO_CHECK(c->hasTiledFp8());
}

void testNormAndElementwiseLaunches() {
    auto c = std::make_unique<DispatchProbe>();
    c->cfg.n_embd = 64;
    c->k.rmsnorm_q8 = fn(0x40);
    c->x16_src = 9;
    fake::recordLaunches(true);
    c->rmsnormQ8(0x100, 0x200, 0x300, 3);
    ISO_CHECK(c->xq_src == 0x300 && c->xq_n == 3 && c->x16_src == 0);
    c->xq_src = c->x16_src = 5;
    c->elementwise(fn(0x50), 0x400, 0x500, 600);
    ISO_CHECK(c->xq_src == 0 && c->x16_src == 0);
    const auto l = fake::recordedLaunches();
    fake::recordLaunches(false);
    ISO_CHECK(l.size() == 2);
    if (l.size() == 2) {
        ISO_CHECK(l[0].f == fn(0x40) && l[0].grid.x == 3 && l[0].block.x == 1024);
        ISO_CHECK(l[1].f == fn(0x50) && l[1].grid.x == 3 && l[1].block.x == 256);  // ceil(600 / 256)
    }
}

// v0252-pp2k2 (PLAN V1-139, F1 GEMM tail split): Context::gemmF16, the one funnel of the f16 GEMM, launches what
// ops::planGemmF16 returns. A 2058-row prefill GEMM of a Swift-shaped MXFP4 matrix (ffn_gate 17408 x 5120, tuned dequant
// choice) on gfx1201: dequant_f16, gemmh_f16 over 8 tiles (2048 rows) and ONE gemms launch (slot 4, 128 x 16) for the
// 10 tail rows at x16in + 2048 rows and y + 2048 rows; WHIRL_GEMM_TAIL=0 (Context::gemm_tail 0) keeps the shipped 9-tile
// launch, the gfx1151 profile row is off, gemm_tail 1 forces the split.
void testGemmTailLaunches() {
    namespace kn = whirl::kernels;
    auto c = std::make_unique<DispatchProbe>();
    const std::size_t mx = static_cast<std::size_t>(GgmlType::mxfp4);
    const std::uint32_t G = static_cast<std::uint32_t>(kn::kGemmCfgs.size());
    c->k.gemmh_f16 = fn(0x60);
    c->k.dequant_f16[mx] = fn(0x61);
    for (std::size_t s = 0; s < kn::kGemmsCfgs.size(); ++s) c->k.gemms[s][mx] = fn(0x70 + s);
    const whirl::hip::Function tail = fn(0x74);  // slot 4: the narrowest token tile (16)
    q::Mat w;
    w.ty = GgmlType::mxfp4;
    w.nrows = 17408;
    w.ncols = 5120;
    w.row_bytes = 5120 / 256 * 136;
    w.ptr = 0x10000000;
    w.tune.fill(static_cast<std::uint8_t>(G + 2));  // dequant + f16 GEMM (gemmh_f16 from 512 rows)
    c->w16 = 0x20000000;
    const whirl::hip::DevPtr x16in = 0x30000000, y = 0x40000000;
    auto run = [&](std::uint32_t n, int arg_index) {
        fake::recordLaunches(true);
        fake::recordArgWord(tail, arg_index);
        c->gemmF16(w, x16in, y, n, 0);
        auto l = fake::recordedLaunches();
        auto ctx = fake::recordedLaunchCtx();
        fake::recordLaunches(false);
        return std::make_pair(l, ctx);
    };
    ISO_CHECK(c->gemmFlags().gemm_tail);  // gfx1201 profile default
    {
        const auto [l, ctx] = run(2058, 2);
        ISO_CHECK(l.size() == 3);
        if (l.size() == 3) {
            ISO_CHECK(l[0].f == fn(0x61));
            ISO_CHECK(l[1].f == fn(0x60) && l[1].grid.x == 8 && l[1].grid.y == 136);
            ISO_CHECK(l[2].f == tail && l[2].grid.x == 1 && l[2].grid.y == 17408 / 128 && l[2].block.x == 256);
            ISO_CHECK(ctx.size() == 3 && ctx[2].word == x16in + 2048ull * 5120 * 2);  // X of row 2048
        }
    }
    {
        const auto [l, ctx] = run(2058, 3);
        ISO_CHECK(l.size() == 3 && ctx.size() == 3 && ctx[2].word == y + 2048ull * 17408 * 4);  // Y of row 2048 (f32 out)
    }
    {
        const auto [l, ctx] = run(1033, 2);  // the MTP batch after the capture row: 4 tiles + 9 rows
        ISO_CHECK(l.size() == 3 && l[1].grid.x == 4 && l[2].f == tail && ctx[2].word == x16in + 1024ull * 5120 * 2);
    }
    for (const std::uint32_t n : {2048u, 2041u}) {  // whole tiles / a 249-row last tile: the shipped single launch
        const auto [l, ctx] = run(n, 2);
        ISO_CHECK(l.size() == 2 && l[1].f == fn(0x60) && l[1].grid.x == 8);
    }
    c->gemm_tail = 0;  // WHIRL_GEMM_TAIL=0
    {
        const auto [l, ctx] = run(2058, 2);
        ISO_CHECK(l.size() == 2 && l[1].f == fn(0x60) && l[1].grid.x == 9);
    }
    // the 8060S row is off (its gemmFlags reads the device's gemvh cap, a device call this test program cannot make: the
    // profile value is what gemmFlags uses when gemm_tail is -1, as the gfx1201 case above shows)
    ISO_CHECK(!whirl::device::profile(whirl::hip::Arch::gfx1151).gemm_tail);
    c->gemm_tail = 1;  // WHIRL_GEMM_TAIL=1 (on for any family)
    {
        const auto [l, ctx] = run(2058, 2);
        ISO_CHECK(l.size() == 3 && l[1].grid.x == 8 && l[2].f == tail);
    }
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("forward.dispatch",
                    {{"fused_rows", testFusedRows},
                     {"wide_needs_replay", testWideNeedsReplay},
                     {"mat_shape", testMatShape},
                     {"matmul_flags", testMatmulFlags},
                     {"prec_missing", testPrecMissing},
                     {"has_tiled_fp8", testHasTiledFp8},
                     {"norm_and_elementwise_launches", testNormAndElementwiseLaunches},
                     {"gemm_tail_launches", testGemmTailLaunches}},  // v0252-pp2k2
                    argc, argv);
}
