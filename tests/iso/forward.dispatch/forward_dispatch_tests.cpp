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
using whirl::gguf::GgmlType;

struct DispatchProbe : q::Context {
    using q::Context::elementwise;
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

}  // namespace

int main(int argc, char** argv) {
    return iso::run("forward.dispatch",
                    {{"fused_rows", testFusedRows},
                     {"wide_needs_replay", testWideNeedsReplay},
                     {"mat_shape", testMatShape},
                     {"matmul_flags", testMatmulFlags},
                     {"prec_missing", testPrecMissing},
                     {"has_tiled_fp8", testHasTiledFp8},
                     {"norm_and_elementwise_launches", testNormAndElementwiseLaunches}},
                    argc, argv);
}
