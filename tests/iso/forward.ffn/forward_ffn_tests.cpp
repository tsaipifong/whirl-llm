// whirl-iso-forward-ffn-tests: the module forward.ffn (src/model/qwen35/forward/ffn.cpp: the dense SwiGLU FFN ffnBlock and the
// MoE FFN moeBlock) built over forward.dispatch (dispatch.cpp) and forward.state (context.cpp), WITHOUT any other forward source
// (ARCH-1 8.5 isolation, arch025-n8b3, PLAN N8 batch 3): a call from ffn.cpp into run / mtp_verify / prefill / attn / gdn /
// vision_rows, class Model, the engine or the server log is LNK2019 here. ffn.cpp only launches: the cases read what it launches
// through the launch recorder of tests/iso/fake_hip.cpp (kernel handle, grid, block; never a GPU result).
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "model/qwen35/context.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
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

// v0252-8sdet round 3 (the 8060S crash of WHIRL_MOEP_COMB=6, 2026-11-24 12:3x: 0xC0000005 inside hipModuleLaunchKernel at
// the first prefill): the MoE prefill of an Ornith-shaped forward (E 2048, 8 of 256 MXFP4 experts, F 512, 4096 rows, no
// shared expert) on the gfx1151 moep plan, for every MoE prefill combine (WHIRL_MOEP_COMB 0..6) x the fused-dequant twins
// (WHIRL_MOEP_FQ) x moeskip, in fast. The down projection's kernel must be launched EXACTLY ONCE per block (a second launch
// with another argument list = the crash: the moep step launched every non-fused, non-atomic fdn with the plain-down
// arguments (8), so COMB 6's 17-argument gemm_moepdi read past the argument array), never with a null handle, and, for
// COMB 6, after moe_fix_zero. With the fused-dequant twins on, COMB 6 takes the fq twin gemm_moepfqdi_<v> as COMB 5 takes
// gemm_moepfqda_<v>. RED on the tree of the first two rounds (COMB 6: the down handle launched twice).
struct MoeProbe : q::Context {
    using q::Context::moeBlock;
    whirl::numerics::Plan& plan() { return num_plan; }
    void rows() {
        small_max = 16;
        // scratch buffers the moep plan asks for (fake addresses: nothing is dereferenced)
        moe_sx = 0x20000;
        moe_tiles64 = 0x21000;
        moe_ntiles64 = 0x22000;
        moe_cnt = 0x23000;
        moe_yd = 0x24000;
        moe_yg = 0x25000;
        moe_x16 = 0x26000;
    }
};

void testMoepDownLaunchedOnce() {
    namespace kn = whirl::kernels;
    int fails = 0;
    for (const unsigned comb : {0u, 1u, 2u, 3u, 4u, 5u, 6u})
        for (const bool fq : {false, true})
            for (const float skip : {0.f, 1.f}) {
                auto c = std::make_unique<MoeProbe>();
                c->cfg.moe = true;
                c->cfg.n_embd = 2048;
                c->cfg.n_expert_used = 8;
                c->cfg.n_ff_exp = 512;
                c->cfg.n_expert = 256;
                c->cfg.n_ff = 512;
                c->rows();
                std::uintptr_t h = 0x1000;
                auto nf = [&]() { return fn(h += 0x10); };
                kn::KernelTable& k = c->k;
                k.rmsnorm = nf();
                k.moe_topk = nf();
                k.moe_topk_skip = nf();
                k.moe_route = nf();
                k.moe_route_skip = nf();
                k.moe_tiles = nf();
                k.moe_tiles_skip = nf();
                k.moe_gather_f16 = nf();
                k.moe_combine = nf();
                k.moe_combine_skip = nf();
                k.moe_combine4 = nf();
                k.moe_combine4_skip = nf();
                k.moe_combine4h = nf();
                k.moe_fix_zero = nf();
                const std::size_t mx = static_cast<std::size_t>(whirl::gguf::GgmlType::mxfp4);
                k.gemm_moegu[mx] = nf();
                k.gemm_moe32r[mx] = nf();
                for (int v = 0; v < kn::kMoepGuN; ++v) k.moep_gu[v] = nf(), k.moep_fq_gu[v] = nf();
                for (int v = 0; v < kn::kMoepDnN; ++v) {
                    k.moep_dn[v] = nf(), k.moep_dc[v] = nf(), k.moep_dh[v] = nf(), k.moep_dhc[v] = nf(), k.moep_da[v] = nf(), k.moep_di[v] = nf();
                    k.moep_fq_dn[v] = nf(), k.moep_fq_dh[v] = nf(), k.moep_fq_da[v] = nf(), k.moep_fq_di[v] = nf();
                }
                whirl::numerics::Flags& f = c->plan().flags;
                c->plan().mode = whirl::numerics::Mode::fast;
                f.prec_dec = false;
                f.moep_gu = 1;
                f.moep_dn = 3;  // m128w4n128k64
                f.moep_comb = comb;
                f.moep_fq = fq;
                f.moe_skip_thr = skip;
                f.moe_skip_min = 6;
                q::MoeW mo;
                mo.router.ty = whirl::gguf::GgmlType::f32;
                mo.router.ncols = 2048;
                mo.router.nrows = 256;
                for (q::Mat* m : {&mo.gate, &mo.up, &mo.down}) {
                    m->ty = whirl::gguf::GgmlType::mxfp4;
                    m->ptr = 0x5000;
                }
                mo.gate.ncols = mo.up.ncols = 2048, mo.gate.nrows = mo.up.nrows = 256 * 512;
                mo.down.ncols = 512, mo.down.nrows = 256 * 2048;
                mo.gate.row_bytes = mo.up.row_bytes = 2048 / 32 * 17;
                mo.down.row_bytes = 512 / 32 * 17;
                fake::recordLaunches(true);
                std::string why;
                try {
                    c->moeBlock(0x100, q::Mat{}, q::Mat{}, q::Mat{}, mo, 4096);
                } catch (const std::exception& e) {
                    why = e.what();
                }
                const auto l = fake::recordedLaunches();
                fake::recordLaunches(false);
                // the down-projection handles of the table and how often each was launched
                std::vector<whirl::hip::Function> downs;
                for (int v = 0; v < kn::kMoepDnN; ++v)
                    for (const auto* a : {&k.moep_dn, &k.moep_dc, &k.moep_dh, &k.moep_dhc, &k.moep_da, &k.moep_di, &k.moep_fq_dn, &k.moep_fq_dh,
                                          &k.moep_fq_da, &k.moep_fq_di})
                        downs.push_back((*a)[v]);
                int n_down = 0, n_null = 0, i_zero = -1, i_down = -1;
                whirl::hip::Function the_down = nullptr;
                // from the routing on (the router matmul before it has no kernel in this table: not what is checked here)
                std::size_t i0 = l.size();
                for (std::size_t i = 0; i < l.size() && i0 == l.size(); ++i)
                    if (l[i].f == k.moe_topk || l[i].f == k.moe_topk_skip) i0 = i;
                for (std::size_t i = i0; i < l.size(); ++i) {
                    if (l[i].f == nullptr) ++n_null;
                    if (l[i].f == k.moe_fix_zero && i_zero < 0) i_zero = static_cast<int>(i);
                    if (std::find(downs.begin(), downs.end(), l[i].f) != downs.end()) {
                        ++n_down;
                        the_down = l[i].f;
                        if (i_down < 0) i_down = static_cast<int>(i);
                    }
                }
                bool ok = why.empty() && i0 < l.size() && n_null == 0 && n_down == 1;
                if (comb == 6) ok = ok && i_zero >= 0 && i_zero < i_down && the_down == (fq ? k.moep_fq_di[2] : k.moep_di[2]);
                if (comb == 5) ok = ok && the_down == (fq ? k.moep_fq_da[2] : k.moep_da[2]);
                if (!ok) {
                    ++fails;
                    std::printf("  comb %u fq %d skip %d: %zu launches, down launches %d, null %d, fix_zero at %d, down at %d%s%s\n", comb, fq ? 1 : 0,
                                static_cast<int>(skip), l.size(), n_down, n_null, i_zero, i_down, why.empty() ? "" : ", threw: ", why.c_str());
                    std::printf("    launches:");
                    for (const auto& x : l) std::printf(" %llx", static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(x.f)));
                    std::printf("\n");
                }
            }
    ISO_CHECK(fails == 0);
}

// v0252-8sdet round 4 (the 8060S screen 20261124-1301: COMB 6 paid ~0.3-0.7 % for moe_fix_zero in every MoE layer, 120
// launches per 2048-token bench prefill): the fixed-point sums + counters in moe_yd are left zero by every gemm_moep[fq]di
// launch (last arriver: atomic swap / counter reset), so they are cleared only when they may be dirty: before the FIRST
// COMB 6 block after a MoE prefill block of another plan wrote moe_yd (or after allocation), and when a block needs more
// words than the last clear covered. Sequence of MoE blocks on ONE context (the forward's layers, then further forwards):
//   3 COMB 6 blocks n 4096           -> moe_fix_zero once (the first), the down once per block;
//   a block n 128 (below the moep row floor: the fused_gu plan writes moe_yd)  -> dirty;
//   2 COMB 6 blocks n 4096           -> moe_fix_zero once more;
//   1 COMB 6 block n 2048 (fewer words: inside the cleared prefix) -> none;
//   1 COMB 6 block n 8192 (more words) -> once more.
// RED on the round-3 tree (moe_fix_zero in every COMB 6 block: 7 instead of 3).
void testMoepFixZeroOncePerDirty() {
    namespace kn = whirl::kernels;
    auto c = std::make_unique<MoeProbe>();
    c->cfg.moe = true;
    c->cfg.n_embd = 2048;
    c->cfg.n_expert_used = 8;
    c->cfg.n_ff_exp = 512;
    c->cfg.n_expert = 256;
    c->cfg.n_ff = 512;
    c->rows();
    std::uintptr_t h = 0x1000;
    auto nf = [&]() { return fn(h += 0x10); };
    kn::KernelTable& k = c->k;
    k.rmsnorm = nf();
    k.moe_topk = nf(), k.moe_topk_skip = nf(), k.moe_route = nf(), k.moe_route_skip = nf(), k.moe_tiles = nf(), k.moe_tiles_skip = nf();
    k.moe_gather_f16 = nf(), k.moe_combine = nf(), k.moe_combine_skip = nf(), k.moe_combine4 = nf(), k.moe_combine4_skip = nf();
    k.moe_combine4h = nf(), k.moe_fix_zero = nf();
    const std::size_t mx = static_cast<std::size_t>(whirl::gguf::GgmlType::mxfp4);
    k.gemm_moegu[mx] = nf();
    k.gemm_moe32r[mx] = nf();
    for (int v = 0; v < kn::kMoepGuN; ++v) k.moep_gu[v] = nf(), k.moep_fq_gu[v] = nf();
    for (int v = 0; v < kn::kMoepDnN; ++v) k.moep_dn[v] = nf(), k.moep_di[v] = nf(), k.moep_fq_dn[v] = nf(), k.moep_fq_di[v] = nf();
    whirl::numerics::Flags& f = c->plan().flags;
    c->plan().mode = whirl::numerics::Mode::fast;
    f.prec_dec = false;
    f.moep_gu = 1;
    f.moep_dn = 3;
    f.moep_comb = 6;
    f.moep_fq = true;
    f.moe_skip_thr = 1.f;
    f.moe_skip_min = 6;
    q::MoeW mo;
    mo.router.ty = whirl::gguf::GgmlType::f32;
    mo.router.ncols = 2048;
    mo.router.nrows = 256;
    for (q::Mat* m : {&mo.gate, &mo.up, &mo.down}) {
        m->ty = whirl::gguf::GgmlType::mxfp4;
        m->ptr = 0x5000;
    }
    mo.gate.ncols = mo.up.ncols = 2048, mo.gate.nrows = mo.up.nrows = 256 * 512;
    mo.down.ncols = 512, mo.down.nrows = 256 * 2048;
    mo.gate.row_bytes = mo.up.row_bytes = 2048 / 32 * 17;
    mo.down.row_bytes = 512 / 32 * 17;
    struct Step {
        std::uint32_t n;
        int want_zero;  // moe_fix_zero launches in this block
        bool fix;       // a COMB 6 block (down = gemm_moepfqdi_m128w4n128k64 once)
    };
    const Step steps[] = {{4096, 1, true}, {4096, 0, true}, {4096, 0, true}, {128, 0, false}, {4096, 1, true},
                          {4096, 0, true}, {2048, 0, true}, {8192, 1, true}};
    int total_zero = 0, fails = 0;
    for (std::size_t s = 0; s < sizeof(steps) / sizeof(steps[0]); ++s) {
        fake::recordLaunches(true);
        std::string why;
        try {
            c->moeBlock(0x100, q::Mat{}, q::Mat{}, q::Mat{}, mo, steps[s].n);
        } catch (const std::exception& e) {
            why = e.what();
        }
        const auto l = fake::recordedLaunches();
        fake::recordLaunches(false);
        int nz = 0, nd = 0;
        for (const auto& x : l) {
            nz += x.f == k.moe_fix_zero;
            nd += x.f == k.moep_fq_di[2];
        }
        total_zero += nz;
        const bool ok = why.empty() && nz == steps[s].want_zero && nd == (steps[s].fix ? 1 : 0);
        if (!ok) {
            ++fails;
            std::printf("  step %zu n %u: moe_fix_zero %d (want %d), gemm_moepfqdi %d (want %d)%s%s\n", s, steps[s].n, nz, steps[s].want_zero, nd,
                        steps[s].fix ? 1 : 0, why.empty() ? "" : ", threw: ", why.c_str());
        }
    }
    ISO_CHECK(fails == 0);
    ISO_CHECK(total_zero == 3);
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("forward.ffn",
                    {{"fused_norm_first", testFusedNormFirst}, {"class_reset", testClassReset}, {"moep_down_launched_once", testMoepDownLaunchedOnce},
                     {"moep_fix_zero_once_per_dirty", testMoepFixZeroOncePerDirty}},
                    argc, argv);
}
