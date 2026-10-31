// Unit tests of the kernel-choice module (src/ops, ARCH-1 K-1..), CPU only, no GPU: fake kernel tables (a non-null
// Function is a name tag), the choice functions, and the selection tables (problem x flags x kernel set -> kernel,
// grid). The real launches are pinned on the GPUs by the K-0 launch tables (tools/impact/launch_table.py); these
// tests pin the rules: precise picks only the precise bit class, the tuned tables only move within a class, the grids.
// SPDX-License-Identifier: Apache-2.0
//
// Written for WHIRL. Run: whirl-ops-tests.exe (exit 0 = all passed).

// arch025-n8b1: the per-module cases are in ops_<module>_cases.inc (also built alone by tests/iso/ops.<module>/); this file
// keeps the cross-module cases (testOpTables, the testGdnChoice selection-table snapshot) and the order of every case.
#include "whirl/device_profile.h"
#include "whirl/ops/act.h"
#include "whirl/ops/attention.h"
#include "whirl/ops/gdn.h"
#include "whirl/ops/matmul.h"
#include "whirl/ops/moe.h"
#include "whirl/ops/mtp.h"
#include "../test_check.h"

#include <cstdio>
#include <cstring>
#include <deque>
#include <string>
#include <type_traits>

#include "ops_fakes.h"
#include "ops_mat.h"
#include "ops_matmul_cases.inc"
#include "ops_attention_cases.inc"
#include "ops_moe_cases.inc"
#include "ops_act_cases.inc"
#include "ops_gdn_cases.inc"
#include "ops_mtp_cases.inc"

namespace {

using whirl::test::g_fail;
using whirl::test::g_pass;
// CHECK / CHECK_EQ / the counters: tests/test_check.h (arch025 N16)


namespace ops = whirl::ops;
// ---- per-op kernel tables (K-5): KernelTable is the union of the op tables; each choice reads only its own
void testOpTables() {
    static_assert(std::is_base_of_v<kn::MatmulKernels, kn::KernelTable> && std::is_base_of_v<kn::AttnKernels, kn::KernelTable> &&
                  std::is_base_of_v<kn::MoeKernels, kn::KernelTable> && std::is_base_of_v<kn::DeltaNetKernels, kn::KernelTable> &&
                  std::is_base_of_v<kn::MiscKernels, kn::KernelTable>);
    // an attention table alone is enough for the attention choice (no matmul / MoE entries needed)
    kn::AttnKernels a;
    a.attn_prefill_wmma = tag("attn_prefill_wmma");
    ops::AttnPrefillProblem p;
    p.n = 256;
    p.n_head = 24;
    p.n_head_kv = 4;
    p.scale = 0.1f;
    const auto pl = ops::chooseAttnPrefill(p, ops::AttnPrefillFlags{}, a);
    CHECK(pl.steps.size() == 1 && nameOf(pl.steps[0].f) == "attn_prefill_wmma");
    // a matmul table alone for the matmul path
    kn::MatmulKernels m;
    m.gemvq[ti(GgmlType::q4_k)] = tag("gemvq");
    CHECK(ops::chooseMatmulPath(mat(GgmlType::q4_k, 64, 256), 1, ops::MatmulFlags{}, m) == ops::MatmulPath::gemvq);
    // the MoE choice takes the fp8 activation kernel from the misc table explicitly
    kn::MoeKernels e;
    ops::MoeProblem q;
    q.n = 1;
    q.E = 256;
    q.K = 2;
    q.F = 64;
    q.R = 8;
    q.gate_ty = q.up_ty = q.down_ty = GgmlType::q4_k;
    e.moe_gu[ti(GgmlType::q4_k)] = tag("moe_gu");
    e.moe_down[ti(GgmlType::q4_k)] = tag("moe_down");
    CHECK(nameOf(ops::chooseMoe(q, ops::MoeFlags{}, e, nullptr).gu) == "moe_gu");
}

// ---- arch025-F6: the DeltaNet / attention-prep / MTP choice (src/ops/gdn.cpp, attention.cpp, mtp.cpp). Selection-table
// snapshot: two device profiles x three modes on a Swift-shaped DeltaNet layer (48 v heads, 16 k heads, d_state 128,
// conv 10240; MXFP4 qkv / gate / out with load-time row exponents, contiguous q8_0 [beta; alpha]). Each row is one
// string; any changed choice changes the string and fails here (the F6 mutants prove it). The mode flags follow
// tune.cpp: gdnwmma / h16 are balance items (ffn_h16, gdn_wmma), gdn_strix needs gdnwmma and Profile::gdn_strix.
// Witnessed by the S0b-1 B golden launch logs (both cards, Swift and Ornith; scratch\a25_f6\lt_gdn.py):
//   gfx1201 balance / fast: gdn_conv_l2n_h + gdn_conv_state_h (h16), gdn_wprep + gdn_wscan8, gated_norm_x8ht,
//     decode gdn_abconv_q8_0, gdn_gates_ba (in2)
//   gfx1201 / gfx1151 precise: gdn_conv_l2n (+ gdn_conv_par + l2norm per segment), gdn_chunk_prep / scan,
//     gated_norm_x16, decode gdn_gates_ba + gdn_conv_l2 (precise [beta; alpha] GEMM)
//   gfx1151 balance / fast: gdn_sx_prep / scan (>= 256 rows) and gdn_chunk_prep / scan (shorter runs), gdn_conv_l2n,
//     gated_norm_x16, decode gdn_abconv_q8_0
//   every card / mode: the fused attention prep (attn_prep / _q8h / _q4; never rope_neox / kv_store), the 2-bit draft
//     head (gemv_d2_nt*) with draft_pick / draft_pick_rows
std::string gdnRow(const whirl::device::Profile& prof, const ActMode& md) {
    const bool r9 = prof.arch == whirl::hip::Arch::gfx1201;
    const kn::KernelTable k = gdnTable(r9);
    const bool balance = !md.prec_dec;  // balance / fast: the gdnwmma and h16 items
    ops::MatmulFlags f;
    f.prec_dec = md.prec_dec;
    f.fp8_prefill = md.fp8_prefill && r9;
    f.fp8_mask = 7;
    f.gv_max = 16;
    const bool g8t = md.g8t && r9;
    const std::uint32_t G = static_cast<std::uint32_t>(kn::kGemmCfgs.size());
    ops::H16Tune h;
    h.from_candidates = prof.h16_from_candidates;
    h.n_choices = 2 * G + static_cast<std::uint32_t>(kn::kGemmsCfgs.size());
    const std::uint8_t tuned[3] = {static_cast<std::uint8_t>(G + 3), static_cast<std::uint8_t>(G + 1), static_cast<std::uint8_t>(G + 3)};
    const ops::MatShape qkv = mat(GgmlType::mxfp4, 10240, 5120, false, true), gate = mat(GgmlType::mxfp4, 6144, 5120, false, true);
    const ops::MatShape out = mat(GgmlType::mxfp4, 5120, 6144, false, true);
    ops::MatShape beta = mat(GgmlType::q8_0, 48, 5120), alpha = mat(GgmlType::q8_0, 48, 5120);
    beta.row_bytes = alpha.row_bytes = 5440;
    ops::GdnShape s;
    s.conv_ch = 10240;
    s.n_v_heads = 48;
    s.n_k_heads = 16;
    ops::GdnBaProblem ba;
    ba.ba_buf = ba.ba_contiguous = true;
    ops::GdnH16Flags hf;
    hf.ffn_h16 = balance;
    auto h16At = [&](std::uint32_t n) {
        ops::GdnH16Problem p;
        p.qkv = qkv;
        p.gate = gate;
        p.out = out;
        p.qkv_tune = tuned;
        p.gate_tune = tuned;
        p.n = n;
        p.s = s;
        p.fused = ops::fusedRows(n, 16, false, false);
        return ops::chooseGdnH16(p, hf, f, h, k);
    };
    ops::GdnIn2Problem g2;
    g2.qkv = qkv;
    g2.gate = gate;
    g2.beta = beta;
    g2.alpha = alpha;
    g2.n = 512;
    g2.n_embd = 5120;
    g2.fused = false;
    g2.ba_buf = g2.ba_contiguous = true;
    ops::GdnIn2Flags gf;
    gf.g8t = g8t;
    ops::GdnChunkFlags cf;
    cf.gdn_wmma = balance;
    cf.gdn_strix = balance && prof.gdn_strix;
    ops::GdnChunkProblem cp;
    cp.n_v_heads = 48;
    cp.n_k_heads = 16;
    cp.gc_m_bytes = cp.gc_w_bytes = cp.gc_u_bytes = 1ull << 30;
    auto chunk = [&](std::uint32_t n) {
        cp.n = n;
        return std::string(ops::gdnChunkName(ops::chooseGdnChunk(cp, cf, k).kind));
    };
    auto conv = [&](bool seg, bool h16, std::uint32_t n) {
        const ops::GdnConvPlan c = ops::chooseGdnConv(seg, h16, n, s, true, k);
        return std::string(ops::gdnConvName(c.conv)) + (c.l2norm ? "+l2norm" : "") + (c.state_h ? "/h" : "");
    };
    auto norm = [&](bool h16, std::uint32_t n) {
        const ops::GdnNormPlan p = ops::chooseGdnGatedNorm(out, n, s, h16, true, g8t, f, k);
        return nameOf(p.f) + (p.fp8 ? "/fp8" : "");
    };
    auto dec = [&](std::uint32_t n) {
        if (!ops::gdnFusable(ops::fusedRows(n, 16, false, false), beta, alpha, s, k)) return std::string("unfused");
        const ops::GdnDecodePlan p = ops::chooseGdnDecode(beta.ty, n, 16, md.prec_dec, 0, ba, k);
        return std::string(ops::gdnDecodeGatesName(p.gates)) + "/" + nameOf(p.f);
    };
    const bool h512 = h16At(512);
    const bool kv8 = !md.prec_dec;  // balance: q8 KV, fast: q4 KV (both "kv_q8" in the prep rule); precise: f16 KV
    ops::DraftHeadProblem dh;
    dh.head_rows = 248320;
    dh.draft_rows = 65536;
    dh.d2 = dh.map = true;
    const ops::DraftHeadPlan dp = ops::chooseDraftHead(dh);
    std::string r = std::string(r9 ? "gfx1201 " : "gfx1151 ") + md.name + ":";
    r += " pre.h16=" + std::to_string(h512);
    r += " pre.gates=" + std::string(ops::gdnGatesName(ops::chooseGdnGates(ops::chooseGdnIn2(g2, gf, f, k), ba, k)));
    r += " pre.conv=" + conv(false, h512, 512);
    r += " seg.conv=" + conv(true, h512, 512);
    r += " pre.scan=" + std::string(ops::chooseGdnScan(true, 512, s) == ops::GdnScan::chunked ? "chunked" : "seq");
    r += " pre.chunk=" + chunk(512) + " short.chunk=" + chunk(128);
    r += " pre.norm=" + norm(h512, 512);
    r += " dec=" + dec(1) + " ver=" + dec(4);
    r += " prep=" + std::string(ops::attnPrepName(ops::chooseAttnPrep(false, 256, kv8, false, k)));
    r += " core.pre=" + std::string(ops::attnCoreName(ops::chooseAttnCore(512, 16, 256, false)));
    r += " core.ver=" + std::string(ops::attnCoreName(ops::chooseAttnCore(4, 16, 256, false)));
    r += " head=" + std::to_string(dp.nrows) + (dp.map ? "/map" : "");
    return r;
}

void testGdnChoice() {
    const ActMode modes[3] = {{"precise", true, false, false}, {"balance", false, true, true}, {"fast", false, true, true}};
    const whirl::device::Profile* profs[2] = {&whirl::device::k_profile_gfx1201, &whirl::device::k_profile_gfx1151};
    // SNAPSHOT (arch025-F6): change only with a launch-table reason, in the same commit as the rule change
    const char* expect[6] = {
        "gfx1201 precise: pre.h16=0 pre.gates=ba pre.conv=l2n seg.conv=seg_par+l2norm pre.scan=chunked pre.chunk=f32 "
        "short.chunk=f32 pre.norm=gated_norm_x16 dec=precise_ba/null ver=precise_ba/null prep=fused core.pre=prefill "
        "core.ver=decode head=65536/map",
        "gfx1201 balance: pre.h16=1 pre.gates=in2 pre.conv=l2n_h/h seg.conv=seg_h16/h pre.scan=chunked pre.chunk=wmma "
        "short.chunk=wmma pre.norm=gated_norm_x8ht/fp8 dec=abconv/gdn_abconv_q8_0 ver=abconv/gdn_abconv_q8_0 prep=fused "
        "core.pre=prefill core.ver=decode head=65536/map",
        "gfx1201 fast: pre.h16=1 pre.gates=in2 pre.conv=l2n_h/h seg.conv=seg_h16/h pre.scan=chunked pre.chunk=wmma "
        "short.chunk=wmma pre.norm=gated_norm_x8ht/fp8 dec=abconv/gdn_abconv_q8_0 ver=abconv/gdn_abconv_q8_0 prep=fused "
        "core.pre=prefill core.ver=decode head=65536/map",
        "gfx1151 precise: pre.h16=0 pre.gates=ba pre.conv=l2n seg.conv=seg_par+l2norm pre.scan=chunked pre.chunk=f32 "
        "short.chunk=f32 pre.norm=gated_norm_x16 dec=precise_ba/null ver=precise_ba/null prep=fused core.pre=prefill "
        "core.ver=decode head=65536/map",
        "gfx1151 balance: pre.h16=0 pre.gates=ba pre.conv=l2n seg.conv=seg_par+l2norm pre.scan=chunked pre.chunk=strix "
        "short.chunk=f32 pre.norm=gated_norm_x16 dec=abconv/gdn_abconv_q8_0 ver=abconv/gdn_abconv_q8_0 prep=fused "
        "core.pre=prefill core.ver=decode head=65536/map",
        "gfx1151 fast: pre.h16=0 pre.gates=ba pre.conv=l2n seg.conv=seg_par+l2norm pre.scan=chunked pre.chunk=strix "
        "short.chunk=f32 pre.norm=gated_norm_x16 dec=abconv/gdn_abconv_q8_0 ver=abconv/gdn_abconv_q8_0 prep=fused "
        "core.pre=prefill core.ver=decode head=65536/map",
    };
    for (int pi = 0; pi < 2; ++pi)
        for (int mi = 0; mi < 3; ++mi) {
            const std::string got = gdnRow(*profs[pi], modes[mi]);
            const bool ok = got == expect[pi * 3 + mi];
            CHECK(ok);
            if (!ok) std::printf("  gdn row %d.%d\n    got:    %s\n    expect: %s\n", pi, mi, got.c_str(), expect[pi * 3 + mi]);
        }
    testGdnRules();
    testAttnPrepRules();
    testMtpRules();
}

}  // namespace

int main() {
    testBitClasses();
    testPreciseRule();
    testMatmulPath();
    testGemvTile();
    testGemvWide();
    testOpsGemm();
    testAttnDecode();
    testAttnPrefill();
    testOpsMoe();
    testOpTables();
    testActChoice();
    testGdnChoice();
    std::printf("ops tests: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
