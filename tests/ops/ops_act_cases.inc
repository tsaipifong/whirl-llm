// arch025-n8b1: cases moved verbatim from tests/ops/ops_tests.cpp; included by whirl-ops-tests (tests/ops/ops_tests.cpp, same
// cases, same order) and by the module's isolation program tests/iso/ops.act/ (ARCH-1 8.5).
// SPDX-License-Identifier: Apache-2.0

#include "whirl/device_profile.h"
#include "whirl/ops/act.h"
#include "../test_check.h"
#include "ops_fakes.h"
#include "ops_mat.h"

#include <cstdint>
#include <cstdio>
#include <optional>
#include <span>
#include <string>

namespace {

namespace ops = whirl::ops;

// ---- arch025-F5: the activation-input / fused-path choice (src/ops/act.cpp). Selection-table snapshot: two device
// profiles x three modes on a Swift-shaped layer (MXFP4 weights with load-time row exponents, q8_0 [beta; alpha]).
// Each row is one string; any changed choice changes the string and fails here (the F5 mutants prove it).
// Derived from the S0b-1 golden launch tables (both cards, B = S0a exe):
//   gfx1201 balance / fast: prefill norms rmsnorm_x8t + rmsnorm_x8h16t, fp8 GEMMs with f16 out (gemm8th), silu_mul_x8ht
//   gfx1201 precise, gfx1151 every mode: rmsnorm_x16, f16 GEMM input, no h16 activation kernels
//   every mode: decode rows on the fused int8 path (rmsnorm_q8)
// The h16 / wide / [beta; alpha] cells with no launch-table witness follow the rules of act.h on these tables.
std::string actRow(const whirl::device::Profile& prof, const ActMode& md) {
    const bool r9 = prof.arch == whirl::hip::Arch::gfx1201;
    const kn::KernelTable k = actTable(r9);
    // the numerics plan: fp8 prefill only where the fp8 GEMMs exist (balance / fast); g8t when the tiled set is complete
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
    // tuned on the gemms slot 0 (f16-out twin present on both): the tuned choices alone would allow h16 on both
    // families; the 8060S profile row (h16_from_candidates) is what turns it off there
    const std::uint8_t tuned[3] = {static_cast<std::uint8_t>(2 * G), static_cast<std::uint8_t>(2 * G), static_cast<std::uint8_t>(2 * G)};
    const ops::MatShape q = mat(GgmlType::mxfp4, 6144, 5120, false, true), kk = mat(GgmlType::mxfp4, 1024, 5120, false, true);
    const ops::MatShape qkv = mat(GgmlType::mxfp4, 10240, 5120, false, true), gate = mat(GgmlType::mxfp4, 6144, 5120, false, true);
    ops::MatShape beta = mat(GgmlType::q8_0, 48, 5120), alpha = mat(GgmlType::q8_0, 48, 5120);
    beta.row_bytes = alpha.row_bytes = 5440;
    const ops::MatShape fg = mat(GgmlType::mxfp4, 17408, 5120, false, true), fd = mat(GgmlType::mxfp4, 5120, 17408, false, true);
    const ops::MatShape attn[3] = {q, kk, kk}, gu[2] = {fg, fg}, gdn4[4] = {qkv, gate, beta, alpha};
    auto ai = [](ops::ActIn a) { return std::string(ops::actInName(a)); };
    auto oc = [](std::optional<ops::ActIn> a) { return a ? std::string(ops::actInName(*a)) : std::string("-"); };
    auto rn = [&](std::span<const ops::MatShape> c, std::uint32_t n, std::uint8_t cls) {
        const ops::RmsnormInPlan p = ops::chooseRmsnormIn(5120, c, n, cls, true, g8t, f, k);
        return p.f == nullptr ? std::string("rmsnorm") : nameOf(p.f) + (p.fp8 ? "/fp8" : "/f16");
    };
    ops::GdnIn2Problem g2;
    g2.qkv = qkv;
    g2.gate = gate;
    g2.beta = beta;
    g2.alpha = alpha;
    g2.n = 512;
    g2.n_embd = 5120;
    g2.fused = ops::fusedRows(512, 16, false, false);
    g2.ba_buf = g2.ba_contiguous = true;
    ops::GdnIn2Flags gf;
    gf.g8t = g8t;
    std::string r = std::string(r9 ? "gfx1201 " : "gfx1151 ") + md.name + ":";
    // prefill (512 rows)
    r += " pre.q=" + ai(ops::chooseActIn(q, 512, 1, f, k));
    r += " pre.beta=" + ai(ops::chooseActIn(beta, 512, 1, f, k));
    r += " pre.down=" + ai(ops::chooseActIn(fd, 512, 4, f, k));
    r += " pre.attn_norm=" + rn(attn, 512, 1);
    r += " pre.ffn_norm=" + rn(gu, 512, 2);
    r += " pre.gdn4=" + oc(ops::chooseActCommon(gdn4, 512, 1, true, f, k));
    r += " pre.gdn_pc=" + oc(ops::chooseActCommon(std::span<const ops::MatShape>(gdn4, 2), 512, 1, true, f, k));
    r += " pre.h16_gate=" + std::to_string(ops::chooseH16Out(fg, tuned, 512, 2, f, h, k));
    r += " pre.h16_qkv=" + std::to_string(ops::chooseH16Out(qkv, tuned, 512, 1, f, h, k));
    r += " pre.gdn_in2=" + std::to_string(ops::chooseGdnIn2(g2, gf, f, k));
    // verify (4 rows) and decode (1 row)
    r += " ver.q=" + ai(ops::chooseActIn(q, 4, 1, f, k));
    r += " ver.common=" + oc(ops::chooseActCommon(attn, 4, 1, true, f, k));
    r += " ver.h16_gate=" + std::to_string(ops::chooseH16Out(fg, tuned, 4, 2, f, h, k));
    r += " dec.q=" + ai(ops::chooseActIn(q, 1, 1, f, k));
    r += " dec.fused=" + std::to_string(ops::fusedRows(1, 16, false, false));
    r += " ab_fusable=" + std::to_string(ops::gdnAbFusable(beta, alpha, k));
    r += " wide.dense=" + std::to_string(ops::wideAllowed(true, false, false, md.prec_dec) && ops::wideKernels(k));
    r += " wide.moe=" + std::to_string(ops::wideAllowed(true, true, true, md.prec_dec) && ops::wideKernels(k));
    return r;
}

void testActChoice() {
    const ActMode modes[3] = {{"precise", true, false, false}, {"balance", false, true, true}, {"fast", false, true, true}};
    const whirl::device::Profile* profs[2] = {&whirl::device::k_profile_gfx1201, &whirl::device::k_profile_gfx1151};
    // SNAPSHOT (arch025-F5): change only with a launch-table reason, in the same commit as the rule change
    const char* expect[6] = {
        "gfx1201 precise: pre.q=f16 pre.beta=f16 pre.down=f16 pre.attn_norm=rmsnorm_x16/f16 pre.ffn_norm=rmsnorm_x16/f16 pre.gdn4=f16 "
        "pre.gdn_pc=f16 pre.h16_gate=1 pre.h16_qkv=1 pre.gdn_in2=0 ver.q=f16 ver.common=f16 ver.h16_gate=1 dec.q=f16 dec.fused=1 "
        "ab_fusable=1 wide.dense=1 wide.moe=0",
        "gfx1201 balance: pre.q=fp8 pre.beta=f16 pre.down=fp8 pre.attn_norm=rmsnorm_x8t/fp8 pre.ffn_norm=rmsnorm_x8t/fp8 pre.gdn4=- "
        "pre.gdn_pc=fp8 pre.h16_gate=1 pre.h16_qkv=1 pre.gdn_in2=1 ver.q=gemv ver.common=- ver.h16_gate=0 dec.q=gemv dec.fused=1 "
        "ab_fusable=1 wide.dense=1 wide.moe=1",
        "gfx1201 fast: pre.q=fp8 pre.beta=f16 pre.down=fp8 pre.attn_norm=rmsnorm_x8t/fp8 pre.ffn_norm=rmsnorm_x8t/fp8 pre.gdn4=- "
        "pre.gdn_pc=fp8 pre.h16_gate=1 pre.h16_qkv=1 pre.gdn_in2=1 ver.q=gemv ver.common=- ver.h16_gate=0 dec.q=gemv dec.fused=1 "
        "ab_fusable=1 wide.dense=1 wide.moe=1",
        "gfx1151 precise: pre.q=f16 pre.beta=f16 pre.down=f16 pre.attn_norm=rmsnorm_x16/f16 pre.ffn_norm=rmsnorm_x16/f16 pre.gdn4=f16 "
        "pre.gdn_pc=f16 pre.h16_gate=0 pre.h16_qkv=0 pre.gdn_in2=0 ver.q=f16 ver.common=f16 ver.h16_gate=0 dec.q=f16 dec.fused=1 "
        "ab_fusable=1 wide.dense=1 wide.moe=0",
        "gfx1151 balance: pre.q=f16 pre.beta=f16 pre.down=f16 pre.attn_norm=rmsnorm_x16/f16 pre.ffn_norm=rmsnorm_x16/f16 pre.gdn4=f16 "
        "pre.gdn_pc=f16 pre.h16_gate=0 pre.h16_qkv=0 pre.gdn_in2=0 ver.q=gemv ver.common=- ver.h16_gate=0 dec.q=gemv dec.fused=1 "
        "ab_fusable=1 wide.dense=1 wide.moe=1",
        "gfx1151 fast: pre.q=f16 pre.beta=f16 pre.down=f16 pre.attn_norm=rmsnorm_x16/f16 pre.ffn_norm=rmsnorm_x16/f16 pre.gdn4=f16 "
        "pre.gdn_pc=f16 pre.h16_gate=0 pre.h16_qkv=0 pre.gdn_in2=0 ver.q=gemv ver.common=- ver.h16_gate=0 dec.q=gemv dec.fused=1 "
        "ab_fusable=1 wide.dense=1 wide.moe=1",
    };
    for (int pi = 0; pi < 2; ++pi)
        for (int mi = 0; mi < 3; ++mi) {
            const std::string got = actRow(*profs[pi], modes[mi]);
            const bool ok = got == expect[pi * 3 + mi];
            CHECK(ok);
            if (!ok) std::printf("  act row %d.%d\n    got:    %s\n    expect: %s\n", pi, mi, got.c_str(), expect[pi * 3 + mi]);
        }
    // the 8060S h16 rule reads the profile field (h16_from_candidates): on the gfx1151 kernel set the candidate walk
    // meets a fused quantized candidate with no gemm_ch twin -> false, whatever the tuned choice
    {
        const kn::KernelTable k = actTable(false);
        ops::MatmulFlags f;
        ops::H16Tune h;
        h.n_choices = 2 * static_cast<std::uint32_t>(kn::kGemmCfgs.size()) + static_cast<std::uint32_t>(kn::kGemmsCfgs.size());
        const std::uint8_t tuned_gs[1] = {static_cast<std::uint8_t>(2 * kn::kGemmCfgs.size())};  // a gemms slot
        const ops::MatShape w = mat(GgmlType::mxfp4, 256, 5120, false, true);
        h.from_candidates = false;
        const bool off = ops::chooseH16Out(w, tuned_gs, 512, 1, f, h, k);
        h.from_candidates = true;
        const bool on = ops::chooseH16Out(w, tuned_gs, 512, 1, f, h, k);
        CHECK(!on);
        CHECK(off);
        CHECK(whirl::device::k_profile_gfx1151.h16_from_candidates && !whirl::device::k_profile_gfx1201.h16_from_candidates);
    }
    // actCommon: act_fuse off / no consumers -> none; gemv / f32 inputs are never "common"
    {
        const kn::KernelTable k = actTable(true);
        ops::MatmulFlags f;
        const ops::MatShape w = mat(GgmlType::mxfp4, 256, 5120, false, true);
        CHECK(!ops::chooseActCommon(std::span<const ops::MatShape>(&w, 1), 512, 1, false, f, k));
        CHECK(!ops::chooseActCommon(std::span<const ops::MatShape>(), 512, 1, true, f, k));
        CHECK(!ops::chooseActCommon(std::span<const ops::MatShape>(&w, 1), 1, 1, true, f, k));
        CHECK(ops::chooseActIn(mat(GgmlType::f32, 64, 256), 1, 1, f, k) == ops::ActIn::f32in);
        // rmsnormIn: n_embd above 8192 never fuses
        CHECK(ops::chooseRmsnormIn(8193, std::span<const ops::MatShape>(&w, 1), 512, 1, true, false, f, k).f == nullptr);
        CHECK(ops::chooseRmsnormIn(8192, std::span<const ops::MatShape>(&w, 1), 512, 1, true, false, f, k).f != nullptr);
    }
    // fused rows / DeltaNet decode shape / wide gates
    CHECK(ops::fusedRows(16, 16, false, false) && !ops::fusedRows(17, 16, false, false) && !ops::fusedRows(1, 16, true, false) &&
          !ops::fusedRows(1, 16, false, true));
    CHECK(ops::gdnDecodeShape(4, 128, 128, 10240) && !ops::gdnDecodeShape(3, 128, 128, 10240) && !ops::gdnDecodeShape(4, 128, 128, 10200));
    CHECK(!ops::wideAllowed(false, false, false, false) && !ops::wideAllowed(true, true, false, false) &&
          !ops::wideAllowed(true, true, true, true) && ops::wideAllowed(true, false, false, true));
}

}  // namespace
