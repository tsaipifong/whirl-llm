// arch025-n8b1: cases moved verbatim from tests/ops/ops_tests.cpp; included by whirl-ops-tests (tests/ops/ops_tests.cpp, same
// cases, same order) and by the module's isolation program tests/iso/ops.moe/ (ARCH-1 8.5).
// SPDX-License-Identifier: Apache-2.0

#include "whirl/ops/moe.h"
#include "../test_check.h"
#include "ops_fakes.h"

#include <cstdint>
#include <string>

namespace {

namespace ops = whirl::ops;

// ---- MoE (K-4)
kn::KernelTable moeTable(bool g1151) {
    kn::KernelTable k;
    const auto mx = ti(GgmlType::mxfp4), q8 = ti(GgmlType::q8_0);
    k.moe_gu[mx] = tag("moe_gu_mxfp4");
    k.moe_down[mx] = tag("moe_down_mxfp4");
    k.moe_gu[q8] = tag("moe_gu_q8_0");
    k.moe_down[q8] = tag("moe_down_q8_0");
    k.moe_gu_mxw = tag("moe_gu_mxw");
    k.moe_down_mxw = tag("moe_down_mxw");
    k.moe_gu_f[mx] = tag("moe_gu_f_mxfp4");
    k.moe_down_f[mx] = tag("moe_down_f_mxfp4");
    k.moe_gu_f[q8] = tag("moe_gu_f_q8_0");
    k.gemm_moe[mx] = tag("gemm_moe_mxfp4");
    k.gemm_moe32[mx] = tag("gemm_moe32_mxfp4");
    // v0252-f5 moeskip entry points: both code objects (routing, prefill grouping / combine, generic MXFP4 decode twins)
    k.moe_topk_skip = tag("moe_topk_skip");
    k.moe_skip_apply = tag("moe_skip_apply");
    k.moe_route_skip = tag("moe_route_skip");
    k.moe_combine_skip = tag("moe_combine_skip");
    k.moe_gu_skip_mx = tag("moe_gu_skip_mx");
    k.moe_down_skip_mx = tag("moe_down_skip_mx");
    if (g1151) {
        k.moe_tiles = tag("moe_tiles");
        k.moe_tiles_skip = tag("moe_tiles_skip");  // v0252-f5: gfx1151 only
        k.moe_combine4_skip = tag("moe_combine4_skip");
        k.gemm_moegu[mx] = tag("gemm_moegu_mxfp4");
        k.gemm_moe32r[mx] = tag("gemm_moe32r_mxfp4");
        k.moe_router_fused_i16 = tag("moe_router_fused_i16");  // v0251-a11port: gfx1151 code object only
        k.moe_router_fused_i24 = tag("moe_router_fused_i24");
    } else {
        k.gemmr_moegu = tag("gemmr_moegu");
        k.gemmr_moegu32 = tag("gemmr_moegu32");
        k.gemmr_moedn = tag("gemmr_moedn");
        k.gemmr_moedn32 = tag("gemmr_moedn32");
        k.gemm8_moe = tag("gemm8_moe");
        k.gemm8_moe32 = tag("gemm8_moe32");
        k.gemm8_moeh = tag("gemm8_moeh");
        k.gemm8_moe32h = tag("gemm8_moe32h");
        k.gemm8r_moe = tag("gemm8r_moe");
        k.gemm8r_moe32 = tag("gemm8r_moe32");
        k.gemm8r_moeh = tag("gemm8r_moeh");
        k.gemm8r_moe32h = tag("gemm8r_moe32h");
        k.moe_gather_fp8 = tag("moe_gather_fp8");
        k.silu_mul_x8h = tag("silu_mul_x8h");
        k.moe_gu_skip_mxw = tag("moe_gu_skip_mxw");  // v0252-f5: gfx1201 only (whole-block decode experts)
        k.moe_down_skip_mxw = tag("moe_down_skip_mxw");
    }
    return k;
}

// v0251-a11port (feat026-router): the fused router choice (ops::MoePlan::router), qwen35::Context::moeBlock's launch
// decision, as the screened host condition (feat026-router-p25 f06b245 forward/ffn.cpp): flag on, a small batch of one
// piece (n <= max_small_batch), R <= 256, K <= 32, E % 4 == 0, ceil(E / 128) = 16 (i16) or 24 (i24), the counter
// buffer, and the kernel in the code object (only the gfx1151 one has it). Everything else: null = moe_logits_f32 +
// moe_topk, the 0.2.5 path. Fails before the port (chooseMoe never sets router).
void testOpsMoeRouter(ops::MoeProblem p, const kn::KernelTable& k97, const kn::KernelTable& k8) {
    p.rcnt_buf = true;
    ops::MoeFlags bal, pre;
    bal.moe_fp8 = true;
    bal.moe_rfuse = pre.moe_rfuse = true;
    pre.prec_dec = true;
    CHECK(!ops::MoeFlags{}.moe_rfuse && !ops::MoeProblem{}.rcnt_buf);  // the defaults: off (callers set them)
    // 8060S, balance and precise: decode and verify pieces up to 16 rows take the i16 entry (Ornith E 2048, R 256, K 8)
    for (const std::uint32_t n : {1u, 2u, 4u, 5u, 16u}) {
        p.n = n;
        CHECK(nameOf(ops::chooseMoe(p, bal, k8, k8.silu_mul_x8h).router) == "moe_router_fused_i16");
        CHECK(nameOf(ops::chooseMoe(p, pre, k8, k8.silu_mul_x8h).router) == "moe_router_fused_i16");
    }
    // the plan's other fields do not move with the router choice
    {
        p.n = 2;
        ops::MoeFlags off = bal;
        off.moe_rfuse = false;
        const ops::MoePlan a = ops::chooseMoe(p, off, k8, k8.silu_mul_x8h), b = ops::chooseMoe(p, bal, k8, k8.silu_mul_x8h);
        CHECK(a.router == nullptr && nameOf(b.router) == "moe_router_fused_i16");  // WHIRL_MOE_RFUSE=0: the 0.2.5 path
        CHECK(a.small == b.small && a.small_q8 == b.small_q8 && a.gu == b.gu && a.down == b.down && a.logits_grid.x == b.logits_grid.x &&
              b.logits_grid.x == 32);
    }
    // more than one piece: the two-kernel path (balance wide verify 17..32 rows; precise verify up to max_verify_rows)
    {
        ops::MoeFlags wide = bal;
        wide.small_max = 32;
        p.n = 17;
        const ops::MoePlan w = ops::chooseMoe(p, wide, k8, k8.silu_mul_x8h);
        CHECK(w.small && w.router == nullptr);
        p.n = 24;
        const ops::MoePlan v = ops::chooseMoe(p, pre, k8, k8.silu_mul_x8h);
        CHECK(v.small && v.router == nullptr);
    }
    // prefill: never
    p.n = 4096;
    CHECK(ops::chooseMoe(p, bal, k8, k8.silu_mul_x8h).router == nullptr && ops::chooseMoe(p, pre, k8, k8.silu_mul_x8h).router == nullptr);
    // the R9700 table has no fused router: the two-kernel path even with the flag on
    p.n = 2;
    CHECK(ops::chooseMoe(p, bal, k97, k97.silu_mul_x8h).router == nullptr && ops::chooseMoe(p, pre, k97, k97.silu_mul_x8h).router == nullptr);
    // no counter buffer: the two-kernel path
    {
        ops::MoeProblem q = p;
        q.rcnt_buf = false;
        CHECK(ops::chooseMoe(q, bal, k8, k8.silu_mul_x8h).router == nullptr);
    }
    // shapes: E 3072 -> i24, E 2000 -> i16 (ceil 16), E 2050 (not % 4) / 4096 (32 iterations) / R 257 / K 33 -> null
    struct S {
        std::uint32_t E, R, K;
        const char* want;
    };
    for (const S s : {S{3072, 256, 8, "moe_router_fused_i24"}, S{2000, 256, 8, "moe_router_fused_i16"}, S{2048, 128, 8, "moe_router_fused_i16"},
                      S{2048, 256, 32, "moe_router_fused_i16"}, S{2050, 256, 8, "null"}, S{4096, 256, 8, "null"}, S{1024, 256, 8, "null"},
                      S{2048, 257, 8, "null"}, S{2048, 256, 33, "null"}}) {
        ops::MoeProblem q = p;
        q.E = s.E;
        q.R = s.R;
        q.K = s.K;
        q.n = 3;
        const std::string got = nameOf(ops::chooseMoe(q, bal, k8, k8.silu_mul_x8h).router);
        CHECK_DETAIL(got == s.want, "router shape E " + std::to_string(s.E) + " R " + std::to_string(s.R) + " K " + std::to_string(s.K) +
                                        ": got " + got + ", want " + s.want);
    }
}

// v0252-f5 (F5 moeskip, from feat027-moeskip-r1 4b6474b; DISCOVERY v4 rule 3.0-1: new entry points only): ops::chooseMoe's
// moeskip choice (MoePlan::skip). A block skips only when the flag is on (moe_skip_thr > 0, never with prec_dec) and EVERY
// kernel of its path has a skip entry point: the routing (moe_topk_skip, or moe_skip_apply after the fused router), the
// decode experts (the whole-block or the generic MXFP4 twins, swapped into gu / down with the same grids), the prefill
// grouping / combine (moe_route_skip, moe_tiles_skip on the 64-token list paths, moe_combine_skip / moe_combine4_skip; never
// the fused or f16 combines). Otherwise the block keeps the plan it had (moe_topk; no id is ever -1). Tables: the gfx1201
// object (whole-block twins) and the gfx1151 one (no whole-block kernels, generic twins, moe_tiles_skip, moe_combine4_skip,
// the moep default). Fails before the port (chooseMoe never sets skip).
void testOpsMoeSkip(ops::MoeProblem p) {
    p.rcnt_buf = true;
    p.cnt_buf = true;
    const kn::KernelTable k97 = moeTable(false);
    kn::KernelTable k8 = moeTable(true);
    k8.moe_gu_mxw = k8.moe_down_mxw = nullptr;  // the real gfx1151 object has no whole-block MXFP4 decode experts
    k8.moep_gu[kn::kMoepDefault.gu] = tag("gemm_moepgu_def");
    k8.moep_dn[kn::kMoepDefault.dn] = tag("gemm_moepdn_def");
    k8.moep_dc[kn::kMoepDefault.dn] = tag("gemm_moepdc_def");
    k8.moep_dh[kn::kMoepDefault.dn] = tag("gemm_moepdh_def");
    k8.moep_dhc[kn::kMoepDefault.dn] = tag("gemm_moepdhc_def");
    k8.moep_da[kn::kMoepDefault.dn] = tag("gemm_moepda_def");
    k8.moe_combine4 = tag("moe_combine4");
    k8.moe_combine4h = tag("moe_combine4h");
    CHECK(k97.moe_gu_skip_mxw != nullptr && k8.moe_gu_skip_mxw == nullptr && k8.moe_tiles_skip != nullptr && k97.moe_tiles_skip == nullptr);
    ops::MoeFlags fast;  // R9700 fast: int8 decode, fp8 prefill, whole-block decode experts
    fast.moe_fp8 = true;
    fast.moep_comb_ok = ops::moepCombFor(false, true);
    fast.moe_skip_thr = 0.05f;
    ops::MoeFlags noskip = fast;
    noskip.moe_skip_thr = 0.f;
    CHECK(ops::MoeFlags{}.moe_skip_thr == 0.f && ops::MoeFlags{}.moe_skip_min == 1u && !ops::MoePlan{}.skip);
    auto same = [](const ops::MoePlan& a, const ops::MoePlan& b) {
        return a.small == b.small && a.small_q8 == b.small_q8 && a.bits == b.bits && a.gu_split == b.gu_split && a.gu_up == b.gu_up &&
               a.gu_grid.x == b.gu_grid.x && a.gu_grid.y == b.gu_grid.y && a.down_grid.x == b.down_grid.x && a.down_grid.y == b.down_grid.y &&
               a.router == b.router && a.kind == b.kind && a.bn == b.bn && a.max_tiles == b.max_tiles && a.fgu == b.fgu && a.fdn == b.fdn &&
               a.g_gate == b.g_gate && a.g_up == b.g_up && a.g_down == b.g_down && a.gu_pgrid.x == b.gu_pgrid.x && a.dn_pgrid.x == b.dn_pgrid.x &&
               a.max_tiles64 == b.max_tiles64 && a.gu_bn == b.gu_bn && a.comb_fused == b.comb_fused && a.comb4 == b.comb4 && a.ydh == b.ydh &&
               a.comb_atomic == b.comb_atomic && a.comb_grid.x == b.comb_grid.x && a.comb_grid.y == b.comb_grid.y && a.rbf == b.rbf;
    };
    // ---- R9700: decode (n = 1), MTP verify rows (2..16) and the wide verify (17..32): the whole-block twins, same grids
    ops::MoeFlags wide = fast;
    wide.small_max = 32;
    for (std::uint32_t n = 1; n <= 32; ++n) {
        p.n = n;
        const ops::MoeFlags& f = n <= 16 ? fast : wide;
        ops::MoeFlags f0 = f;
        f0.moe_skip_thr = 0.f;
        const ops::MoePlan a = ops::chooseMoe(p, f, k97, k97.silu_mul_x8h), b = ops::chooseMoe(p, f0, k97, k97.silu_mul_x8h);
        CHECK_DETAIL(a.skip && a.small_q8 && nameOf(a.gu) == "moe_gu_skip_mxw" && nameOf(a.down) == "moe_down_skip_mxw" && a.router == nullptr &&
                         !b.skip && nameOf(b.gu) == "moe_gu_mxw" && same(a, b),
                     "R9700 decode n=" + std::to_string(n) + ": " + nameOf(a.gu) + " / " + nameOf(a.down));
    }
    // WHIRL_MOE_MXW=0: the generic MXFP4 twins
    p.n = 1;
    {
        ops::MoeFlags g = fast;
        g.moe_mxw = false;
        const ops::MoePlan a = ops::chooseMoe(p, g, k97, k97.silu_mul_x8h);
        CHECK(a.skip && nameOf(a.gu) == "moe_gu_skip_mx" && nameOf(a.down) == "moe_down_skip_mx");
    }
    // the flag off (balance / precise / fast without the item): exactly the old plan, decode and prefill, both tables
    for (const std::uint32_t n : {1u, 2u, 16u, 512u, 4096u})
        for (const kn::KernelTable* k : {&k97, static_cast<const kn::KernelTable*>(&k8)}) {
            p.n = n;
            ops::MoeFlags pre;
            pre.prec_dec = true;
            pre.moe_skip_thr = 0.05f;  // never with prec_dec (precise, f16 decode activations): the precise experts never see -1
            const ops::MoePlan a = ops::chooseMoe(p, noskip, *k, k->silu_mul_x8h), c = ops::chooseMoe(p, pre, *k, k->silu_mul_x8h);
            ops::MoeFlags pre0 = pre;
            pre0.moe_skip_thr = 0.f;
            CHECK_DETAIL(!a.skip && !c.skip && same(c, ops::chooseMoe(p, pre0, *k, k->silu_mul_x8h)), "flag off / prec_dec n=" + std::to_string(n));
        }
    // gate / up of different types (two launches): no twin for that path -> no skip, the old plan
    {
        ops::MoeProblem q = p;
        q.n = 2;
        q.up_ty = GgmlType::q8_0;
        const ops::MoePlan a = ops::chooseMoe(q, fast, k97, k97.silu_mul_x8h);
        CHECK(!a.skip && a.gu_split && nameOf(a.gu) == "moe_gu_mxw" && same(a, ops::chooseMoe(q, noskip, k97, k97.silu_mul_x8h)));
        q.n = 4096;
        CHECK(!ops::chooseMoe(q, fast, k97, k97.silu_mul_x8h).skip);  // not MXFP4 experts: never (the item needs MXFP4 experts)
    }
    // ---- R9700 prefill: fp8 (512 rows: 32-token tiles, 4096: 64), gemmr (WHIRL_MOE_FP8=0), generic (WHIRL_MOE_RBF=0)
    for (const std::uint32_t n : {512u, 4096u}) {
        p.n = n;
        const ops::MoePlan a = ops::chooseMoe(p, fast, k97, k97.silu_mul_x8h), b = ops::chooseMoe(p, noskip, k97, k97.silu_mul_x8h);
        CHECK_DETAIL(a.skip && a.kind == ops::MoePrefillKind::fp8 && !a.comb4 && same(a, b), "R9700 prefill fp8 n=" + std::to_string(n));
        ops::MoeFlags g = fast;
        g.moe_fp8 = false;
        CHECK_DETAIL(ops::chooseMoe(p, g, k97, k97.silu_mul_x8h).skip && ops::chooseMoe(p, g, k97, k97.silu_mul_x8h).kind == ops::MoePrefillKind::gemmr,
                     "R9700 prefill gemmr n=" + std::to_string(n));
        g.moe_rbf = false;
        CHECK_DETAIL(ops::chooseMoe(p, g, k97, k97.silu_mul_x8h).skip && ops::chooseMoe(p, g, k97, k97.silu_mul_x8h).kind == ops::MoePrefillKind::generic,
                     "R9700 prefill generic n=" + std::to_string(n));
    }
    // ---- 8060S: decode with the fused router (+ moe_skip_apply) and without it (moe_topk_skip); generic MXFP4 twins
    ops::MoeFlags f8 = fast;
    f8.moe_fp8 = false;
    f8.moe_rfuse = true;
    ops::setMoep(f8, static_cast<std::uint32_t>(kn::kMoepDefault.gu) + 1, static_cast<std::uint32_t>(kn::kMoepDefault.dn) + 1, kn::kMoepDefault.comb);
    ops::MoeFlags f80 = f8;
    f80.moe_skip_thr = 0.f;
    for (const std::uint32_t n : {1u, 2u, 16u}) {
        p.n = n;
        const ops::MoePlan a = ops::chooseMoe(p, f8, k8, k8.silu_mul_x8h), b = ops::chooseMoe(p, f80, k8, k8.silu_mul_x8h);
        CHECK_DETAIL(a.skip && nameOf(a.router) == "moe_router_fused_i16" && nameOf(a.gu) == "moe_gu_skip_mx" && nameOf(a.down) == "moe_down_skip_mx" &&
                         !b.skip && nameOf(b.gu) == "moe_gu_mxfp4" && same(a, b),
                     "8060S decode n=" + std::to_string(n) + ": " + nameOf(a.gu));
        ops::MoeFlags nr = f8;
        nr.moe_rfuse = false;
        const ops::MoePlan c = ops::chooseMoe(p, nr, k8, k8.silu_mul_x8h);
        CHECK_DETAIL(c.skip && c.router == nullptr && nameOf(c.gu) == "moe_gu_skip_mx", "8060S decode, two-kernel router n=" + std::to_string(n));
    }
    // ---- 8060S prefill: fused_gu (< 1024 rows), moep with the default moe_combine4, COMB 5 (atomic) yes, COMB 2 / 3 / 4 no
    p.n = 512;
    {
        const ops::MoePlan a = ops::chooseMoe(p, f8, k8, k8.silu_mul_x8h);
        CHECK(a.skip && a.kind == ops::MoePrefillKind::fused_gu && !a.comb4 && same(a, ops::chooseMoe(p, f80, k8, k8.silu_mul_x8h)));
    }
    p.n = 4096;
    {
        const ops::MoePlan a = ops::chooseMoe(p, f8, k8, k8.silu_mul_x8h);
        CHECK(a.skip && a.kind == ops::MoePrefillKind::moep && a.comb4 && !a.comb_fused && same(a, ops::chooseMoe(p, f80, k8, k8.silu_mul_x8h)));
        for (const std::uint32_t cm : {2u, 3u, 4u, 5u}) {
            ops::MoeFlags g = f8;
            g.moep_comb = cm;
            ops::MoeFlags g0 = g;
            g0.moe_skip_thr = 0.f;
            const ops::MoePlan c = ops::chooseMoe(p, g, k8, k8.silu_mul_x8h), c0 = ops::chooseMoe(p, g0, k8, k8.silu_mul_x8h);
            CHECK_DETAIL(c.skip == (cm == 5) && same(c, c0), "8060S moep COMB " + std::to_string(cm) + ": skip " + (c.skip ? "yes" : "no"));
        }
    }
    // ---- a missing entry point of the path: no skip, the old plan
    {
        p.n = 1;
        kn::KernelTable m = k97;
        m.moe_down_skip_mxw = nullptr;
        CHECK(!ops::chooseMoe(p, fast, m, m.silu_mul_x8h).skip && nameOf(ops::chooseMoe(p, fast, m, m.silu_mul_x8h).gu) == "moe_gu_mxw");
        m = k97;
        m.moe_topk_skip = nullptr;
        CHECK(!ops::chooseMoe(p, fast, m, m.silu_mul_x8h).skip);
        m = k8;
        m.moe_skip_apply = nullptr;  // the fused router without the apply pass
        CHECK(!ops::chooseMoe(p, f8, m, m.silu_mul_x8h).skip);
        p.n = 4096;
        m = k97;
        m.moe_route_skip = nullptr;
        CHECK(!ops::chooseMoe(p, fast, m, m.silu_mul_x8h).skip);
        m = k97;
        m.moe_combine_skip = nullptr;
        CHECK(!ops::chooseMoe(p, fast, m, m.silu_mul_x8h).skip);
        m = k8;
        m.moe_combine4_skip = nullptr;  // moep's moe_combine4
        CHECK(!ops::chooseMoe(p, f8, m, m.silu_mul_x8h).skip);
        m = k8;
        m.moe_tiles_skip = nullptr;
        CHECK(!ops::chooseMoe(p, f8, m, m.silu_mul_x8h).skip);
        p.n = 512;
        CHECK(!ops::chooseMoe(p, f8, m, m.silu_mul_x8h).skip);  // fused_gu's 64-token list
    }
}

void testOpsMoe() {
    ops::MoeProblem p;  // Ornith-like: E 2048, 8 of 256 experts, F 512, all MXFP4 with row exponents
    p.E = 2048;
    p.K = 8;
    p.F = 512;
    p.R = 256;
    p.gate_ty = p.up_ty = p.down_ty = GgmlType::mxfp4;
    p.gate_rb = p.up_rb = 2048 / 32 * 17;
    p.gate_ncols = 2048;
    p.down_ncols = 512;
    p.refs = true;
    p.sx_buf = true;
    p.tiles64_buf = true;
    ops::MoeFlags bal;
    bal.moe_fp8 = true;
    ops::MoeFlags pre;
    pre.prec_dec = true;
    const kn::KernelTable k97 = moeTable(false), k8 = moeTable(true);
    // decode, balance: int8 experts, the MXFP4 whole-block twins, gate + up in one launch
    p.n = 1;
    auto pl = ops::chooseMoe(p, bal, k97, k97.silu_mul_x8h);
    CHECK(pl.small && pl.small_q8 && pl.bits == ops::MoeBits::moe_q8_row && nameOf(pl.gu) == "moe_gu_mxw" && nameOf(pl.down) == "moe_down_mxw");
    CHECK(!pl.gu_split && pl.gu_grid.x == (2 * 512 + 7) / 8 && pl.gu_grid.y == 8 && pl.down_grid.x == 256 && pl.down_grid.y == 1 && pl.logits_grid.x == 32);
    ops::MoeFlags nomxw = bal;
    nomxw.moe_mxw = false;
    CHECK(nameOf(ops::chooseMoe(p, nomxw, k97, k97.silu_mul_x8h).gu) == "moe_gu_mxfp4");
    // decode, precise: the f32-activation experts, the precise class
    pl = ops::chooseMoe(p, pre, k97, k97.silu_mul_x8h);
    CHECK(pl.small && !pl.small_q8 && pl.bits == ops::MoeBits::moe_f32_row && nameOf(pl.gu) == "moe_gu_f_mxfp4" && nameOf(pl.down) == "moe_down_f_mxfp4");
    // precise: a 24-row verify (> 16) still takes the decode experts (MTP + n-gram == plain); balance does not
    p.n = 24;
    CHECK(ops::chooseMoe(p, pre, k97, k97.silu_mul_x8h).small && !ops::chooseMoe(p, bal, k97, k97.silu_mul_x8h).small);
    // gate and up of different types: two gate / up launches of F rows each
    ops::MoeProblem q = p;
    q.n = 2;
    q.up_ty = GgmlType::q8_0;
    pl = ops::chooseMoe(q, bal, k97, k97.silu_mul_x8h);
    CHECK(pl.gu_split && nameOf(pl.gu) == "moe_gu_mxw" && nameOf(pl.gu_up) == "moe_gu_q8_0" && pl.gu_grid.x == 64 && pl.gu_grid.y == 16);
    // prefill on gfx1201, balance: MXFP4 x fp8, row-block-fast 1-D grids; 64-token tiles for a long prompt
    p.n = 4096;
    pl = ops::chooseMoe(p, bal, k97, k97.silu_mul_x8h);
    CHECK(!pl.small && pl.kind == ops::MoePrefillKind::fp8 && pl.rbf && pl.bn == 64 && nameOf(pl.fgu) == "gemm8r_moeh" && nameOf(pl.fdn) == "gemm8r_moe");
    CHECK(pl.max_tiles == 4096 * 8 / 64 + 256 && pl.gu_pgrid.x == pl.max_tiles * 2 * 4 && pl.gu_pgrid.y == 1 && pl.dn_pgrid.x == pl.max_tiles * 16);
    // a short prompt: 32-token tiles
    p.n = 512;
    pl = ops::chooseMoe(p, bal, k97, k97.silu_mul_x8h);
    CHECK(pl.bn == 32 && nameOf(pl.fgu) == "gemm8r_moe32h");
    // precise prefill on gfx1201: no fp8, the f16 MXFP4 gemmr grouped GEMMs
    p.n = 4096;
    pl = ops::chooseMoe(p, pre, k97, k97.silu_mul_x8h);
    CHECK(pl.kind == ops::MoePrefillKind::gemmr && nameOf(pl.fgu) == "gemmr_moegu" && pl.bits == ops::MoeBits::moe_f16_gemm);
    // WHIRL_MOE_RBF=0: the generic grouped GEMM (gate, up, down separately; 2-D grids)
    ops::MoeFlags norbf = pre;
    norbf.moe_rbf = false;
    pl = ops::chooseMoe(p, norbf, k97, k97.silu_mul_x8h);
    CHECK(pl.kind == ops::MoePrefillKind::generic && nameOf(pl.g_gate) == "gemm_moe_mxfp4" && pl.gu_pgrid.y == 4 && pl.dn_pgrid.y == 16);
    // the 8060S: the fused gate + up + SwiGLU over 64-token tiles, down over 32-token tiles (bn 32)
    pl = ops::chooseMoe(p, bal, k8, k8.silu_mul_x8h);
    CHECK(pl.kind == ops::MoePrefillKind::fused_gu && pl.bn == 32 && pl.max_tiles64 == 4096 * 8 / 64 + 256 && nameOf(pl.fgu) == "gemm_moegu_mxfp4");
    CHECK(pl.gu_pgrid.x == pl.max_tiles64 * 8 && pl.dn_pgrid.x == pl.max_tiles * 16);
    // WHIRL_MOE_BN forces the tile only where the fused path does not set it
    ops::MoeFlags bn64 = norbf;
    bn64.moe_bn_force = 64;
    p.n = 512;
    CHECK(ops::chooseMoe(p, bn64, k97, k97.silu_mul_x8h).bn == 64);
    CHECK(ops::moeTile(512, 8, 256) == 32 && ops::moeTile(4096, 8, 256) == 64);
    // feat026-moep (source wt_026_moep_p25 8c57223, verbatim): the multi-fragment expert GEMMs on the 8060S only when
    // both variants are named and in the code object; the combine choice (moe_combine / moe_combine4 / fused into the
    // down epilogue)
    {
        kn::KernelTable km = k8;
        km.moep_gu[1] = tag("gemm_moepgu_g64n128k64_mxfp4");
        km.moep_dn[2] = tag("gemm_moepdn_m128w4n128k64_mxfp4");
        km.moep_dc[2] = tag("gemm_moepdc_m128w4n128k64_mxfp4");
        km.moe_combine4 = tag("moe_combine4");
        CHECK(std::string(kn::kMoepGu[1].name) == "g64n128k64" && std::string(kn::kMoepDn[2].name) == "m128w4n128k64");
        ops::MoeProblem pm = p;
        pm.n = 4096;
        pm.cnt_buf = true;
        ops::MoeFlags fm = bal;
        // off (MoeFlags default): the fused_gu plan, moe_combine on (E / 256, n)
        pl = ops::chooseMoe(pm, fm, km, km.silu_mul_x8h);
        CHECK(pl.kind == ops::MoePrefillKind::fused_gu && !pl.comb4 && !pl.comb_fused && pl.comb_grid.x == 8 && pl.comb_grid.y == 4096);
        fm.moep_gu = 1;
        fm.moep_dn = 2;
        pl = ops::chooseMoe(pm, fm, km, km.silu_mul_x8h);
        CHECK(pl.kind == ops::MoePrefillKind::moep && pl.bits == ops::MoeBits::moe_f16_gemm && pl.bn == 128 && pl.gu_bn == 128);
        CHECK(pl.max_tiles == 4096 * 8 / 128 + 256 && pl.max_tiles64 == 4096 * 8 / 128 + 256 && pl.gu_pgrid.x == pl.max_tiles64 * 8 &&
              pl.dn_pgrid.x == pl.max_tiles * 16);
        CHECK(nameOf(pl.fgu) == "gemm_moepgu_g64n128k64_mxfp4" && nameOf(pl.fdn) == "gemm_moepdn_m128w4n128k64_mxfp4" && !pl.comb_fused);
        fm.moep_comb = 2;
        pl = ops::chooseMoe(pm, fm, km, km.silu_mul_x8h);
        CHECK(pl.comb_fused && !pl.comb4 && nameOf(pl.fdn) == "gemm_moepdc_m128w4n128k64_mxfp4");
        fm.moep_comb = 1;
        pl = ops::chooseMoe(pm, fm, km, km.silu_mul_x8h);
        CHECK(!pl.comb_fused && pl.comb4 && pl.comb_grid.x == 2 && pl.comb_grid.y == 4096);
        // not bit-identical: f16 expert outputs (balance item), atomic scatter (fast candidate); never in precise
        km.moep_dh[2] = tag("gemm_moepdh_m128w4n128k64_mxfp4");
        km.moep_dhc[2] = tag("gemm_moepdhc_m128w4n128k64_mxfp4");
        km.moep_da[2] = tag("gemm_moepda_m128w4n128k64_mxfp4");
        km.moe_combine4h = tag("moe_combine4h");
        fm.moep_comb_ok = ops::MoeFlags::MoepComb::atomic;  // v0251-a7pol: fast's class (the combines below need it)
        fm.moep_comb = 3;
        pl = ops::chooseMoe(pm, fm, km, km.silu_mul_x8h);
        CHECK(pl.ydh && pl.comb4 && !pl.comb_fused && pl.bits == ops::MoeBits::moe_lossy_gemm && nameOf(pl.fdn) == "gemm_moepdh_m128w4n128k64_mxfp4");
        fm.moep_comb = 4;
        pl = ops::chooseMoe(pm, fm, km, km.silu_mul_x8h);
        CHECK(pl.ydh && pl.comb_fused && pl.bits == ops::MoeBits::moe_lossy_gemm && nameOf(pl.fdn) == "gemm_moepdhc_m128w4n128k64_mxfp4");
        fm.moep_comb = 5;
        pl = ops::chooseMoe(pm, fm, km, km.silu_mul_x8h);
        CHECK(pl.comb_atomic && !pl.comb_fused && pl.comb_grid.x == 2 && nameOf(pl.fdn) == "gemm_moepda_m128w4n128k64_mxfp4");
        ops::MoeFlags fp = pre;  // precise: the lossy combines are ignored -> bit-identical plain down + moe_combine
        fp.moep_gu = 1;
        fp.moep_dn = 2;
        for (const unsigned cm : {3u, 4u, 5u}) {
            fp.moep_comb = cm;
            pl = ops::chooseMoe(pm, fp, km, km.silu_mul_x8h);
            CHECK(pl.kind == ops::MoePrefillKind::moep && pl.bits == ops::MoeBits::moe_f16_gemm && !pl.ydh && !pl.comb_atomic && !pl.comb_fused &&
                  nameOf(pl.fdn) == "gemm_moepdn_m128w4n128k64_mxfp4");
        }
        fp.moep_comb = 2;  // the bit-identical fused combine is allowed in precise
        CHECK(ops::chooseMoe(pm, fp, km, km.silu_mul_x8h).comb_fused);
        // v0251-a7pol (B26-41): the combine class by the MODE (MoeFlags::moep_comb_ok), not prec_dec
        {
            CHECK(ops::moepCombFor(true, false) == ops::MoeFlags::MoepComb::bitwise && ops::moepCombFor(false, false) == ops::MoeFlags::MoepComb::f16 &&
                  ops::moepCombFor(false, true) == ops::MoeFlags::MoepComb::atomic);
            CHECK(ops::MoeFlags{}.moep_comb_ok == ops::MoeFlags::MoepComb::bitwise);
            ops::MoeFlags fb = bal;  // balance: COMB 5 (atomic scatter) refused -> the plain moep down + moe_combine (bitwise)
            fb.moep_gu = 1;
            fb.moep_dn = 2;
            fb.moep_comb_ok = ops::moepCombFor(false, false);
            fb.moep_comb = 5;
            pl = ops::chooseMoe(pm, fb, km, km.silu_mul_x8h);
            CHECK(pl.kind == ops::MoePrefillKind::moep && !pl.comb_atomic && !pl.ydh && !pl.comb_fused && !pl.comb4 &&
                  pl.bits == ops::MoeBits::moe_f16_gemm && nameOf(pl.fdn) == "gemm_moepdn_m128w4n128k64_mxfp4");
            for (const unsigned cm : {3u, 4u}) {  // balance keeps COMB 3 / 4 (f16 intermediates)
                fb.moep_comb = cm;
                pl = ops::chooseMoe(pm, fb, km, km.silu_mul_x8h);
                CHECK(pl.ydh && !pl.comb_atomic && pl.bits == ops::MoeBits::moe_lossy_gemm);
            }
            // precise with prec_dec cleared (WHIRL_Q8DEC=1): still bitwise only
            ops::MoeFlags fq = bal;
            fq.moep_gu = 1;
            fq.moep_dn = 2;
            fq.moep_comb_ok = ops::moepCombFor(true, false);
            for (const unsigned cm : {3u, 4u, 5u}) {
                fq.moep_comb = cm;
                pl = ops::chooseMoe(pm, fq, km, km.silu_mul_x8h);
                CHECK(pl.kind == ops::MoePrefillKind::moep && pl.bits == ops::MoeBits::moe_f16_gemm && !pl.ydh && !pl.comb_atomic &&
                      nameOf(pl.fdn) == "gemm_moepdn_m128w4n128k64_mxfp4");
            }
            // fast: COMB 5 applies
            ops::MoeFlags ff = fb;
            ff.moep_comb_ok = ops::moepCombFor(false, true);
            ff.moep_comb = 5;
            pl = ops::chooseMoe(pm, ff, km, km.silu_mul_x8h);
            CHECK(pl.comb_atomic && pl.bits == ops::MoeBits::moe_lossy_gemm && nameOf(pl.fdn) == "gemm_moepda_m128w4n128k64_mxfp4");
            // fast below the token floor (B26-43): no moep, so no atomic combine either
            ops::MoeProblem ps = pm;
            ps.n = 128;
            pl = ops::chooseMoe(ps, ff, km, km.silu_mul_x8h);
            CHECK(pl.kind == ops::MoePrefillKind::fused_gu && !pl.comb_atomic && !pl.ydh && !pl.comb4);
        }
        // no counter buffer: the fused combine falls back to the plain down twin + moe_combine
        fm.moep_comb = 2;
        pm.cnt_buf = false;
        pl = ops::chooseMoe(pm, fm, km, km.silu_mul_x8h);
        CHECK(pl.kind == ops::MoePrefillKind::moep && !pl.comb_fused && nameOf(pl.fdn) == "gemm_moepdn_m128w4n128k64_mxfp4");
        // a variant not in the code object: off
        fm.moep_gu = 0;
        CHECK(ops::chooseMoe(pm, fm, km, km.silu_mul_x8h).kind == ops::MoePrefillKind::fused_gu);
        // gfx1201 table (no moep kernels): unchanged fp8 plan
        fm.moep_gu = 1;
        CHECK(ops::chooseMoe(pm, fm, k97, k97.silu_mul_x8h).kind == ops::MoePrefillKind::fp8);
        // non-MXFP4 down experts: off
        pm.down_ty = GgmlType::q8_0;
        CHECK(ops::chooseMoe(pm, fm, km, km.silu_mul_x8h).kind != ops::MoePrefillKind::moep);
    }
    // v0251-a7port: the gfx1151 default (kernels::kMoepDefault: g64n64k64 + m128w4n128k64 + moe_combine4), assembled from
    // the plan's flags (variant index + 1) by ops::setMoep as qwen35::Context::moeBlock does, on a gfx1151-like table that
    // carries the whole variant set the gfx1151 object exports; and the plan's 0 / 0 / 0 (R9700, dense, WHIRL_MOEP_*=0)
    {
        CHECK(std::string(kn::kMoepGu[kn::kMoepDefault.gu].name) == "g64n64k64" && std::string(kn::kMoepDn[kn::kMoepDefault.dn].name) == "m128w4n128k64" &&
              kn::kMoepDefault.comb == 1u);
        kn::KernelTable km = k8;
        for (int v = 0; v < kn::kMoepGuN; ++v) km.moep_gu[v] = tag(std::string("gemm_moepgu_") + kn::kMoepGu[v].name + "_mxfp4");
        for (int v = 0; v < kn::kMoepDnN; ++v) {
            km.moep_dn[v] = tag(std::string("gemm_moepdn_") + kn::kMoepDn[v].name + "_mxfp4");
            km.moep_dc[v] = tag(std::string("gemm_moepdc_") + kn::kMoepDn[v].name + "_mxfp4");
        }
        km.moe_combine4 = tag("moe_combine4");
        ops::MoeProblem pm = p;
        pm.down_ty = GgmlType::mxfp4;
        ops::MoeFlags fa;
        ops::setMoep(fa, 0, 0, 0);
        CHECK(fa.moep_gu == -1 && fa.moep_dn == -1 && fa.moep_comb == 0);
        ops::setMoep(fa, static_cast<unsigned>(kn::kMoepDefault.gu) + 1, static_cast<unsigned>(kn::kMoepDefault.dn) + 1, kn::kMoepDefault.comb);
        CHECK(fa.moep_gu == 0 && fa.moep_dn == 2 && fa.moep_comb == 1);
        for (const bool precise : {false, true}) {
            ops::MoeFlags f = precise ? pre : bal;
            ops::setMoep(f, 1, 3, 1);
            // v0251-a7pol (B26-43): from 32 tokens per expert (Ornith 8 of 256: n >= 1024) exactly the cfce7ea default plan
            for (const unsigned n : {1024u, 1280u, 2048u, 4096u, 8192u, 32768u}) {
                pm.n = n;
                pl = ops::chooseMoe(pm, f, km, km.silu_mul_x8h);
                CHECK(pl.kind == ops::MoePrefillKind::moep && pl.bits == ops::MoeBits::moe_f16_gemm && pl.bn == 128 && pl.gu_bn == 64);
                CHECK(nameOf(pl.fgu) == "gemm_moepgu_g64n64k64_mxfp4" && nameOf(pl.fdn) == "gemm_moepdn_m128w4n128k64_mxfp4");
                CHECK(pl.comb4 && !pl.comb_fused && !pl.ydh && !pl.comb_atomic && pl.comb_grid.x == 2 && pl.comb_grid.y == n);
                CHECK(pl.max_tiles == (n * 8 + 127) / 128 + 256 && pl.max_tiles64 == (n * 8 + 63) / 64 + 256 && pl.gu_pgrid.x == pl.max_tiles64 * 8 &&
                      pl.dn_pgrid.x == pl.max_tiles * 16);
            }
            // v0251-a7pol (B26-43): below the floor the shipped 0.2.5 plan, field for field as the plan's 0 / 0 / 0 at that n
            CHECK(ops::kMoepMinTokensPerExpert == 32u && !ops::moepWide(1023, 8, 256) && ops::moepWide(1024, 8, 256) &&
                  !ops::moepWide(511, 8, 128) && ops::moepWide(512, 8, 128) && ops::moepWide(0xffffffffu, 8, 256));
            for (const unsigned n : {33u, 128u, 512u, 1000u, 1023u}) {  // > max_verify_rows: prefill in precise too
                pm.n = n;
                ops::MoeFlags off = precise ? pre : bal;
                ops::setMoep(off, 0, 0, 0);
                const ops::MoePlan a = ops::chooseMoe(pm, f, km, km.silu_mul_x8h), b = ops::chooseMoe(pm, off, km, km.silu_mul_x8h);
                CHECK(!a.small && a.kind == ops::MoePrefillKind::fused_gu && !a.comb4 && !a.comb_fused && !a.ydh && !a.comb_atomic);
                CHECK(a.kind == b.kind && a.bits == b.bits && a.bn == b.bn && a.max_tiles == b.max_tiles && a.max_tiles64 == b.max_tiles64 &&
                      a.fgu == b.fgu && a.fdn == b.fdn && a.gu_bn == b.gu_bn && a.comb4 == b.comb4 && a.comb_grid.x == b.comb_grid.x &&
                      a.comb_grid.y == b.comb_grid.y && a.gu_pgrid.x == b.gu_pgrid.x && a.dn_pgrid.x == b.dn_pgrid.x);
                CHECK(nameOf(a.fgu) == "gemm_moegu_mxfp4" && nameOf(a.fdn) == "gemm_moe32r_mxfp4");
                // an explicit WHIRL_MOEP_* combination (or COMB alone, e.g. GU=0 + COMB=1) is floored the same way
                ops::MoeFlags x = precise ? pre : bal;
                ops::setMoep(x, 2, 2, 2);
                CHECK(ops::chooseMoe(pm, x, km, km.silu_mul_x8h).kind == ops::MoePrefillKind::fused_gu);
                ops::setMoep(x, 0, 3, 1);
                CHECK(!ops::chooseMoe(pm, x, km, km.silu_mul_x8h).comb4);
            }
            // decode / verify batches never take it (the decode experts; no combine launch on that path)
            pm.n = 1;
            CHECK(ops::chooseMoe(pm, f, km, km.silu_mul_x8h).small);
            // the plan's 0 / 0 / 0: the shipped 0.2.5 plan (fused_gu, moe_combine)
            ops::MoeFlags off = precise ? pre : bal;
            ops::setMoep(off, 0, 0, 0);
            pm.n = 4096;
            pl = ops::chooseMoe(pm, off, km, km.silu_mul_x8h);
            CHECK(pl.kind == ops::MoePrefillKind::fused_gu && !pl.comb4 && pl.comb_grid.x == 8 && nameOf(pl.fdn) == "gemm_moe32r_mxfp4");
            // WHIRL_MOEP_COMB=0 alone: the moep GEMMs with moe_combine
            ops::MoeFlags c0 = precise ? pre : bal;
            ops::setMoep(c0, 1, 3, 0);
            pl = ops::chooseMoe(pm, c0, km, km.silu_mul_x8h);
            CHECK(pl.kind == ops::MoePrefillKind::moep && !pl.comb4 && pl.comb_grid.x == 8);
            // WHIRL_MOEP_GU=0 alone: the fused_gu GEMMs with moe_combine4 (both bit-identical to the shipped path)
            ops::MoeFlags g0 = precise ? pre : bal;
            ops::setMoep(g0, 0, 3, 1);
            pl = ops::chooseMoe(pm, g0, km, km.silu_mul_x8h);
            CHECK(pl.kind == ops::MoePrefillKind::fused_gu && pl.comb4 && pl.comb_grid.x == 2);
            // a gfx1201 table with the default flags (the plan never sets them there; if it did): no moep kernels, no
            // moe_combine4 -> the unchanged R9700 plan
            ops::MoeFlags r = precise ? pre : bal;
            ops::MoeFlags r0 = r;
            ops::setMoep(r, 1, 3, 1);
            const ops::MoePlan a = ops::chooseMoe(pm, r, k97, k97.silu_mul_x8h), b = ops::chooseMoe(pm, r0, k97, k97.silu_mul_x8h);
            CHECK(a.kind == b.kind && a.bn == b.bn && a.fgu == b.fgu && a.fdn == b.fdn && !a.comb4 && a.comb_grid.x == b.comb_grid.x &&
                  a.gu_pgrid.x == b.gu_pgrid.x && a.dn_pgrid.x == b.dn_pgrid.x && a.dn_pgrid.y == b.dn_pgrid.y);
        }
    }
    testOpsMoeRouter(p, k97, k8);
    testOpsMoeSkip(p);  // v0252-f5
}

}  // namespace
