// arch025-n8b1: cases moved verbatim from tests/ops/ops_tests.cpp; included by whirl-ops-tests (tests/ops/ops_tests.cpp, same
// cases, same order) and by the module's isolation program tests/iso/ops.moe/ (ARCH-1 8.5).
// SPDX-License-Identifier: Apache-2.0

#include "whirl/ops/moe.h"
#include "../test_check.h"
#include "ops_fakes.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

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
// buffer, and the kernel in the code object (gfx1151 since 0.2.5.1, gfx1201 since v0252-c5port). Everything else: null =
// moe_logits_f32 + moe_topk, the 0.2.5 path. Fails before the a11 port (chooseMoe never sets router).
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
    // the 0.2.5.1 R9700 table has no fused router: the two-kernel path even with the flag on
    p.n = 2;
    CHECK(ops::chooseMoe(p, bal, k97, k97.silu_mul_x8h).router == nullptr && ops::chooseMoe(p, pre, k97, k97.silu_mul_x8h).router == nullptr);
    // v0252-c5port: the gfx1201 table with the fused router (kernels/moe_router.hip in the gfx1201 code object) takes it
    // on the same pieces as the 8060S (the selection does not depend on the card), and the rest of its plan is the
    // 0.2.5.1 R9700 plan; with the flag off it is the 0.2.5.1 plan exactly
    {
        kn::KernelTable k97r = k97;
        k97r.moe_router_fused_i16 = tag("moe_router_fused_i16");
        k97r.moe_router_fused_i24 = tag("moe_router_fused_i24");
        for (const std::uint32_t n : {1u, 2u, 4u, 5u, 16u}) {
            p.n = n;
            for (const ops::MoeFlags* f : {&bal, &pre}) {
                const ops::MoePlan a = ops::chooseMoe(p, *f, k97, k97.silu_mul_x8h), b = ops::chooseMoe(p, *f, k97r, k97r.silu_mul_x8h);
                CHECK_DETAIL(nameOf(b.router) == "moe_router_fused_i16", "r9700 router n " + std::to_string(n) + ": " + nameOf(b.router));
                CHECK(a.small == b.small && a.small_q8 == b.small_q8 && a.gu == b.gu && a.gu_up == b.gu_up && a.down == b.down &&
                      a.bits == b.bits && a.logits_grid.x == b.logits_grid.x && a.gu_grid.x == b.gu_grid.x && a.gu_grid.y == b.gu_grid.y);
                ops::MoeFlags off = *f;
                off.moe_rfuse = false;
                const ops::MoePlan c0 = ops::chooseMoe(p, off, k97r, k97r.silu_mul_x8h);
                CHECK(c0.router == nullptr && c0.small == a.small && c0.gu == a.gu && c0.down == a.down);
            }
        }
        p.n = 17;
        ops::MoeFlags wide = bal;
        wide.small_max = 32;
        CHECK(ops::chooseMoe(p, wide, k97r, k97r.silu_mul_x8h).router == nullptr);  // more than one piece: two kernels
        p.n = 4096;
        CHECK(ops::chooseMoe(p, bal, k97r, k97r.silu_mul_x8h).router == nullptr);  // prefill: never
        p.n = 2;
    }
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

// v0252-c3port (feat026-moedec-p25 0ae5530 testMoeFuse, gfx1201 only): the fused precise decode MoE launch
// (ops::chooseMoeFuse) - taken only for the precise f32 decode experts with one moe_gu_f_mxfp4 gate+up launch, MXFP4
// routed and shared matrices on the gemvh path, the shapes the kernel needs (F % 4, n_embd % 256, shared rows % 64),
// 1..32 rows, and only when enabled; grid = shared blocks + routed blocks. Per-arch selector (DISCOVERY 2e C3): with the
// gfx1151 kernel table (no moe_gus*_f) the plan is never used, so the 8060S keeps the 0.2.5.1 chain; the plans chooseMoe
// itself gives (precise decode on both tables, balance q8, prefill) are the inputs. Fails before the port (stub).
void testOpsMoeFuse(ops::MoeProblem mp, const kn::KernelTable& k97, const kn::KernelTable& k8) {
    kn::KernelTable k = k97;  // the gfx1201 table with the port's kernels (kernels_abi.cpp loadMoe)
    k.moe_gus_f = tag("moe_gus_f_mxfp4");
    k.moe_gus2_f = tag("moe_gus2_f_mxfp4");
    CHECK(k97.moe_gus_f == nullptr && k8.moe_gus_f == nullptr && k8.moe_gus2_f == nullptr);  // the shipped tables
    ops::MoeFlags pre;
    pre.prec_dec = true;
    mp.n = 1;
    const ops::MoePlan pl = ops::chooseMoe(mp, pre, k, k.silu_mul_x8h);
    CHECK(pl.small && !pl.small_q8 && pl.bits == ops::MoeBits::moe_f32_row && !pl.gu_split && nameOf(pl.gu) == "moe_gu_f_mxfp4");
    ops::MoeFuseProblem p;  // Ornith-1.5-35B-A3B: n_embd 2048, 8 of 256 experts, F 512, shared 512 rows
    p.n = 1;
    p.C = 2048;
    p.F = 512;
    p.K = 8;
    p.gate_ty = p.up_ty = GgmlType::mxfp4;
    p.gate_rb = p.up_rb = 2048 / 32 * 17;
    p.shared = true;
    p.shared_gemvh = true;
    p.sg_ty = p.su_ty = GgmlType::mxfp4;
    p.sg_rb = p.su_rb = 2048 / 32 * 17;
    p.sg_nrows = p.su_nrows = 512;
    p.sg_ncols = p.su_ncols = 2048;
    ops::MoeFusePlan f = ops::chooseMoeFuse(p, true, pl, k);
    CHECK(f.use && nameOf(f.f) == "moe_gus_f_mxfp4" && f.nsb == 8 && f.nrb == 128 && f.grid.x == 8 + 128 * 8 && f.grid.y == 1 && f.grid.z == 1);
    // off unless enabled (WHIRL_MOE_FUSE, default off)
    CHECK(!ops::chooseMoeFuse(p, false, pl, k).use);
    // per-arch selector: the gfx1151 table (and the shipped gfx1201 table) never fuse, even enabled, with either plan
    {
        const ops::MoePlan pl8 = ops::chooseMoe(mp, pre, k8, k8.silu_mul_x8h);
        CHECK(!ops::chooseMoeFuse(p, true, pl8, k8).use && !ops::chooseMoeFuse(p, true, pl, k8).use && !ops::chooseMoeFuse(p, true, pl, k97).use);
        for (const std::uint32_t n : {1u, 4u, 16u, 17u, 32u}) {
            ops::MoeFuseProblem q = p;
            q.n = n;
            mp.n = n;
            CHECK(!ops::chooseMoeFuse(q, true, ops::chooseMoe(mp, pre, k8, k8.silu_mul_x8h), k8).use);
        }
        mp.n = 1;
    }
    // never for the balance / fast int8 experts or a prefill batch (the plans chooseMoe gives)
    {
        ops::MoeFlags bal;
        bal.moe_fp8 = true;
        CHECK(!ops::chooseMoeFuse(p, true, ops::chooseMoe(mp, bal, k, k.silu_mul_x8h), k).use);
        ops::MoeProblem big = mp;
        big.n = 4096;
        ops::MoeFuseProblem q = p;
        q.n = 4096;
        CHECK(!ops::chooseMoeFuse(q, true, ops::chooseMoe(big, pre, k, k.silu_mul_x8h), k).use);
    }
    ops::MoePlan q8 = pl;
    q8.small_q8 = true;
    q8.bits = ops::MoeBits::moe_q8_row;
    CHECK(!ops::chooseMoeFuse(p, true, q8, k).use);
    ops::MoePlan pf = pl;
    pf.small = false;
    CHECK(!ops::chooseMoeFuse(p, true, pf, k).use);
    ops::MoePlan split = pl;
    split.gu_split = true;
    CHECK(!ops::chooseMoeFuse(p, true, split, k).use);
    // the plan's gate+up kernel must be the one the fused routed blocks reproduce (moe_gu_f_mxfp4)
    ops::MoePlan other = pl;
    other.gu = tag("moe_gu_f_q8_0");
    CHECK(!ops::chooseMoeFuse(p, true, other, k).use);
    // rows: 17..32 take the two-token-tile twin, 0 / 33 none
    p.n = 16;
    CHECK(nameOf(ops::chooseMoeFuse(p, true, pl, k).f) == "moe_gus_f_mxfp4");
    p.n = 17;
    f = ops::chooseMoeFuse(p, true, pl, k);
    CHECK(f.use && nameOf(f.f) == "moe_gus2_f_mxfp4" && f.grid.x == 8 + 128 * 17 * 8);
    p.n = 32;
    CHECK(ops::chooseMoeFuse(p, true, pl, k).use);
    p.n = 33;
    CHECK(!ops::chooseMoeFuse(p, true, pl, k).use);
    p.n = 0;
    CHECK(!ops::chooseMoeFuse(p, true, pl, k).use);
    p.n = 1;
    // shapes and types
    ops::MoeFuseProblem b = p;
    b.F = 510;
    CHECK(!ops::chooseMoeFuse(b, true, pl, k).use);
    b = p;
    b.C = 2048 + 128;
    CHECK(!ops::chooseMoeFuse(b, true, pl, k).use);
    b = p;
    b.gate_ty = b.up_ty = GgmlType::q4_k;
    CHECK(!ops::chooseMoeFuse(b, true, pl, k).use);
    b = p;
    b.up_rb = b.gate_rb + 17;
    CHECK(!ops::chooseMoeFuse(b, true, pl, k).use);
    b = p;
    b.sg_nrows = b.su_nrows = 512 + 32;
    CHECK(!ops::chooseMoeFuse(b, true, pl, k).use);
    b = p;
    b.su_ty = GgmlType::q8_0;
    CHECK(!ops::chooseMoeFuse(b, true, pl, k).use);
    b = p;
    b.shared_gemvh = false;  // the old chain would not run gemvh on the shared matrices -> other bits, never fused
    CHECK(!ops::chooseMoeFuse(b, true, pl, k).use);
    b = p;
    b.sg_ncols = 1024;
    CHECK(!ops::chooseMoeFuse(b, true, pl, k).use);
    // no shared expert: routed blocks only (the shared checks do not apply)
    b = p;
    b.shared = false;
    b.shared_gemvh = false;
    b.sg_ty = b.su_ty = GgmlType::f32;
    f = ops::chooseMoeFuse(b, true, pl, k);
    CHECK(f.use && f.nsb == 0 && f.grid.x == 128 * 8);
    // a code object with only the 16-row entry: 17..32 rows keep the old chain
    kn::KernelTable only16 = k;
    only16.moe_gus2_f = nullptr;
    p.n = 20;
    CHECK(!ops::chooseMoeFuse(p, true, pl, only16).use);
    p.n = 16;
    CHECK(ops::chooseMoeFuse(p, true, pl, only16).use);
}

// v0252-precfix (PLAN V1-85; KB v0252-precconc:moe-segjoin-window): R9700 2026-11-11 hostwait proof, Ornith precise server
// gate basic: "5 concurrent requests == each alone (cold)" and "same seed under load (4 concurrent)" FAIL, deterministic; the
// slot whose prompt was prefilled alone matched, the slots of the segmented prefill forward differed. Cause: since 92aa626
// precise MoE gives every forward of <= max_verify_rows (32) rows the f32 decode experts, but the server let chunks of
// > max_small_batch (16) rows join a segmented forward (> 32 rows: the f16 grouped GEMM). A chunk that may join (more than
// moeSegJoinMin rows) must take the same MoE bit class alone as in the smallest segmented forward it can be part of
// (itself + another chunk of moeSegJoinMin + 1 rows), in every mode, on both code objects (gfx1201, gfx1151 tables).
void testOpsMoeSegJoin(ops::MoeProblem p, const kn::KernelTable& k97, const kn::KernelTable& k8) {
    CHECK_EQ(ops::moeSegJoinMin(false, 16, 32), 16u);  // balance / fast: the decode experts stop at 16 rows in a prefill forward
    CHECK_EQ(ops::moeSegJoinMin(true, 16, 32), 32u);   // precise: they serve up to max_verify_rows (92aa626)
    CHECK_EQ(ops::moeSegJoinMin(true, 16, 16), 16u);   // no wider precise range: the server's rule
    for (const bool precise : {false, true}) {
        ops::MoeFlags f;
        f.prec_dec = precise;
        f.moe_fp8 = !precise;  // balance / fast prefill: MXFP4 x fp8 (numerics plan); precise: f16
        const std::uint32_t jmin = ops::moeSegJoinMin(f.prec_dec, f.max_small_batch, f.max_verify_rows);
        for (const kn::KernelTable* k : {&k97, &k8}) {
            std::uint32_t bad = 0;
            for (std::uint32_t n = jmin + 1; n <= 2 * f.max_verify_rows + 1; ++n) {
                p.n = n;
                const ops::MoePlan alone = ops::chooseMoe(p, f, *k, k->silu_mul_x8h);
                p.n = n + jmin + 1;
                const ops::MoePlan joined = ops::chooseMoe(p, f, *k, k->silu_mul_x8h);
                if (alone.small != joined.small || alone.bits != joined.bits) ++bad;
            }
            CHECK_EQ(bad, 0u);
        }
    }
}

// v0252-moedf16 (PLAN V1-76): on the gfx1201 fp8 MoE prefill (balance / fast), COMB 3 (the balance item f16
// intermediates) stores the down projection's output moe_yd as f16 - the down launch is gemm8r_moeh / gemm8r_moe32h with
// W2 = null (one GEMM, f16 epilogue) - and combines it with moe_combine4h (grid (E / 1024, n)). Precise (prec_dec, or the
// bitwise class with WHIRL_Q8DEC=1) and the 8060S plans never change; without moe_combine4h / the rbf kernels / E % 4: the
// f32 path. Not floored by the moep token floor (that floor is about the moep tile sharing): any prefill n.
void testOpsMoeYdh(ops::MoeProblem p, const kn::KernelTable& k97_in, const kn::KernelTable& k8) {
    kn::KernelTable k97 = k97_in;
    k97.moe_combine4h = tag("moe_combine4h");  // the gfx1201 code object carries it from v0252-moedf16 on
    ops::MoeFlags bal, fst, pre, q8p;
    bal.moe_fp8 = fst.moe_fp8 = q8p.moe_fp8 = true;
    bal.moep_comb_ok = ops::moepCombFor(false, false);
    fst.moep_comb_ok = ops::moepCombFor(false, true);
    pre.prec_dec = true;
    pre.moep_comb_ok = ops::moepCombFor(true, false);
    q8p.moep_comb_ok = ops::moepCombFor(true, false);  // precise with WHIRL_Q8DEC=1 (prec_dec cleared), WHIRL_MOE_FP8=1
    for (ops::MoeFlags* f : {&bal, &fst, &pre, &q8p}) f->moep_comb = 3;
    for (const std::uint32_t n : {4096u, 2048u, 512u, 200u}) {
        p.n = n;
        const ops::MoeFlags bal0 = [&] {
            ops::MoeFlags g = bal;
            g.moep_comb = 0;
            return g;
        }();
        const ops::MoePlan a = ops::chooseMoe(p, bal0, k97, k97.silu_mul_x8h);  // A arm (WHIRL_MOEP_COMB=0): the f32 path
        CHECK(a.kind == ops::MoePrefillKind::fp8 && !a.ydh && !a.comb4 && a.bits == ops::MoeBits::moe_fp8_gemm);
        CHECK(nameOf(a.fdn) == (a.bn == 32 ? "gemm8r_moe32" : "gemm8r_moe") && a.comb_grid.x == 8 && a.comb_grid.y == n);
        for (const ops::MoeFlags* f : {&bal, &fst}) {
            const ops::MoePlan b = ops::chooseMoe(p, *f, k97, k97.silu_mul_x8h);
            // only the down kernel, the combine and the bit class move; the rest of the plan is the A arm's
            CHECK(b.kind == ops::MoePrefillKind::fp8 && b.ydh && b.comb4 && !b.comb_fused && !b.comb_atomic && b.bits == ops::MoeBits::moe_lossy_gemm);
            CHECK(nameOf(b.fdn) == (b.bn == 32 ? "gemm8r_moe32h" : "gemm8r_moeh") && b.comb_grid.x == 2 && b.comb_grid.y == n);
            CHECK(b.bn == a.bn && b.max_tiles == a.max_tiles && b.fgu == a.fgu && b.rbf && b.gu_pgrid.x == a.gu_pgrid.x && b.dn_pgrid.x == a.dn_pgrid.x &&
                  b.dn_pgrid.y == a.dn_pgrid.y);
        }
        // precise: the gemmr (f16) kind, f32 moe_yd + moe_combine, whatever COMB says
        const ops::MoePlan c = ops::chooseMoe(p, pre, k97, k97.silu_mul_x8h);
        CHECK(c.kind == ops::MoePrefillKind::gemmr && !c.ydh && !c.comb4 && c.bits == ops::MoeBits::moe_f16_gemm && nameOf(c.fdn) != "gemm8r_moeh");
        // precise with prec_dec cleared and fp8 forced: still the f32 moe_yd (the mode's class is bitwise)
        const ops::MoePlan d = ops::chooseMoe(p, q8p, k97, k97.silu_mul_x8h);
        CHECK(d.kind == ops::MoePrefillKind::fp8 && !d.ydh && !d.comb4 && d.bits == ops::MoeBits::moe_fp8_gemm &&
              nameOf(d.fdn) == (d.bn == 32 ? "gemm8r_moe32" : "gemm8r_moe"));
    }
    p.n = 4096;
    // COMB 1 / 2 / 4 / 5 on the fp8 kind: the f32 path (moe_combine; no moe_combine4 in the gfx1201 object)
    for (const std::uint32_t cm : {1u, 2u, 4u, 5u}) {
        ops::MoeFlags f = fst;
        f.moep_comb = cm;
        const ops::MoePlan b = ops::chooseMoe(p, f, k97, k97.silu_mul_x8h);
        CHECK(b.kind == ops::MoePrefillKind::fp8 && !b.ydh && !b.comb_fused && !b.comb_atomic && nameOf(b.fdn) == "gemm8r_moe");
    }
    // a code object without moe_combine4h (the gfx1201 object before this change) or E % 4 != 0: the f32 path
    {
        const ops::MoePlan b = ops::chooseMoe(p, bal, k97_in, k97_in.silu_mul_x8h);
        CHECK(b.kind == ops::MoePrefillKind::fp8 && !b.ydh && nameOf(b.fdn) == "gemm8r_moe");
        ops::MoeProblem q = p;
        q.E = 2050;
        CHECK(!ops::chooseMoe(q, bal, k97, k97.silu_mul_x8h).ydh);
    }
    // WHIRL_MOE_RBF=0 (no row-block-fast entries): the tile-major f32 path, no ydh (gemm8_moeh's 2-D grid is not wired)
    {
        ops::MoeFlags f = bal;
        f.moe_rbf = false;
        const ops::MoePlan b = ops::chooseMoe(p, f, k97, k97.silu_mul_x8h);
        CHECK(b.kind == ops::MoePrefillKind::fp8 && !b.rbf && !b.ydh && nameOf(b.fdn) == "gemm8_moe");
    }
    // decode / verify (small) batches: no combine, no change
    p.n = 1;
    CHECK(ops::chooseMoe(p, bal, k97, k97.silu_mul_x8h).small && !ops::chooseMoe(p, bal, k97, k97.silu_mul_x8h).ydh);
    // the 8060S table with COMB 3 and no moep variants (the plan's moep off): the fused_gu plan, f32 moe_yd (unchanged)
    p.n = 4096;
    kn::KernelTable k8h = k8;
    k8h.moe_combine4h = tag("moe_combine4h");
    const ops::MoePlan e = ops::chooseMoe(p, bal, k8h, k8h.silu_mul_x8h);
    CHECK(e.kind == ops::MoePrefillKind::fused_gu && !e.ydh && !e.comb4 && nameOf(e.fdn) == "gemm_moe32r_mxfp4");
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
    // v0252-f5fix: a kn::KernelTable is ~140 KB; main -> testOpsMoe -> testOpsMoeSkip -> moeTable held 7 of them on the 1 MB
    // main stack (1.12 MB of frames) and whirl-ops-tests / whirl-iso-ops-moe-tests died with 0xC00000FD before any output.
    // Tables live in static storage (built once) or on the heap (the mutated copies), as in testAttnKgVariant (v0252-c1asan)
    // and llama.cpp test-backend-ops (test cases in std::unique_ptr).
    static const kn::KernelTable k97 = moeTable(false);
    const std::unique_ptr<kn::KernelTable> k8p(new kn::KernelTable(moeTable(true)));  // direct-initialized on the heap
    kn::KernelTable& k8 = *k8p;
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
        const auto mp = std::make_unique<kn::KernelTable>(k97);  // heap, see the note at the top of the function
        kn::KernelTable& m = *mp;
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

// v0252-combfix (PLAN V1-119; KB v0252-ornbal:moep-comb3-row-floor-schedule-dependence): 8060S release gate step
// seq_moe_balance FAIL 2026-11-13 - 2 of 7 server chunk schedules gave other last-row logits bits than the cold reference
// (Ornith-1.5-35B-A3B MXFP4, balance). Cause: v0252-ornkv2 (PLAN V1-94) made COMB 3 (f16 expert outputs moe_yd +
// moe_combine4h) the gfx1151 balance default (and the fast one where moeskip is off), but chooseMoe's row floor (moepWide,
// Ornith: n >= 1024 rows) dropped the combine class together with the moep tiling, so a forward of 284 rows stored moe_yd
// as f32 and one of 1308 rows as f16: a token's MoE numeric class followed the row count of the forward it landed in.
// The rule (docs/guide/en/kv-and-caching.md section 7, FIX-TEST-B; the same rule form as testOpsMoeSegJoin): within ONE
// mode the MoE numeric class of a prefill row is the same for every forward size, on the gfx1151 table as on the gfx1201
// one. The floor may still change the TILING (fused_gu gate + up below it, the moep GEMMs from it: bitwise equal), never
// the stored type of the expert outputs. Not a balance == precise requirement: one mode against itself.
// Fails before the fix (n = 284: bits moe_f16_gemm / ydh 0, n = 1308: moe_lossy_gemm / ydh 1).
void testOpsMoeCombRows(ops::MoeProblem p) {
    p.rcnt_buf = true;
    p.cnt_buf = true;
    static const kn::KernelTable k97 = moeTable(false);
    // the gfx1151 code object: every moep variant with all of its epilogue twins (kernels/gfx1151/moe_prefill_mt.hip
    // MOEPGU_ENTRY / MOEPDN_ENTRY), no whole-block MXFP4 decode experts; heap (a kn::KernelTable is ~140 KB, v0252-f5fix)
    const std::unique_ptr<kn::KernelTable> k8p(new kn::KernelTable(moeTable(true)));
    kn::KernelTable& k8 = *k8p;
    k8.moe_gu_mxw = k8.moe_down_mxw = nullptr;
    for (int v = 0; v < kn::kMoepGuN; ++v) k8.moep_gu[v] = tag(std::string("gemm_moepgu_") + kn::kMoepGu[v].name + "_mxfp4");
    for (int v = 0; v < kn::kMoepDnN; ++v) {
        k8.moep_dn[v] = tag(std::string("gemm_moepdn_") + kn::kMoepDn[v].name + "_mxfp4");
        k8.moep_dc[v] = tag(std::string("gemm_moepdc_") + kn::kMoepDn[v].name + "_mxfp4");
        k8.moep_dh[v] = tag(std::string("gemm_moepdh_") + kn::kMoepDn[v].name + "_mxfp4");
        k8.moep_dhc[v] = tag(std::string("gemm_moepdhc_") + kn::kMoepDn[v].name + "_mxfp4");
        k8.moep_da[v] = tag(std::string("gemm_moepda_") + kn::kMoepDn[v].name + "_mxfp4");
    }
    k8.moe_combine4 = tag("moe_combine4");
    k8.moe_combine4h = tag("moe_combine4h");
    // what decides a row's bits: the experts' class, the stored type of the expert outputs and the combine's arithmetic
    // (moe_combine4 == moe_combine and fused_gu == moep bitwise, so kind / comb4 / tiles are not part of it), and whether
    // the block drops slots
    auto cls = [](const ops::MoePlan& a) {
        return std::string(a.small ? (a.small_q8 ? "q8row" : "f32row") : "gemm") + " bits " + std::to_string(static_cast<int>(a.bits)) + " ydh " +
               (a.ydh ? "1" : "0") + " fusedh " + (a.ydh && a.comb_fused ? "1" : "0") + " atomic " + (a.comb_atomic ? "1" : "0") + " skip " +
               (a.skip ? "1" : "0");
    };
    const unsigned gu1 = static_cast<unsigned>(kn::kMoepDefault.gu) + 1, dn1 = static_cast<unsigned>(kn::kMoepDefault.dn) + 1;
    // the three flag sets that reach COMB 3 on the 8060S (numerics.cpp moepCombDefault): balance; fast with moeskip off;
    // fast with moeskip on and an explicit WHIRL_MOEP_COMB=3 (the screen's other arm)
    ops::MoeFlags bal, fst, fsk;
    for (ops::MoeFlags* f : {&bal, &fst, &fsk}) ops::setMoep(*f, gu1, dn1, 3);
    bal.moep_comb_ok = ops::moepCombFor(false, false);
    fst.moep_comb_ok = fsk.moep_comb_ok = ops::moepCombFor(false, true);
    fsk.moe_skip_thr = 0.05f;
    const struct {
        const char* name;
        const ops::MoeFlags* f;
    } arms[] = {{"balance", &bal}, {"fast, moeskip off", &fst}, {"fast, moeskip on + COMB 3", &fsk}};
    for (const auto& arm : arms) {
        const ops::MoeFlags& f = *arm.f;
        // the two forward sizes of the failing gate run: [9216,9500] = 284 rows and [8192,9500] = 1308 rows
        p.n = 284;
        const ops::MoePlan a = ops::chooseMoe(p, f, k8, k8.silu_mul_x8h);
        p.n = 1308;
        const ops::MoePlan b = ops::chooseMoe(p, f, k8, k8.silu_mul_x8h);
        CHECK_DETAIL(a.bits == b.bits && a.ydh == b.ydh,
                     std::string(arm.name) + ": n=284 [" + cls(a) + "] n=1308 [" + cls(b) + "]");
        CHECK_DETAIL(cls(a) == cls(b), std::string(arm.name) + ": n=284 [" + cls(a) + "] n=1308 [" + cls(b) + "]");
        // every prefill forward size (above the decode experts' rows) against the wide one
        std::uint32_t bad = 0, first_bad = 0;
        for (std::uint32_t n = f.small_max + 1; n <= 4096; ++n) {
            p.n = n;
            if (cls(ops::chooseMoe(p, f, k8, k8.silu_mul_x8h)) != cls(b)) {
                if (bad++ == 0) first_bad = n;
            }
        }
        CHECK_DETAIL(bad == 0u, std::string(arm.name) + ": " + std::to_string(bad) + " prefill forward sizes of 17..4096 with another class than n=1308, first n=" +
                                    std::to_string(first_bad));
        // the wide forward keeps the V1-94 plan: the moep GEMMs, gemm_moepdh of the default DN variant, moe_combine4h's grid
        CHECK_DETAIL(b.kind == ops::MoePrefillKind::moep && b.ydh && b.comb4 && !b.comb_fused && !b.comb_atomic && !b.skip &&
                         b.bits == ops::MoeBits::moe_lossy_gemm && nameOf(b.fgu) == "gemm_moepgu_g64n64k64_mxfp4" &&
                         nameOf(b.fdn) == "gemm_moepdh_m128w4n128k64_mxfp4" && b.bn == 128 && b.comb_grid.x == 2 && b.comb_grid.y == 1308,
                     std::string(arm.name) + ": wide plan " + nameOf(b.fgu) + " / " + nameOf(b.fdn));
        // the narrow forward keeps the 0.2.5 tiling (B26-43: the fused gate + up over 64-token tiles, the down GEMM over
        // 32-token tiles) and takes the f16-output down twin of that tile (gemm_moepdh_m128w4n32k64 = gemm_moe32r's tile,
        // gemm_moe32r's arguments, so ffn.cpp's fused_gu launch is unchanged) + moe_combine4h
        CHECK_DETAIL(a.kind == ops::MoePrefillKind::fused_gu && a.ydh && a.comb4 && !a.comb_fused && !a.comb_atomic && !a.skip &&
                         a.bits == ops::MoeBits::moe_lossy_gemm && nameOf(a.fgu) == "gemm_moegu_mxfp4" &&
                         nameOf(a.fdn) == "gemm_moepdh_m128w4n32k64_mxfp4",
                     std::string(arm.name) + ": narrow plan " + nameOf(a.fgu) + " / " + nameOf(a.fdn));
        CHECK(a.bn == 32 && a.max_tiles == (284 * 8 + 31) / 32 + 256 && a.max_tiles64 == (284 * 8 + 63) / 64 + 256 &&
              a.gu_pgrid.x == a.max_tiles64 * 8 && a.dn_pgrid.x == a.max_tiles * 16 && a.comb_grid.x == 2 && a.comb_grid.y == 284);
    }
    // a code object without the 32-token f16 twin: the narrow forward takes the caller's DN variant's (its tile list, its
    // row blocks) - the class never waits for a kernel the wide path does not need
    {
        const auto kmp = std::make_unique<kn::KernelTable>(k8);
        kn::KernelTable& km = *kmp;
        for (int v = 0; v < kn::kMoepDnN; ++v)
            if (v != kn::kMoepDefault.dn) km.moep_dh[v] = nullptr;
        p.n = 284;
        const ops::MoePlan a = ops::chooseMoe(p, bal, km, km.silu_mul_x8h);
        CHECK_DETAIL(a.kind == ops::MoePrefillKind::fused_gu && a.ydh && a.comb4 && a.bits == ops::MoeBits::moe_lossy_gemm &&
                         nameOf(a.fgu) == "gemm_moegu_mxfp4" && nameOf(a.fdn) == "gemm_moepdh_m128w4n128k64_mxfp4" && a.bn == 128 &&
                         a.max_tiles == (284 * 8 + 127) / 128 + 256 && a.dn_pgrid.x == a.max_tiles * 16,
                     "narrow plan without the 32-token twin: " + nameOf(a.fdn) + " bn " + std::to_string(a.bn));
    }
    // a code object with the moep GEMMs but no fused gate + up (none ships): no narrow-tile plan can carry the f16 class, so
    // the moep GEMMs serve every forward size (moepFloor keep) rather than the class changing with the rows
    {
        const auto kmp = std::make_unique<kn::KernelTable>(k8);
        kn::KernelTable& km = *kmp;
        km.gemm_moegu[ti(GgmlType::mxfp4)] = nullptr;
        p.n = 284;
        const ops::MoePlan a = ops::chooseMoe(p, bal, km, km.silu_mul_x8h);
        p.n = 1308;
        const ops::MoePlan b = ops::chooseMoe(p, bal, km, km.silu_mul_x8h);
        CHECK_DETAIL(a.kind == ops::MoePrefillKind::moep && a.ydh && cls(a) == cls(b) && nameOf(a.fgu) == "gemm_moepgu_g64n64k64_mxfp4" &&
                         nameOf(a.fdn) == "gemm_moepdh_m128w4n128k64_mxfp4" && a.fgu == b.fgu && a.fdn == b.fdn,
                     "no fused gate + up: n=284 [" + cls(a) + "] " + nameOf(a.fdn) + " n=1308 [" + cls(b) + "] " + nameOf(b.fdn));
        // the bitwise default (COMB 1) is still floored there: the generic grouped GEMM below the floor, the same bits
        ops::MoeFlags c1 = bal;
        c1.moep_comb = 1;
        p.n = 284;
        const ops::MoePlan c = ops::chooseMoe(p, c1, km, km.silu_mul_x8h);
        CHECK_DETAIL(c.kind == ops::MoePrefillKind::generic && !c.ydh && c.bits == ops::MoeBits::moe_f16_gemm,
                     "no fused gate + up, COMB 1: n=284 [" + cls(c) + "]");
    }
    // where the wide forward does not get the f16 class, the narrow one does not either (and the other way round):
    // precise (prec_dec, or the bitwise class with WHIRL_Q8DEC=1), no gemm_moepdh of the caller's variant, no moe_combine4h,
    // the moep variants off, E % 4 != 0 - and the bit-identical combines 0 / 1 / 2 keep the 0.2.5.1 plans on both sides
    {
        struct Case {
            std::string name;
            ops::MoeFlags f;
            std::unique_ptr<kn::KernelTable> k;
            ops::MoeProblem p;
        };
        std::vector<Case> cs;
        auto add = [&](const std::string& name, const ops::MoeFlags& f) -> Case& {
            cs.push_back(Case{name, f, std::make_unique<kn::KernelTable>(k8), p});
            return cs.back();
        };
        ops::MoeFlags pre = bal;
        pre.prec_dec = true;
        pre.moep_comb_ok = ops::moepCombFor(true, false);
        add("precise", pre);
        ops::MoeFlags q8p = bal;
        q8p.moep_comb_ok = ops::moepCombFor(true, false);
        add("precise + WHIRL_Q8DEC=1", q8p);
        add("no gemm_moepdh of the DN variant", bal).k->moep_dh[kn::kMoepDefault.dn] = nullptr;
        add("no moe_combine4h", bal).k->moe_combine4h = nullptr;
        ops::MoeFlags nogu = bal;
        nogu.moep_gu = -1;
        add("WHIRL_MOEP_GU=0", nogu);
        add("E % 4 != 0", bal).p.E = 2050;
        for (const unsigned cm : {0u, 1u, 2u}) {
            ops::MoeFlags c = bal;
            c.moep_comb = cm;
            add("COMB " + std::to_string(cm), c);
            ops::MoeFlags s = fsk;  // fast with moeskip: COMB 1 is its default (numerics.cpp moepCombMoeSkip)
            s.moep_comb = cm;
            add("fast moeskip COMB " + std::to_string(cm), s);
        }
        for (Case& c : cs) {
            c.p.n = 284;
            const ops::MoePlan a = ops::chooseMoe(c.p, c.f, *c.k, c.k->silu_mul_x8h);
            c.p.n = 1308;
            const ops::MoePlan b = ops::chooseMoe(c.p, c.f, *c.k, c.k->silu_mul_x8h);
            CHECK_DETAIL(!a.ydh && !b.ydh && a.bits == ops::MoeBits::moe_f16_gemm && b.bits == ops::MoeBits::moe_f16_gemm &&
                             a.kind == ops::MoePrefillKind::fused_gu && nameOf(a.fdn) == "gemm_moe32r_mxfp4" && a.bn == 32,
                         c.name + ": n=284 [" + cls(a) + "] " + nameOf(a.fdn) + " n=1308 [" + cls(b) + "] " + nameOf(b.fdn));
        }
    }
    // the gfx1201 table (R9700): COMB 3 from the caller's flag at any prefill n (v0252-moedf16), unchanged
    {
        const auto k97p = std::make_unique<kn::KernelTable>(k97);
        k97p->moe_combine4h = tag("moe_combine4h");
        ops::MoeFlags r = bal;
        ops::setMoep(r, 0, 0, 3);
        r.moe_fp8 = true;
        std::uint32_t bad = 0;
        p.n = 1308;
        const std::string want = cls(ops::chooseMoe(p, r, *k97p, k97p->silu_mul_x8h));
        for (std::uint32_t n = r.small_max + 1; n <= 4096; ++n) {
            p.n = n;
            const ops::MoePlan a = ops::chooseMoe(p, r, *k97p, k97p->silu_mul_x8h);
            if (cls(a) != want || !a.ydh || a.kind != ops::MoePrefillKind::fp8) ++bad;
        }
        CHECK_EQ(bad, 0u);
    }
}

// v0252-8smoe (B, PLAN V1-147): the routing-aware wide gate + up tile (MoeFlags::moep_guw, ops::moepGuFor): a moep forward
// whose experts average at least the wide variant's token tile (n * K >= bn * R; Ornith 8 of 256 with g64n128k64: n >= 4096
// rows) runs the wide gate + up variant (gu_bn, max_tiles64 and the grid follow it); below it, with the flag off (-1), with
// the wide kernel missing from the code object or an index out of range, the plan is field for field the moep_gu plan.
// Nothing else moves: the down GEMM, its tile list, the combine (every COMB) and the skip choice. The R9700 table (no moep
// kernels) is unchanged. Fails to build before the change (no MoeFlags::moep_guw / setMoepGuw / moepGuFor).
void testOpsMoeGuw(ops::MoeProblem p, const kn::KernelTable& k97, const kn::KernelTable& k8) {
    const auto kmp = std::make_unique<kn::KernelTable>(k8);  // heap (v0252-f5fix)
    kn::KernelTable& km = *kmp;
    for (int v = 0; v < kn::kMoepGuN; ++v) km.moep_gu[v] = tag(std::string("gemm_moepgu_") + kn::kMoepGu[v].name + "_mxfp4");
    for (int v = 0; v < kn::kMoepDnN; ++v) {
        km.moep_dn[v] = tag(std::string("gemm_moepdn_") + kn::kMoepDn[v].name + "_mxfp4");
        km.moep_dh[v] = tag(std::string("gemm_moepdh_") + kn::kMoepDn[v].name + "_mxfp4");
        km.moep_da[v] = tag(std::string("gemm_moepda_") + kn::kMoepDn[v].name + "_mxfp4");
    }
    km.moe_combine4 = tag("moe_combine4");
    km.moe_combine4h = tag("moe_combine4h");
    ops::MoeFlags f0;  // the 8060S default moep plan (g64n64k64 + m128w4n128k64 + moe_combine4)
    ops::setMoep(f0, 1, 3, 1);
    CHECK(f0.moep_guw == -1);
    ops::setMoepGuw(f0, 0);
    CHECK(f0.moep_guw == -1);
    ops::MoeFlags fw = f0;
    ops::setMoepGuw(fw, 2);  // the profile row: g64n128k64
    CHECK(fw.moep_guw == 1 && kn::kMoepGu[1].bn == 128);
    auto same = [](const ops::MoePlan& a, const ops::MoePlan& b) {
        return a.kind == b.kind && a.bits == b.bits && a.bn == b.bn && a.max_tiles == b.max_tiles && a.max_tiles64 == b.max_tiles64 && a.fgu == b.fgu &&
               a.fdn == b.fdn && a.gu_bn == b.gu_bn && a.comb4 == b.comb4 && a.comb_fused == b.comb_fused && a.ydh == b.ydh &&
               a.comb_atomic == b.comb_atomic && a.comb_grid.x == b.comb_grid.x && a.comb_grid.y == b.comb_grid.y && a.gu_pgrid.x == b.gu_pgrid.x &&
               a.dn_pgrid.x == b.dn_pgrid.x && a.skip == b.skip;
    };
    // the threshold: n * 8 >= 128 * 256 <=> n >= 4096 (the 8060S MoE forward size); 2058 = the pp 2k forward
    CHECK(ops::moepGuFor(4095, 8, 256, fw, km) == 0 && ops::moepGuFor(4096, 8, 256, fw, km) == 1 && ops::moepGuFor(2058, 8, 256, fw, km) == 0);
    CHECK(ops::moepGuFor(0xffffffffu, 8, 256, fw, km) == 1);  // no u32 overflow in n * K
    CHECK(ops::moepGuFor(1u << 20, 8, 256, f0, km) == 0);     // off
    for (const unsigned mode : {0u, 1u, 2u}) {  // precise / balance / fast class (prec_dec, moep_comb_ok)
        ops::MoeFlags a = f0, b = fw;
        for (ops::MoeFlags* x : {&a, &b}) {
            x->prec_dec = mode == 0;
            x->moep_comb_ok = ops::moepCombFor(mode == 0, mode == 2);
        }
        for (const unsigned comb : {0u, 1u, 2u, 3u, 5u}) {
            a.moep_comb = b.moep_comb = comb;
            for (const float skip : {0.f, 1.f}) {
                a.moe_skip_thr = b.moe_skip_thr = mode == 2 ? skip : 0.f;
                a.moe_skip_min = b.moe_skip_min = 6;
                for (const unsigned n : {33u, 1023u, 1024u, 2058u, 4095u, 4096u, 4106u, 8192u, 32768u}) {
                    ops::MoeProblem q = p;
                    q.n = n;
                    q.cnt_buf = true;
                    const ops::MoePlan pa = ops::chooseMoe(q, a, km, km.silu_mul_x8h), pb = ops::chooseMoe(q, b, km, km.silu_mul_x8h);
                    const std::string cell = "guw mode " + std::to_string(mode) + " comb " + std::to_string(comb) + " skip " +
                                             std::to_string(static_cast<int>(skip)) + " n " + std::to_string(n);
                    if (pa.kind == ops::MoePrefillKind::moep && n >= 4096) {
                        CHECK_DETAIL(pb.kind == ops::MoePrefillKind::moep && nameOf(pb.fgu) == "gemm_moepgu_g64n128k64_mxfp4" && pb.gu_bn == 128 &&
                                         pb.max_tiles64 == (n * 8 + 127) / 128 + 256 && pb.gu_pgrid.x == pb.max_tiles64 * (512 / 64),
                                     cell + ": the wide gate + up tile");
                        CHECK_DETAIL(nameOf(pa.fgu) == "gemm_moepgu_g64n64k64_mxfp4" && pa.gu_bn == 64, cell + ": the A arm keeps g64n64k64");
                        // everything but the gate + up launch is the A arm's
                        CHECK_DETAIL(pa.bits == pb.bits && pa.bn == pb.bn && pa.max_tiles == pb.max_tiles && pa.fdn == pb.fdn && pa.comb4 == pb.comb4 &&
                                         pa.comb_fused == pb.comb_fused && pa.ydh == pb.ydh && pa.comb_atomic == pb.comb_atomic &&
                                         pa.comb_grid.x == pb.comb_grid.x && pa.comb_grid.y == pb.comb_grid.y && pa.dn_pgrid.x == pb.dn_pgrid.x &&
                                         pa.skip == pb.skip,
                                     cell + ": only the gate + up launch moves");
                    } else {
                        CHECK_DETAIL(same(pa, pb), cell + ": the plan of moep_gu");
                    }
                }
            }
        }
    }
    // the wide kernel not in the code object, or an index outside kMoepGu: the moep_gu plan
    ops::MoeProblem q = p;
    q.n = 8192;
    const ops::MoePlan base = ops::chooseMoe(q, f0, km, km.silu_mul_x8h);
    {
        const auto kxp = std::make_unique<kn::KernelTable>(km);
        kxp->moep_gu[1] = nullptr;
        CHECK(same(ops::chooseMoe(q, fw, *kxp, kxp->silu_mul_x8h), ops::chooseMoe(q, f0, *kxp, kxp->silu_mul_x8h)));
        ops::MoeFlags fx = f0;
        fx.moep_guw = kn::kMoepGuN;
        CHECK(same(ops::chooseMoe(q, fx, km, km.silu_mul_x8h), base));
    }
    // another wide variant by WHIRL_MOEP_GUW (g64n256k32: from 256 tokens per expert = 8192 rows)
    ops::MoeFlags f256 = f0;
    ops::setMoepGuw(f256, 5);
    CHECK(ops::moepGuFor(8191, 8, 256, f256, km) == 0 && ops::moepGuFor(8192, 8, 256, f256, km) == 4);
    CHECK(nameOf(ops::chooseMoe(q, f256, km, km.silu_mul_x8h).fgu) == "gemm_moepgu_g64n256k32_mxfp4");
    // GU off (WHIRL_MOEP_GU=0): no moep plan, the wide flag alone changes nothing
    ops::MoeFlags g0 = fw;
    g0.moep_gu = -1;
    ops::MoeFlags g00 = f0;
    g00.moep_gu = -1;
    CHECK(same(ops::chooseMoe(q, g0, km, km.silu_mul_x8h), ops::chooseMoe(q, g00, km, km.silu_mul_x8h)));
    // the R9700 table (no moep kernels): unchanged
    CHECK(same(ops::chooseMoe(q, fw, k97, k97.silu_mul_x8h), ops::chooseMoe(q, f0, k97, k97.silu_mul_x8h)));
}

// v0252-8smoe2 (PLAN V1-147, levers C / D / E; all opt-in): MoeFlags moep_dhs / moep_dna / moep_fq (ops::setMoep2).
//   C moep_dhs: where the plan stores f16 expert outputs (ydh: moep COMB 3, and the fused_gu kind below the row floor with
//     COMB 3) fdn is gemm_moepdhs_<the same variant>; nothing else moves; without the twin in the table: unchanged.
//   D moep_dna: where the plan runs the atomic scatter (comb_atomic: COMB 5, fast) fdn is gemm_moepda_<dna> when dna has
//     the plan's token tile (else the plan's variant); the down grid follows dna's BM; nothing else moves.
//   E moep_fq: on the moep / narrow-f16 plans fgu / fdn become their gemm_moepfq* twins (same variant, same epilogue) where
//     the table has them and gate / down ncols are multiples of 256; nothing else moves (grids, tile lists, bits class).
//   Off (the default) = the plan before the change, field for field; the R9700 table (no moep kernels) is unchanged.
// Fails to build before the change (no MoeFlags::moep_dhs / moep_dna / moep_fq, setMoep2, kernel table twins).
void testOpsMoep2(ops::MoeProblem p, const kn::KernelTable& k97, const kn::KernelTable& k8) {
    const auto kmp = std::make_unique<kn::KernelTable>(k8);  // heap (v0252-f5fix)
    kn::KernelTable& km = *kmp;
    for (int v = 0; v < kn::kMoepGuN; ++v) km.moep_gu[v] = tag(std::string("gemm_moepgu_") + kn::kMoepGu[v].name + "_mxfp4");
    for (int v = 0; v < kn::kMoepDnN; ++v) {
        const std::string vn = kn::kMoepDn[v].name;
        km.moep_dn[v] = tag("gemm_moepdn_" + vn + "_mxfp4");
        km.moep_dh[v] = tag("gemm_moepdh_" + vn + "_mxfp4");
        km.moep_da[v] = tag("gemm_moepda_" + vn + "_mxfp4");
        km.moep_dhs[v] = tag("gemm_moepdhs_" + vn + "_mxfp4");
        // the code object's fused-dequant twins: the variants with one lane per MXFP4 sub-block (BM * BK / 32 % 256 == 0)
        if (kn::kMoepDn[v].bm * kn::kMoepDn[v].bk / 32 % 256 == 0) {
            km.moep_fq_dn[v] = tag("gemm_moepfqdn_" + vn + "_mxfp4");
            km.moep_fq_dh[v] = tag("gemm_moepfqdh_" + vn + "_mxfp4");
            km.moep_fq_dhs[v] = tag("gemm_moepfqdhs_" + vn + "_mxfp4");
            km.moep_fq_da[v] = tag("gemm_moepfqda_" + vn + "_mxfp4");
        }
    }
    for (int v = 0; v < kn::kMoepGuN; ++v)
        if (2 * kn::kMoepGu[v].gr * kn::kMoepGu[v].bk / 32 % 256 == 0) km.moep_fq_gu[v] = tag(std::string("gemm_moepfqgu_") + kn::kMoepGu[v].name + "_mxfp4");
    CHECK(km.moep_fq_gu[0] != nullptr && km.moep_fq_gu[1] != nullptr && km.moep_fq_gu[2] == nullptr && km.moep_fq_dn[2] != nullptr &&
          km.moep_fq_dn[3] != nullptr && km.moep_fq_dn[4] == nullptr);  // = kernels/gfx1151/moe_prefill_fq.hip's entry list
    km.moe_combine4 = tag("moe_combine4");
    km.moe_combine4h = tag("moe_combine4h");
    ops::MoeFlags f0;  // the 8060S default moep plan (g64n64k64 + m128w4n128k64 + moe_combine4)
    ops::setMoep(f0, 1, 3, 1);
    CHECK(!f0.moep_dhs && f0.moep_dna == -1 && !f0.moep_fq);
    ops::setMoep2(f0, false, 0, false);
    CHECK(!f0.moep_dhs && f0.moep_dna == -1 && !f0.moep_fq);
    {
        ops::MoeFlags x = f0;
        ops::setMoep2(x, true, 4, true);
        CHECK(x.moep_dhs && x.moep_dna == 3 && x.moep_fq && std::string(kn::kMoepDn[3].name) == "m128w8n128k64");
    }
    auto same = [](const ops::MoePlan& a, const ops::MoePlan& b) {
        return a.kind == b.kind && a.bits == b.bits && a.bn == b.bn && a.max_tiles == b.max_tiles && a.max_tiles64 == b.max_tiles64 && a.fgu == b.fgu &&
               a.fdn == b.fdn && a.gu_bn == b.gu_bn && a.comb4 == b.comb4 && a.comb_fused == b.comb_fused && a.ydh == b.ydh &&
               a.comb_atomic == b.comb_atomic && a.comb_grid.x == b.comb_grid.x && a.comb_grid.y == b.comb_grid.y && a.gu_pgrid.x == b.gu_pgrid.x &&
               a.dn_pgrid.x == b.dn_pgrid.x && a.skip == b.skip;
    };
    // every field but fdn / fgu / dn_pgrid
    auto rest = [](const ops::MoePlan& a, const ops::MoePlan& b) {
        return a.kind == b.kind && a.bits == b.bits && a.bn == b.bn && a.max_tiles == b.max_tiles && a.max_tiles64 == b.max_tiles64 &&
               a.gu_bn == b.gu_bn && a.comb4 == b.comb4 && a.comb_fused == b.comb_fused && a.ydh == b.ydh && a.comb_atomic == b.comb_atomic &&
               a.comb_grid.x == b.comb_grid.x && a.comb_grid.y == b.comb_grid.y && a.gu_pgrid.x == b.gu_pgrid.x && a.skip == b.skip;
    };
    auto swapPrefix = [](const std::string& n, const std::string& from, const std::string& to) {
        return n.rfind(from, 0) == 0 ? to + n.substr(from.size()) : n;
    };
    // the fused-dequant twin name the table holds for a base kernel name ("" = none)
    auto fqName = [&](const std::string& n) -> std::string {
        for (const char* b : {"gu_", "dn_", "dhs_", "dh_", "da_"}) {
            const std::string from = std::string("gemm_moep") + b, to = std::string("gemm_moepfq") + b;
            if (n.rfind(from, 0) == 0) {
                const std::string t = swapPrefix(n, from, to);
                for (int v = 0; v < kn::kMoepGuN; ++v)
                    if (km.moep_fq_gu[v] != nullptr && nameOf(km.moep_fq_gu[v]) == t) return t;
                for (int v = 0; v < kn::kMoepDnN; ++v)
                    for (const auto* a : {&km.moep_fq_dn, &km.moep_fq_dh, &km.moep_fq_dhs, &km.moep_fq_da})
                        if ((*a)[v] != nullptr && nameOf((*a)[v]) == t) return t;
                return "";
            }
        }
        return "";
    };
    std::size_t n_dhs = 0, n_dna = 0, n_fq = 0;
    for (const unsigned mode : {0u, 1u, 2u}) {  // precise / balance / fast class (prec_dec, moep_comb_ok)
        for (const unsigned comb : {1u, 3u, 5u}) {
            for (const float skip : {0.f, 1.f}) {
                for (const unsigned n : {33u, 1023u, 1024u, 2058u, 4096u, 8192u}) {
                    ops::MoeFlags a = f0;
                    a.prec_dec = mode == 0;
                    a.moep_comb_ok = ops::moepCombFor(mode == 0, mode == 2);
                    a.moep_comb = comb;
                    a.moe_skip_thr = mode == 2 ? skip : 0.f;
                    a.moe_skip_min = 6;
                    ops::MoeProblem q = p;
                    q.n = n;
                    q.cnt_buf = true;
                    const std::string cell = "moep2 mode " + std::to_string(mode) + " comb " + std::to_string(comb) + " skip " +
                                             std::to_string(static_cast<int>(skip)) + " n " + std::to_string(n);
                    const ops::MoePlan pa = ops::chooseMoe(q, a, km, km.silu_mul_x8h);
                    // C
                    ops::MoeFlags c = a;
                    c.moep_dhs = true;
                    const ops::MoePlan pc = ops::chooseMoe(q, c, km, km.silu_mul_x8h);
                    const std::string an = nameOf(pa.fdn);
                    if (pa.ydh && an.rfind("gemm_moepdh_", 0) == 0) {
                        ++n_dhs;
                        CHECK_DETAIL(nameOf(pc.fdn) == swapPrefix(an, "gemm_moepdh_", "gemm_moepdhs_") && pc.fgu == pa.fgu && pc.dn_pgrid.x == pa.dn_pgrid.x &&
                                         rest(pa, pc),
                                     cell + ": C takes gemm_moepdhs of the same variant, nothing else moves");
                    } else {
                        CHECK_DETAIL(same(pa, pc), cell + ": C changes nothing off the f16 expert-output plans");
                    }
                    // D (m128w8n128k64: the plan's token tile 128; m128w4n64k64: another tile -> ignored)
                    ops::MoeFlags d = a, d64 = a;
                    d.moep_dna = 3;
                    d64.moep_dna = 1;
                    const ops::MoePlan pd = ops::chooseMoe(q, d, km, km.silu_mul_x8h);
                    CHECK_DETAIL(same(ops::chooseMoe(q, d64, km, km.silu_mul_x8h), pa), cell + ": D with another token tile is ignored");
                    if (pa.comb_atomic) {
                        ++n_dna;
                        CHECK_DETAIL(nameOf(pd.fdn) == "gemm_moepda_m128w8n128k64_mxfp4" && pd.dn_pgrid.x == pa.dn_pgrid.x && pd.fgu == pa.fgu && rest(pa, pd),
                                     cell + ": D takes gemm_moepda_m128w8n128k64 (same BM: same grid), nothing else moves");
                    } else {
                        CHECK_DETAIL(same(pa, pd), cell + ": D changes nothing off the atomic scatter");
                    }
                    // E
                    ops::MoeFlags e = a;
                    e.moep_fq = true;
                    const ops::MoePlan pe = ops::chooseMoe(q, e, km, km.silu_mul_x8h);
                    const std::string wg = fqName(nameOf(pa.fgu)), wd = fqName(an);
                    const bool on = pa.kind == ops::MoePrefillKind::moep || (pa.kind == ops::MoePrefillKind::fused_gu && pa.ydh);
                    if (on && (!wg.empty() || !wd.empty())) ++n_fq;
                    CHECK_DETAIL(nameOf(pe.fgu) == (on && pa.kind == ops::MoePrefillKind::moep && !wg.empty() ? wg : nameOf(pa.fgu)) &&
                                     nameOf(pe.fdn) == (on && !wd.empty() ? wd : an) && pe.dn_pgrid.x == pa.dn_pgrid.x && rest(pa, pe),
                                 cell + ": E takes the fused-dequant twins of the same kernels, nothing else moves");
                    // all three together: C then E on the same variant
                    ops::MoeFlags all = a;
                    all.moep_dhs = all.moep_fq = true;
                    all.moep_dna = 3;
                    const ops::MoePlan px = ops::chooseMoe(q, all, km, km.silu_mul_x8h);
                    CHECK_DETAIL(nameOf(px.fdn) == (fqName(nameOf(pc.ydh ? pc.fdn : pd.fdn)).empty() || !on ? nameOf(pc.ydh ? pc.fdn : pd.fdn)
                                                                                                         : fqName(nameOf(pc.ydh ? pc.fdn : pd.fdn))) &&
                                     rest(pa, px),
                                 cell + ": C + D + E compose");
                    // the R9700 table (no moep kernels): unchanged
                    CHECK_DETAIL(same(ops::chooseMoe(q, all, k97, k97.silu_mul_x8h), ops::chooseMoe(q, a, k97, k97.silu_mul_x8h)), cell + ": R9700 unchanged");
                }
            }
        }
    }
    CHECK(n_dhs > 0 && n_dna > 0 && n_fq > 0);  // each lever is exercised
    // E: shapes outside whole MXFP4 super-blocks keep the base kernels; a missing twin keeps the base kernel
    {
        ops::MoeProblem q = p;
        q.n = 8192;
        ops::MoeFlags a = f0, e = f0;
        e.moep_fq = true;
        q.gate_ncols = 2048 + 32;
        CHECK(same(ops::chooseMoe(q, e, km, km.silu_mul_x8h), ops::chooseMoe(q, a, km, km.silu_mul_x8h)));
        q.gate_ncols = 2048;
        const auto kxp = std::make_unique<kn::KernelTable>(km);
        kxp->moep_fq_gu[0] = nullptr;
        const ops::MoePlan px = ops::chooseMoe(q, e, *kxp, kxp->silu_mul_x8h);
        CHECK(nameOf(px.fgu) == "gemm_moepgu_g64n64k64_mxfp4" && nameOf(px.fdn) == "gemm_moepfqdn_m128w4n128k64_mxfp4");
        CHECK(ops::moepFqTwin(nullptr, km) == nullptr && ops::moepFqTwin(km.moe_combine4, km) == km.moe_combine4);
    }
}

// v0252-8sdet (PLAN V1-140): COMB 6 = the order-independent fixed-point combine (fast only, opt-in WHIRL_MOEP_COMB=6):
// fdn = gemm_moepdi_<the plan's down variant> (i32 atomic adds of w * y into moe_yd's first n * E words, the last arriver
// of each (token, row block) adds fxValue(sum) + sg * ysh into x), moe_fix_zero clears the sums + counters first; no
// moe_combine4 launch (comb4 false). Plan values: the counters at byte n * E * 4 of moe_yd, the zeroed words / 4, the
// clamp fxLimit(K) (K = 8: 268435440 = the largest float with 8 * lim <= INT32_MAX). Never in precise (also with
// WHIRL_Q8DEC=1) or balance, never below the moep row floor, never without moep_di / moe_fix_zero in the table or with
// K < 2 (moe_yd must hold n * E + n * nrb words), never with moeskip min-keep 0 (a token with no kept slot would never
// be finalized); with moeskip (min >= 1) it keeps the skip path; D (moep_dna) picks the gemm_moepdi twin of that variant;
// COMB 5 and every other COMB unchanged by the new table entries; the R9700 table never takes it.
// Fails to build before the change (no MoePlan::comb_fix / fix_*, KernelTable moep_di / moe_fix_zero).
void testOpsMoepFix(ops::MoeProblem p, const kn::KernelTable& k97, const kn::KernelTable& k8) {
    const auto kmp = std::make_unique<kn::KernelTable>(k8);
    kn::KernelTable& km = *kmp;
    for (int v = 0; v < kn::kMoepGuN; ++v) km.moep_gu[v] = tag(std::string("gemm_moepgu_") + kn::kMoepGu[v].name + "_mxfp4");
    for (int v = 0; v < kn::kMoepDnN; ++v) {
        const std::string vn = kn::kMoepDn[v].name;
        km.moep_dn[v] = tag("gemm_moepdn_" + vn + "_mxfp4");
        km.moep_dh[v] = tag("gemm_moepdh_" + vn + "_mxfp4");
        km.moep_da[v] = tag("gemm_moepda_" + vn + "_mxfp4");
    }
    km.moe_combine4 = tag("moe_combine4");
    km.moe_combine4h = tag("moe_combine4h");
    const auto knp = std::make_unique<kn::KernelTable>(km);  // the table before the change (no COMB 6 kernels)
    for (int v = 0; v < kn::kMoepDnN; ++v) km.moep_di[v] = tag(std::string("gemm_moepdi_") + kn::kMoepDn[v].name + "_mxfp4");
    km.moe_fix_zero = tag("moe_fix_zero");
    ops::MoeFlags f0;
    ops::setMoep(f0, 1, 3, 1);
    ops::MoeProblem q = p;
    q.n = 4096;
    ops::MoeFlags ff = f0;
    ff.moep_comb_ok = ops::moepCombFor(false, true);
    ff.moep_comb = 5;
    const ops::MoePlan p5 = ops::chooseMoe(q, ff, km, km.silu_mul_x8h);
    CHECK(p5.comb_atomic && !p5.comb_fix && nameOf(p5.fdn) == "gemm_moepda_m128w4n128k64_mxfp4");
    ff.moep_comb = 6;
    const ops::MoePlan pl = ops::chooseMoe(q, ff, km, km.silu_mul_x8h);
    CHECK_DETAIL(pl.kind == ops::MoePrefillKind::moep && pl.comb_fix && !pl.comb_atomic && !pl.comb_fused && !pl.ydh && !pl.comb4 &&
                     pl.bits == ops::MoeBits::moe_lossy_gemm && nameOf(pl.fdn) == "gemm_moepdi_m128w4n128k64_mxfp4",
                 "COMB 6 fast: gemm_moepdi, no combine launch, lossy class");
    CHECK(pl.fgu == p5.fgu && pl.dn_pgrid.x == p5.dn_pgrid.x && pl.bn == p5.bn && pl.max_tiles == p5.max_tiles && pl.gu_pgrid.x == p5.gu_pgrid.x);
    const std::uint64_t words = 4096ull * 2048 + 4096ull * (2048 / 128);
    CHECK_DETAIL(pl.fix_cnt_off == 4096ull * 2048 * 4 && pl.fix_words4 == words / 4 && pl.fix_lim == 268435440.f &&
                     static_cast<double>(pl.fix_lim) * 8 <= 2147483647.0 && pl.fix_zero_grid.x == std::min<std::uint64_t>((words / 4 + 255) / 256, 1024),
                 "COMB 6 plan values: counter offset " + std::to_string(pl.fix_cnt_off) + " words4 " + std::to_string(pl.fix_words4) + " lim " +
                     std::to_string(pl.fix_lim) + " zero grid " + std::to_string(pl.fix_zero_grid.x));
    // the other combines do not move with the new table entries
    for (const unsigned cm : {0u, 1u, 2u, 3u, 4u, 5u}) {
        ops::MoeFlags a = ff;
        a.moep_comb = cm;
        q.cnt_buf = cm == 2 || cm == 4;
        const ops::MoePlan x = ops::chooseMoe(q, a, km, km.silu_mul_x8h), y = ops::chooseMoe(q, a, *knp, knp->silu_mul_x8h);
        CHECK_DETAIL(!x.comb_fix && x.fdn == y.fdn && x.fgu == y.fgu && x.comb4 == y.comb4 && x.comb_atomic == y.comb_atomic && x.ydh == y.ydh &&
                         x.comb_fused == y.comb_fused && x.bits == y.bits && x.skip == y.skip && x.comb_grid.x == y.comb_grid.x,
                     "COMB " + std::to_string(cm) + " unchanged by the COMB 6 table entries");
    }
    q.cnt_buf = false;
    // not fast: the plain moep down + moe_combine (the bitwise fallback, as COMB 5 there)
    for (const int mode : {0, 1, 2}) {  // precise, balance, precise with prec_dec cleared (WHIRL_Q8DEC=1)
        ops::MoeFlags a = f0;
        a.prec_dec = mode == 0;
        a.moep_comb_ok = ops::moepCombFor(mode != 1, false);
        a.moep_comb = 6;
        const ops::MoePlan x = ops::chooseMoe(q, a, km, km.silu_mul_x8h);
        CHECK_DETAIL(x.kind == ops::MoePrefillKind::moep && !x.comb_fix && !x.comb_atomic && x.bits == ops::MoeBits::moe_f16_gemm &&
                         nameOf(x.fdn) == "gemm_moepdn_m128w4n128k64_mxfp4",
                     "COMB 6 refused outside fast, mode " + std::to_string(mode));
    }
    // kernels missing: not taken (the plain moep down)
    {
        const ops::MoePlan x = ops::chooseMoe(q, ff, *knp, knp->silu_mul_x8h);
        CHECK(!x.comb_fix && !x.comb_atomic && nameOf(x.fdn) == "gemm_moepdn_m128w4n128k64_mxfp4");
        const auto kz = std::make_unique<kn::KernelTable>(km);
        kz->moe_fix_zero = nullptr;
        CHECK(!ops::chooseMoe(q, ff, *kz, kz->silu_mul_x8h).comb_fix);
    }
    // below the moep row floor: no moep, no COMB 6
    {
        ops::MoeProblem s = q;
        s.n = 128;
        const ops::MoePlan x = ops::chooseMoe(s, ff, km, km.silu_mul_x8h);
        CHECK(x.kind == ops::MoePrefillKind::fused_gu && !x.comb_fix && !x.comb4 && !x.ydh);
    }
    // K < 2: moe_yd cannot hold the sums + counters
    {
        ops::MoeProblem s = q;
        s.K = 1;
        CHECK(!ops::chooseMoe(s, ff, km, km.silu_mul_x8h).comb_fix);
    }
    // moeskip: kept with min-keep >= 1 (the skip path stays), refused with min-keep 0
    {
        ops::MoeFlags a = ff;
        a.moe_skip_thr = 1.f;
        a.moe_skip_min = 6;
        const ops::MoePlan x = ops::chooseMoe(q, a, km, km.silu_mul_x8h);
        CHECK_DETAIL(x.comb_fix && x.skip && nameOf(x.fdn) == "gemm_moepdi_m128w4n128k64_mxfp4", "COMB 6 with moeskip keeps the skip path");
        a.moe_skip_min = 0;
        CHECK(!ops::chooseMoe(q, a, km, km.silu_mul_x8h).comb_fix);
    }
    // D: the down variant of the atomic forms (same token tile) -> its gemm_moepdi twin
    {
        ops::MoeFlags a = ff;
        a.moep_dna = 3;
        const ops::MoePlan x = ops::chooseMoe(q, a, km, km.silu_mul_x8h);
        CHECK(x.comb_fix && nameOf(x.fdn) == "gemm_moepdi_m128w8n128k64_mxfp4" && x.dn_pgrid.x == pl.dn_pgrid.x);
    }
    // the R9700 table: never
    CHECK(!ops::chooseMoe(q, ff, k97, k97.silu_mul_x8h).comb_fix);
    // v0252-8sdet round 3: the fused-dequant twin (the shipping fast plan has moep_fq on): COMB 6 takes gemm_moepfqdi_<v> as
    // COMB 5 takes gemm_moepfqda_<v>; a variant without the twin keeps gemm_moepdi_<v>; and for EVERY COMB x fq x moeskip the
    // down handle exists and the step that launches it matches its argument list: downInCombine() exactly for the fused
    // (moep_dc / moep_dhc), atomic (moep_da / moep_fq_da) and fixed-point (moep_di / moep_fq_di) kernels, the plain-list
    // kernels (moep_dn / dh / dhs and their fq twins) otherwise
    {
        const auto kf = std::make_unique<kn::KernelTable>(km);
        for (int v = 0; v < kn::kMoepDnN; ++v)
            if (kn::kMoepDn[v].bm * kn::kMoepDn[v].bk / 32 % 256 == 0) {
                const std::string vn = kn::kMoepDn[v].name;
                kf->moep_fq_dn[v] = tag("gemm_moepfqdn_" + vn + "_mxfp4");
                kf->moep_fq_dh[v] = tag("gemm_moepfqdh_" + vn + "_mxfp4");
                kf->moep_fq_da[v] = tag("gemm_moepfqda_" + vn + "_mxfp4");
                kf->moep_fq_di[v] = tag("gemm_moepfqdi_" + vn + "_mxfp4");
            }
        for (int v = 0; v < kn::kMoepDnN; ++v) {
            kf->moep_dc[v] = tag(std::string("gemm_moepdc_") + kn::kMoepDn[v].name + "_mxfp4");
            kf->moep_dhc[v] = tag(std::string("gemm_moepdhc_") + kn::kMoepDn[v].name + "_mxfp4");
        }
        ops::MoeFlags a = ff;
        a.moep_fq = true;
        const ops::MoePlan x6 = ops::chooseMoe(q, a, *kf, kf->silu_mul_x8h);
        CHECK_DETAIL(x6.comb_fix && x6.downInCombine() && nameOf(x6.fdn) == "gemm_moepfqdi_m128w4n128k64_mxfp4", "COMB 6 + fq: gemm_moepfqdi, " + nameOf(x6.fdn));
        a.moep_comb = 5;
        CHECK(nameOf(ops::chooseMoe(q, a, *kf, kf->silu_mul_x8h).fdn) == "gemm_moepfqda_m128w4n128k64_mxfp4");
        a.moep_comb = 6;
        kf->moep_fq_di[2] = nullptr;  // no twin of this variant: the base kernel, never a null launch
        CHECK(nameOf(ops::chooseMoe(q, a, *kf, kf->silu_mul_x8h).fdn) == "gemm_moepdi_m128w4n128k64_mxfp4");
        kf->moep_fq_di[2] = tag("gemm_moepfqdi_m128w4n128k64_mxfp4");
        auto inFam = [&](Fn f, std::initializer_list<const std::array<Fn, kn::kMoepDnN>*> fams) {
            for (const auto* a2 : fams)
                for (const Fn g : *a2)
                    if (g != nullptr && g == f) return true;
            return false;
        };
        int cells = 0;
        for (const unsigned cm : {0u, 1u, 2u, 3u, 4u, 5u, 6u})
            for (const bool fq : {false, true})
                for (const float sk : {0.f, 1.f})
                    for (const unsigned n : {1024u, 4096u, 8192u}) {
                        ops::MoeFlags b = ff;
                        b.moep_comb = cm;
                        b.moep_fq = fq;
                        b.moe_skip_thr = sk;
                        b.moe_skip_min = 6;
                        ops::MoeProblem qq = q;
                        qq.n = n;
                        qq.cnt_buf = true;
                        const ops::MoePlan pp = ops::chooseMoe(qq, b, *kf, kf->silu_mul_x8h);
                        if (pp.kind != ops::MoePrefillKind::moep) continue;
                        ++cells;
                        const bool comb_list = inFam(pp.fdn, {&kf->moep_dc, &kf->moep_dhc, &kf->moep_da, &kf->moep_fq_da, &kf->moep_di, &kf->moep_fq_di});
                        const bool plain_list = inFam(pp.fdn, {&kf->moep_dn, &kf->moep_dh, &kf->moep_dhs, &kf->moep_fq_dn, &kf->moep_fq_dh, &kf->moep_fq_dhs});
                        CHECK_DETAIL(pp.fdn != nullptr && pp.downInCombine() == comb_list && pp.downInCombine() != plain_list,
                                     "COMB " + std::to_string(cm) + " fq " + std::to_string(fq) + " skip " + std::to_string(static_cast<int>(sk)) + " n " +
                                         std::to_string(n) + ": down " + nameOf(pp.fdn) + " launched by the " +
                                         (pp.downInCombine() ? "combine" : "grouped-GEMM") + " step");
                    }
        CHECK(cells >= 7 * 2 * 2);
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
    // v0252-f5fix: static / heap tables (a kn::KernelTable is ~140 KB; see testOpsMoeSkip)
    static const kn::KernelTable k97 = moeTable(false), k8 = moeTable(true);
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
        const auto kmp = std::make_unique<kn::KernelTable>(k8);  // heap (v0252-f5fix)
        kn::KernelTable& km = *kmp;
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
        const auto kmp = std::make_unique<kn::KernelTable>(k8);  // heap (v0252-f5fix)
        kn::KernelTable& km = *kmp;
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
    testOpsMoeFuse(p, k97, k8);  // v0252-c3port (fails before the port: chooseMoeFuse is a stub)
    testOpsMoeSegJoin(p, k97, k8);  // v0252-precfix (fails before the fix: moeSegJoinMin is a stub returning 16)
    testOpsMoeYdh(p, k97, k8);   // v0252-moedf16 (fails before the change: the fp8 kind never sets ydh)
    testOpsMoeSkip(p);  // v0252-f5
    testOpsMoeCombRows(p);  // v0252-combfix (fails before the fix: the row floor drops COMB 3 below 1024 rows)
    testOpsMoeGuw(p, k97, k8);  // v0252-8smoe (fails to build before the change: no MoeFlags::moep_guw)
    testOpsMoep2(p, k97, k8);  // v0252-8smoe2 (fails to build before the change: no MoeFlags::moep_dhs / moep_dna / moep_fq)
    testOpsMoepFix(p, k97, k8);  // v0252-8sdet (fails to build before the change: no MoePlan::comb_fix, KernelTable moep_di)
}

}  // namespace
