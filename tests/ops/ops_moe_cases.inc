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
    if (g1151) {
        k.moe_tiles = tag("moe_tiles");
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
    testOpsMoeFuse(p, k97, k8);  // v0252-c3port (fails before the port: chooseMoeFuse is a stub)
    testOpsMoeSegJoin(p, k97, k8);  // v0252-precfix (fails before the fix: moeSegJoinMin is a stub returning 16)
    testOpsMoeYdh(p, k97, k8);   // v0252-moedf16 (fails before the change: the fp8 kind never sets ydh)
}

}  // namespace
