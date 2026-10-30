// arch025-n8b1: cases moved verbatim from tests/ops/ops_tests.cpp; included by whirl-ops-tests (tests/ops/ops_tests.cpp, same
// cases, same order) and by the module's isolation program tests/iso/ops.matmul/ (ARCH-1 8.5).
// SPDX-License-Identifier: Apache-2.0

#include "whirl/ops/matmul.h"
#include "../test_check.h"
#include "ops_fakes.h"
#include "ops_mat.h"

#include <array>
#include <cstdint>
#include <string>

namespace {

namespace ops = whirl::ops;

void testBitClasses() {
    CHECK(ops::bitClassOf(ops::MatmulPath::gemvq) == ops::BitClass::q8_row);
    CHECK(ops::bitClassOf(ops::MatmulPath::gemv1) == ops::BitClass::f32_row);
    CHECK(ops::bitClassOf(ops::MatmulPath::float_gemv) == ops::BitClass::f32_row);
    CHECK(ops::bitClassOf(ops::MatmulPath::gemm_fp8) == ops::BitClass::fp8_gemm);
    CHECK(ops::bitClassOf(ops::MatmulPath::gemm_f16) == ops::BitClass::f16_gemm);
    CHECK(ops::preciseClass(ops::BitClass::f16_gemm));
    CHECK(!ops::preciseClass(ops::BitClass::q8_row) && !ops::preciseClass(ops::BitClass::f32_row) && !ops::preciseClass(ops::BitClass::fp8_gemm));
}

// REVIEW-ARCH4 P3-1: the one precise-class rule every row-count gate asks (matmul.h preciseRows / preciseAllows).
// Its truth table, exhaustively over the row ranges the callers pass (gv_max 16 / 32, small_max 16 / 32).
void testPreciseRule() {
    const ops::BitClass all[] = {ops::BitClass::q8_row, ops::BitClass::f32_row, ops::BitClass::f16_gemm, ops::BitClass::fp8_gemm};
    for (std::uint32_t rmax : {16u, 32u})
        for (std::uint32_t n = 1; n <= 64; ++n)
            for (bool pd : {false, true}) {
                const bool rows = pd && n <= rmax;
                CHECK(ops::preciseRows(pd, n, rmax) == rows);
                for (ops::BitClass b : all) CHECK(ops::preciseAllows(b, pd, n, rmax) == (!rows || b == ops::BitClass::f16_gemm));
            }
    // the boundary rows the callers hit: the last precise row and the first one past it
    CHECK(ops::preciseRows(true, 16, 16) && !ops::preciseRows(true, 17, 16));
    CHECK(!ops::preciseAllows(ops::BitClass::q8_row, true, 32, 32) && ops::preciseAllows(ops::BitClass::q8_row, true, 33, 32));
}

// the matmul path table: rows x flags -> path; precise rows only ever get the precise class
void testMatmulPath() {
    const kn::KernelTable k = fakeTable(true, true, true, true);
    ops::MatmulFlags bal;
    bal.gv_max = 16;
    bal.fp8_prefill = true;
    bal.fp8_mask = 7;
    bal.mm_class = 1;
    ops::MatmulFlags pre;
    pre.prec_dec = true;
    pre.gv_max = 16;
    const auto q4 = mat(GgmlType::q4_k, 4096, 5120), mx = mat(GgmlType::mxfp4, 4096, 5120, false, true), f32 = mat(GgmlType::f32, 48, 5120);
    using P = ops::MatmulPath;
    // balance / fast
    CHECK(ops::chooseMatmulPath(q4, 1, bal, k) == P::gemvq);
    CHECK(ops::chooseMatmulPath(q4, 16, bal, k) == P::gemvq);
    CHECK(ops::chooseMatmulPath(q4, 17, bal, k) == P::gemm_f16);
    CHECK(ops::chooseMatmulPath(mx, 512, bal, k) == P::gemm_fp8);
    CHECK(ops::chooseMatmulPath(mat(GgmlType::mxfp4, 4096, 5120), 512, bal, k) == P::gemm_f16);  // no row exponents
    CHECK(ops::chooseMatmulPath(f32, 1, bal, k) == P::gemv1);
    CHECK(ops::chooseMatmulPath(f32, 4, bal, k) == P::float_gemv);
    CHECK(ops::chooseMatmulPath(f32, 17, bal, k) == P::gemm_f16);
    ops::MatmulFlags fg = bal;
    fg.float_gemv = true;
    CHECK(ops::chooseMatmulPath(q4, 1, fg, k) == P::gemv1);
    CHECK(ops::chooseMatmulPath(q4, 8, fg, k) == P::gemm_f16);
    ops::MatmulFlags wide = bal;
    wide.gv_max = 32;
    CHECK(ops::chooseMatmulPath(q4, 32, wide, k) == P::gemvq);
    // fp8 only for the matmul classes in the mask
    ops::MatmulFlags m2 = bal;
    m2.mm_class = 2;
    m2.fp8_mask = 1;
    CHECK(ops::chooseMatmulPath(mx, 512, m2, k) == P::gemm_f16);
    // precise: every decode / verify row count on the f16 GEMM, whatever the kernel set or weights
    for (std::uint32_t n : {1u, 2u, 7u, 16u}) {
        for (const auto& w : {q4, mx, f32}) {
            const P p = ops::chooseMatmulPath(w, n, pre, k);
            CHECK(p == P::gemm_f16 && ops::preciseClass(ops::bitClassOf(p)));
        }
    }
    // precise prefill (n > gv_max) never takes fp8 when the plan has it off
    CHECK(ops::chooseMatmulPath(mx, 512, pre, k) == P::gemm_f16);
    // floatGemvN: f32 / f16 weights without an int8 GEMV, n = 2..gv_max
    CHECK(ops::floatGemvN(f32, 2, 16, k) && ops::floatGemvN(f32, 16, 16, k) && !ops::floatGemvN(f32, 1, 16, k) && !ops::floatGemvN(f32, 17, 16, k));
    CHECK(!ops::floatGemvN(q4, 4, 16, k));
}

// the GEMV tile table: n x matrices x tuned variant -> kernel, grouping, grid
void testGemvTile() {
    ops::GemvR r{};
    r.fill(1);
    for (int n = 6; n <= 16; ++n) r[n] = 2;  // the gfx1201 profile row
    std::array<ops::GemvR, kn::kNTypes> w{};
    ops::GemvR wh{};
    const ops::GemvTune tn{&r, &w, &wh, 0};
    const kn::KernelTable k = fakeTable(true, true, false, false);
    const std::string q4 = std::to_string(static_cast<int>(GgmlType::q4_k)), q6 = std::to_string(static_cast<int>(GgmlType::q6_k)),
                      mx = std::to_string(static_cast<int>(GgmlType::mxfp4));
    const ops::MatShape a = mat(GgmlType::q4_k, 6144, 5120), b = mat(GgmlType::q4_k, 1024, 5120), c = mat(GgmlType::q4_k, 1024, 5120);
    const ops::MatShape abc[3] = {a, b, c};
    // n = 1: the one-token entry, grouped through GvArgs (gv_grp), 8 rows per block
    auto p = ops::chooseGemvTile(abc, 3, 1, k, tn);
    CHECK(nameOf(p.f) == "gemvq_" + q4 && p.grouped && p.rows_per_block == 8 && p.grid[0].x == 768 + 128 + 128 && p.block.x == 256);
    p = ops::chooseGemvTile(abc, 1, 1, k, tn);
    CHECK(nameOf(p.f) == "gemvq_" + q4 && p.grouped && p.grid[0].x == 768);
    // n = 4: plain multi-token; a group takes the "g" twin
    p = ops::chooseGemvTile(abc, 3, 4, k, tn);
    CHECK(nameOf(p.f) == "ggemvq_nt4_" + q4 && p.grouped && p.grid[0].x == 1024);
    p = ops::chooseGemvTile(abc, 1, 4, k, tn);
    CHECK(nameOf(p.f) == "gemvq_nt4_" + q4 && !p.grouped && p.grid[0].x == 768);
    // n = 8: the multi-row R = 2 entry of the profile row (16 rows per block)
    p = ops::chooseGemvTile(abc, 3, 8, k, tn);
    CHECK(nameOf(p.f) == "ggemvq_nt8r2_" + q4 && p.grouped && p.rows_per_block == 16 && p.grid[0].x == 384 + 64 + 64);
    // n = 12 > gvNmax (11 for q4_k): no twin, one launch per matrix with its own grid
    p = ops::chooseGemvTile(abc, 3, 12, k, tn);
    CHECK(nameOf(p.f) == "gemvq_nt12r2_" + q4 && !p.grouped && p.grid[0].x == 384 && p.grid[1].x == 64 && p.grid[2].x == 64);
    // mxfp4: twins up to 13 rows
    const ops::MatShape m3[3] = {mat(GgmlType::mxfp4, 6144, 5120), mat(GgmlType::mxfp4, 1024, 5120), mat(GgmlType::mxfp4, 1024, 5120)};
    p = ops::chooseGemvTile(m3, 3, 13, k, tn);
    CHECK(nameOf(p.f) == "ggemvq_nt13r2_" + mx && p.grouped);
    CHECK(ops::gvNmax(GgmlType::mxfp4, 0) == 13 && ops::gvNmax(GgmlType::q4_k, 0) == 11 && ops::gvNmax(GgmlType::q4_k, 5) == 5);
    // WHIRL_GV_NMAX override
    const ops::GemvTune tn5{&r, &w, &wh, 5};
    p = ops::chooseGemvTile(abc, 3, 6, k, tn5);
    CHECK(!p.grouped && nameOf(p.f) == "gemvq_nt6r2_" + q4);
    // a tuned gemvw variant wins over the multi-row entry (same class, 16 rows per block for v1)
    w[ti(GgmlType::q4_k)][8] = 1;
    p = ops::chooseGemvTile(abc, 3, 8, k, tn);
    CHECK(nameOf(p.f) == "ggemvw_nt8v1_" + q4 && p.grouped && p.rows_per_block == 16);
    // ... but not where the code object lacks it (mxfp4 has no gemvw here): back to the multi-row entry
    w[ti(GgmlType::mxfp4)][8] = 1;
    p = ops::chooseGemvTile(m3, 1, 8, k, tn);
    CHECK(nameOf(p.f) == "gemvq_nt8r2_" + mx);
    // the q6_k head reads its own variant table (gemv_w_head), never gemv_w
    w[ti(GgmlType::q6_k)][4] = 1;
    const ops::MatShape head = mat(GgmlType::q6_k, 248320, 5120, true);
    p = ops::chooseGemvTile(&head, 1, 4, k, tn);
    CHECK(nameOf(p.f) == "gemvq_nt4_" + q6 && p.grid[0].x == 248320 / 8);
    // without gv_grp a one-token launch is per matrix
    const kn::KernelTable k0 = fakeTable(false, false, false, false);
    p = ops::chooseGemvTile(abc, 3, 1, k0, tn);
    CHECK(!p.grouped && p.grid[0].x == 768 && p.grid[1].x == 128);
    // grids round up: a matrix whose rows are not a multiple of the rows per block keeps its last rows
    const ops::MatShape odd[2] = {mat(GgmlType::q4_k, 1001, 5120), mat(GgmlType::q4_k, 17, 5120)};
    p = ops::chooseGemvTile(odd, 2, 1, k, tn);
    CHECK(p.grouped && p.grid[0].x == 126 + 3);
    p = ops::chooseGemvTile(odd, 2, 12, k, tn);  // R = 2: 16 rows per block, one launch per matrix
    CHECK(!p.grouped && p.grid[0].x == 63 && p.grid[1].x == 2);
    // every plan is class q8_row
    CHECK(p.bits == ops::BitClass::q8_row);
}

void testGemvWide() {
    const kn::KernelTable kx = fakeTable(true, true, true, false), k = fakeTable(true, true, false, false);
    const auto q4 = mat(GgmlType::q4_k, 4096, 5120), head = mat(GgmlType::q6_k, 248320, 5120, true), hq4 = mat(GgmlType::q4_k, 248320, 5120, true);
    CHECK(ops::chooseGemvWide(q4, 3, kx, false) == ops::GemvWide::gemvx);
    CHECK(ops::chooseGemvWide(q4, 3, k, false) == ops::GemvWide::pieces);
    CHECK(ops::chooseGemvWide(head, 1, kx, false) == ops::GemvWide::head_split);
    CHECK(ops::chooseGemvWide(head, 2, kx, false) == ops::GemvWide::pieces);
    CHECK(ops::chooseGemvWide(hq4, 1, kx, false) == ops::GemvWide::pieces);  // a q4_k head without WHIRL_HEAD_GEMVX
    CHECK(ops::chooseGemvWide(hq4, 1, kx, true) == ops::GemvWide::gemvx);
    CHECK(ops::gemvxGrid(4096).x == 256 && ops::gemvxGrid(4097).x == 257 && ops::headSplitGrid(248320).x == 7760);
}

// ---- f16 / fp8 GEMM (K-2)
kn::KernelTable gemmTable(bool gemvh, bool gemms, bool gemmh) {
    kn::KernelTable k;
    for (GgmlType t : {GgmlType::q4_k, GgmlType::mxfp4, GgmlType::f16}) {
        const std::string s = std::to_string(static_cast<int>(t));
        for (std::size_t c = 0; c < kn::kGemmCfgs.size(); ++c) k.gemmc[c][ti(t)] = tag("gemm_c" + std::to_string(c) + "_" + s);
        if (t != GgmlType::f16) k.dequant_f16[ti(t)] = tag("dequant_f16_" + s);
        if (gemvh) {
            k.gemvh[ti(t)] = tag("gemvh_" + s);
            k.gemvh2[ti(t)] = tag("gemvh2_" + s);
        }
        if (gemms && t != GgmlType::f16)
            for (std::size_t c : {0u, 1u, 4u}) k.gemms[c][ti(t)] = tag("gemms_c" + std::to_string(c) + "_" + s);
    }
    if (gemmh) k.gemmh_f16 = tag("gemmh_f16");
    for (std::size_t c = 0; c < kn::kGemm8Cfgs.size(); ++c) {
        k.gemm8[c] = tag("gemm8_c" + std::to_string(c));
        k.gemm8h[c] = tag("gemm8h_c" + std::to_string(c));
        k.gemm8t[c] = tag("gemm8t_c" + std::to_string(c));
    }
    return k;
}

void testOpsGemm() {
    const std::uint32_t G = static_cast<std::uint32_t>(kn::kGemmCfgs.size());
    auto q4 = mat(GgmlType::q4_k, 5120, 5120);
    q4.row_bytes = 5120 / 256 * 144;
    auto f16 = mat(GgmlType::f16, 4096, 5120);
    f16.row_bytes = 5120 * 2;
    ops::GemmFlags pre;
    pre.prec_dec = true;
    ops::GemmFlags bal;
    // precise decode rows: gemvh (16 rows) / gemvh2 (17..32), one block per 16 rows, 32 threads; the profile's
    // grid where it says so
    kn::KernelTable k = gemmTable(true, true, true);
    auto p = ops::chooseGemmF16(q4, 4, 0, pre, k);
    CHECK(p.kind == ops::GemmKind::gemvh && nameOf(p.f).rfind("gemvh_", 0) == 0 && p.grid.x == 320 && p.block.x == 32 && p.bits == ops::BitClass::f16_gemm);
    ops::GemmFlags pre8 = pre;
    pre8.gemvh_cap = 64;  // fewer blocks than the 320 row tiles: 5 tiles per block
    CHECK(ops::chooseGemmF16(q4, 4, 0, pre8, k).grid.x == kn::gemvhGridCap(5120, 64) && kn::gemvhGridCap(5120, 64) == 64);
    // WHIRL_PREC_GEMVH=0: the precise gemms choice (narrowest tile that holds n, then the smaller row block)
    ops::GemmFlags nog = pre;
    nog.prec_gemvh = false;
    p = ops::chooseGemmF16(q4, 4, 0, nog, k);
    CHECK(p.kind == ops::GemmKind::gemms && nameOf(p.f).rfind("gemms_c4_", 0) == 0);  // c4 = 128 x 16
    CHECK(ops::precChoice(q4, 4, 0, nog, k) == 2 * G + 4);
    CHECK(ops::precChoice(q4, 20, 0, nog, k) == 2 * G + 0);  // 32-token tiles: c0 (64 x 32) over c1 (128 x 32)
    ops::GemmFlags tie = nog;
    tie.gemms_tie_larger_bm = true;
    CHECK(ops::precChoice(q4, 20, 0, tie, k) == 2 * G + 1);  // the 8060S rule: ties to the larger row block
    ops::GemmFlags gs = nog;
    gs.prec_gs = 1;
    CHECK(ops::precChoice(q4, 4, 0, gs, k) == 2 * G + 1);    // WHIRL_PREC_GS
    // no gemms for the type (f16 weights): the tuned fused config, never the dequant choice
    CHECK(ops::precChoice(f16, 4, static_cast<std::uint8_t>(G + 3), nog, k) == 3);
    // prefill: the tuned choice; a dequant choice dequantizes into w16 then runs gemm_c on f16 (or gemmh for n >= 512)
    p = ops::chooseGemmF16(q4, 256, 2, bal, k);
    CHECK(p.kind == ops::GemmKind::gemmc && !p.dequant && nameOf(p.f) == "gemm_c2_" + std::to_string(static_cast<int>(GgmlType::q4_k)));
    const kn::GemmCfg c2 = kn::kGemmCfgs[2];
    CHECK(p.grid.x == (256 + c2.bn - 1) / c2.bn && p.grid.y == (5120 + c2.bm - 1) / c2.bm && p.block.x == static_cast<unsigned>(c2.nth));
    p = ops::chooseGemmF16(q4, 256, static_cast<std::uint8_t>(G + 2), bal, k);
    CHECK(p.kind == ops::GemmKind::gemmc && p.dequant && nameOf(p.dq).rfind("dequant_f16_", 0) == 0 &&
          nameOf(p.f) == "gemm_c2_" + std::to_string(static_cast<int>(GgmlType::f16)));
    p = ops::chooseGemmF16(q4, 1024, static_cast<std::uint8_t>(G + 2), bal, k);
    CHECK(p.kind == ops::GemmKind::gemmh && p.dequant && nameOf(p.f) == "gemmh_f16" && p.grid.x == 4 && p.grid.y == 40);
    // f16 weights with padded rows are not gemmh's layout
    auto f16p = f16;
    f16p.row_bytes = 5120 * 2 + 64;
    CHECK(ops::chooseGemmF16(f16p, 1024, 1, bal, k).kind == ops::GemmKind::gemmc);
    CHECK(ops::chooseGemmF16(f16, 1024, 1, bal, k).kind == ops::GemmKind::gemmh);
    // a tuned gemms slot runs with the code object's geometry
    k.gemms_geom[1] = kn::GemmCfg{64, 16, 128};
    p = ops::chooseGemmF16(q4, 40, static_cast<std::uint8_t>(2 * G + 1), bal, k);
    CHECK(p.kind == ops::GemmKind::gemms && p.grid.x == 3 && p.grid.y == 80 && p.block.x == 128);
    // fp8: the configuration per batch, f16-out twin where present, tiled set when on
    auto g = ops::chooseGemm8(4096, 32, false, false, k);
    CHECK(nameOf(g.f) == "gemm8_c3" && g.bits == ops::BitClass::fp8_gemm);
    CHECK(nameOf(ops::chooseGemm8(4096, 512, true, false, k).f) == "gemm8h_c0");
    CHECK(nameOf(ops::chooseGemm8(4096, 40, true, true, k).f) == "gemm8t_c4");  // no gemm8th: the plain tiled one
    CHECK(ops::gemm8Cfg(64) == 3 && ops::gemm8Cfg(65) == 1 && ops::gemm8Cfg(257) == 0 && ops::gemm8tCfg(48) == 4 && ops::gemm8tCfg(193) == 0);
}

}  // namespace
