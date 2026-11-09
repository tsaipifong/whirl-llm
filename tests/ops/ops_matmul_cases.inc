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

// feat026-ggemv / v0251-a6port: the gemvk selector (TuneTable::gemv_k through GemvTune::gemv_k) and its auto rule;
// nullptr or an all-0 row = the 0.2.5 gemvq path; the 8060S dense profile row routes 5 / 6 tokens to v7
void testGemvK() {
    ops::GemvR r{};
    r.fill(1);
    for (int n = 2; n <= 16; ++n) r[n] = 2;  // the gfx1151 profile row
    std::array<ops::GemvR, kn::kNTypes> w{};
    ops::GemvR wh{}, gk{};
    kn::KernelTable k = fakeTable(true, true, false, false);
    const std::string mx = std::to_string(static_cast<int>(GgmlType::mxfp4)), q4 = std::to_string(static_cast<int>(GgmlType::q4_k));
    for (int nt = 2; nt <= kn::kGemvkMaxNt; ++nt)
        for (int v = 1; v < kn::kNGemvk; ++v) {
            k.gemvk[v][nt - 2] = tag("gemvk_nt" + std::to_string(nt) + "v" + std::to_string(v));
            k.gemvk_g[v][nt - 2] = tag("ggemvk_nt" + std::to_string(nt) + "v" + std::to_string(v));
        }
    const ops::MatShape s3[3] = {mat(GgmlType::mxfp4, 6144, 5120), mat(GgmlType::mxfp4, 1024, 5120), mat(GgmlType::mxfp4, 1024, 5120)};
    // no table / all 0: the old multi-row entry
    const ops::GemvTune t_null{&r, &w, &wh, 0};
    auto p = ops::chooseGemvTile(s3, 3, 4, k, t_null);
    CHECK(nameOf(p.f) == "ggemvq_nt4r2_" + mx && p.rows_per_block == 16);
    const ops::GemvTune tn{&r, &w, &wh, 0, &gk};
    p = ops::chooseGemvTile(s3, 3, 4, k, tn);
    CHECK(nameOf(p.f) == "ggemvq_nt4r2_" + mx && p.rows_per_block == 16 && p.grid[0].x == 384 + 64 + 64);
    // forced v2 (KW 4, 32 rows): twin for a group, plain for one matrix, grids by 32 rows
    gk[4] = 2;
    p = ops::chooseGemvTile(s3, 3, 4, k, tn);
    CHECK(nameOf(p.f) == "ggemvk_nt4v2" && p.grouped && p.rows_per_block == 32 && p.grid[0].x == 192 + 32 + 32);
    p = ops::chooseGemvTile(s3, 1, 4, k, tn);
    CHECK(nameOf(p.f) == "gemvk_nt4v2" && !p.grouped && p.grid[0].x == 192);
    // v4 (KW 2): 64 rows per block; v6 / v7 (KW 8): 16
    gk[4] = 4;
    p = ops::chooseGemvTile(s3, 3, 4, k, tn);
    CHECK(nameOf(p.f) == "ggemvk_nt4v4" && p.rows_per_block == 64 && p.grid[0].x == 96 + 16 + 16);
    gk[4] = 7;
    p = ops::chooseGemvTile(s3, 3, 4, k, tn);
    CHECK(nameOf(p.f) == "ggemvk_nt4v7" && p.rows_per_block == 16 && p.grid[0].x == 384 + 64 + 64);
    // other types, other n and n > 8 (no entry) keep the old path
    const ops::MatShape a = mat(GgmlType::q4_k, 6144, 5120);
    p = ops::chooseGemvTile(&a, 1, 4, k, tn);
    CHECK(nameOf(p.f) == "gemvq_nt4r2_" + q4);
    p = ops::chooseGemvTile(s3, 1, 3, k, tn);
    CHECK(nameOf(p.f) == "gemvq_nt3r2_" + mx);
    gk[12] = 2;
    p = ops::chooseGemvTile(s3, 1, 12, k, tn);
    CHECK(nameOf(p.f) == "gemvq_nt12r2_" + mx);
    // a code object without gemvk (the gfx1201 object, the 0.2.5 gfx1151 object): the old path
    const kn::KernelTable k0 = fakeTable(true, true, false, false);
    gk[4] = 2;
    p = ops::chooseGemvTile(s3, 3, 4, k0, tn);
    CHECK(nameOf(p.f) == "ggemvq_nt4r2_" + mx && p.rows_per_block == 16);
    // auto: Swift K = 5120 (20 superblocks) -> KW 4 (v2); Ornith 2048 (8) -> KW 2 (v4); 16384 (64) -> KW 8 (v6);
    // few rows widen KW back
    CHECK(ops::gemvkAuto(5120, 8192) == 2);
    CHECK(ops::gemvkAuto(2048, 8192) == 4);
    CHECK(ops::gemvkAuto(6144, 5120) == 2);
    CHECK(ops::gemvkAuto(17408, 5120) == 2);
    CHECK(ops::gemvkAuto(16384, 5120) == 6);
    CHECK(ops::gemvkAuto(512, 8192) == 4);
    CHECK(ops::gemvkAuto(2048, 1024) == 6);
    CHECK(ops::gemvkAuto(5120, 2048) == 6);
    gk[4] = ops::kGemvkAuto;
    p = ops::chooseGemvTile(s3, 3, 4, k, tn);  // 8192 rows, K 5120 -> v2
    CHECK(nameOf(p.f) == "ggemvk_nt4v2" && p.rows_per_block == 32);
    // v0251-a6port: the table the 8060S dense profile row holds (v7 at 5 and 6 tokens only) and an all-0 row (the MoE row,
    // the R9700 rows). ops does not depend on device (iso ops.matmul): the rows themselves are pinned in device.profile
    // (tests/iso/device.profile/device_profile_tests.cpp, tests/device/device_tests.cpp testProfiles) and model testGemvkDefault
    {
        ops::GemvR dense{};
        dense[5] = dense[6] = 7;
        const ops::GemvTune td{&r, &w, &wh, 0, &dense};
        for (std::uint32_t n = 2; n <= 16; ++n) {
            const auto pg = ops::chooseGemvTile(s3, 3, n, k, td);
            // a group takes the twin up to gvNmax (13 for mxfp4), above it one plain launch per matrix
            const std::string g = n <= 13 ? "g" : "";
            const std::string want = n == 5 || n == 6 ? "ggemvk_nt" + std::to_string(n) + "v7" : g + "gemvq_nt" + std::to_string(n) + "r2_" + mx;
            CHECK(nameOf(pg.f) == want);
            const auto p1 = ops::chooseGemvTile(s3, 1, n, k, td);
            const std::string want1 = n == 5 || n == 6 ? "gemvk_nt" + std::to_string(n) + "v7" : "gemvq_nt" + std::to_string(n) + "r2_" + mx;
            CHECK(nameOf(p1.f) == want1);
        }
        const ops::GemvR zero{};
        const ops::GemvTune t0{&r, &w, &wh, 0, &zero};
        for (std::uint32_t n = 2; n <= 8; ++n) CHECK(nameOf(ops::chooseGemvTile(s3, 3, n, k, t0).f) == "ggemvq_nt" + std::to_string(n) + "r2_" + mx);
    }
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

// v0251-a3port: the gfx1151 fused-dequant MXFP4 GEMM (gemmfq) replaces an MXFP4 DEQUANT choice (dequant_f16 + gemm_c<i>
// on f16) at n >= fdq_min rows when the slot is on and the code object has the kernel: slot 2 = t2 (128 x 256 tiles,
// the gfx1151 default), slot 1 = c2 (256 x 128). Below the threshold, for a fused or gemms choice, another weight type,
// f16 outputs, slot <= 0 or out of range, or a code object without gemmfq (gfx1201) the old plan stays. Shapes: Swift
// ffn_gate 17408 x 5120 / ffn_down 5120 x 17408 (tuned dequant + gemm_c2 at > 1024 rows on the 8060S), Ornith 512 x
// 2048 (tuned dequant + gemm_c1); rows 2048 (Swift forward), 4096 (MoE forward), 1025 (a tail just above 1024).
void testOpsGemmFq() {
    const std::uint32_t G = static_cast<std::uint32_t>(kn::kGemmCfgs.size());
    auto gate = mat(GgmlType::mxfp4, 17408, 5120);
    gate.row_bytes = 5120 / 256 * 136;
    auto down = mat(GgmlType::mxfp4, 5120, 17408);
    down.row_bytes = 17408 / 256 * 136;
    auto shg = mat(GgmlType::mxfp4, 512, 2048);
    shg.row_bytes = 2048 / 256 * 136;
    auto q4 = mat(GgmlType::q4_k, 5120, 5120);
    q4.row_bytes = 5120 / 256 * 144;
    kn::KernelTable k51 = gemmTable(true, true, false);  // gfx1151-like: no gemmh_f16, gemmfq present
    k51.gemmfq[0] = tag("gemmfq_c2_mxfp4");
    k51.gemmfq[1] = tag("gemmfq_t2_mxfp4");
    const kn::KernelTable k01 = gemmTable(true, true, true);  // gfx1201-like: gemmh_f16, no gemmfq
    const std::uint8_t deq2 = static_cast<std::uint8_t>(G + 2), deq1 = static_cast<std::uint8_t>(G + 1);
    const std::string mx = std::to_string(static_cast<int>(GgmlType::mxfp4)), f16 = std::to_string(static_cast<int>(GgmlType::f16));
    CHECK(kn::kGemmFqCfgs.size() == 2 && std::string(kn::kGemmFqNames[0]) == "c2" && std::string(kn::kGemmFqNames[1]) == "t2");
    ops::GemmFlags t2;
    t2.fdq = 2;
    CHECK(t2.fdq_min == 1025);
    // t2 at the Swift forward size: 128 x 256 tiles, TOK_X grid, 256 threads, no dequant pass
    auto p = ops::chooseGemmF16(gate, 2048, deq2, t2, k51);
    CHECK(p.kind == ops::GemmKind::gemmfq && !p.dequant && nameOf(p.f) == "gemmfq_t2_mxfp4" && p.grid.x == 8 && p.grid.y == 136 && p.grid.z == 1 &&
          p.block.x == 256 && p.bits == ops::BitClass::f16_gemm);
    p = ops::chooseGemmF16(down, 2048, deq2, t2, k51);
    CHECK(p.kind == ops::GemmKind::gemmfq && nameOf(p.f) == "gemmfq_t2_mxfp4" && p.grid.x == 8 && p.grid.y == 40);
    // the Ornith forward (4096 rows) on its dequant shape (tuned dequant + gemm_c1: any dequant choice is replaced)
    p = ops::chooseGemmF16(shg, 4096, deq1, t2, k51);
    CHECK(p.kind == ops::GemmKind::gemmfq && nameOf(p.f) == "gemmfq_t2_mxfp4" && p.grid.x == 16 && p.grid.y == 4);
    // a tail forward just above the threshold: partial token tile
    p = ops::chooseGemmF16(gate, 1025, deq2, t2, k51);
    CHECK(p.kind == ops::GemmKind::gemmfq && p.grid.x == 5 && p.grid.y == 136);
    // c2 (WHIRL_FDQ8S=c2): 256 x 128 tiles
    ops::GemmFlags c2 = t2;
    c2.fdq = 1;
    p = ops::chooseGemmF16(gate, 2048, deq2, c2, k51);
    CHECK(p.kind == ops::GemmKind::gemmfq && nameOf(p.f) == "gemmfq_c2_mxfp4" && p.grid.x == 16 && p.grid.y == 68 && p.block.x == 256);
    // below the threshold: the old plan (dequant into w16, then gemm_c2 on f16 weights; this table has no gemmh_f16)
    p = ops::chooseGemmF16(gate, 1024, deq2, t2, k51);
    CHECK(p.kind == ops::GemmKind::gemmc && p.dequant && nameOf(p.dq) == "dequant_f16_" + mx && nameOf(p.f) == "gemm_c2_" + f16);
    ops::GemmFlags lo = t2;
    lo.fdq_min = 512;  // WHIRL_FDQ8S_MIN=512
    CHECK(ops::chooseGemmF16(gate, 1024, deq2, lo, k51).kind == ops::GemmKind::gemmfq);
    CHECK(ops::chooseGemmF16(gate, 511, deq2, lo, k51).kind == ops::GemmKind::gemmc);
    // only a dequant choice is replaced: a fused gemm_c choice, a gemms slot and another type keep their plan
    p = ops::chooseGemmF16(gate, 2048, 2, t2, k51);
    CHECK(p.kind == ops::GemmKind::gemmc && !p.dequant && nameOf(p.f) == "gemm_c2_" + mx);
    CHECK(ops::chooseGemmF16(gate, 2048, static_cast<std::uint8_t>(2 * G + 1), t2, k51).kind == ops::GemmKind::gemms);
    CHECK(ops::chooseGemmF16(q4, 2048, deq2, t2, k51).kind == ops::GemmKind::gemmc);
    // f16 outputs (gemmfq has no f16-out twin): not replaced
    ops::GemmFlags h = t2;
    h.out_h16 = true;
    CHECK(ops::chooseGemmF16(gate, 2048, deq2, h, k51).kind != ops::GemmKind::gemmfq);
    // slot 0 (WHIRL_FDQ8S=0, the gfx1201 default), out of range, unresolved -1: the old plan
    for (int s : {0, 3, -1}) {
        ops::GemmFlags off = t2;
        off.fdq = s;
        p = ops::chooseGemmF16(gate, 2048, deq2, off, k51);
        CHECK(p.kind == ops::GemmKind::gemmc && p.dequant);
    }
    // a code object without gemmfq (gfx1201): never, whatever the slot (dequant + gemmh_f16 as before)
    for (int s : {1, 2}) {
        ops::GemmFlags f = t2;
        f.fdq = s;
        p = ops::chooseGemmF16(gate, 2048, deq2, f, k01);
        CHECK(p.kind == ops::GemmKind::gemmh && p.dequant && nameOf(p.f) == "gemmh_f16");
    }
    // one slot missing from the object: that slot falls back, the other still runs
    kn::KernelTable k1 = k51;
    k1.gemmfq[1] = nullptr;
    CHECK(ops::chooseGemmF16(gate, 2048, deq2, t2, k1).kind == ops::GemmKind::gemmc);
    CHECK(ops::chooseGemmF16(gate, 2048, deq2, c2, k1).kind == ops::GemmKind::gemmfq);
    // v0251-a3fix: from fdq_big_min (2305) rows the slot is fdq_big (-1 = fdq, as in every case above). The gfx1151
    // default: t2 up to 2304 rows (the largest dense forward), c2 from 2305 (t2 is 1.33-1.35x the old pair at 4096 rows,
    // c2 0.91-0.98; the tuner times bucket 1 at 4096 rows, the 8060S MoE forward is 4096 rows)
    CHECK(t2.fdq_big == -1 && t2.fdq_big_min == 2305);
    ops::GemmFlags dv = t2;
    dv.fdq_big = 1;
    p = ops::chooseGemmF16(gate, 2304, deq2, dv, k51);  // 2048 + a merged 256-row tail: still t2
    CHECK(p.kind == ops::GemmKind::gemmfq && nameOf(p.f) == "gemmfq_t2_mxfp4" && p.grid.x == 9 && p.grid.y == 136);
    p = ops::chooseGemmF16(gate, 2305, deq2, dv, k51);
    CHECK(p.kind == ops::GemmKind::gemmfq && nameOf(p.f) == "gemmfq_c2_mxfp4" && p.grid.x == 19 && p.grid.y == 68 && p.block.x == 256);
    p = ops::chooseGemmF16(gate, 4096, deq2, dv, k51);  // the bucket-1 tune size
    CHECK(p.kind == ops::GemmKind::gemmfq && nameOf(p.f) == "gemmfq_c2_mxfp4" && p.grid.x == 32 && p.grid.y == 68 && !p.dequant);
    p = ops::chooseGemmF16(down, 4096, deq2, dv, k51);
    CHECK(p.kind == ops::GemmKind::gemmfq && nameOf(p.f) == "gemmfq_c2_mxfp4" && p.grid.x == 32 && p.grid.y == 20);
    p = ops::chooseGemmF16(shg, 4096, deq1, dv, k51);  // the Ornith forward
    CHECK(p.kind == ops::GemmKind::gemmfq && nameOf(p.f) == "gemmfq_c2_mxfp4" && p.grid.x == 32 && p.grid.y == 2);
    // fdq_big 0: the old plan from 2305 rows, gemmfq below
    ops::GemmFlags bo = t2;
    bo.fdq_big = 0;
    p = ops::chooseGemmF16(gate, 4096, deq2, bo, k51);
    CHECK(p.kind == ops::GemmKind::gemmc && p.dequant && nameOf(p.f) == "gemm_c2_" + f16);
    CHECK(ops::chooseGemmF16(gate, 2048, deq2, bo, k51).kind == ops::GemmKind::gemmfq);
    // the big-row slot missing from the object: the old plan there (never the other slot)
    kn::KernelTable kc = k51;
    kc.gemmfq[0] = nullptr;
    p = ops::chooseGemmF16(gate, 4096, deq2, dv, kc);
    CHECK(p.kind == ops::GemmKind::gemmc && p.dequant);
    // the row threshold still applies first (WHIRL_FDQ8S_MIN above 4096): the old plan
    ops::GemmFlags hi = dv;
    hi.fdq_min = 8192;
    CHECK(ops::chooseGemmF16(gate, 4096, deq2, hi, k51).kind == ops::GemmKind::gemmc);
    // a fused choice, f16 outputs and gfx1201 stay as they were at big rows too
    CHECK(ops::chooseGemmF16(gate, 4096, 2, dv, k51).kind == ops::GemmKind::gemmc && !ops::chooseGemmF16(gate, 4096, 2, dv, k51).dequant);
    ops::GemmFlags dh = dv;
    dh.out_h16 = true;
    CHECK(ops::chooseGemmF16(gate, 4096, deq2, dh, k51).kind != ops::GemmKind::gemmfq);
    CHECK(ops::chooseGemmF16(gate, 4096, deq2, dv, k01).kind == ops::GemmKind::gemmh);
    // v0252-8sbadopt: GemmFlags::fq_grp bit (slot - 1) runs the slot's grouped-raster twin (MatmulKernels::gemmfq_grp,
    // gemmfq_<slot>g8_mxfp4: same tile, grid and bits) - the gfx1151 default per model class (device::Profile::fq_grp_dense
    // = t2 bit, fq_grp_moe = c2 bit; job 94 re-judged ADOPT). No bit, no twin in the object or the old plan: unchanged.
    CHECK(t2.fq_grp == 0);
    kn::KernelTable kg = k51;
    kg.gemmfq_grp[0] = tag("gemmfq_c2g8_mxfp4");
    kg.gemmfq_grp[1] = tag("gemmfq_t2g8_mxfp4");
    ops::GemmFlags gd = dv;  // the gfx1151 dense row: t2 twin up to 2304 rows, c2 (plain) from 2305
    gd.fq_grp = 2;
    p = ops::chooseGemmF16(gate, 2048, deq2, gd, kg);  // the Swift forward
    CHECK(p.kind == ops::GemmKind::gemmfq && !p.dequant && nameOf(p.f) == "gemmfq_t2g8_mxfp4" && p.grid.x == 8 && p.grid.y == 136 && p.block.x == 256);
    p = ops::chooseGemmF16(down, 2304, deq2, gd, kg);
    CHECK(p.kind == ops::GemmKind::gemmfq && nameOf(p.f) == "gemmfq_t2g8_mxfp4" && p.grid.x == 9 && p.grid.y == 40);
    p = ops::chooseGemmF16(gate, 4096, deq2, gd, kg);
    CHECK(p.kind == ops::GemmKind::gemmfq && nameOf(p.f) == "gemmfq_c2_mxfp4" && p.grid.x == 32 && p.grid.y == 68);
    ops::GemmFlags gm = dv;  // the gfx1151 MoE row: c2 twin from 2305 rows, t2 (plain) below
    gm.fq_grp = 1;
    p = ops::chooseGemmF16(shg, 4096, deq1, gm, kg);  // the Ornith forward
    CHECK(p.kind == ops::GemmKind::gemmfq && nameOf(p.f) == "gemmfq_c2g8_mxfp4" && p.grid.x == 32 && p.grid.y == 2 && p.block.x == 256);
    p = ops::chooseGemmF16(gate, 4096, deq2, gm, kg);
    CHECK(p.kind == ops::GemmKind::gemmfq && nameOf(p.f) == "gemmfq_c2g8_mxfp4" && p.grid.x == 32 && p.grid.y == 68);
    p = ops::chooseGemmF16(shg, 2048, deq1, gm, kg);  // an Ornith tail chunk: the plain t2
    CHECK(p.kind == ops::GemmKind::gemmfq && nameOf(p.f) == "gemmfq_t2_mxfp4");
    // flags without a bit (every non-gfx1151 row, WHIRL_FQ_RASTER set): the product kernels, also when the twins exist
    CHECK(nameOf(ops::chooseGemmF16(gate, 2048, deq2, dv, kg).f) == "gemmfq_t2_mxfp4");
    CHECK(nameOf(ops::chooseGemmF16(gate, 4096, deq2, dv, kg).f) == "gemmfq_c2_mxfp4");
    // a bit without the twin in the object: the slot's product kernel
    kn::KernelTable kn1 = kg;
    kn1.gemmfq_grp[1] = nullptr;
    p = ops::chooseGemmF16(gate, 2048, deq2, gd, kn1);
    CHECK(p.kind == ops::GemmKind::gemmfq && nameOf(p.f) == "gemmfq_t2_mxfp4" && p.grid.x == 8 && p.grid.y == 136);
    // the bits never reach the old plan, a fused choice, f16 outputs or a code object without gemmfq
    ops::GemmFlags gb = bo;
    gb.fq_grp = 3;
    p = ops::chooseGemmF16(gate, 4096, deq2, gb, kg);
    CHECK(p.kind == ops::GemmKind::gemmc && p.dequant && nameOf(p.f) == "gemm_c2_" + f16);
    ops::GemmFlags ga = dv;
    ga.fq_grp = 3;
    CHECK(ops::chooseGemmF16(gate, 2048, 2, ga, kg).kind == ops::GemmKind::gemmc);
    ga.out_h16 = true;
    CHECK(ops::chooseGemmF16(gate, 2048, deq2, ga, kg).kind != ops::GemmKind::gemmfq);
    ga.out_h16 = false;
    CHECK(ops::chooseGemmF16(gate, 2048, deq2, ga, k01).kind == ops::GemmKind::gemmh);
}

// v0252-c3port (feat026-moedec-p25 0ae5530 dispatch.cpp, gfx1201 only): ops::gemvhF32, Context::matmul's gemvhf choice -
// the f32-input twin of the plan's gemvh kernel (gemvhf_<T> for gemvh_<T>, gemvhf2_<T> for gemvh2_<T>) when enabled,
// the plan is gemvh with one block per 16 rows (no gemvh_cap: the gfx1151 grid-capped gemvh loops over row tiles, the
// twin does not), and the code object has the twin; else null (f32_to_f16 + the plan's kernel, the 0.2.5.1 path).
// Per-arch selector: a table without gemvhf (gfx1151, the shipped gfx1201) always null. Fails before the port (stub).
void testGemvhF32() {
    auto mx = mat(GgmlType::mxfp4, 512, 2048);
    mx.row_bytes = 2048 / 32 * 17;
    auto q4 = mat(GgmlType::q4_k, 5120, 5120);
    q4.row_bytes = 5120 / 256 * 144;
    auto f16 = mat(GgmlType::f16, 4096, 5120);
    f16.row_bytes = 5120 * 2;
    ops::GemmFlags pre;
    pre.prec_dec = true;
    const kn::KernelTable k0 = gemmTable(true, true, true);  // shipped: no gemvhf
    kn::KernelTable k = k0;
    for (GgmlType t : {GgmlType::q4_k, GgmlType::mxfp4}) {  // no f16 twin (it spills; f16 keeps f32_to_f16 + gemvh)
        const std::string s = std::to_string(static_cast<int>(t));
        k.gemvhf[ti(t)] = tag("gemvhf_" + s);
        k.gemvhf2[ti(t)] = tag("gemvhf2_" + s);
    }
    const std::string smx = std::to_string(static_cast<int>(GgmlType::mxfp4)), sq4 = std::to_string(static_cast<int>(GgmlType::q4_k));
    for (const std::uint32_t n : {1u, 4u, 16u}) {
        const ops::GemmPlan p = ops::chooseGemmF16(mx, n, 0, pre, k);
        CHECK(p.kind == ops::GemmKind::gemvh && nameOf(p.f) == "gemvh_" + smx);
        CHECK(nameOf(ops::gemvhF32(p, GgmlType::mxfp4, true, pre, k)) == "gemvhf_" + smx);
        CHECK(ops::gemvhF32(p, GgmlType::mxfp4, false, pre, k) == nullptr);  // WHIRL_GEMVH_F32 off (default)
        CHECK(ops::gemvhF32(p, GgmlType::mxfp4, true, pre, k0) == nullptr);  // no twin in the code object
    }
    ops::GemmFlags pre32 = pre;
    pre32.gv_max = 32;  // a wide precise verify (Context::gvMax() = small_max 32): 17..32 rows on gemvh2
    for (const std::uint32_t n : {17u, 32u}) {
        const ops::GemmPlan p = ops::chooseGemmF16(q4, n, 0, pre32, k);
        CHECK(p.kind == ops::GemmKind::gemvh && nameOf(p.f) == "gemvh2_" + sq4);
        CHECK(nameOf(ops::gemvhF32(p, GgmlType::q4_k, true, pre32, k)) == "gemvhf2_" + sq4);
    }
    // f16 weights: gemvh, but no twin -> null
    {
        const ops::GemmPlan p = ops::chooseGemmF16(f16, 4, 0, pre, k);
        CHECK(p.kind == ops::GemmKind::gemvh && ops::gemvhF32(p, GgmlType::f16, true, pre, k) == nullptr);
    }
    // not a gemvh plan (WHIRL_PREC_GEMVH=0 -> gemms; balance prefill rows -> gemm_c): null
    {
        ops::GemmFlags nog = pre;
        nog.prec_gemvh = false;
        const ops::GemmPlan p = ops::chooseGemmF16(q4, 4, 0, nog, k);
        CHECK(p.kind == ops::GemmKind::gemms && ops::gemvhF32(p, GgmlType::q4_k, true, nog, k) == nullptr);
        ops::GemmFlags bal;
        const ops::GemmPlan b = ops::chooseGemmF16(q4, 256, 2, bal, k);
        CHECK(b.kind == ops::GemmKind::gemmc && ops::gemvhF32(b, GgmlType::q4_k, true, bal, k) == nullptr);
    }
    // a grid-capped gemvh (device profile gemvh_grid_by_rows, the gfx1151 rule): null
    {
        ops::GemmFlags cap = pre;
        cap.gemvh_cap = 64;
        const ops::GemmPlan p = ops::chooseGemmF16(q4, 4, 0, cap, k);
        CHECK(p.kind == ops::GemmKind::gemvh && ops::gemvhF32(p, GgmlType::q4_k, true, cap, k) == nullptr);
    }
    // the plan's kernel must be the table's gemvh entry of that type (a kernel the twin does not reproduce: null)
    {
        ops::GemmPlan p = ops::chooseGemmF16(mx, 4, 0, pre, k);
        p.f = tag("gemvh_other");
        CHECK(ops::gemvhF32(p, GgmlType::mxfp4, true, pre, k) == nullptr);
        const ops::GemmPlan p2 = ops::chooseGemmF16(mx, 4, 0, pre, k);
        CHECK(ops::gemvhF32(p2, GgmlType::q4_k, true, pre, k) == nullptr);  // wrong type index
    }
}

}  // namespace
