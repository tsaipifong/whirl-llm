// arch025-n8b1: cases moved verbatim from tests/ops/ops_tests.cpp; included by whirl-ops-tests (tests/ops/ops_tests.cpp, same
// cases, same order) and by the module's isolation program tests/iso/ops.matmul/ (ARCH-1 8.5).
// SPDX-License-Identifier: Apache-2.0

#include "whirl/ops/matmul.h"
#include "../test_check.h"
#include "ops_fakes.h"
#include "ops_mat.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

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
    // v0252-opsasan: a kn::KernelTable is about 140 KB and under MSVC ASan every local of that type gets its own stack slot;
    // with six of them this frame was 0.81 MiB of the 1 MB stack (cpu_asan). The tables built once are static (built in
    // static storage), every table the case changes is a heap copy (same shape as v0252-c1asan, ops_attention_cases.inc).
    static const kn::KernelTable k51b = gemmTable(true, true, false);  // gfx1151-like: no gemmh_f16 (gemmfq: added below)
    const auto k51p = std::make_unique<kn::KernelTable>(k51b);
    kn::KernelTable& k51 = *k51p;
    k51.gemmfq[0] = tag("gemmfq_c2_mxfp4");
    k51.gemmfq[1] = tag("gemmfq_t2_mxfp4");
    static const kn::KernelTable k01 = gemmTable(true, true, true);  // gfx1201-like: gemmh_f16, no gemmfq
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
    const auto k1p = std::make_unique<kn::KernelTable>(k51);
    kn::KernelTable& k1 = *k1p;
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
    const auto kcp = std::make_unique<kn::KernelTable>(k51);
    kn::KernelTable& kc = *kcp;
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
    const auto kgp = std::make_unique<kn::KernelTable>(k51);
    kn::KernelTable& kg = *kgp;
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
    const auto kn1p = std::make_unique<kn::KernelTable>(kg);
    kn::KernelTable& kn1 = *kn1p;
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


// v0252-8sdec (PLAN V1-142): ops::chooseGemvhGroup, Context::matmulGroup's precise grouped gemvh - the same-input matrices
// of one matmulGroup call (GDN {qkv, z}, attention {q, k, v}, FFN {gate, up}) in one gemvhg_<T> launch when enabled
// (Flags::gemvh_grp, WHIRL_GEMVH_GRP), 2..3 matrices of one type and ncols, each of which would run gemvh_<T> on its own
// (MatmulPath::gemm_f16 and chooseGemmF16 kind gemvh with the table's gemvh_<T>: precise rows, n <= 16, no f16 outputs),
// and the code object has gemvhg_<T> (gfx1151 only). Tiles of 16 rows are numbered matrix by matrix (b1 / b2 = first
// tile of matrix 1 / 2, 0x7fffffff absent, the int8 GvArgs rule); the grid is the single gemvh rule on the total
// (kernels::gemvhGridCap with the device cap, else one block per tile). Anything else: null = one matmul per matrix
// (the 0.2.5.1 path). Fails to build before the change (no chooseGemvhGroup / MatmulKernels::gemvhg).
void testGemvhGroup() {
    auto mxm = [](std::uint32_t rows, std::uint32_t cols) {
        auto m = mat(GgmlType::mxfp4, rows, cols);
        m.row_bytes = cols / 32 * 17;
        return m;
    };
    const ops::MatShape qkv = mxm(10240, 5120), z = mxm(6144, 5120), q = mxm(12288, 5120), kk = mxm(1024, 5120), v = mxm(1024, 5120);
    const ops::MatShape gate = mxm(17408, 5120), up = mxm(17408, 5120), down = mxm(5120, 17408), odd = mxm(1000, 5120);
    ops::GemmFlags pre;
    pre.prec_dec = true;
    ops::MatmulFlags mpre;
    mpre.prec_dec = true;
    mpre.gv_max = 16;
    const kn::KernelTable k0 = gemmTable(true, true, true);  // the gfx1201 shape: no gemvhg
    kn::KernelTable k = k0;
    const std::string smx = std::to_string(static_cast<int>(GgmlType::mxfp4));
    k.gemvhg[ti(GgmlType::mxfp4)] = tag("gemvhg_" + smx);
    constexpr std::int32_t none = 0x7fffffff;
    {
        const ops::MatShape ws[2] = {qkv, z};
        for (const std::uint32_t n : {1u, 4u, 9u, 16u}) {
            if (ops::chooseMatmulPath(qkv, n, mpre, k) != ops::MatmulPath::gemm_f16) continue;  // n = 1 may take gemv1
            const ops::GemvhGroupPlan p = ops::chooseGemvhGroup(ws, 2, n, true, mpre, pre, k);
            CHECK(nameOf(p.f) == "gemvhg_" + smx && p.b1 == 640 && p.b2 == none && p.tiles == 1024 && p.grid.x == 1024 && p.block.x == 32);
            CHECK(ops::chooseGemvhGroup(ws, 2, n, false, mpre, pre, k).f == nullptr);  // WHIRL_GEMVH_GRP=0
            CHECK(ops::chooseGemvhGroup(ws, 2, n, true, mpre, pre, k0).f == nullptr);   // no gemvhg in the code object
        }
    }
    {
        const ops::MatShape ws[3] = {q, kk, v};
        const ops::GemvhGroupPlan p = ops::chooseGemvhGroup(ws, 3, 4, true, mpre, pre, k);
        CHECK(p.f != nullptr && p.b1 == 768 && p.b2 == 832 && p.tiles == 896 && p.grid.x == 896);
        ops::GemmFlags cap = pre;
        cap.gemvh_cap = 640;  // the gfx1151 grid: gemvhGridCap(tiles * 16, cap) = 448 blocks of 2 tiles
        const ops::GemvhGroupPlan pc = ops::chooseGemvhGroup(ws, 3, 4, true, mpre, cap, k);
        CHECK(pc.f == p.f && pc.tiles == 896 && pc.grid.x == kn::gemvhGridCap(896 * 16, 640) && pc.grid.x == 448);
    }
    {
        const ops::MatShape ws[2] = {gate, up}, wo[2] = {odd, qkv};
        const ops::GemvhGroupPlan p = ops::chooseGemvhGroup(ws, 2, 7, true, mpre, pre, k);
        CHECK(p.f != nullptr && p.b1 == 1088 && p.tiles == 2176);
        const ops::GemvhGroupPlan po = ops::chooseGemvhGroup(wo, 2, 7, true, mpre, pre, k);  // 1000 rows = 63 tiles (last partial)
        CHECK(po.f != nullptr && po.b1 == 63 && po.tiles == 63 + 640);
    }
    // not grouped: one matrix, other ncols, other type, balance rows, f16 outputs, WHIRL_PREC_GEMVH=0, 17+ rows (gemvh2)
    {
        const ops::MatShape one[1] = {qkv}, dif[2] = {qkv, down}, typ[2] = {qkv, mat(GgmlType::q4_k, 6144, 5120)};
        CHECK(ops::chooseGemvhGroup(one, 1, 4, true, mpre, pre, k).f == nullptr);
        CHECK(ops::chooseGemvhGroup(dif, 2, 4, true, mpre, pre, k).f == nullptr);
        CHECK(ops::chooseGemvhGroup(typ, 2, 4, true, mpre, pre, k).f == nullptr);
        const ops::MatShape ws[2] = {qkv, z};
        ops::GemmFlags bal;
        ops::MatmulFlags mbal;
        mbal.gv_max = 16;
        CHECK(ops::chooseGemvhGroup(ws, 2, 4, true, mbal, bal, k).f == nullptr);
        ops::GemmFlags h16 = pre;
        h16.out_h16 = true;
        CHECK(ops::chooseGemvhGroup(ws, 2, 4, true, mpre, h16, k).f == nullptr);
        ops::GemmFlags nog = pre;
        nog.prec_gemvh = false;
        CHECK(ops::chooseGemvhGroup(ws, 2, 4, true, mpre, nog, k).f == nullptr);
        ops::GemmFlags pre32 = pre;
        pre32.gv_max = 32;
        ops::MatmulFlags mpre32 = mpre;
        mpre32.gv_max = 32;
        CHECK(ops::chooseGemvhGroup(ws, 2, 17, true, mpre32, pre32, k).f == nullptr);
        CHECK(ops::chooseGemvhGroup(ws, 2, 16, true, mpre32, pre32, k).f != nullptr);
    }
}

// v0252-pp2k2 (PLAN V1-139, F1 "GEMM tail split"): a precise f16 prefill GEMM on the 256-token tile (gemmh_f16 / gemmhq) whose
// rows r = n mod 256 past the last full tile fit the widest gemms token tile runs as two pieces: rows [0, n - r) on the
// unchanged 256 plan (grid x (n - r) / 256) and rows [n - r, n) on precChoice's narrowest gemms slot holding r (raw
// quantized bytes, the h16 twin when the main kernel writes f16). Same bits (class f16_gemm: each output element keeps its
// k order), so the rule only moves speed. Shapes: Swift ffn_gate 17408 x 5120 MXFP4, tuned dequant choice; gfx1201 gemms
// geometry (kGemmsCfgs: bn 32 / 32 / 48 / 64 / 16 / 64 / 96 / 128).
void testOpsGemmTail() {
    const std::uint32_t G = static_cast<std::uint32_t>(kn::kGemmCfgs.size());
    auto gate = mat(GgmlType::mxfp4, 17408, 5120);
    gate.row_bytes = 5120 / 256 * 136;
    auto f16 = mat(GgmlType::f16, 4096, 5120);
    f16.row_bytes = 5120 * 2;
    static const kn::KernelTable kb = gemmTable(true, true, true);  // gemmh_f16, gemms slots 0 / 1 / 4
    const auto kp = std::make_unique<kn::KernelTable>(kb);
    kn::KernelTable& k = *kp;
    const std::string mx = std::to_string(static_cast<int>(GgmlType::mxfp4));
    const std::string f16s = std::to_string(static_cast<int>(GgmlType::f16));
    for (std::size_t c = 0; c < kn::kGemmsCfgs.size(); ++c) {
        k.gemms[c][ti(GgmlType::mxfp4)] = tag("gemms_c" + std::to_string(c) + "_" + mx);
        k.gemmsh[c][ti(GgmlType::mxfp4)] = tag("gemmsh_c" + std::to_string(c) + "_" + mx);
    }
    const std::uint8_t deq2 = static_cast<std::uint8_t>(G + 2);
    ops::GemmFlags on;  // a prefill (the f16 GEMM path) with the split on
    on.gemm_tail = true;
    ops::GemmFlags off = on;
    off.gemm_tail = false;
    CHECK(ops::GemmFlags{}.gemm_tail == false);  // off unless the caller (device profile / WHIRL_GEMM_TAIL) turns it on
    CHECK(ops::gemmh_bn == 256);
    // expected pieces: {row0, rows, kernel name, grid x}
    struct Want {
        std::uint32_t row0, rows;
        std::string f;
        unsigned gx;
    };
    auto check = [&](const ops::MatShape& w, std::uint32_t n, std::uint8_t tuned, const ops::GemmFlags& f, const std::vector<Want>& want,
                     const char* what) {
        const ops::GemmPieces pc = ops::planGemmF16(w, n, tuned, f, k);
        bool ok = pc.n == want.size();
        for (std::size_t i = 0; ok && i < want.size(); ++i) {
            const ops::GemmPiece& g = pc.p[i];
            ok = g.row0 == want[i].row0 && g.rows == want[i].rows && nameOf(g.plan.f) == want[i].f && g.plan.grid.x == want[i].gx &&
                 g.plan.bits == ops::BitClass::f16_gemm;
        }
        if (!ok)
            std::printf("  gemm tail: %s (n %u): %zu pieces, first %s rows %u grid %u\n", what, n, pc.n, nameOf(pc.p[0].plan.f).c_str(),
                        pc.p[0].rows, pc.p[0].plan.grid.x);
        CHECK(ok);
    };
    const std::string c4 = "gemms_c4_" + mx;  // 128 x 16: the narrowest tile
    // the 2k cell's prompt: 8 full tiles + a 10-row gemms tail (was 9 tiles of 256)
    check(gate, 2058, deq2, on, {{0, 2048, "gemmh_f16", 8}, {2048, 10, c4, 1}}, "2058");
    {
        const ops::GemmPieces pc = ops::planGemmF16(gate, 2058, deq2, on, k);
        CHECK(pc.n == 2 && pc.p[0].plan.dequant && nameOf(pc.p[0].plan.dq) == "dequant_f16_" + mx && pc.p[0].plan.grid.y == 136);
        CHECK(pc.n == 2 && pc.p[1].plan.kind == ops::GemmKind::gemms && !pc.p[1].plan.dequant && pc.p[1].plan.grid.y == 136 &&
              pc.p[1].plan.block.x == static_cast<unsigned>(kn::kGemmsCfgs[4].nth) && !pc.p[1].plan.y_h16 && !pc.p[0].plan.y_h16);
    }
    check(gate, 2058, deq2, off, {{0, 2058, "gemmh_f16", 9}}, "2058 split off (WHIRL_GEMM_TAIL=0 / gfx1151)");
    check(gate, 2048, deq2, on, {{0, 2048, "gemmh_f16", 8}}, "2048: whole tiles");
    check(gate, 2049, deq2, on, {{0, 2048, "gemmh_f16", 8}, {2048, 1, c4, 1}}, "2049");
    check(gate, 1033, deq2, on, {{0, 1024, "gemmh_f16", 4}, {1024, 9, c4, 1}}, "1033 (the MTP batch after the capture row)");
    check(gate, 4096, deq2, on, {{0, 4096, "gemmh_f16", 16}}, "4096");
    check(gate, 4097, deq2, on, {{0, 4096, "gemmh_f16", 16}, {4096, 1, c4, 1}}, "4097");
    check(gate, 2041, deq2, on, {{0, 2041, "gemmh_f16", 8}}, "2041: 249-row last tile, wider than every gemms tile");
    // below gemmh_min the plan is gemm_c (its own tile): no split
    const unsigned c2bn = static_cast<unsigned>(kn::kGemmCfgs[2].bn);
    check(gate, 257, deq2, on, {{0, 257, "gemm_c2_" + f16s, (257 + c2bn - 1) / c2bn}}, "257");
    check(gate, 255, deq2, on, {{0, 255, "gemm_c2_" + f16s, (255 + c2bn - 1) / c2bn}}, "255");
    // the narrowest slot that holds r (then the smaller row block): 20 -> c0 (64 x 32), 40 -> c2 (48), 60 -> c5 (64 x 64),
    // 90 -> c6 (96), 128 -> c7 (128); 129 -> no gemms tile holds it: the old plan
    check(gate, 2048 + 20, deq2, on, {{0, 2048, "gemmh_f16", 8}, {2048, 20, "gemms_c0_" + mx, 1}}, "r 20");
    check(gate, 2048 + 40, deq2, on, {{0, 2048, "gemmh_f16", 8}, {2048, 40, "gemms_c2_" + mx, 1}}, "r 40");
    check(gate, 2048 + 60, deq2, on, {{0, 2048, "gemmh_f16", 8}, {2048, 60, "gemms_c5_" + mx, 1}}, "r 60");
    check(gate, 2048 + 90, deq2, on, {{0, 2048, "gemmh_f16", 8}, {2048, 90, "gemms_c6_" + mx, 1}}, "r 90");
    check(gate, 2048 + 128, deq2, on, {{0, 2048, "gemmh_f16", 8}, {2048, 128, "gemms_c7_" + mx, 1}}, "r 128");
    check(gate, 2048 + 129, deq2, on, {{0, 2177, "gemmh_f16", 9}}, "r 129");
    // the widest gemms tile is a kernel property (KernelTable::gemms_geom), not a number of the rule
    {
        const auto k2p = std::make_unique<kn::KernelTable>(k);
        k2p->gemms_geom[7] = kn::GemmCfg{64, 160, 256};
        const ops::GemmPieces pc = ops::planGemmF16(gate, 2048 + 150, deq2, on, *k2p);
        CHECK(pc.n == 2 && pc.p[1].rows == 150 && nameOf(pc.p[1].plan.f) == "gemms_c7_" + mx);
    }
    // gemmhq (dequant fused into gemmh) is split the same way
    k.gemmhq[ti(GgmlType::mxfp4)] = tag("gemmhq_" + mx);
    ops::GemmFlags hq = on;
    hq.gemmhq_on = true;
    check(gate, 2058, deq2, hq, {{0, 2048, "gemmhq_" + mx, 8}, {2048, 10, c4, 1}}, "gemmhq 2058");
    // f16 outputs: the h16 main kernel's tail is the gemmsh twin; without that twin, no split
    k.gemmhqh[ti(GgmlType::mxfp4)] = tag("gemmhqh_" + mx);
    k.gemmch[2] = tag("gemm_ch2_f16");
    ops::GemmFlags h = hq;
    h.out_h16 = true;
    check(gate, 2058, deq2, h, {{0, 2048, "gemmhqh_" + mx, 8}, {2048, 10, "gemmsh_c4_" + mx, 1}}, "gemmhqh 2058");
    {
        const ops::GemmPieces pc = ops::planGemmF16(gate, 2058, deq2, h, k);
        CHECK(pc.n == 2 && pc.p[0].plan.y_h16 && pc.p[1].plan.y_h16);
        const auto k3p = std::make_unique<kn::KernelTable>(k);
        k3p->gemmsh[4][ti(GgmlType::mxfp4)] = nullptr;
        CHECK(ops::planGemmF16(gate, 2058, deq2, h, *k3p).n == 1);
    }
    // f16 weights (no gemms kernel for the type), a fused gemm_c choice, a tuned gemms choice: one piece, the old plan
    check(f16, 2058, 1, on, {{0, 2058, "gemmh_f16", 9}}, "f16 weights");
    CHECK(ops::planGemmF16(gate, 2058, 2, on, k).n == 1);
    CHECK(ops::planGemmF16(gate, 2058, static_cast<std::uint8_t>(2 * G + 1), on, k).n == 1);
    // WHIRL_PREC_GS forces a slot: used when it holds r, else no split
    ops::GemmFlags gs = on;
    gs.prec_gs = 1;
    check(gate, 2058, deq2, gs, {{0, 2048, "gemmh_f16", 8}, {2048, 10, "gemms_c1_" + mx, 1}}, "PREC_GS=1");
    gs.prec_gs = 4;
    check(gate, 2048 + 40, deq2, gs, {{0, 2088, "gemmh_f16", 9}}, "PREC_GS=4 cannot hold 40");
    // every row count of a forward: the pieces cover [0, n) in order, the main piece is whole 256 tiles, the tail fits one
    // gemms tile, and a one-piece plan is exactly chooseGemmF16's
    for (std::uint32_t n = 1; n <= 4096 + 300; ++n) {
        const ops::GemmPieces pc = ops::planGemmF16(gate, n, deq2, on, k);
        const ops::GemmPlan one = ops::chooseGemmF16(gate, n, deq2, on, k);
        bool ok = pc.n >= 1 && pc.n <= 2 && pc.p[0].row0 == 0;
        if (ok && pc.n == 1) ok = pc.p[0].rows == n && pc.p[0].plan.f == one.f && pc.p[0].plan.grid.x == one.grid.x;
        if (ok && pc.n == 2)
            ok = pc.p[0].rows % 256 == 0 && pc.p[0].plan.grid.x * 256 == pc.p[0].rows && pc.p[1].row0 == pc.p[0].rows &&
                 pc.p[0].rows + pc.p[1].rows == n && pc.p[1].rows > 0 && pc.p[1].plan.grid.x == 1 && pc.p[0].plan.f == one.f;
        if (!ok) {
            std::printf("  gemm tail: cover rule fails at n %u\n", n);
            CHECK(false);
            break;
        }
    }
}

// v0252-w4a4k2 (PLAN V1-144, kill-screen K2): GemmFlags::w4s_w = the W4A4 fake-quant arm's weight half (fast, gfx1151 dense
// prefill; the numerics flag w4a4_sim, WHIRL_W4A4_SIM). An MXFP4 matrix with whole superblocks then ALWAYS runs as
// w4s_dequant_f16_mxfp4 (the per-32 int4 re-quantized grid, MatmulKernels::w4s_dq) into the f16 weight scratch + the f16 GEMM
// of the tuned slot: never gemmfq / gemmhq / a fused gemm_c / gemms / gemvh (they decode the original MXFP4 values), and no
// token-tail split (its gemms tail would read the original bytes). Off (the default), another type or a table without the
// kernel (gfx1201): the plan is exactly the one without the flag. Fails to build before the change (no GemmFlags::w4s_w /
// MatmulKernels::w4s_dq).
void testOpsGemmW4s() {
    const std::uint32_t G = static_cast<std::uint32_t>(kn::kGemmCfgs.size());
    auto gate = mat(GgmlType::mxfp4, 17408, 5120);
    gate.row_bytes = 5120 / 256 * 136;
    auto q4 = mat(GgmlType::q4_k, 5120, 5120);
    q4.row_bytes = 5120 / 256 * 144;
    static const kn::KernelTable kb = gemmTable(true, true, false);  // gfx1151-like
    const auto kp = std::make_unique<kn::KernelTable>(kb);
    kn::KernelTable& k = *kp;
    k.gemmfq[0] = tag("gemmfq_c2_mxfp4");
    k.gemmfq[1] = tag("gemmfq_t2_mxfp4");
    k.gemmhq[ti(GgmlType::mxfp4)] = tag("gemmhq_mxfp4");
    for (std::size_t g = 0; g < k.w4s_dq.size(); ++g) k.w4s_dq[g] = tag(kn::kW4sDqNames[g]);
    const std::string mx = std::to_string(static_cast<int>(GgmlType::mxfp4)), f16 = std::to_string(static_cast<int>(GgmlType::f16));
    CHECK(ops::GemmFlags{}.w4s_w == false);
    ops::GemmFlags base;  // the gfx1151 dense plan: gemmfq slot t2, gemmhq off (device default)
    base.fdq = 2;
    ops::GemmFlags sim = base;
    sim.w4s_w = true;
    sim.gemmhq_on = true;  // the arm must skip gemmhq too (it decodes the original MXFP4 values)
    const std::uint8_t deq2 = static_cast<std::uint8_t>(G + 2);
    // every tuned choice (fused gemm_c 0..G-1, dequant G..2G-1, gemms 2G..) and row count -> dequant through w4s_dq + f16 GEMM
    for (std::uint32_t n : {17u, 300u, 1024u, 2048u, 2058u, 4096u})
        for (std::uint32_t tuned : {0u, 2u, G + 1, G + 2, 2 * G + 1}) {
            const ops::GemmPlan p = ops::chooseGemmF16(gate, n, static_cast<std::uint8_t>(tuned), sim, k);
            const std::size_t gc = tuned < 2 * G ? tuned % G : 0;
            CHECK(p.dequant && nameOf(p.dq) == "w4s_dequant_f16_mxfp4" && p.kind == ops::GemmKind::gemmc && nameOf(p.f) == "gemm_c" + std::to_string(gc) + "_" + f16);
        }
    // precise rows with the flag (never set by the plan there): still the re-quantized weights (no gemvh / gemms on raw bytes)
    ops::GemmFlags sp = sim;
    sp.prec_dec = true;
    CHECK(ops::chooseGemmF16(gate, 8, deq2, sp, k).dequant && nameOf(ops::chooseGemmF16(gate, 8, deq2, sp, k).dq) == "w4s_dequant_f16_mxfp4");
    // no token-tail split under the arm (the tail would read the original MXFP4 bytes)
    ops::GemmFlags tail = sim;
    tail.gemm_tail = true;
    k.gemmh_f16 = tag("gemmh_f16");
    const ops::GemmPieces pc = ops::planGemmF16(gate, 2058, deq2, tail, k);
    CHECK(pc.n == 1 && pc.p[0].rows == 2058 && nameOf(pc.p[0].plan.dq) == "w4s_dequant_f16_mxfp4" && nameOf(pc.p[0].plan.f) == "gemmh_f16");
    ops::GemmFlags tail0 = base;
    tail0.gemm_tail = true;
    CHECK(ops::planGemmF16(gate, 2058, deq2, tail0, k).n == 1);  // the gfx1151 gemmfq plan: no split there either (unchanged)
    k.gemmh_f16 = nullptr;
    // off: the plans without the flag (gemmfq at 2048 rows, gemmhq below 1025, fused gemm_c for a fused choice)
    for (std::uint32_t n : {300u, 2048u})
        for (std::uint32_t tuned : {2u, G + 2}) {
            const ops::GemmPlan a = ops::chooseGemmF16(gate, n, static_cast<std::uint8_t>(tuned), base, k);
            CHECK(a.dq == nullptr || nameOf(a.dq) != "w4s_dequant_f16_mxfp4");
        }
    CHECK(ops::chooseGemmF16(gate, 2048, deq2, base, k).kind == ops::GemmKind::gemmfq);
    // another weight type, or a table without the kernel (gfx1201): the flag changes nothing (same flags but w4s_w)
    ops::GemmFlags s1 = base;
    s1.w4s_w = true;
    {
        const ops::GemmPlan a = ops::chooseGemmF16(q4, 2048, deq2, base, k), b = ops::chooseGemmF16(q4, 2048, deq2, s1, k);
        CHECK(a.kind == b.kind && a.f == b.f && a.dq == b.dq && a.dequant == b.dequant && a.grid.x == b.grid.x && a.grid.y == b.grid.y);
    }
    // round 2: GemmFlags::w4s_w = grid + 1 picks MatmulKernels::w4s_dq[grid]; out of range (6) = off
    for (std::uint8_t g = 1; g <= 5; ++g) {
        ops::GemmFlags sg = s1;
        sg.w4s_w = g;
        const ops::GemmPlan p = ops::chooseGemmF16(gate, 2048, deq2, sg, k);
        CHECK(p.dequant && nameOf(p.dq) == kn::kW4sDqNames[g - 1]);
        ops::GemmFlags tg = sg;
        tg.gemm_tail = true;
        k.gemmh_f16 = tag("gemmh_f16");
        CHECK(ops::planGemmF16(gate, 2058, deq2, tg, k).n == 1);  // no tail split for any grid
        k.gemmh_f16 = nullptr;
    }
    {
        ops::GemmFlags s6 = s1;
        s6.w4s_w = 6;
        CHECK(ops::chooseGemmF16(gate, 2048, deq2, s6, k).kind == ops::GemmKind::gemmfq);
    }
    k.w4s_dq = {};
    {
        const ops::GemmPlan a = ops::chooseGemmF16(gate, 2048, deq2, base, k), b = ops::chooseGemmF16(gate, 2048, deq2, s1, k);
        CHECK(a.kind == b.kind && a.f == b.f && a.dq == b.dq && a.dequant == b.dequant && a.grid.x == b.grid.x && a.grid.y == b.grid.y);
    }
}

// v0252-w4a4k2 round 3: the least-squares gain of the W4A4 fake-quant weight grids. ops::w4sGains turns the per-row terms
// num_r = <w_r, w_hat_r>, den_r = <w_hat_r, w_hat_r> (double, computed by the w4s_gain_terms kernel / its host reference) into
// one f32 gain per row: mode 1 (g) = sum num / sum den (rows in order) for every row, mode 2 (r) = num_r / den_r. A row (or
// tensor) with den 0 or a non-finite ratio gets 1. GemmFlags::w4s_gain (1 / 2) makes the plan's dequant the gain entry
// MatmulKernels::w4s_dqg[grid] (GemmPlan::dq_gain: the launch adds the gain array); without the entry the arm stays off.
// Fails to build before the change (no ops::w4sGains / GemmFlags::w4s_gain / MatmulKernels::w4s_dqg / GemmPlan::dq_gain).
void testOpsW4sGains() {
    // known scale errors: w_hat = 2 w (row 0), w_hat = w / 4 (row 1), a zero row (row 2), a NaN term (row 3)
    const double num[4] = {2.0 * 10.0, 0.25 * 10.0, 0.0, std::nan("")};
    const double den[4] = {4.0 * 10.0, 0.0625 * 10.0, 0.0, 1.0};
    float g[4] = {-1, -1, -1, -1};
    ops::w4sGains(num, den, 4, 2, g);
    CHECK(g[0] == 0.5f && g[1] == 4.0f && g[2] == 1.0f && g[3] == 1.0f);
    ops::w4sGains(num, den, 3, 1, g);  // tensor over rows 0..2: (20 + 2.5 + 0) / (40 + 0.625 + 0)
    const float t = static_cast<float>((20.0 + 2.5 + 0.0) / (40.0 + 0.625 + 0.0));
    CHECK(g[0] == t && g[1] == t && g[2] == t);
    ops::w4sGains(num, den, 4, 1, g);  // a NaN anywhere in the tensor -> 1 everywhere
    CHECK(g[0] == 1.0f && g[3] == 1.0f);
    const double z[2] = {0.0, 0.0};
    ops::w4sGains(z, z, 2, 1, g);
    CHECK(g[0] == 1.0f && g[1] == 1.0f);
    for (int m = 1; m <= 2; ++m) {
        const double hn[1] = {1e300}, hd[1] = {1e-300};  // ratio overflows to inf -> 1
        float o[1] = {0};
        ops::w4sGains(hn, hd, 1, m, o);
        CHECK(o[0] == 1.0f);
    }
    // plan: the gain entry of the grid, dq_gain set; no gain entry -> the arm stays off (the original plan)
    const std::uint32_t G = static_cast<std::uint32_t>(kn::kGemmCfgs.size());
    auto gate = mat(GgmlType::mxfp4, 17408, 5120);
    gate.row_bytes = 5120 / 256 * 136;
    static const kn::KernelTable kb = gemmTable(true, true, false);
    const auto kp = std::make_unique<kn::KernelTable>(kb);
    kn::KernelTable& k = *kp;
    k.gemmfq[1] = tag("gemmfq_t2_mxfp4");
    for (std::size_t i = 0; i < k.w4s_dq.size(); ++i) {
        k.w4s_dq[i] = tag(kn::kW4sDqNames[i]);
        k.w4s_dqg[i] = tag(kn::kW4sDqgNames[i]);
    }
    const std::uint8_t deq2 = static_cast<std::uint8_t>(G + 2);
    ops::GemmFlags f;
    f.fdq = 2;
    f.w4s_w = 4;  // grid w1d
    for (std::uint8_t m : {1, 2}) {
        f.w4s_gain = m;
        const ops::GemmPlan p = ops::chooseGemmF16(gate, 2048, deq2, f, k);
        CHECK(p.dequant && p.dq_gain && nameOf(p.dq) == kn::kW4sDqgNames[3]);
    }
    f.w4s_gain = 0;
    {
        const ops::GemmPlan p = ops::chooseGemmF16(gate, 2048, deq2, f, k);
        CHECK(p.dequant && !p.dq_gain && nameOf(p.dq) == kn::kW4sDqNames[3]);
    }
    CHECK(!ops::GemmPlan{}.dq_gain && ops::GemmFlags{}.w4s_gain == 0);
    f.w4s_gain = 2;
    k.w4s_dqg[3] = nullptr;
    CHECK(ops::chooseGemmF16(gate, 2048, deq2, f, k).kind == ops::GemmKind::gemmfq);
}

}  // namespace
