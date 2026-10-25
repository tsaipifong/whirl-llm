// Unit tests of the kernel-choice module (src/ops, ARCH-1 K-1..), CPU only, no GPU: fake kernel tables (a non-null
// Function is a name tag), the choice functions, and the selection tables (problem x flags x kernel set -> kernel,
// grid). The real launches are pinned on the GPUs by the K-0 launch tables (tools/impact/launch_table.py); these
// tests pin the rules: precise picks only the precise bit class, the tuned tables only move within a class, the grids.
// SPDX-License-Identifier: Apache-2.0
//
// Written for WHIRL. Run: whirl-ops-tests.exe (exit 0 = all passed).

#include "whirl/ops/attention.h"
#include "whirl/ops/matmul.h"
#include "whirl/ops/moe.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <deque>
#include <string>
#include <type_traits>

namespace {

int g_pass = 0, g_fail = 0;

void check(bool ok, const char* what, int line) {
    if (ok) {
        ++g_pass;
    } else {
        ++g_fail;
        std::printf("FAIL %s:%d: %s\n", __FILE__, line, what);
    }
}
#define CHECK(c) check((c), #c, __LINE__)

namespace ops = whirl::ops;
namespace kn = whirl::kernels;
using whirl::gguf::GgmlType;
using Fn = whirl::hip::Function;

// name tags: a fake Function points at its name
std::deque<std::string> g_names;
Fn tag(const std::string& n) {
    g_names.push_back(n);
    return const_cast<char*>(g_names.back().c_str());
}
std::string nameOf(Fn f) { return f == nullptr ? std::string("null") : std::string(static_cast<const char*>(f)); }
constexpr std::size_t ti(GgmlType t) { return static_cast<std::size_t>(t); }

// a code object with the int8 GEMV set for q4_k / q6_k / mxfp4 (one-token, multi-token, "g" twins, multi-row R=2),
// the f16 GEMM, gemv_f32_1, and optionally gemvw variant 1 for q4_k and gemvx for q4_k
kn::KernelTable fakeTable(bool gv_grp, bool gemvw, bool gemvx, bool fp8) {
    kn::KernelTable k;
    k.gv_grp = gv_grp;
    for (GgmlType t : {GgmlType::q4_k, GgmlType::q6_k, GgmlType::mxfp4}) {
        const std::string s = std::to_string(static_cast<int>(t));
        k.gemvq[ti(t)] = tag("gemvq_" + s);
        for (int nt = 2; nt <= 16; ++nt) {
            k.gemvq_nt[nt - 2][ti(t)] = tag("gemvq_nt" + std::to_string(nt) + "_" + s);
            k.gemvq_nt_g[nt - 2][ti(t)] = tag("ggemvq_nt" + std::to_string(nt) + "_" + s);
            k.gemvq_mr[0][nt - 2][ti(t)] = tag("gemvq_nt" + std::to_string(nt) + "r2_" + s);
            k.gemvq_mr_g[0][nt - 2][ti(t)] = tag("ggemvq_nt" + std::to_string(nt) + "r2_" + s);
            if (gemvw && t == GgmlType::q4_k) {
                k.gemvw[1][nt - 2][ti(t)] = tag("gemvw_nt" + std::to_string(nt) + "v1_" + s);
                k.gemvw_g[1][nt - 2][ti(t)] = tag("ggemvw_nt" + std::to_string(nt) + "v1_" + s);
            }
        }
    }
    k.gemv1[ti(GgmlType::f32)] = tag("gemv_f32_1");
    if (gemvx) k.gemvx[ti(GgmlType::q4_k)] = tag("gemvx_q4_k");
    k.gemvw_head_s = tag("gemvw_head_s");
    if (fp8) k.gemm8[0] = tag("gemm8_c0");
    return k;
}

ops::MatShape mat(GgmlType t, std::uint32_t rows, std::uint32_t cols, bool head = false, bool ref = false) {
    ops::MatShape m;
    m.ty = t;
    m.nrows = rows;
    m.ncols = cols;
    m.head = head;
    m.has_ref = ref;
    return m;
}

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

// ---- attention (K-3)
kn::KernelTable attnTable(bool ws4, bool group1, bool dq4, bool kg) {
    kn::KernelTable k;
    k.attn_split = tag("attn_split");
    k.attn_wsplit1 = tag("attn_wsplit1");
    k.attn_wsplit2 = tag("attn_wsplit2");
    k.attn_combine = tag("attn_combine");
    k.attn_combine_q8 = tag("attn_combine_q8");
    if (ws4) {
        k.attn_ws4 = tag("attn_ws4");
        for (int i = 0; i < 3; ++i) k.attn_ws4g[i] = tag("attn_ws4g" + std::to_string(i + 1));
    }
    if (dq4) k.attn_dq4 = tag("attn_dq4");
    k.caps.attn_group1 = group1;
    k.attn_prefill_wmma = tag("attn_prefill_wmma");
    k.attn_kx = tag("attn_kx");
    if (kg) {
        k.attn_kg2 = tag("attn_kg2");
        k.attn_kg4 = tag("attn_kg4");
        k.attn_kgf4 = tag("attn_kgf4");
        k.attn_kgs4 = tag("attn_kgs4");
        k.kv_dq_rows = tag("kv_dq_rows");
        k.kv_dq_rows_r = tag("kv_dq_rows_r");
    } else {
        k.attn_pwf = tag("attn_pwf");
        k.attn_pws = tag("attn_pws");
        k.kv_dq_rows_r = tag("kv_dq_rows_r");
    }
    return k;
}

void testAttnDecode() {
    const auto& p97 = whirl::device::k_profile_gfx1201;
    const auto& p8s = whirl::device::k_profile_gfx1151;
    ops::AttnDecodeProblem p;
    p.n = 1;
    p.head_dim = 256;
    p.n_head = 24;
    p.n_head_kv = 4;  // GQA group 6
    p.max_ctx = 32768;
    ops::AttnDecodeFlags f;
    // gfx1201-like set (no ws4, groups allowed): one decode row on attn_wsplit1, 64 splits, combine_q8 when the
    // o-projection is on the int8 GEMV
    const kn::KernelTable k97 = attnTable(false, false, true, true);
    p.o_on_gemv = true;
    auto pl = ops::chooseAttnDecode(p, f, k97, k97.caps, p97);
    CHECK(nameOf(pl.f) == "attn_wsplit1" && pl.kind == ops::AttnDecodeKind::wsplit && pl.grid.x == 4 && pl.grid.y == 64 && pl.grid.z == 1);
    CHECK(pl.block.x == 128 && pl.combine_q8 && pl.combine_grid.x == 24 && pl.combine_grid.y == 1);
    // precise never takes the int8 combine
    ops::AttnDecodeFlags fp = f;
    fp.prec_dec = true;
    CHECK(!ops::chooseAttnDecode(p, fp, k97, k97.caps, p97).combine_q8);
    // a 4-row verify at consecutive positions of one sequence: one group of 2 (16 / 6 columns), then 2
    const std::int32_t pos[4] = {1000, 1001, 1002, 1003}, base[4] = {0, 0, 0, 0};
    p.n = 4;
    p.rows_known = true;
    p.row_pos = pos;
    p.row_base = base;
    f.attn_wide = false;
    pl = ops::chooseAttnDecode(p, f, k97, k97.caps, p97);
    CHECK(nameOf(pl.f) == "attn_wsplit1" && pl.grid.z == 2 && pl.groups.first[1] == 2 && pl.groups.count[0] == 2);
    // with attn_wide: attn_wsplit2 takes 5 rows per group (32 / 6): one group of 4
    f.attn_wide = true;
    pl = ops::chooseAttnDecode(p, f, k97, k97.caps, p97);
    CHECK(nameOf(pl.f) == "attn_wsplit2" && pl.grid.z == 1 && pl.groups.count[0] == 4);
    // q4 KV with attn_dq4: one query per group on attn_dq4, 256 threads; the R9700 (no attn_group1) takes the
    // dynamic split count too (feat027-dq4r9700: same partition as the fixed 256 grid, so the same bits)
    p.kv_q4 = true;
    f.fd_splits = 256;
    pl = ops::chooseAttnDecode(p, f, k97, k97.caps, p97);
    CHECK(nameOf(pl.f) == "attn_dq4" && pl.kind == ops::AttnDecodeKind::dq4 && pl.grid.z == 4 && pl.block.x == 256);
    CHECK(pl.n_split == 8 && pl.grid.y == 8 && pl.n_split == ops::dq4Splits(std::span<const std::int32_t>(pos, 4), 0, 256));
    // WHIRL_DQ4_DYN=0: the fixed grid
    {
        ops::AttnDecodeFlags f0 = f;
        f0.dq4_dyn = false;
        CHECK(ops::chooseAttnDecode(p, f0, k97, k97.caps, p97).n_split == 256);
    }
    // the 8060S (attn_group1 set): the dynamic split count for short rows (same partition), FAST-8S-1
    const kn::KernelTable k8 = attnTable(true, true, true, false);
    pl = ops::chooseAttnDecode(p, f, k8, k8.caps, p8s);
    CHECK(nameOf(pl.f) == "attn_dq4" && pl.n_split < 256 && pl.n_split == ops::dq4Splits(std::span<const std::int32_t>(pos, 4), 0, 256));
    // ... a profile row decides it, not the kernel set: a row with attn_dq4_dynamic_splits off keeps the fixed grid
    {
        whirl::device::Profile off = p97;
        off.attn_dq4_dynamic_splits = false;
        CHECK(ops::chooseAttnDecode(p, f, k8, k8.caps, off).n_split == 256);
        CHECK(ops::chooseAttnDecode(p, f, k97, k97.caps, off).n_split == 256);
    }
    // host-planned rows only: device-position decode (the captured decode graph, rows_known false) keeps the grid
    {
        ops::AttnDecodeProblem pd = p;
        pd.rows_known = false;
        CHECK(ops::chooseAttnDecode(pd, f, k97, k97.caps, p97).n_split == 256);
    }
    // f16 KV on the 8060S (an f16-KV kernel table has no attn_dq4): attn_ws4g for a multi-row verify group (64 splits)
    const kn::KernelTable k8f = attnTable(true, true, false, false);
    p.kv_q4 = false;
    f.fd_splits = 64;
    pl = ops::chooseAttnDecode(p, f, k8f, k8f.caps, p8s);
    CHECK(pl.kind == ops::AttnDecodeKind::ws4g && nameOf(pl.f).rfind("attn_ws4g", 0) == 0 && pl.grid.y == 64 && pl.grid.z == 1 && pl.block.x == 256);
    // rows of two sequences never share a group
    const std::int32_t base2[4] = {0, 0, 4096, 4096};
    p.row_base = base2;
    pl = ops::chooseAttnDecode(p, f, k8f, k8f.caps, p8s);
    CHECK(pl.grid.z == 2);
    // WHIRL_ATTN_WS4=0 on the 8060S: attn_wsplit1, one query per group (attn_group1)
    f.attn_ws4 = false;
    pl = ops::chooseAttnDecode(p, f, k8f, k8f.caps, p8s);
    CHECK(nameOf(pl.f) == "attn_wsplit1" && pl.grid.z == 4);
    // dbg_flags bit 2 forces attn_split (grid.z = rows, 256 threads)
    f.dbg_flags = 2;
    pl = ops::chooseAttnDecode(p, f, k8f, k8f.caps, p8s);
    CHECK(nameOf(pl.f) == "attn_split" && pl.kind == ops::AttnDecodeKind::split && pl.grid.z == 4 && pl.block.x == 256);
    CHECK(ops::awPer(1, 64) == 64 && ops::awPer(4096, 64) == 64 && ops::awPer(4097, 64) == 128);
}

// feat027-dq4r9700: host model of attn_dq4's split ranges, written from kernels/attn.hip (gfx1201) and
// kernels/gfx1151/attn.hip (the same code: dq_per, DQ_NW 8, DQ_MAXS 128, the MTP draft-window branch), not from
// ops::dq4Part. A dynamic grid of s splits is bitwise the fixed grid of `full` splits when every split i < s has
// the same [b0, b1) under both grids (same partial) and every split i >= s of the full grid is empty (b0 >= n_pos:
// it writes m = -inf, which attn_combine / attn_combine_q8 skip, and fmaxf(M, -inf) == M): the combine then sees
// the same partials in the same order.
struct KRange {
    std::int32_t b0, b1;
};
std::int32_t kDqPer(std::int32_t n_keys, std::int32_t n_split) {
    const std::int32_t ns = std::min<std::int32_t>(n_split, 128);
    return std::max<std::int32_t>(8 * 32, (n_keys + ns * 64 - 1) / (ns * 64) * 64);
}
KRange kRange(std::int32_t n_pos, std::int32_t split, std::int32_t n_split, std::int32_t win) {
    std::int32_t per = kDqPer(n_pos, n_split);
    std::int32_t b0 = split * per, b1 = std::min(n_pos, b0 + per);
    if (win != 0 && n_split > 1) {
        const std::int32_t w = (win & 0xffff) * 64, min_pos = (win >> 16) * 1024;
        const std::int32_t w0 = std::max<std::int32_t>(256, (n_pos - w) & ~63);
        if (n_pos >= min_pos && w0 > 256) {
            if (split == 0) {
                b0 = 0;
                b1 = 256;
            } else {
                per = kDqPer(n_pos - w0, n_split - 1);
                b0 = w0 + (split - 1) * per;
                b1 = std::min(n_pos, b0 + per);
            }
        }
    }
    return {b0, b1};
}
// the dynamic grid s gives row n_pos the fixed grid's partials and combine sequence
bool samePartials(std::int32_t n_pos, std::uint32_t s, std::uint32_t full, std::int32_t win) {
    if (s > full) return false;
    for (std::uint32_t i = 0; i < full; ++i) {
        const KRange b = kRange(n_pos, static_cast<std::int32_t>(i), static_cast<std::int32_t>(full), win);
        if (i < s) {
            const KRange a = kRange(n_pos, static_cast<std::int32_t>(i), static_cast<std::int32_t>(s), win);
            const bool ea = a.b0 >= n_pos, eb = b.b0 >= n_pos;
            if (ea != eb) return false;
            if (!ea && (a.b0 != b.b0 || a.b1 != b.b1)) return false;
        } else if (b.b0 < n_pos) {
            return false;  // a keyed split the dynamic grid does not launch
        }
    }
    return true;
}
std::int32_t winArg(std::uint32_t w, std::uint32_t min_ctx) {  // qwen35::draftWindowArg
    if (w == 0) return 0;
    const std::uint32_t lo = std::min<std::uint32_t>((w + 63) / 64, 0xffffu), hi = std::min<std::uint32_t>(min_ctx / 1024, 0x7fffu);
    return static_cast<std::int32_t>(lo | hi << 16);
}

void testDq4Partition() {
    // windows: off, the default (16384 keys from 64k), a short one from 0, one from 16k
    const std::int32_t wins[4] = {0, winArg(16384, 65536), winArg(4096, 0), winArg(1024, 16384)};
    // every n_pos 1 .. 256k for the q4 default grid (256) and a non-power-of-two WHIRL_FD_SPLITS (100); step 7 for
    // the smaller grids (max_ctx / 64 caps the grid at short contexts)
    const std::uint32_t fulls[6] = {256, 100, 128, 64, 32, 16};
    long long rows = 0, bad = 0, smaller = 0, notmin = 0;
    for (const std::uint32_t full : fulls) {
        const std::int32_t step = (full == 256 || full == 100) ? 1 : 7;
        for (const std::int32_t win : wins) {
            for (std::int32_t n_pos = 1; n_pos <= 262144; n_pos += step) {
                const std::int32_t pos = n_pos - 1;
                const std::uint32_t s = ops::dq4Splits(std::span<const std::int32_t>(&pos, 1), win, full);
                rows += 1;
                if (!samePartials(n_pos, s, full, win)) bad += 1;
                if (s < full) smaller += 1;
                // minimal: every smaller candidate (8, 16, ... < s) changes some partial (kernel model), so the
                // choice is the cheapest bitwise grid (a mutant that keeps the full grid or a larger count fails)
                for (std::uint32_t c = 8; c < s; c *= 2)
                    if (samePartials(n_pos, c, full, win)) {
                        notmin += 1;
                        break;
                    }
            }
        }
    }
    CHECK(bad == 0);
    CHECK(smaller > 0);
    CHECK(notmin == 0);
    // Why dq4Splits' third test (keyed splits a[2] <= s) never decides (REVIEW feat027 mutcheck: the mutant that
    // drops it survived): when the window branch and per match, per = dq_per(n, min(s, 128)) >= ceil(n / s) (dq_per
    // rounds n / (ns * 64) up, times 64), so ceil(n / per) <= s; in the window branch per = dq_per(n - w0, s - 1)
    // gives 1 + ceil((n - w0) / per) <= s. The kernel model below shows it for every n_pos 1 .. 256k, s = 8 .. 128
    // and the four windows: a matching per with more keyed splits than s never occurs. The check stays in
    // dq4Splits as a guard; mutations.json carries killable mutants instead (per / branch unchecked, always full).
    long long redundant_hits = 0;
    for (const std::int32_t win : wins)
        for (std::int32_t n_pos = 1; n_pos <= 262144; ++n_pos)
            for (std::uint32_t c = 8; c <= 128; c *= 2) {
                // same branch and per under grid c and grid 256 <=> splits 0 and 1 have the same ranges (split 1
                // starts at per, or at w0 with length per in the window branch; split 1 empty: one keyed split)
                const KRange a0 = kRange(n_pos, 0, static_cast<std::int32_t>(c), win), b0 = kRange(n_pos, 0, 256, win);
                const KRange a1 = kRange(n_pos, 1, static_cast<std::int32_t>(c), win), b1 = kRange(n_pos, 1, 256, win);
                if (a0.b0 != b0.b0 || a0.b1 != b0.b1 || a1.b0 != b1.b0 || a1.b1 != b1.b1) continue;
                // keyed splits of grid 256 beyond c
                if (kRange(n_pos, static_cast<std::int32_t>(c), 256, win).b0 < n_pos) redundant_hits += 1;
            }
    CHECK(redundant_hits == 0);
    // verify / MTP row sets: k consecutive rows of one sequence (k = 2, 9, 16) and two sequences far apart; every
    // row of the set must keep its partials under the set's one split count
    long long sets = 0, sbad = 0;
    for (const std::int32_t win : wins) {
        for (std::int32_t p0 = 0; p0 < 262144; p0 += 61) {
            std::int32_t rp[16];
            for (std::int32_t k = 2; k <= 16; k += 7) {
                for (std::int32_t r = 0; r < k; ++r) rp[r] = std::min(p0 + r, 262143);
                const std::uint32_t s = ops::dq4Splits(std::span<const std::int32_t>(rp, static_cast<std::size_t>(k)), win, 256);
                sets += 1;
                for (std::int32_t r = 0; r < k; ++r)
                    if (!samePartials(rp[r] + 1, s, 256, win)) sbad += 1;
            }
            const std::int32_t two[2] = {p0, (p0 * 7 + 4099) % 262144};
            const std::uint32_t s2 = ops::dq4Splits(std::span<const std::int32_t>(two, 2), win, 256);
            sets += 1;
            for (const std::int32_t q : two)
                if (!samePartials(q + 1, s2, 256, win)) sbad += 1;
        }
    }
    CHECK(sbad == 0);
    // n_pos 3000: grid 8 has per 384 (the 256 grid: 256), so its partials differ; grid 16 has per 256 = the 256 grid's
    CHECK(!samePartials(3000, 8, 256, 0) && samePartials(3000, 16, 256, 0));
    {
        const std::int32_t pos = 2999;
        CHECK(ops::dq4Splits(std::span<const std::int32_t>(&pos, 1), 0, 256) == 16);
    }
    // launch-table snapshot (gfx1201 / gfx1151 profile rows, q4 KV, fd_splits 256, one decode / verify row at
    // n_pos keys): the attn_dq4 grid.y and the combine's n_split. Computed by an independent Python model
    // (progress/feat027-dq4r9700.HANDOFF.md); window off and the default window give the same column.
    struct Snap {
        std::int32_t n_pos;
        std::uint32_t s;
    };
    const Snap snap[] = {{1, 8},       {256, 8},     {257, 8},     {2048, 8},    {2049, 16},    {4096, 16},    {4097, 32},     {8192, 32},    {8193, 64},
                         {16384, 64},  {16385, 128}, {32768, 128}, {65536, 128}, {65537, 128}, {131072, 128}, {131073, 128}, {262144, 128}};
    ops::AttnDecodeProblem p;
    p.n = 1;
    p.head_dim = 256;
    p.n_head = 24;
    p.n_head_kv = 4;
    p.max_ctx = 262144;
    p.kv_q4 = true;
    p.rows_known = true;
    const std::int32_t base[1] = {0};
    p.row_base = base;
    ops::AttnDecodeFlags f;
    f.fd_splits = 256;
    const kn::KernelTable k97 = attnTable(false, false, true, true);
    const kn::KernelTable k8 = attnTable(true, true, true, false);
    long long snap_bad = 0;
    for (const Snap& e : snap) {
        const std::int32_t pos[1] = {e.n_pos - 1};
        p.row_pos = pos;
        for (const std::int32_t win : {wins[0], wins[1]}) {
            p.win = win;
            const auto a = ops::chooseAttnDecode(p, f, k97, k97.caps, whirl::device::k_profile_gfx1201);
            const auto b = ops::chooseAttnDecode(p, f, k8, k8.caps, whirl::device::k_profile_gfx1151);
            if (nameOf(a.f) != "attn_dq4" || a.n_split != e.s || a.grid.y != e.s || nameOf(b.f) != "attn_dq4" || b.n_split != e.s) {
                std::printf("  snapshot n_pos %d win %d: gfx1201 %u, gfx1151 %u, want %u\n", e.n_pos, win, a.n_split, b.n_split, e.s);
                snap_bad += 1;
            }
        }
    }
    CHECK(snap_bad == 0);
    std::printf("  dq4 partition: %lld rows (%lld on a smaller grid), %lld row sets, mismatches %lld / %lld, not minimal %lld, keyed > s %lld\n",
                rows, smaller, sets, bad, sbad, notmin, redundant_hits);
}

void testAttnPrefill() {
    ops::AttnPrefillProblem p;
    p.n = 512;
    p.pos0 = 0;
    p.head_dim = 256;
    p.n_head = 24;
    p.n_head_kv = 4;
    p.scale = 1.0f / 16.0f;  // 1 / sqrt(256): a power of two
    ops::AttnPrefillFlags f;
    // attn_kg set, f16 KV: attn_kg4 (group 6 is not a multiple of 4 -> kg2), one launch, from the cache
    kn::KernelTable k = attnTable(false, false, false, true);
    auto pl = ops::chooseAttnPrefill(p, f, k);
    CHECK(pl.steps.size() == 1 && nameOf(pl.steps[0].f) == "attn_kg2" && !pl.steps[0].from_copy && pl.steps[0].grid.x == 32 && pl.steps[0].grid.y == 12 &&
          pl.steps[0].block.x == 128);
    // q8 KV with the dequant scratch: kv_dq_rows once, then attn_kgf (only kgf4 exists here -> kg2's kgf2 is null -> falls through)
    p.kv_q8 = true;
    p.dq_rows = 8192;
    p.dq_ptab = true;
    p.dq_st_ml = true;
    pl = ops::chooseAttnPrefill(p, f, k);
    CHECK(!pl.steps.empty());
    // a non-power-of-two scale: no attn_kg; attn_kx (512 threads)
    p.kv_q8 = false;
    p.scale = 0.1f;
    pl = ops::chooseAttnPrefill(p, f, k);
    CHECK(pl.steps.size() == 1 && nameOf(pl.steps[0].f) == "attn_kx" && pl.steps[0].block.x == 512 && pl.steps[0].grid.x == 4 && pl.steps[0].grid.y == 24);
    // the 8060S-like set (no attn_kg, pwf / pws): q8 KV in one dequantized range -> kv_dq_rows_r + attn_pwf
    k = attnTable(true, true, true, false);
    p.scale = 1.0f / 16.0f;
    p.kv_q8 = true;
    pl = ops::chooseAttnPrefill(p, f, k);
    CHECK(pl.steps.size() == 2 && pl.steps[0].op == ops::AttnStep::Op::dq_rows_r && nameOf(pl.steps[1].f) == "attn_pwf" && pl.steps[1].from_copy &&
          !pl.steps[1].ranged);
    // keys past the scratch: ranges of dq_rows, each dequantized and walked by attn_pws with the state carried
    p.pos0 = 20000;
    pl = ops::chooseAttnPrefill(p, f, k);
    std::size_t dq = 0, ranged = 0;
    for (const auto& s : pl.steps) {
        dq += s.op == ops::AttnStep::Op::dq_rows_r;
        ranged += s.op == ops::AttnStep::Op::attn && s.ranged && nameOf(s.f) == "attn_pws";
    }
    CHECK(dq == 3 && ranged == 3 && pl.steps.back().r1 == 20512);
    // a long prefill splits the heads (Windows TDR budget): 4096 x 128k query-key pairs per launch
    p.kv_q8 = false;
    p.pos0 = 0;
    p.n = 8192;
    p.dq_rows = 0;
    k = attnTable(false, false, false, true);
    p.scale = 0.1f;
    p.n = 4096;
    p.pos0 = 131072;
    pl = ops::chooseAttnPrefill(p, f, k);
    CHECK(pl.steps.size() == 2 && pl.steps[1].head0 == 12 && pl.steps[0].grid.y == 12);
}

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
}

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

}  // namespace

int main() {
    testBitClasses();
    testPreciseRule();
    testMatmulPath();
    testGemvTile();
    testGemvWide();
    testOpsGemm();
    testAttnDecode();
    testDq4Partition();
    testAttnPrefill();
    testOpsMoe();
    testOpTables();
    std::printf("ops tests: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
