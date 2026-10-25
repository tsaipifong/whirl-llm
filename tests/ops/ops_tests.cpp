// Unit tests of the kernel-choice module (src/ops, ARCH-1 K-1..), CPU only, no GPU: fake kernel tables (a non-null
// Function is a name tag), the choice functions, and the selection tables (problem x flags x kernel set -> kernel,
// grid). The real launches are pinned on the GPUs by the K-0 launch tables (tools/impact/launch_table.py); these
// tests pin the rules: precise picks only the precise bit class, the tuned tables only move within a class, the grids.
// SPDX-License-Identifier: Apache-2.0
//
// Written for WHIRL. Run: whirl-ops-tests.exe (exit 0 = all passed).

#include "whirl/ops/attention.h"
#include "whirl/ops/cascade.h"
#include "whirl/ops/cascade_kernels.h"
#include "whirl/ops/matmul.h"
#include "whirl/ops/moe.h"

#include <cmath>
#include <cstdio>
#include <map>
#include <random>
#include <vector>
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
    // q4 KV with attn_dq4: one query per group on attn_dq4, 256 threads; the R9700 keeps the fixed grid
    p.kv_q4 = true;
    f.fd_splits = 256;
    pl = ops::chooseAttnDecode(p, f, k97, k97.caps, p97);
    CHECK(nameOf(pl.f) == "attn_dq4" && pl.kind == ops::AttnDecodeKind::dq4 && pl.grid.z == 4 && pl.block.x == 256 && pl.n_split == 256);
    // the 8060S (attn_group1 set): the dynamic split count for short rows (same partition), FAST-8S-1
    const kn::KernelTable k8 = attnTable(true, true, true, false);
    pl = ops::chooseAttnDecode(p, f, k8, k8.caps, p8s);
    CHECK(nameOf(pl.f) == "attn_dq4" && pl.n_split < 256 && pl.n_split == ops::dq4Splits(std::span<const std::int32_t>(pos, 4), 0, 256));
    // ... a profile row decides it, not the kernel set: the same kernels under the gfx1201 row keep the fixed grid
    CHECK(ops::chooseAttnDecode(p, f, k8, k8.caps, p97).n_split == 256);
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

// ---- feat026-cascade: shared-prefix planner, prefix kernel choice, merge numerics
void testCascadePlan() {
    // page tables by base: A, B, C share pages 10..17 (8 pages = 2048 keys), D shares only 3 of them
    std::map<std::int32_t, std::vector<std::int32_t>> pt;
    auto mk = [](std::int32_t priv0, std::int32_t shared_n) {
        std::vector<std::int32_t> v;
        for (std::int32_t i = 0; i < 16; ++i) v.push_back(i < shared_n ? 10 + i : priv0 + i);
        return v;
    };
    pt[0] = mk(100, 8);
    pt[64] = mk(200, 8);
    pt[128] = mk(300, 8);
    pt[192] = mk(400, 3);
    const ops::CascadePages pages = [&](std::int32_t b) -> std::span<const std::int32_t> {
        auto it = pt.find(b);
        return it == pt.end() ? std::span<const std::int32_t>{} : std::span<const std::int32_t>(it->second);
    };
    ops::CascadeConfig cc;
    cc.on = true;
    cc.min_keys = 2048;
    {
        const std::vector<std::int32_t> pos = {3000, 2500, 2100, 4000}, base = {0, 64, 128, 192};
        const auto pl = ops::planCascade(pos, base, pages, cc, 32);
        CHECK(pl.groups.size() == 1 && pl.groups[0].first == 0 && pl.groups[0].count == 3 && pl.groups[0].keys == 2048);
        CHECK(pl.keys[0] == 2048 && pl.keys[1] == 2048 && pl.keys[2] == 2048 && pl.keys[3] == 0);
        const auto t = ops::cascadeRows(pl, pos, base);
        CHECK(t.spos[0] == 3000 - 2048 && t.sbase[0] == 0 + 8 && t.ppos[0] == 2047);
        CHECK(t.spos[2] == 2100 - 2048 && t.sbase[2] == 128 + 8 && t.ppos[2] == 2047);
        CHECK(t.spos[3] == 4000 && t.sbase[3] == 192 && t.ppos[3] == -1);
        ops::CascadeConfig off = cc;
        off.on = false;
        CHECK(!ops::planCascade(pos, base, pages, off, 32).any());
        CHECK(!ops::planCascade(pos, base, pages, cc, 1).any());  // no multi-row kernel
        ops::CascadeConfig big = cc;
        big.min_keys = 2049;  // 9 pages: not shared that far
        CHECK(!ops::planCascade(pos, base, pages, big, 32).any());
    }
    {
        // a row whose own position caps the prefix below min_keys breaks the run (it keeps >= 1 suffix key)
        pt[320] = mk(500, 8);
        const std::vector<std::int32_t> pos = {3000, 2500, 2000, 2600}, base = {0, 64, 128, 320};
        const auto pl = ops::planCascade(pos, base, pages, cc, 32);
        CHECK(pl.groups.size() == 1 && pl.groups[0].first == 0 && pl.groups[0].count == 2);
        CHECK(pl.keys[2] == 0 && pl.keys[3] == 0);  // C alone, then the 4th: a run of one sequence
    }
    {
        // a position exactly on a page boundary: P = pos (keys [0, pos) prefix, the row's own key in the suffix)
        const std::vector<std::int32_t> pos = {2048, 2048}, base = {0, 64};
        const auto pl = ops::planCascade(pos, base, pages, cc, 32);
        CHECK(pl.any() && pl.groups[0].keys == 2048);
        CHECK(ops::cascadeRows(pl, pos, base).spos[0] == 0);
    }
    {
        // MTP verify rows: 3 rows per sequence, the group takes whole sequences and must fit max_rows
        const std::vector<std::int32_t> pos = {3000, 3001, 3002, 2500, 2501, 2502}, base = {0, 0, 0, 64, 64, 64};
        const auto pl = ops::planCascade(pos, base, pages, cc, 32);
        CHECK(pl.groups.size() == 1 && pl.groups[0].count == 6 && pl.keys[5] == 2048);
        CHECK(!ops::planCascade(pos, base, pages, cc, 5).any());
    }
    {
        // unmapped page 0 (-1): nothing shared
        pt[0][0] = -1;
        const std::vector<std::int32_t> pos = {3000, 2500}, base = {0, 64};
        CHECK(!ops::planCascade(pos, base, pages, cc, 32).any());
        pt[0][0] = 10;
    }
    {
        // two groups in one batch (two system prompts)
        pt[448] = mk(600, 0);
        pt[512] = mk(700, 0);
        for (std::int32_t i = 0; i < 12; ++i) pt[448][i] = pt[512][i] = 900 + i;
        const std::vector<std::int32_t> pos = {3000, 2500, 3300, 3400}, base = {0, 64, 448, 512};
        const auto pl = ops::planCascade(pos, base, pages, cc, 32);
        CHECK(pl.groups.size() == 2 && pl.groups[0].keys == 2048 && pl.groups[1].first == 2 && pl.groups[1].keys == 12 * 256);
    }
}

void testCascadePrefixChoice() {
    ops::CascadePlan pl;
    auto grp = [&](std::uint32_t first, std::uint32_t count) {
        pl = ops::CascadePlan{};
        pl.groups.push_back(ops::CascadeGroup{first, count, 2048});
    };
    ops::CascadePrefixProblem p;
    p.n_head = 24;
    p.n_head_kv = 4;
    p.n_split = 64;
    {
        const auto k = attnTable(false, false, false, false);  // gfx1201-like
        CHECK(ops::cascadeGroupCap(p, k, k.caps) == 5);
        grp(0, 4);
        auto c = ops::chooseCascadePrefix(pl, p, k, k.caps);
        CHECK(c.kind == ops::CascadePrefixKind::wsplit2 && nameOf(c.f) == "attn_wsplit2" && c.grid.x == 4 && c.grid.y == 64 && c.grid.z == 1 &&
              c.block.x == 128 && c.groups.first[0] == 0 && c.groups.count[0] == 4);
        grp(2, 2);
        c = ops::chooseCascadePrefix(pl, p, k, k.caps);
        CHECK(c.kind == ops::CascadePrefixKind::wsplit1 && nameOf(c.f) == "attn_wsplit1" && c.groups.first[0] == 2);
        grp(0, 6);  // past the cap
        CHECK(ops::chooseCascadePrefix(pl, p, k, k.caps).kind == ops::CascadePrefixKind::none);
        ops::CascadePrefixProblem p8 = p;
        p8.head_dim = 128;
        CHECK(ops::cascadeGroupCap(p8, k, k.caps) == 0);
    }
    {
        const auto k = attnTable(true, true, true, false);  // gfx1151-like
        CHECK(ops::cascadeGroupCap(p, k, k.caps) == 32);
        grp(0, 4);
        auto c = ops::chooseCascadePrefix(pl, p, k, k.caps);
        CHECK(c.kind == ops::CascadePrefixKind::ws4g && nameOf(c.f) == "attn_ws4g2" && c.grid.x == 4 && c.block.x == 256);
        grp(0, 10);  // CG 3, 8 rows per block: two column groups
        c = ops::chooseCascadePrefix(pl, p, k, k.caps);
        CHECK(c.kind == ops::CascadePrefixKind::ws4g && nameOf(c.f) == "attn_ws4g3" && c.grid.x == 8);
        ops::CascadePrefixProblem q4 = p;
        q4.kv_q4 = true;
        CHECK(ops::cascadeGroupCap(q4, k, k.caps) == 0);
        const auto k1 = attnTable(false, true, false, false);  // attn_group1 without attn_ws4g: no cascade
        CHECK(ops::cascadeGroupCap(p, k1, k1.caps) == 0);
    }
    const auto k0 = attnTable(false, false, false, false);
    CHECK(ops::chooseCascadePrefix(ops::CascadePlan{}, p, k0, k0.caps).kind == ops::CascadePrefixKind::none);
}

// feat026-cascade-r1 C2: the planner with the REAL per-arch group cap (cascadeGroupCap of a gfx1201-like / gfx1151-like
// table) for MTP verify rows. gfx1201: attn_wsplit2 holds 32 columns, so 27B (GQA 6) caps a prefix group at 5 rows and
// Ornith (GQA 8) at 4. With MTP every sequence brings nd + 1 rows: two sequences of 4 rows (nd 3) never fit, the plan
// is empty and the cascade does nothing (documented limit: the plain path already reads the prefix once per sequence
// for all of its MTP rows, so grouping across sequences needs a wider prefix kernel). nd 1 (2 rows) still fits.
// gfx1151 (attn_ws4g with column-group blocks, cap 32) groups MTP rows of 4 sequences.
void testCascadeCapMtp() {
    // S sequences x r rows (consecutive positions), all sharing pages 0..15 (4096 keys), then own pages
    auto planFor = [](std::uint32_t S, std::uint32_t r, std::uint32_t cap) {
        std::vector<std::int32_t> rp, rb;
        std::vector<std::vector<std::int32_t>> tabs(S);
        for (std::uint32_t s = 0; s < S; ++s) {
            for (std::int32_t i = 0; i < 20; ++i) tabs[s].push_back(i < 16 ? i : 100 + static_cast<std::int32_t>(s) * 10 + i);
            for (std::uint32_t j = 0; j < r; ++j) {
                rp.push_back(4096 + 300 + static_cast<std::int32_t>(7 * s + j));
                rb.push_back(static_cast<std::int32_t>(s) * 64);
            }
        }
        const ops::CascadePages pages = [&tabs](std::int32_t b) -> std::span<const std::int32_t> {
            const std::size_t s = static_cast<std::size_t>(b / 64);
            return s < tabs.size() ? std::span<const std::int32_t>(tabs[s]) : std::span<const std::int32_t>{};
        };
        ops::CascadeConfig cc;
        cc.on = true;
        return ops::planCascade(rp, rb, pages, cc, cap);
    };
    ops::CascadePrefixProblem p27, p35;
    p27.n_head = 24;
    p27.n_head_kv = 4;
    p35.n_head = 16;
    p35.n_head_kv = 2;
    const auto k01 = attnTable(false, false, false, false);  // gfx1201-like
    const auto k51 = attnTable(true, true, true, false);     // gfx1151-like
    const std::uint32_t c27 = ops::cascadeGroupCap(p27, k01, k01.caps), c35 = ops::cascadeGroupCap(p35, k01, k01.caps);
    CHECK(c27 == 5 && c35 == 4);
    CHECK(ops::cascadeGroupCap(p27, k51, k51.caps) == 32 && ops::cascadeGroupCap(p35, k51, k51.caps) == 32);
    // gfx1201, MTP nd 3 (4 rows / sequence), 4 slots: no group fits -> empty plan; with the cap lifted the rows
    // share the prefix (what Model::cascadeBegin counts as "nofit" and logs)
    for (const std::uint32_t cap : {c27, c35}) {
        CHECK(!planFor(4, 4, cap).any());
        CHECK(!planFor(2, 4, cap).any());
        CHECK(planFor(4, 4, ops::kCascadeMaxRows).any());
    }
    // nd 2 (3 rows): 27B 3 + 3 > 5, empty; nd 1 (2 rows): 27B two sequences per group (4 rows), Ornith two (4 rows)
    CHECK(!planFor(4, 3, c27).any());
    {
        const ops::CascadePlan pl = planFor(4, 2, c27);
        CHECK(pl.groups.size() == 2 && pl.groups[0].count == 4 && pl.groups[1].first == 4 && pl.groups[1].count == 4 && pl.groups[0].keys == 4096);
        const ops::CascadePlan p8 = planFor(4, 2, c35);
        CHECK(p8.groups.size() == 2 && p8.groups[0].count == 4);
    }
    // no MTP (1 row / sequence): 27B 4 slots in one group, Ornith 4 in one group (cap 4)
    CHECK(planFor(4, 1, c27).groups.size() == 1 && planFor(4, 1, c27).groups[0].count == 4);
    CHECK(planFor(5, 1, c35).groups.size() == 1 && planFor(5, 1, c35).groups[0].count == 4);  // the 5th row alone: no group
    // gfx1151: MTP nd 3, 4 slots -> one group of 16 rows; chooseCascadePrefix takes it (attn_ws4g3, column-group blocks)
    {
        const ops::CascadePlan pl = planFor(4, 4, 32);
        CHECK(pl.groups.size() == 1 && pl.groups[0].count == 16);
        p27.n_split = 64;
        const auto c = ops::chooseCascadePrefix(pl, p27, k51, k51.caps);
        CHECK(c.kind == ops::CascadePrefixKind::ws4g && nameOf(c.f) == "attn_ws4g3" && c.grid.x == 4 * 2 && c.grid.y == 64);
    }
}

// cascade (prefix splits merged into suffix split 0, then the combine) vs the plain split-K path vs an exact
// double softmax: the same attention up to f32 rounding (not bitwise: balance / fast only)
void testCascadeNumerics() {
    const std::size_t hd = 32;
    const std::int32_t nk = 1200;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<std::vector<float>> K(nk, std::vector<float>(hd)), V(nk, std::vector<float>(hd));
    for (auto& r : K)
        for (float& x : r) x = nd(rng) * 1.5f;
    for (auto& r : V)
        for (float& x : r) x = nd(rng);
    const std::function<std::span<const float>(std::int32_t)> key = [&](std::int32_t p) { return std::span<const float>(K[p]); };
    const std::function<std::span<const float>(std::int32_t)> val = [&](std::int32_t p) { return std::span<const float>(V[p]); };
    const float scale = 1.f / std::sqrt(static_cast<float>(hd));
    auto splits = [&](std::span<const float> q, std::int32_t k0, std::int32_t n_keys, std::int32_t pos, std::uint32_t ns) {
        // attn_wsplit split rule on keys [k0, k0 + n_keys): per = ceil(n / (ns 64)) 64
        std::vector<ops::AttnPartial> v;
        const std::int32_t per = ops::awPer(n_keys, ns);
        for (std::uint32_t s = 0; s < ns; ++s) {
            const std::int32_t b0 = static_cast<std::int32_t>(s) * per, b1 = std::min(n_keys, b0 + per);
            if (b0 >= n_keys) {
                v.push_back(ops::AttnPartial{});
                v.back().acc.assign(hd, 12345.f);  // never written by the kernels: must be ignored
                continue;
            }
            v.push_back(ops::cascadePartialRef(q, key, val, k0 + b0, k0 + b1, pos, scale));
        }
        return v;
    };
    float worst_cas = 0.f, worst_flat = 0.f, worst_diff = 0.f;
    for (const std::int32_t pos : {512, 700, 1023, 1199}) {
        for (const std::int32_t P : {256, 512}) {
            if (P > pos) continue;
            std::vector<float> q(hd);
            for (float& x : q) x = nd(rng);
            // plain
            const auto flat = splits(q, 0, pos + 1, pos, 8);
            const auto of = ops::cascadeCombineRef(flat, hd);
            // cascade: suffix keys [P, pos], prefix keys [0, P) (position P - 1: all visible), merge, combine
            auto suf = splits(q, P, pos + 1 - P, pos, 8);
            const auto pre = splits(q, 0, P, P - 1, 8);
            ops::cascadeMergeRef(suf[0], pre);
            const auto oc = ops::cascadeCombineRef(suf, hd);
            // exact
            std::vector<double> w(pos + 1);
            double mx = -1e300, sum = 0;
            for (std::int32_t p = 0; p <= pos; ++p) {
                double sc = 0;
                for (std::size_t d = 0; d < hd; ++d) sc += static_cast<double>(q[d]) * K[p][d];
                w[p] = sc * scale;
                mx = std::max(mx, w[p]);
            }
            for (double& x : w) sum += (x = std::exp(x - mx));
            for (std::size_t d = 0; d < hd; ++d) {
                double e = 0;
                for (std::int32_t p = 0; p <= pos; ++p) e += w[p] * V[p][d];
                e /= sum;
                worst_cas = std::max(worst_cas, static_cast<float>(std::fabs(oc[d] - e)));
                worst_flat = std::max(worst_flat, static_cast<float>(std::fabs(of[d] - e)));
                worst_diff = std::max(worst_diff, std::fabs(oc[d] - of[d]));
            }
        }
    }
    std::printf("  cascade numerics: max |cascade - exact| %.3g, |plain - exact| %.3g, |cascade - plain| %.3g\n", worst_cas, worst_flat, worst_diff);
    CHECK(worst_cas < 1e-5f);
    CHECK(worst_diff < 1e-5f);
    CHECK(std::isfinite(worst_cas) && std::isfinite(worst_diff));
    // an all-empty prefix leaves the suffix partial as it was; an empty suffix split 0 takes the prefix alone
    ops::AttnPartial a;
    a.m = 1.f;
    a.l = 2.f;
    a.acc = {3.f, 4.f};
    std::vector<ops::AttnPartial> none(2);
    for (auto& x : none) x.acc = {99.f, 99.f};
    ops::cascadeMergeRef(a, none);
    CHECK(a.m == 1.f && a.l == 2.f && a.acc[0] == 3.f && a.acc[1] == 4.f);
    ops::AttnPartial e;
    e.acc = {std::nanf(""), std::nanf("")};  // unwritten acc of an empty split
    std::vector<ops::AttnPartial> one(1, a);
    ops::cascadeMergeRef(e, one);
    CHECK(e.m == 1.f && e.l == 2.f && e.acc[0] == 3.f && e.acc[1] == 4.f);
}

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
    testCascadePlan();
    testCascadePrefixChoice();
    testCascadeCapMtp();
    testCascadeNumerics();
    std::printf("ops tests: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
