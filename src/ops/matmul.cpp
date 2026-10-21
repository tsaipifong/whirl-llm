// Matmul kernel choice (ARCH-1 K-1), see matmul.h. Moved from src/model/forward.cpp (Model::matmulPath,
// floatGemvN, gemvKernels, gemvLaunch); the choices and grids are the same, pinned by the K-0 launch tables.
// SPDX-License-Identifier: Apache-2.0

#include "whirl/ops/matmul.h"

#include <algorithm>
#include <stdexcept>

namespace whirl::ops {

namespace {
constexpr std::size_t ti(GgmlType t) { return static_cast<std::size_t>(t); }
constexpr Dim3 D(std::uint64_t x) { return Dim3{static_cast<unsigned>(x), 1, 1}; }
}  // namespace

const char* bitClassName(BitClass b) {
    switch (b) {
        case BitClass::q8_row: return "q8_row";
        case BitClass::f32_row: return "f32_row";
        case BitClass::f16_gemm: return "f16_gemm";
        case BitClass::fp8_gemm: return "fp8_gemm";
    }
    return "?";
}

BitClass bitClassOf(MatmulPath p) {
    switch (p) {
        case MatmulPath::gemvq: return BitClass::q8_row;
        case MatmulPath::gemv1:
        case MatmulPath::float_gemv: return BitClass::f32_row;
        case MatmulPath::gemm_fp8: return BitClass::fp8_gemm;
        case MatmulPath::gemm_f16: return BitClass::f16_gemm;
    }
    return BitClass::f16_gemm;
}

bool floatGemvN(const MatShape& w, std::uint32_t n, std::uint32_t gv_max, const kernels::MatmulKernels& k) {
    const std::size_t t = ti(w.ty);
    return n >= 2 && n <= gv_max && (w.ty == GgmlType::f32 || w.ty == GgmlType::f16) && k.gemvq[t] == nullptr && k.gemv1[t] != nullptr;
}

// Candidates in order of preference, each with its class; precise decode / verify rows (prec_dec, n <= gv_max)
// skip every candidate outside the precise class, so they always reach the f16 GEMM.
MatmulPath chooseMatmulPath(const MatShape& w, std::uint32_t n, const MatmulFlags& f, const kernels::MatmulKernels& k) {
    const std::size_t t = ti(w.ty);
    auto allowed = [&](MatmulPath p) { return preciseAllows(bitClassOf(p), f.prec_dec, n, f.gv_max); };
    if (allowed(MatmulPath::gemvq) && n <= f.gv_max && k.gemvq[t] != nullptr && !f.float_gemv) return MatmulPath::gemvq;
    if (allowed(MatmulPath::gemv1) && n == 1) return MatmulPath::gemv1;
    if (allowed(MatmulPath::float_gemv) && floatGemvN(w, n, f.gv_max, k)) return MatmulPath::float_gemv;
    if (allowed(MatmulPath::gemm_fp8) && f.fp8_prefill && (f.fp8_mask & f.mm_class) != 0 && w.ty == GgmlType::mxfp4 && w.has_ref &&
        k.gemm8[0] != nullptr)
        return MatmulPath::gemm_fp8;
    return MatmulPath::gemm_f16;
}

std::uint32_t gvNmax(GgmlType ty, std::uint32_t gv_nmax_override) {
    if (gv_nmax_override != 0) return gv_nmax_override;
    return ty == GgmlType::mxfp4 ? 13 : 11;
}

// n <= 16: the one-token / multi-token int8 GEMV, its int8-WMMA (gemvw) or multi-row (gemvq_mr) variant from the
// tuned tables, grouped over the matrices through the "g" twin when the group fits gvNmax. Every variant computes
// each token row as the one-token gemvq does (class q8_row), so the tables only move speed.
GemvPlan chooseGemvTile(const MatShape* mats, std::size_t n_mats, std::uint32_t n, const kernels::MatmulKernels& k, const GemvTune& tn) {
    if (n_mats < 1 || n_mats > 3 || n < 1 || n > static_cast<std::uint32_t>(kernels::kMaxSmallBatch))
        throw std::invalid_argument("ops::chooseGemvTile: 1..3 matrices and 1..16 rows");
    const MatShape& w = mats[0];
    const std::size_t t = ti(w.ty);
    Function f = n == 1 ? k.gemvq[t] : k.gemvq_nt[n - 2][t];
    Function fg = n == 1 ? nullptr : k.gemvq_nt_g[n - 2][t];
    std::uint32_t rows_per_block = 8;
    const std::uint8_t wv = n < 2 ? 0 : (w.ty == GgmlType::q6_k && w.head) ? (*tn.gemv_w_head)[n] : (*tn.gemv_w)[t % kernels::kNTypes][n];
    const Function wf = wv > 0 ? k.gemvw[wv][n - 2][t] : nullptr;
    if (wf != nullptr) {
        f = wf;
        fg = k.gemvw_g[wv][n - 2][t];
        rows_per_block = static_cast<std::uint32_t>(kernels::kGemvwRows[wv]);
    } else if (n >= 2 && (*tn.gemv_r)[n] > 1) {
        const Function mr = k.gemvq_mr[(*tn.gemv_r)[n] / 2 - 1][n - 2][t];
        if (mr != nullptr) {
            f = mr;
            fg = k.gemvq_mr_g[(*tn.gemv_r)[n] / 2 - 1][n - 2][t];
            rows_per_block = 8 * static_cast<std::uint32_t>((*tn.gemv_r)[n]);
        }
    }
    GemvPlan p;
    // one-token entries always take GvArgs (gv_grp); a multi-token group takes the twin
    const bool twin = n >= 2 && n_mats > 1 && fg != nullptr && n <= gvNmax(w.ty, tn.gv_nmax_override);
    if (twin) f = fg;
    p.f = f;
    p.rows_per_block = rows_per_block;
    p.grouped = (k.gv_grp && n == 1) || twin;
    if (p.grouped) {
        std::uint32_t nb = 0;
        for (std::size_t si = 0; si < n_mats; ++si) nb += (mats[si].nrows + rows_per_block - 1) / rows_per_block;
        p.grid[0] = D(nb);
    } else {
        for (std::size_t si = 0; si < n_mats; ++si) p.grid[si] = D((mats[si].nrows + rows_per_block - 1) / rows_per_block);
    }
    return p;
}

GemvWide chooseGemvWide(const MatShape& w0, std::size_t n_mats, const kernels::MatmulKernels& k, bool head_gemvx) {
    const std::size_t t = ti(w0.ty);
    if (k.gemvx[t] != nullptr && (!w0.head || head_gemvx)) return GemvWide::gemvx;
    if (w0.head && w0.ty == GgmlType::q6_k && k.gemvw_head_s != nullptr && n_mats == 1) return GemvWide::head_split;
    return GemvWide::pieces;
}

Dim3 gemvxGrid(std::uint32_t nrows) { return D((nrows + 15) / 16); }
Dim3 headSplitGrid(std::uint32_t chunk_rows) { return D((chunk_rows + 31) / 32); }


std::size_t gemm8Cfg(std::uint32_t n) { return n <= 64 ? 3 : n <= 256 ? 1 : 0; }
std::size_t gemm8tCfg(std::uint32_t n) { return n <= 32 ? 3 : n <= 48 ? 4 : n <= 64 ? 2 : n <= 192 ? 1 : 0; }

Gemm8Plan chooseGemm8(std::uint32_t nrows, std::uint32_t n, bool out_h16, bool g8t, const kernels::MatmulKernels& k) {
    const std::size_t ci = g8t ? gemm8tCfg(n) : gemm8Cfg(n);
    const kernels::GemmCfg c = g8t ? kernels::kGemm8tCfgs[ci] : kernels::kGemm8Cfgs[ci];
    Gemm8Plan p;
    p.f = g8t ? ((out_h16 && k.gemm8th[ci] != nullptr) ? k.gemm8th[ci] : k.gemm8t[ci]) : ((out_h16 && k.gemm8h[ci] != nullptr) ? k.gemm8h[ci] : k.gemm8[ci]);
    p.grid = Dim3{static_cast<unsigned>((n + c.bn - 1) / c.bn), static_cast<unsigned>((nrows + c.bm - 1) / c.bm), 1};
    p.block = D(c.nth);
    return p;
}

// Precise rows: the small-batch WMMA GEMM on the raw quantized bytes (gemms) with the narrowest token tile that
// holds n (then the smaller row block; the larger one where the profile says so, GEMVH-2), else the tuned fused
// config (gemm_c on the quantized bytes) - never the whole-matrix dequant into w16.
std::uint32_t precChoice(const MatShape& w, std::uint32_t n, std::uint8_t tuned, const GemmFlags& f, const kernels::MatmulKernels& k) {
    const std::size_t G = kernels::kGemmCfgs.size();
    const std::size_t t = ti(w.ty);
    if (w.ty != GgmlType::f16 && w.ty != GgmlType::f32 && w.ncols % 256 == 0) {
        if (f.prec_gs >= 0 && static_cast<std::size_t>(f.prec_gs) < k.gemms.size() && k.gemms[f.prec_gs][t] != nullptr)
            return static_cast<std::uint32_t>(2 * G + f.prec_gs);
        int best = -1;
        for (std::size_t si = 0; si < k.gemms.size(); ++si) {
            if (k.gemms[si][t] == nullptr) continue;
            const kernels::GemmCfg c = k.gemms_geom[si];
            if (best < 0) {
                best = static_cast<int>(si);
                continue;
            }
            const kernels::GemmCfg b = k.gemms_geom[best];
            const bool c_fits = static_cast<std::uint32_t>(c.bn) >= n, b_fits = static_cast<std::uint32_t>(b.bn) >= n;
            if (c_fits != b_fits) {
                if (c_fits) best = static_cast<int>(si);
                continue;
            }
            // both hold n: narrower tile, then smaller row block (larger on gfx1151); neither: wider tile
            const bool tie_bigger_bm = f.gemms_tie_larger_bm;
            if (c_fits ? (c.bn < b.bn || (c.bn == b.bn && (tie_bigger_bm ? c.bm > b.bm : c.bm < b.bm))) : c.bn > b.bn) best = static_cast<int>(si);
        }
        if (best >= 0) return static_cast<std::uint32_t>(2 * G + best);
    }
    const std::uint32_t c = tuned;
    return c >= 2 * G ? 0u : static_cast<std::uint32_t>(c % G);
}

GemmPlan chooseGemmF16(const MatShape& w, std::uint32_t n, std::uint8_t tuned, const GemmFlags& f, const kernels::MatmulKernels& k) {
    const std::size_t t = ti(w.ty);
    const std::size_t G = kernels::kGemmCfgs.size();
    GemmPlan p;
    const bool precise_rows = preciseRows(f.prec_dec, n, f.gv_max);
    if (precise_rows && n <= 32 && w.ncols % 256 == 0 && !f.out_h16) {
        // precise decode / verify: the decode-shaped f16 GEMM (bitwise == every choice below)
        const Function hv = n <= 16 ? k.gemvh[t] : k.gemvh2[t];
        if (f.prec_gemvh && hv != nullptr) {
            p.kind = GemmKind::gemvh;
            p.f = hv;
            p.grid = D(f.gemvh_cap != 0 ? kernels::gemvhGridCap(w.nrows, f.gemvh_cap) : (w.nrows + 15) / 16);
            p.block = D(32);
            return p;
        }
    }
    std::uint32_t choice = precise_rows ? precChoice(w, n, tuned, f, k) : tuned;
    // h16_dq: a fused quantized choice cannot write f16; its dequant twin (same bits) can
    if (f.out_h16 && f.h16_dq && choice < G && w.ty != GgmlType::f16) choice += static_cast<std::uint32_t>(G);
    const std::size_t gcfg = choice % G;
    const kernels::GemmCfg c = kernels::kGemmCfgs[gcfg];
    if (choice >= 2 * G) {
        // small-batch GEMM straight from the quantized weights (bitwise == gemm_cN)
        const std::size_t si = choice - 2 * G;
        const kernels::GemmCfg sc = k.gemms_geom[si];  // a code object may export its own geometry (gfx1151)
        p.kind = GemmKind::gemms;
        p.f = f.out_h16 ? k.gemmsh[si][t] : k.gemms[si][t];
        p.grid = Dim3{static_cast<unsigned>((n + sc.bn - 1) / sc.bn), static_cast<unsigned>((w.nrows + sc.bm - 1) / sc.bm), 1};
        p.block = D(sc.nth);
        return p;
    }
    const Function hq = (f.out_h16 && k.gemmch[gcfg] != nullptr) ? k.gemmhqh[t] : k.gemmhq[t];
    if (choice >= G && w.ty != GgmlType::f16 && f.gemmh_on && f.gemmhq_on && hq != nullptr && n >= gemmh_min && w.ncols % 32 == 0) {
        // dequant fused into the fragment-order f16 GEMM (bitwise == dequant_f16 + gemm_cN_f16)
        p.kind = GemmKind::gemmhq;
        p.f = hq;
        p.grid = Dim3{(n + 255) / 256, (w.nrows + 127) / 128, 1};
        return p;
    }
    std::uint64_t wrb = w.row_bytes;
    GgmlType wty = w.ty;
    if (choice >= G && w.ty != GgmlType::f16) {
        const std::uint64_t groups = static_cast<std::uint64_t>(w.ncols / 8) * w.nrows;
        p.dequant = true;
        p.dq = k.dequant_f16[t];
        p.dq_grid = D(std::min<std::uint64_t>((groups + 255) / 256, 65535));
        wrb = static_cast<std::uint64_t>(w.ncols) * 2;
        wty = GgmlType::f16;
    }
    const bool use_ch = f.out_h16 && wty == GgmlType::f16 && k.gemmch[gcfg] != nullptr;
    const Function gf = use_ch ? k.gemmch[gcfg] : k.gemmc[gcfg][ti(wty)];
    const Function hf = use_ch ? k.gemmhh_f16 : k.gemmh_f16;
    if (f.gemmh_on && hf != nullptr && wty == GgmlType::f16 && n >= gemmh_min && w.ncols % 32 == 0 && wrb == static_cast<std::uint64_t>(w.ncols) * 2) {
        // 128 rows x 256 tokens, bitwise the same as gf
        p.kind = GemmKind::gemmh;
        p.f = hf;
        p.grid = Dim3{(n + 255) / 256, (w.nrows + 127) / 128, 1};
    } else {
        p.kind = GemmKind::gemmc;
        p.f = gf;
        p.grid = Dim3{static_cast<unsigned>((n + c.bn - 1) / c.bn), static_cast<unsigned>((w.nrows + c.bm - 1) / c.bm), 1};
        p.block = D(c.nth);
    }
    return p;
}

}  // namespace whirl::ops
