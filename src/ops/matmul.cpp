// Matmul kernel choice (ARCH-1 K-1), see matmul.h. Moved from src/model/forward.cpp (Model::matmulPath,
// floatGemvN, gemvKernels, gemvLaunch); the choices and grids are the same, pinned by the K-0 launch tables.
// SPDX-License-Identifier: Apache-2.0

#include "whirl/ops/matmul.h"

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

bool floatGemvN(const MatShape& w, std::uint32_t n, std::uint32_t gv_max, const kernels::KernelTable& k) {
    const std::size_t t = ti(w.ty);
    return n >= 2 && n <= gv_max && (w.ty == GgmlType::f32 || w.ty == GgmlType::f16) && k.gemvq[t] == nullptr && k.gemv1[t] != nullptr;
}

// Candidates in order of preference, each with its class; precise decode / verify rows (prec_dec, n <= gv_max)
// skip every candidate outside the precise class, so they always reach the f16 GEMM.
MatmulPath chooseMatmulPath(const MatShape& w, std::uint32_t n, const MatmulFlags& f, const kernels::KernelTable& k) {
    const std::size_t t = ti(w.ty);
    const bool precise_rows = f.prec_dec && n <= f.gv_max;
    auto allowed = [&](MatmulPath p) { return !precise_rows || preciseClass(bitClassOf(p)); };
    if (allowed(MatmulPath::gemvq) && n <= f.gv_max && k.gemvq[t] != nullptr && !f.float_gemv && !f.prec_dec) return MatmulPath::gemvq;
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
GemvPlan chooseGemvTile(const MatShape* mats, std::size_t n_mats, std::uint32_t n, const kernels::KernelTable& k, const GemvTune& tn) {
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

GemvWide chooseGemvWide(const MatShape& w0, std::size_t n_mats, const kernels::KernelTable& k, bool head_gemvx) {
    const std::size_t t = ti(w0.ty);
    if (k.gemvx[t] != nullptr && (!w0.head || head_gemvx)) return GemvWide::gemvx;
    if (w0.head && w0.ty == GgmlType::q6_k && k.gemvw_head_s != nullptr && n_mats == 1) return GemvWide::head_split;
    return GemvWide::pieces;
}

Dim3 gemvxGrid(std::uint32_t nrows) { return D((nrows + 15) / 16); }
Dim3 headSplitGrid(std::uint32_t chunk_rows) { return D((chunk_rows + 31) / 32); }

}  // namespace whirl::ops
