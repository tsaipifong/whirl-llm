    const int m0 = tile * 16;  // shared gfx1151 gemv_h row tile setup (ADR-0005); see the end of this file
    const int nsb = ncols / (16 * D::UK);
    const int rl = lane % 16;
    const int row = min(m0 + rl, nrows - 1);
    const uint8_t *rowp = W + (size_t)row * row_bytes;
    float8 acc[TN];
    const _Float16 *xr[TN];
#pragma unroll
    for (int j = 0; j < TN; ++j) {
        acc[j] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
        const int t = j * 16 + rl;
        xr[j] = X + (size_t)min(t, n_tok - 1) * ncols;
    }

// kernels/shared/gemvh11_tile.h -- start of one 16-row tile of the gfx1151 f16-activation GEMVs
// gemvh_direct_impl and gemvh_dual_impl (kernels/gfx1151/gemv_h.hip): tile rows m0.., superblock count nsb,
// lane row rl (clamped to nrows - 1), its weight row rowp, zeroed accumulators and the token rows xr of the TN
// 16-token fragments (clamped to n_tok - 1). #included inside both tile loops; ADR-0005 / arch025-sc11d. The
// includer provides tile, lane, W, row_bytes, X, ncols, nrows, n_tok, TN and D (GhW / GhQ).
// Line 1 stays a code line and this comment stays at the end (clang -E -P whitespace, see gemmh_impl.h).
