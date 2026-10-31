    constexpr int ROWS = HALF ? 8 : 16 * TILES;  // shared gfx1151 WMMA GEMV block setup (ADR-0005); see the end of this file
    constexpr int NTV = (NT + 1) / 2;
    __shared__ float red[8][NT][ROWS];
    if (skip != nullptr && *skip) return;
    const int lane = threadIdx.x % WARP, w = threadIdx.x / WARP, h = lane >> 4, n = lane & 15;
    const int r0 = (bid < 0 ? (int)blockIdx.x : bid) * ROWS;
    const uint8_t *wr[TILES];
#pragma unroll
    for (int tt = 0; tt < TILES; ++tt) wr[tt] = W + (size_t)min(r0 + tt * 16 + (HALF ? (n & 7) : n), nrows - 1) * row_bytes;
    const bool tok_ok = n < NT;
    const int8_t *xt = xq + (size_t)(tok_ok ? n : 0) * ncols;
    const uint32_t *xp = (const uint32_t *)xd;
    const int nb32 = ncols / 32;
    const int nsb = ncols / 256;

// kernels/shared/gemvw11_setup.h -- setup of the gfx1151 int8 WMMA GEMVs gemvw6_impl (kernels/gfx1151/
// gemv_q6.hip) and gemvw_impl (kernels/gfx1151/gemv_wunit.hip): block rows (HALF -> 8, else 16 TILES), the
// cross-wave reduction buffer red, lane roles (h = lane half, n = matrix row / token column), the clamped
// weight row pointers, the token row xt of lane n (tokens past NT read row 0 and are never stored) and the
// activation scales xp. #included in both bodies; ADR-0005 / arch025-sc11d. The includer provides W,
// row_bytes, xq, xd, ncols, nrows, skip, bid and NT, TILES, HALF.
// Line 1 stays a code line and this comment stays at the end (clang -E -P whitespace, see gemmh_impl.h).
