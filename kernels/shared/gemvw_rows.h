    const int lane = threadIdx.x % WARP, h = lane >> 4, n = lane & 15;  // shared mid-batch WMMA GEMV rows / lanes (ADR-0005); see the end of this file
    const int w = __builtin_amdgcn_readfirstlane(threadIdx.x / WARP);  // wave-uniform: scalar block math, uniform scale branches
    const int r0 = bid * ROWS;
    const uint8_t *wr[TILES];
#pragma unroll
    for (int tt = 0; tt < TILES; ++tt) wr[tt] = W + (size_t)min(r0 + tt * 16 + n, nrows - 1) * row_bytes;
    const int8_t *xt = xq + (size_t)min(n, NT - 1) * ncols;  // A row = token n (tokens past NT repeat the last one, never stored)
    const int nb32 = ncols / 32;

// kernels/shared/gemvw_rows.h -- setup of the gfx1201 mid-batch int8 WMMA GEMVs: lane roles (h = lane half,
// n = matrix row / token column), the wave index w made wave-uniform (readfirstlane), the block's first row r0,
// the clamped weight row pointer of each of the TILES 16-row tiles, the token row xt of lane n and nb32.
// #included in gemvw8_impl and gemvw_impl (kernels/gemv_w.hip), gw2_impl and gw8_impl (kernels/gemv_wp.hip);
// ADR-0005 / arch025-sc11d. The includer provides W, row_bytes, xq, ncols, nrows, bid and NT, TILES, ROWS.
// Line 1 stays a code line and this comment stays at the end (clang -E -P whitespace, see gemmh_impl.h).
