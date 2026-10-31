#pragma unroll  // shared mid-batch WMMA GEMV reduction + store (ADR-0005); see the end of this file
    for (int tt = 0; tt < TILES; ++tt)
#pragma unroll
        for (int v = 0; v < NV; ++v) {
            const int t = t0 + v;
            if (t < NT) red[w][t][tt * 16 + n] = (acc[0][tt][v] + acc[2][tt][v]) + (acc[1][tt][v] + acc[3][tt][v]);
        }
    __syncthreads();
    for (int i = threadIdx.x; i < NT * ROWS; i += blockDim.x) {
        const int t = i / ROWS, nn = i % ROWS, row = r0 + nn;
        const float s = ((red[0][t][nn] + red[4][t][nn]) + (red[2][t][nn] + red[6][t][nn])) +
                        ((red[1][t][nn] + red[5][t][nn]) + (red[3][t][nn] + red[7][t][nn]));
        if (row < nrows) {
            float *dst = y + (size_t)t * nrows + row;
            *dst = accumulate ? *dst + s : s;
        }
    }

// kernels/shared/gemvw_red_store.h -- epilogue of the gfx1201 mid-batch WMMA GEMVs with four slot
// accumulators acc[4][TILES][NV] (slot q = slice w + 8q): slots summed (0+2)+(1+3) (the xor-16 / xor-8 levels
// of warp_sum) into red[w][t][row], then the eight waves summed in the xor-4/2/1 order and stored (accumulate
// or overwrite) to y[t][row]. #included in gemvw8_impl (kernels/gemv_w.hip) and the gw2 / gw8 kernels
// (kernels/gemv_wp.hip); ADR-0005 / arch025-sc11d. The includer provides acc, red[8][NT][ROWS], w, n, t0,
// r0, y, nrows, accumulate and NT, NV, TILES, ROWS.
// Line 1 stays a code line and this comment stays at the end (clang -E -P whitespace, see gemmh_impl.h).
