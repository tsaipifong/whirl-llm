#pragma unroll  // shared gfx1151 WMMA GEMV reduction + store (ADR-0005); see the end of this file
    for (int tt = 0; tt < TILES; ++tt)
#pragma unroll
        for (int v = 0; v < NTV; ++v) {
            const int t = 2 * v + h;
            if (t < NT && (!HALF || n < 8)) red[w][t][tt * 16 + n] = (acc[tt][0][v] + acc[tt][1][v]) + (acc[tt][2][v] + acc[tt][3][v]);
        }
    __syncthreads();
    for (int i = threadIdx.x; i < NT * ROWS; i += blockDim.x) {
        const int t = i / ROWS, nn = i % ROWS, row = r0 + nn;
        const float s = ((red[0][t][nn] + red[1][t][nn]) + (red[2][t][nn] + red[3][t][nn])) +
                        ((red[4][t][nn] + red[5][t][nn]) + (red[6][t][nn] + red[7][t][nn]));
        if (row < nrows) {
            float *dst = y + (size_t)t * nrows + row;
            *dst = accumulate ? *dst + s : s;
        }
    }

// kernels/shared/gemvw11_red_store.h -- epilogue of gemvw6_impl (kernels/gfx1151/gemv_q6.hip) and gemvw_impl
// (kernels/gfx1151/gemv_wunit.hip): the four per-slice accumulators acc[TILES][4][NTV] summed (0+1)+(2+3) into
// red[w][t][row] (lanes of the used tile half only), then the eight waves summed in order and stored
// (accumulate or overwrite) to y[t][row]. #included in both bodies after kernels/shared/gemvw11_setup.h;
// ADR-0005 / arch025-sc11d.
// Line 1 stays a code line and this comment stays at the end (clang -E -P whitespace, see gemmh_impl.h).
