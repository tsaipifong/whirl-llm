            const int t = t0 + (wn * TN + j) * 16 + lane % 16;  // shared f16 WMMA GEMM store of acc[i][j] (ADR-0005); see the end of this file
            if (t >= n_tok) continue;
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int row = m0 + (wm * TM + i) * 16 + 8 * (lane / 16) + v;
                if (row < nrows) {
                    if (OH) {
                        ((_Float16 *)Y)[(size_t)t * nrows + row] = (_Float16)acc[i][j][v];
                    } else {
                        float *dst = Y + (size_t)t * nrows + row;
                        *dst = accumulate ? *dst + acc[i][j][v] : acc[i][j][v];
                    }
                }
            }

// kernels/shared/gemm_store_ij.h -- epilogue step of the gfx1201 f16 WMMA GEMMs for accumulator (i, j): lane
// column = token t, elements v = rows 8 (lane / 16) + v of the 16-row fragment; tokens past n_tok and rows
// past nrows are not stored; OH writes f16 Y, else f32 Y accumulated or overwritten. #included inside the
// j loops of gemm3_impl (kernels/gemm_prefill.hip) and the two kernels of kernels/gemm_small.hip; ADR-0005 /
// arch025-sc11d. The includer provides acc, i, j, t0, m0, wm, wn, TM, TN, lane, n_tok, nrows, Y, OH, accumulate.
// Line 1 stays a code line and this comment stays at the end (clang -E -P whitespace, see gemmh_impl.h).
