            const int t = t0 + (wn * TN + j) * 16 + lane % 16;  // shared f32 store of acc[i][j] (ADR-0005); see the end of this file
            if (t >= n_tok) continue;
#pragma unroll
            for (int v = 0; v < 8; ++v) {
#if defined(__GFX11__)
                const int row = m0 + (wm * TM + i) * 16 + CROW(lane, v);
#else
                const int row = m0 + (wm * TM + i) * 16 + 8 * (lane / 16) + v;
#endif
                if (row < nrows) {
                    float *dst = Y + (size_t)t * nrows + row;
                    *dst = accumulate ? *dst + acc[i][j][v] : acc[i][j][v];
                }
            }

// kernels/shared/gemm_store_ij_f32.h -- epilogue step of the f16 WMMA prefill GEMMs without an f16 output
// option: lane column = token t, element v = fragment row 8 (lane / 16) + v on gfx12 and CROW(lane, v) on gfx11
// (__GFX11__, the gfx1151 set); tokens past n_tok and rows past nrows are not stored; f32 Y accumulated or
// overwritten. #included inside the j loops of gemm4_impl (kernels/gemm_prefill.hip) and gemm3_impl / gemm4_impl
// (kernels/gfx1151/gemm_prefill.hip); ADR-0005 / arch025-sc11d. The f16-output variant (OH) is
// kernels/shared/gemm_store_ij.h.
// Line 1 stays a code line and this comment stays at the end (clang -E -P whitespace, see gemmh_impl.h).
