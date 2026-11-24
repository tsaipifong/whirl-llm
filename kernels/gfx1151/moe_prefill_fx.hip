// v0252-8sdet (PLAN V1-140; KB v0252-8sdet:order-independent-moe-combine): COMB 6, the order-independent form of the fast
// MoE prefill combine on the Radeon 8060S (gfx1151). Opt-in (WHIRL_MOEP_COMB=6, fast only); included LAST by
// whirl_kernels_gfx1151.hip so no existing kernel moves.
//   moe_fix_zero          clears the i32 sums [n][rows_e] and the (token, row block) counters [n][nrb] that follow them
//                         (one int4 store per 4 words, grid-stride)
//   gemm_moepdi_<v>_mxfp4 moepdn_impl EPI 6 (moe_prefill_mt.hip): the moep down GEMM, every routed w * y rounded to the
//                         2^-14 grid and added into the i32 sum by integer atomics (exact: run-to-run identical whatever
//                         the block order), the last arriver of each (token, row block) adds fxValue(sum) + sg * ysh into x
// Replaces COMB 5's moe_combine4 (K = 0) + gemm_moepda_<v> (f32 atomics into x: the sum's bits follow the block order).
// Numeric format, error bound and overflow argument: kernels/moep_map.h (fx*). Fast only: the output differs from COMB 1
// / 5 within kept * 2^-15 absolute per element; what this form guarantees is the SAME output from two identical runs.
// SPDX-License-Identifier: Apache-2.0

extern "C" __global__ void __launch_bounds__(256) moe_fix_zero(int4 *__restrict__ p, int n4) {
    for (int i = blockIdx.x * 256 + threadIdx.x; i < n4; i += gridDim.x * 256) p[i] = make_int4(0, 0, 0, 0);
}

#define MOEPDI_ENTRY(NAME, BM, WM, BN, BK)                                                                          \
    extern "C" __global__ void __launch_bounds__(256)                                                               \
        gemm_moepdi_##NAME##_mxfp4(const uint8_t *W, uint64_t row_bytes, int rows_e, const _Float16 *X, int *acc,   \
                                   int ncols, const int4 *tiles, const int *n_tiles, const int *perm, const int *inv, \
                                   const float *w, const float *sg, const float *ysh, float *x, int *cnt, int K,    \
                                   float qlim) {                                                                    \
        moepdn_impl<QT_MXFP4, BK, BM, WM, BN, 6>(W, row_bytes, rows_e, X, acc, ncols, tiles, n_tiles, perm, inv, w,  \
                                                  sg, ysh, x, cnt, K, qlim);                                        \
    }
// the kMoepDn variant list (include/whirl/kernels_abi.h; same order as MOEPDN_ENTRY in moe_prefill_mt.hip)
MOEPDI_ENTRY(m128w4n32k64, 128, 4, 32, 64)
MOEPDI_ENTRY(m128w4n64k64, 128, 4, 64, 64)
MOEPDI_ENTRY(m128w4n128k64, 128, 4, 128, 64)
MOEPDI_ENTRY(m128w8n128k64, 128, 8, 128, 64)
MOEPDI_ENTRY(m64w4n128k64, 64, 4, 128, 64)
MOEPDI_ENTRY(m128w4n256k32, 128, 4, 256, 32)
#undef MOEPDI_ENTRY
