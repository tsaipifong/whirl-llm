// WHIRL device code (gfx1201 / RDNA 4): v0252-swamax (PLAN V1-77) - the fp8 MoE prefill gate + up GEMM with the
// SwiGLU and the down projection's per-row fp8 activation quantization in its epilogue (one launch instead of
// gemm8r_moeh / gemm8r_moe32h + silu_mul_x8h).
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included LAST in the gfx1201 set by whirl_kernels.hip (uses gemm_fp8.hip / moe_mxfp4.hip
// helpers: k_fp8tab, mx_fp8x4, fp8x8, kernels/shared/gemm8_mainloop.h); no existing kernel moves.
//
// Values, bit for bit (progress\v0252-swamax.PRIOR_ART.md section 3, HANDOFF section 2):
//   * one workgroup owns the WHOLE F row of BN tokens of one tile (2 * ceil(F / 128) passes of the unchanged gemm8
//     main loop: gate slice s, up slice s, ...), so each accumulator element runs the WMMA sequence in k of
//     gemm8r_moeh (an element does not depend on BM / BN / which slice or pass computes it);
//   * g and u are rounded to f16 with gemm8_moe_impl's OH expression ((_Float16)(acc * ldexpf(st, ref - 127)));
//   * y = (float)g / (1 + __expf(-(float)g)) * (float)u, the row amax (fmaxf over |y|, NaN dropped as in block_max;
//     max is order independent, so an LDS reduction over the waves gives block_max's value), sx = amax / 448,
//     q = e4m3(clamp(y * 448 / amax, +-448)) packed exactly as silu_mul_x8h_impl (cvt_pk_fp8_f32 pairs);
//   * output row of pos p at Q + p * F (silu_mul_x8h's non-tiled layout), sx_out[p] (may alias sx_in: every
//     workgroup reads only its own positions, all before its barrier).
// Q must not alias X8 (X8 rows are E bytes, Q rows F bytes: another tile's X8 row would be overwritten).
// Grid: 1-D, n_tile_slots * halves blocks; tile slot b / halves, token half b % halves (tile bn = halves * BN).
// SWA_MUT (kt mutants only, built by scratch\v0252_swamax\mk_mutants.ps1 with -DSWA_MUT=n, never in the product):
//   1 = tile amax (the lane's own partial max, no cross-wave reduction), 2 = reduction over the first wave row only
//   (the "missing cross-tile max"), 3 = wrong order (SwiGLU on the f32 up value, u not rounded to f16 first).
#ifndef SWA_MUT
#define SWA_MUT 0
#endif

template <int BN, int WAVES_M>
__device__ void gemm8_moeswa_impl(const uint8_t *__restrict__ W1, const uint8_t *__restrict__ W2, uint64_t row_bytes,
                                  const uint8_t *__restrict__ ref1, const uint8_t *__restrict__ ref2, int rows_e,
                                  const uint8_t *__restrict__ X8a, const float *sxa, float *sxo, uint8_t *__restrict__ Q,
                                  int ncols, const int4 *__restrict__ tiles, const int *__restrict__ n_tiles, int halves) {
    constexpr int BM = 128, BK = 128, NTH = 256, MAXS = 4;  // F <= MAXS * BM (host check: swamaxShapeOk)
    constexpr int SWN = NTH / 32 / WAVES_M, STM = BM / 16 / WAVES_M;
    static_assert(BN / 16 / SWN == 1, "one 16-token fragment column per wave");
    const int tix = (int)blockIdx.x / halves, hv = (int)blockIdx.x - tix * halves;
    if (tix >= *n_tiles) return;
    const int4 tl = tiles[tix];
    const int base = tl.y + hv * BN;
    const int n_tok = min(BN, tl.z - hv * BN);
    if (n_tok <= 0) return;
    const uint8_t *__restrict__ X8 = X8a + (size_t)base * ncols;
    const int nrows = rows_e;
    const int nby = (rows_e + BM - 1) / BM;
    const int slane = threadIdx.x % WARP, swid = threadIdx.x / WARP;
    const int swm = swid % WAVES_M, swn = swid / WAVES_M;
    const int tt = swn * 16 + slane % 16;  // this lane's token (TN == 1)
    const bool tok_ok = tt < n_tok;
    const float st = tok_ok ? sxa[base + tt] : 0.f;  // read before any sx_out write (barrier below)
    float ys[MAXS][STM][8];
    _Float16 gh[STM][8];
    float m = 0.f;
#pragma unroll
    for (int s2 = 0; s2 < MAXS; ++s2)
#pragma unroll
        for (int i = 0; i < STM; ++i)
#pragma unroll
            for (int v = 0; v < 8; ++v) ys[s2][i][v] = 0.f;
    for (int ps = 0; ps < 2 * nby; ++ps) {
        const int s = ps >> 1;
        const bool up = (ps & 1) != 0;
        const uint8_t *__restrict__ W = (up ? W2 : W1) + (size_t)tl.x * rows_e * row_bytes;
        const uint8_t *__restrict__ wref = (up ? ref2 : ref1) + (size_t)tl.x * rows_e;
        {
#define G8_TILE_ORIGIN m0 = s * BM
#define G8_TROW(t) t
#define G8_TOFF(t) (size_t)t
#define G8_WMMA_IF(j) if ((wn * TN + j) * 16 < n_tok)
#include "shared/gemm8_mainloop.h"
            static_assert(TN == 1 && TM == STM, "tile shape");
            if (tok_ok) {
#pragma unroll
                for (int i = 0; i < TM; ++i) {
                    const int row0 = m0 + (wm * TM + i) * 16 + 8 * (lane / 16);
#pragma unroll
                    for (int v = 0; v < 8; ++v) {
                        const float yv = row0 + v < nrows ? acc[i][0][v] * ldexpf(st, (int)wref[row0 + v] - 127) : 0.f;
                        const _Float16 h = (_Float16)yv;
                        if (!up) {
                            gh[i][v] = h;
                        } else {
                            const float gv = (float)gh[i][v];
#if SWA_MUT == 3
                            const float y = gv / (1.f + __expf(-gv)) * yv;
#else
                            const float y = gv / (1.f + __expf(-gv)) * (float)h;
#endif
#pragma unroll
                            for (int s2 = 0; s2 < MAXS; ++s2)
                                if (s2 == s) ys[s2][i][v] = y;
                            if (row0 + v < nrows) m = fmaxf(m, fabsf(y));
                        }
                    }
                }
            }
        }
    }
    // row amax: lanes l and l + 16 of a wave hold the same token (two 8-row halves), WAVES_M waves split the rows
    __shared__ float red[2 * WAVES_M][BN];
    if (tok_ok) red[swm * 2 + slane / 16][tt] = m;
    __syncthreads();
    float M = 0.f;
    if (tok_ok) {
#if SWA_MUT == 1
        M = m;
#elif SWA_MUT == 2
        for (int r = 0; r < 2; ++r) M = fmaxf(M, red[r][tt]);
#else
        for (int r = 0; r < 2 * WAVES_M; ++r) M = fmaxf(M, red[r][tt]);
#endif
    }
    if (!tok_ok) return;
    const float inv = M > 0.f ? 448.f / M : 0.f;
    if (swm == 0 && slane < 16) sxo[base + tt] = M > 0.f ? M / 448.f : 0.f;
    uint8_t *qr = Q + (size_t)(base + tt) * rows_e;
#pragma unroll
    for (int s2 = 0; s2 < MAXS; ++s2) {
        if (s2 >= nby) break;
#pragma unroll
        for (int i = 0; i < STM; ++i) {
            const int row0 = s2 * BM + (swm * STM + i) * 16 + 8 * (slane / 16);
            if (row0 >= nrows) continue;
            int p[2];
#pragma unroll
            for (int h2 = 0; h2 < 2; ++h2) {
                const float a = fminf(fmaxf(ys[s2][i][4 * h2] * inv, -448.f), 448.f), b = fminf(fmaxf(ys[s2][i][4 * h2 + 1] * inv, -448.f), 448.f);
                const float c = fminf(fmaxf(ys[s2][i][4 * h2 + 2] * inv, -448.f), 448.f), d = fminf(fmaxf(ys[s2][i][4 * h2 + 3] * inv, -448.f), 448.f);
                int w = __builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0, false);
                p[h2] = __builtin_amdgcn_cvt_pk_fp8_f32(c, d, w, true);
            }
            *(int2 *)(qr + row0) = make_int2(p[0], p[1]);
        }
    }
}

// gemm8r_moeswa (token tile 64, 2 x 4 waves) / gemm8r_moeswa32 (token half-tile 32, 4 x 2 waves; tile bn 32 with
// halves 1 or tile bn 64 with halves 2)
#define GEMM8_MOESWA_ENTRY(NAME, BN, WM)                                                                                \
    extern "C" __global__ void __launch_bounds__(256)                                                                   \
        NAME(const uint8_t *W1, const uint8_t *W2, uint64_t row_bytes, const uint8_t *ref1, const uint8_t *ref2,         \
             int rows_e, const uint8_t *X8, const float *sx, float *sxo, uint8_t *Q, int ncols, const int4 *tiles,        \
             const int *n_tiles, int halves) {                                                                          \
        gemm8_moeswa_impl<BN, WM>(W1, W2, row_bytes, ref1, ref2, rows_e, X8, sx, sxo, Q, ncols, tiles, n_tiles, halves); \
    }
GEMM8_MOESWA_ENTRY(gemm8r_moeswa, 64, 2)
GEMM8_MOESWA_ENTRY(gemm8r_moeswa32, 32, 4)
#undef GEMM8_MOESWA_ENTRY
