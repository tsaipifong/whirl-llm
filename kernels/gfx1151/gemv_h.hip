// WHIRL device code (gfx1151 / RDNA 3.5): precise-mode decode / verify matmul (gemvh*).
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included after gemm_small.hip by kernels/gfx1151/whirl_kernels_gfx1151.hip.
//
// gemvh_<type> (n_tok <= 16) / gemvh2_<type> (n_tok <= 32): y[t][r] (+)= sum_k W[r][k] x16[t][k]
// Same ABI and numerics as the gfx1201 kernel (kernels/gemv_h.hip) and gemms11_impl: weights are
// dec_group_h<T> (lane: row lane % 16, joined to the 16-k fragment with join_halves), the f16
// activations are the other operand (WMMA16(b = tokens, a = weights)), one WMMA per 16-k step in
// increasing k from zero, tokens past n_tok are zero rows -> bitwise == gemms_c* == gemm_c*.
// One wave (= block) per 16 weight rows, no K split; the 16 rows' raw 256-k stage(s) go through
// LDS with coalesced 16-byte loads (S = 2 stages per LDS round when ncols % 512 == 0 and a stage is <= 288 bytes,
// i.e. not for f16 weights, whose 17-chunk-per-row-per-stage prefetch registers would not fit).
template <int T, int TN, int S>
__device__ __forceinline__ void gemvh11_impl(uint8_t *__restrict__ Ws, const uint8_t *__restrict__ W, uint64_t row_bytes, const _Float16 *__restrict__ X,
                                             float *__restrict__ Y, int ncols, int nrows, int n_tok, int accumulate) {
    constexpr int SB = S * gs_sbb<T>();
    constexpr int CH = (SB + 15) / 16 + 1;  // 16-byte chunks per row stage (covers the misaligned start)
    constexpr int SLOT = CH * 16;
    constexpr int WCH = 16 * CH;
    constexpr int WPER = (WCH + WARP - 1) / WARP;
    const int lane = threadIdx.x % WARP;
    const int m0 = blockIdx.x * 16;
    if (m0 >= nrows) return;
    const int nst = ncols / (256 * S);
    const int rl = lane % 16;
    const int row = min(m0 + rl, nrows - 1);
    float8 acc[TN];
    const _Float16 *xr[TN];
    bool xv[TN];
#pragma unroll
    for (int j = 0; j < TN; ++j) {
        acc[j] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
        const int t = j * 16 + rl;
        xv[j] = t < n_tok;
        xr[j] = X + (size_t)min(t, n_tok - 1) * ncols;
    }
    gs_u32x4 rw[WPER];
#pragma unroll
    for (int e = 0; e < WPER; ++e) rw[e] = (gs_u32x4){0, 0, 0, 0};
    // (macro, not a lambda: a by-reference capture of rw pushes it to scratch)
#define GEMVH11_FETCH(js_)                                                                                   \
    do {                                                                                                     \
        _Pragma("unroll") for (int e = 0; e < WPER; ++e) {                                                   \
            const int idx = lane + e * WARP;                                                                 \
            if (WCH % WARP == 0 || idx < WCH) {                                                              \
                const size_t st = (size_t)min(m0 + idx / CH, nrows - 1) * row_bytes + (size_t)(js_) * SB;                                             \
                const size_t a = (st & ~(size_t)15) + (idx % CH) * 16;                                       \
                /* no 16-byte chunk wholly past the stage (the matrix may end there) */                      \
                if (a < st + SB) rw[e] = *(const gs_u32x4 *)(W + a);                                         \
            }                                                                                                \
        }                                                                                                    \
    } while (0)
    GEMVH11_FETCH(0);
    for (int js = 0; js < nst; ++js) {
        __syncthreads();
#pragma unroll
        for (int e = 0; e < WPER; ++e) {
            const int idx = lane + e * WARP;
            if (WCH % WARP == 0 || idx < WCH) *(gs_u32x4 *)&Ws[(idx / CH) * SLOT + (idx % CH) * 16] = rw[e];
        }
        __syncthreads();
        if (js + 1 < nst) GEMVH11_FETCH(js + 1);
        const uint8_t *wp = Ws + rl * SLOT + (int)(((size_t)row * row_bytes + (size_t)js * SB) & 15);
        // (partial unroll: a full unroll lets the scheduler hoist every step's decode and X loads and spill)
#pragma unroll 2
        for (int kk = 0; kk < 16 * S; ++kk) {
            const half16 a = join_halves(dec_group_h<T>(wp, 2 * kk + lane / 16), lane);
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                half16 b = ld_frag16(xr[j] + js * 256 * S + kk * 16);
                if (!xv[j]) b = (half16){0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
                acc[j] = WMMA16(b, a, acc[j]);
            }
        }
    }
#undef GEMVH11_FETCH
    const int orow = m0 + rl;
    if (orow >= nrows) return;
#pragma unroll
    for (int j = 0; j < TN; ++j)
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            const int t = j * 16 + CROW(lane, v);
            if (t < n_tok) {
                float *dst = Y + (size_t)t * nrows + orow;
                *dst = accumulate ? *dst + acc[j][v] : acc[j][v];
            }
        }
}

// S = 2 (two 256-k stages per LDS round) when ncols % 512 == 0 and the type's stage is small; one LDS buffer
// per kernel, sized for the S used.
template <int T> constexpr int gv11_s2() { return gs_sbb<T>() <= 288 ? 2 : 1; }
#define GEMVH11_BODY(T, TN)                                                                                      constexpr int S2 = gv11_s2<T>();                                                                             __shared__ __attribute__((aligned(16))) uint8_t Ws[16 * 16 * ((S2 * gs_sbb<T>() + 15) / 16 + 1)];            if (S2 == 2 && ncols % 512 == 0) gemvh11_impl<T, TN, S2>(Ws, W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate);     else gemvh11_impl<T, TN, 1>(Ws, W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate);
#define GEMVH11_TYPE(SUF, T)                                                                                                        extern "C" __global__ void __launch_bounds__(32) gemvh_##SUF(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y,                                                                    int ncols, int nrows, int n_tok, int accumulate) {                   GEMVH11_BODY(T, 1)                                                                                                          }                                                                                                                               extern "C" __global__ void __launch_bounds__(32) gemvh2_##SUF(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y,                                                                     int ncols, int nrows, int n_tok, int accumulate) {                  GEMVH11_BODY(T, 2)                                                                                                          }

GEMVH11_TYPE(f16, QT_F16)
GEMVH11_TYPE(q8_0, QT_Q8_0)
GEMVH11_TYPE(q3_k, QT_Q3_K)
GEMVH11_TYPE(q4_k, QT_Q4_K)
GEMVH11_TYPE(q5_k, QT_Q5_K)
GEMVH11_TYPE(q6_k, QT_Q6_K)
GEMVH11_TYPE(iq4_nl, QT_IQ4_NL)
GEMVH11_TYPE(iq3_s, QT_IQ3_S)
GEMVH11_TYPE(iq4_xs, QT_IQ4_XS)
GEMVH11_TYPE(mxfp4, QT_MXFP4)
