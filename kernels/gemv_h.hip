// WHIRL device code (gfx1201 / RDNA 4): precise-mode decode / verify matmul (gemvh*).
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included after gemm_small.hip by whirl_kernels.hip.
//
// gemvh_<type> (n_tok <= 16) / gemvh2_<type> (n_tok <= 32): y[t][r] (+)= sum_k W[r][k] x16[t][k]
// with f16 activations, the weights decoded to f16 by dec_group_h<T> and f32 accumulation,
// for the few rows of a decode step / MTP or n-gram verify / concurrent batch (precise mode:
// no int8 activations). Numerics are exactly those of gemm_c* / gemms_c* (the prefill GEMM):
// every 16x16 accumulator tile runs wmma(a, b, acc) over k in increasing 16-steps, A = the
// row's dec_group_h fragment (lane: row lane%16, k 8*(lane/16)..+8), B = the f16 activations
// (lane: token lane%16, the same k), tokens past n_tok are zero columns. So a token row's
// result is bitwise the same for every n_tok, every token tile it lands in, and the prefill
// GEMM of any batch size. Decode-shaped: one wave per 16 rows (no K split), its rows' raw
// quantized stages staged through LDS with coalesced loads.
template <int T, int TN>
__device__ __forceinline__ void gemvh_impl(const uint8_t *__restrict__ W, uint64_t row_bytes, const _Float16 *__restrict__ X,
                                           float *__restrict__ Y, int ncols, int nrows, int n_tok, int accumulate) {
    // one wave (= one block) per 16 rows. Each 256-k stage of the 16 rows is copied with
    // coalesced 16-byte loads into LDS (a 16-byte aligned copy keeps every super-block's
    // global alignment mod 16, as gemms_impl), the next stage prefetched into registers.
    constexpr int SB = t2_sbb<T>();
    constexpr int CH = gemms_stage_chunks(SB);  // 16-byte chunks per row stage (covers the misaligned start)
    constexpr int SLOT = CH * 16;
    constexpr int WCH = 16 * CH;
    constexpr int WPER = (WCH + WARP - 1) / WARP;
    __shared__ __attribute__((aligned(16))) uint8_t Ws[16 * SLOT];
    const int lane = threadIdx.x % WARP;
    const int m0 = blockIdx.x * 16;
    if (m0 >= nrows) return;
    const int nst = ncols / 256;
    const int g0 = lane / 16;
    const int rl = lane % 16;
    const int row = min(m0 + rl, nrows - 1);
    float8 acc[TN];
    const _Float16 *xr[TN];
    bool xv[TN];
#pragma unroll
    for (int j = 0; j < TN; ++j) {
        acc[j] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
        const int t = j * 16 + lane % 16;
        xv[j] = t < n_tok;
        xr[j] = X + (size_t)min(t, n_tok - 1) * ncols + 8 * g0;
    }
    t2_u32x4 rw[WPER];
    size_t rbase[WPER];
#pragma unroll
    for (int e = 0; e < WPER; ++e) {
        const int idx = min(lane + e * WARP, WCH - 1);
        rbase[e] = (size_t)min(m0 + idx / CH, nrows - 1) * row_bytes;
    }
    auto fetch = [&](int js) {
#pragma unroll
        for (int e = 0; e < WPER; ++e) {
            const int idx = lane + e * WARP;
            if (WCH % WARP == 0 || idx < WCH) {
                const size_t st = rbase[e] + (size_t)js * SB;
                // only chunks overlapping the stage: the matrix may end right after it (gemm_small_addr.h)
                rw[e] = gemms_chunk_needed(st, idx % CH, SB) ? *(const t2_u32x4 *)(W + (st & ~(size_t)15) + (idx % CH) * 16)
                                                             : (t2_u32x4){0, 0, 0, 0};
            }
        }
    };
    fetch(0);
    for (int js = 0; js < nst; ++js) {
        __syncthreads();
#pragma unroll
        for (int e = 0; e < WPER; ++e) {
            const int idx = lane + e * WARP;
            if (WCH % WARP == 0 || idx < WCH) *(t2_u32x4 *)&Ws[(idx / CH) * SLOT + (idx % CH) * 16] = rw[e];
        }
        __syncthreads();
        if (js + 1 < nst) fetch(js + 1);
        const uint8_t *wp = Ws + rl * SLOT + (int)(((size_t)row * row_bytes + (size_t)js * SB) & 15);
        half8 a[16];
#pragma unroll
        for (int kk = 0; kk < 16; ++kk) a[kk] = dec_group_h<T>(wp, 2 * kk + g0);
#pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                half8 b = *(const half8 *)(xr[j] + js * 256 + kk * 16);
                if (!xv[j]) b = (half8){0, 0, 0, 0, 0, 0, 0, 0};
                acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a[kk], b, acc[j]);
            }
        }
    }
#pragma unroll
    for (int j = 0; j < TN; ++j) {
        const int t = j * 16 + lane % 16;
        if (t >= n_tok) continue;
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            const int r = m0 + 8 * g0 + v;
            if (r < nrows) {
                float *dst = Y + (size_t)t * nrows + r;
                *dst = accumulate ? *dst + acc[j][v] : acc[j][v];
            }
        }
    }
}

#define GEMVH_TYPE(SUF, T)                                                                                                       \
    extern "C" __global__ void __launch_bounds__(32) gemvh_##SUF(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, \
                                                                   int ncols, int nrows, int n_tok, int accumulate) {           \
        gemvh_impl<T, 1>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate);                                                  \
    }                                                                                                                            \
    extern "C" __global__ void __launch_bounds__(32) gemvh2_##SUF(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, \
                                                                    int ncols, int nrows, int n_tok, int accumulate) {          \
        gemvh_impl<T, 2>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate);                                                  \
    }

GEMVH_TYPE(f16, QT_F16)
GEMVH_TYPE(q8_0, QT_Q8_0)
GEMVH_TYPE(q3_k, QT_Q3_K)
GEMVH_TYPE(q4_k, QT_Q4_K)
GEMVH_TYPE(q5_k, QT_Q5_K)
GEMVH_TYPE(q6_k, QT_Q6_K)
GEMVH_TYPE(iq4_nl, QT_IQ4_NL)
GEMVH_TYPE(iq3_s, QT_IQ3_S)
GEMVH_TYPE(iq4_xs, QT_IQ4_XS)
GEMVH_TYPE(mxfp4, QT_MXFP4)
