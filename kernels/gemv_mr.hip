// WHIRL device code (gfx1201 / RDNA 4): multi-row multi-token int8 GEMV (gemvq_nt*r*).
// SPDX-License-Identifier: Apache-2.0
// Ported from the private prototype; not self-contained: included in order by whirl_kernels.hip.

// ---------------------------------------------------------------------------
// Multi-row multi-token int8 GEMV (verify / batched decode, 2..16 tokens):
// each wave owns R rows, so every int8 x unit it loads (the bottleneck once
// NT grows) serves R rows, and row-independent terms (x sums) are computed
// once. Per (row, token) the arithmetic is exactly that of gemvq_impl /
// gemvq_nt_impl (same lane -> unit mapping, same float expressions in the
// same order, same warp_sum), so the results match them bitwise.
// ---------------------------------------------------------------------------
// Activation-scale reads of the shared unit decoders (gfx1201: plain float scales); ADR-0005.
#define GMR_X0 xdt[0]
#define GMR_X1 xdt[1]
#define GMR_XQ(e) xdt[e]
#include "shared/gemv_mr_kw.h"  // MRD, MRD_KW, MRD<Q4_K / Q5_K / IQ4_XS>, both kernel sets (ADR-0005)

// MXFP4 wide unit: as IQ4_XS (sub-blocks 2p, 2p+1), scales 2^(e - 128).
template <> struct MRD<QT_MXFP4> {
    static constexpr int per_sb = 4;
    template <int NT, int R>
    static __device__ __forceinline__ void step(const uint8_t *const *wr, int u, const int8_t *xq, const float *xd,
                                                int ncols, float (*acc)[NT]) {
        const int b = u >> 2, pp = u & 3;
        int lo[R][8], hi[R][8];
        float dl0[R], dl1[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint8_t *blk = wr[r] + (size_t)b * 136;
            const uint32_t ee = *(const uint16_t *)(blk + 2 * pp);
            dl0[r] = e8m0_half(ee & 0xFF);
            dl1[r] = e8m0_half(ee >> 8);
            const uint4 qa = *(const uint4 *)(blk + 8 + 32 * pp);
            const uint4 qb = *(const uint4 *)(blk + 24 + 32 * pp);
            const uint32_t q[8] = {qa.x, qa.y, qa.z, qa.w, qb.x, qb.y, qb.z, qb.w};
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                lo[r][k] = mx_lut4(q[k] & 0x0F0F0F0F);
                hi[r][k] = mx_lut4((q[k] >> 4) & 0x0F0F0F0F);
            }
        }
#include "shared/gemv_mr_unit4_dot.h"  // acc[r][t] += unit dots (ADR-0005)
    }
};

// Q6_K: per row 4 quarters of 2 int8x4 weights, 4 scales and d.
#include "shared/gemv_mr_q6.h"  // MRD<Q6_K>, both kernel sets (ADR-0005)
#undef GMR_X0
#undef GMR_X1
#undef GMR_XQ

template <int T, int NT, int R>
__device__ __forceinline__ void gemvq_mr_impl(const uint8_t *__restrict__ W, uint64_t row_bytes, const int8_t *__restrict__ xq,
                              const float *__restrict__ xd, float *__restrict__ y, int ncols, int nrows, int accumulate,
                              const int *__restrict__ skip, int bid) {
    if (skip != nullptr && *skip) return;
    const int lane = threadIdx.x % WARP;
    const int row0 = (bid * (blockDim.x / WARP) + threadIdx.x / WARP) * R;
    if (row0 >= nrows) return;
    const uint8_t *wr[R];
#pragma unroll
    for (int r = 0; r < R; ++r) wr[r] = W + (size_t)min(row0 + r, nrows - 1) * row_bytes;
    const int nu = ncols / 256 * MRD<T>::per_sb;
    float acc[R][NT];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int t = 0; t < NT; ++t) acc[r][t] = 0.f;
    for (int u = lane; u < nu; u += WARP) MRD<T>::template step<NT, R>(wr, u, xq, xd, ncols, acc);
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const float v = warp_sum(acc[r][t]);
            if (lane == 0 && row0 + r < nrows) {
                float *dst = y + (size_t)t * nrows + row0 + r;
                *dst = accumulate ? *dst + v : v;
            }
        }
    }
}

#define GEMVQ_ENTRY(SUF, T)                                                                      \
    extern "C" __global__ void __launch_bounds__(256)                                            \
        gemvq_##SUF(GV_PARAMS) {          \
        { const GvSel gs = gv_sel(g); gemvq_impl<T>(gs.W, gs.rb, xq, xd, gs.y, ncols, gs.nrows, accumulate, skip, gs.b); }                  \
    }
#define GEMVQ_NT_ENTRY(NT, SUF, T)                                                               \
    extern "C" __global__ void __launch_bounds__(256)                                            \
        gemvq_nt##NT##_##SUF(const uint8_t *W, uint64_t row_bytes, const int8_t *xq,            \
                             const float *xd, float *y, int ncols, int nrows, int accumulate,   \
                             const int *skip) {                                                  \
        gemvq_nt_impl<T, NT>(W, row_bytes, xq, xd, y, ncols, nrows, accumulate, skip, blockIdx.x);           \
    } \
    extern "C" __global__ void __launch_bounds__(256)                                            \
        ggemvq_nt##NT##_##SUF(GV_PARAMS) {                                                  \
        { const GvSel gs = gv_sel(g); gemvq_nt_impl<T, NT>(gs.W, gs.rb, xq, xd, gs.y, ncols, gs.nrows, accumulate, skip, gs.b); }           \
    }
#define GEMVQ_W_ENTRY(SUF, T)                                                                    \
    extern "C" __global__ void __launch_bounds__(256)                                            \
        gemvq_##SUF(GV_PARAMS) {          \
        { const GvSel gs = gv_sel(g); gemvq_mr_impl<T, 1, 1>(gs.W, gs.rb, xq, xd, gs.y, ncols, gs.nrows, accumulate, skip, gs.b); }        \
    }
#define GEMVQ_W_NT_ENTRY(NT, SUF, T)                                                             \
    extern "C" __global__ void __launch_bounds__(256)                                            \
        gemvq_nt##NT##_##SUF(const uint8_t *W, uint64_t row_bytes, const int8_t *xq,            \
                             const float *xd, float *y, int ncols, int nrows, int accumulate,   \
                             const int *skip) {                                                  \
        gemvq_mr_impl<T, NT, 1>(W, row_bytes, xq, xd, y, ncols, nrows, accumulate, skip, blockIdx.x);        \
    } \
    extern "C" __global__ void __launch_bounds__(256)                                            \
        ggemvq_nt##NT##_##SUF(GV_PARAMS) {                                                  \
        { const GvSel gs = gv_sel(g); gemvq_mr_impl<T, NT, 1>(gs.W, gs.rb, xq, xd, gs.y, ncols, gs.nrows, accumulate, skip, gs.b); }        \
    }
// item T1: q3_k / iq3_s one-token and q3_k dp4 few-token entries (bitwise == gemvq_impl / gemvq_nt_impl)
#define GEMVQ_T1_ENTRY(SUF, IMPL)                                                                \
    extern "C" __global__ void __launch_bounds__(256)                                            \
        gemvq_##SUF(GV_PARAMS) {          \
        { const GvSel gs = gv_sel(g); IMPL(gs.W, gs.rb, xq, xd, gs.y, ncols, gs.nrows, accumulate, skip, gs.b); }                           \
    }
#define GEMVQ_T1_Q3K_NT_ENTRY(NT)                                                                \
    extern "C" __global__ void __launch_bounds__(256)                                            \
        gemvq_nt##NT##_q3_k(const uint8_t *W, uint64_t row_bytes, const int8_t *xq,             \
                            const float *xd, float *y, int ncols, int nrows, int accumulate,    \
                            const int *skip) {                                                   \
        t1_q3k_nt_impl<NT>(W, row_bytes, xq, xd, y, ncols, nrows, accumulate, skip, blockIdx.x);             \
    } \
    extern "C" __global__ void __launch_bounds__(256)                                            \
        ggemvq_nt##NT##_q3_k(GV_PARAMS) {                                                   \
        { const GvSel gs = gv_sel(g); t1_q3k_nt_impl<NT>(gs.W, gs.rb, xq, xd, gs.y, ncols, gs.nrows, accumulate, skip, gs.b); }             \
    }
#define GEMVQ_NT_TYPES(NT)                \
    GEMVQ_W_NT_ENTRY(NT, q4_k, QT_Q4_K)     \
    GEMVQ_W_NT_ENTRY(NT, q5_k, QT_Q5_K)     \
    GEMVQ_W_NT_ENTRY(NT, iq4_xs, QT_IQ4_XS) \
    GEMVQ_W_NT_ENTRY(NT, mxfp4, QT_MXFP4)   \
    GEMVQ_NT_ENTRY(NT, q6_k, QT_Q6_K)     \
    GEMVQ_NT_ENTRY(NT, iq4_nl, QT_IQ4_NL) \
    GEMVQ_NT_ENTRY(NT, q8_0, QT_Q8_0)     \
    GEMVQ_T1_Q3K_NT_ENTRY(NT)             \
    GEMVQ_NT_ENTRY(NT, iq3_s, QT_IQ3_S)

GEMVQ_W_ENTRY(q4_k, QT_Q4_K)
GEMVQ_W_ENTRY(q5_k, QT_Q5_K)
GEMVQ_W_ENTRY(iq4_xs, QT_IQ4_XS)
GEMVQ_W_ENTRY(mxfp4, QT_MXFP4)
GEMVQ_ENTRY(q6_k, QT_Q6_K)
GEMVQ_ENTRY(iq4_nl, QT_IQ4_NL)
GEMVQ_ENTRY(q8_0, QT_Q8_0)
GEMVQ_T1_ENTRY(q3_k, t1_q3k_1_impl<4>)
GEMVQ_T1_ENTRY(iq3_s, t1_iq3s_1_impl<4>)
GEMVQ_NT_TYPES(2)
GEMVQ_NT_TYPES(3)
GEMVQ_NT_TYPES(4)
GEMVQ_NT_TYPES(5)
GEMVQ_NT_TYPES(6)
GEMVQ_NT_TYPES(7)
GEMVQ_NT_TYPES(8)
GEMVQ_NT_TYPES(9)
GEMVQ_NT_TYPES(10)
GEMVQ_NT_TYPES(11)
GEMVQ_NT_TYPES(12)
GEMVQ_NT_TYPES(13)
GEMVQ_NT_TYPES(14)
GEMVQ_NT_TYPES(15)
GEMVQ_NT_TYPES(16)

#define GEMVQ_MR_ENTRY(NT, R, SUF, T)                                                            \
    extern "C" __global__ void __launch_bounds__(256)                                            \
        gemvq_nt##NT##r##R##_##SUF(const uint8_t *W, uint64_t row_bytes, const int8_t *xq,      \
                                   const float *xd, float *y, int ncols, int nrows,              \
                                   int accumulate, const int *skip) {                            \
        gemvq_mr_impl<T, NT, R>(W, row_bytes, xq, xd, y, ncols, nrows, accumulate, skip, blockIdx.x);        \
    } \
    extern "C" __global__ void __launch_bounds__(256)                                            \
        ggemvq_nt##NT##r##R##_##SUF(GV_PARAMS) {                            \
        { const GvSel gs = gv_sel(g); gemvq_mr_impl<T, NT, R>(gs.W, gs.rb, xq, xd, gs.y, ncols, gs.nrows, accumulate, skip, gs.b); }        \
    }
#define GEMVQ_MR_TYPES(NT, R)                    \
    GEMVQ_MR_ENTRY(NT, R, q4_k, QT_Q4_K)         \
    GEMVQ_MR_ENTRY(NT, R, q5_k, QT_Q5_K)         \
    GEMVQ_MR_ENTRY(NT, R, iq4_xs, QT_IQ4_XS)     \
    GEMVQ_MR_ENTRY(NT, R, mxfp4, QT_MXFP4)       \
    GEMVQ_MR_ENTRY(NT, R, q6_k, QT_Q6_K)
GEMVQ_MR_TYPES(2, 2)
GEMVQ_MR_TYPES(2, 4)
GEMVQ_MR_TYPES(3, 2)
GEMVQ_MR_TYPES(3, 4)
GEMVQ_MR_TYPES(4, 2)
GEMVQ_MR_TYPES(4, 4)
GEMVQ_MR_TYPES(5, 2)
GEMVQ_MR_TYPES(5, 4)
GEMVQ_MR_TYPES(6, 2)
GEMVQ_MR_TYPES(6, 4)
GEMVQ_MR_TYPES(7, 2)
GEMVQ_MR_TYPES(7, 4)
GEMVQ_MR_TYPES(8, 2)
GEMVQ_MR_TYPES(8, 4)
GEMVQ_MR_TYPES(9, 2)
GEMVQ_MR_TYPES(9, 4)
GEMVQ_MR_TYPES(10, 2)
GEMVQ_MR_TYPES(10, 4)
GEMVQ_MR_TYPES(11, 2)
GEMVQ_MR_TYPES(11, 4)
GEMVQ_MR_TYPES(12, 2)
GEMVQ_MR_TYPES(12, 4)
GEMVQ_MR_TYPES(13, 2)
GEMVQ_MR_TYPES(13, 4)
GEMVQ_MR_TYPES(14, 2)
GEMVQ_MR_TYPES(14, 4)
GEMVQ_MR_TYPES(15, 2)
GEMVQ_MR_TYPES(15, 4)
GEMVQ_MR_TYPES(16, 2)
GEMVQ_MR_TYPES(16, 4)

